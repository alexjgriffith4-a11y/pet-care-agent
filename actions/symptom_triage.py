"""
actions/symptom_triage.py
-------------------------
Multi-turn symptom triage handler.

Collects three pieces of information before generating a possible-causes list:
  Step 1 — species   (detected from context or asked explicitly)
  Step 2 — symptoms  (from initial message or follow-up)
  Step 3 — duration  ("How long has this been going on?")

Once all three are collected, the handler lists likely possible causes and
tells the owner to consult a vet. It does NOT diagnose or recommend treatments.

State is reconstructed from conversation_history on every call using a
marker-based approach (same pattern as pet_profile.py), with keyword
heuristics as fallback. This fixes the previous bug where markers in
is_triage_in_progress() did not match the actual question text.

Public API
----------
    handle_symptom_triage(query, conversation_history, pet_context) -> dict
    is_triage_in_progress(conversation_history) -> bool
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from retriever import retrieve, format_context_for_prompt, RetrievedChunk
from llm_config import build_llm

# ── LLM ───────────────────────────────────────────────────────────────────────
_llm = build_llm(temperature=0.2, max_tokens=2048)

# ── Triage state ──────────────────────────────────────────────────────────────

@dataclass
class TriageState:
    species:  str | None = None
    symptoms: str | None = None
    duration: str | None = None

    @property
    def is_complete(self) -> bool:
        return all([self.species, self.symptoms, self.duration])

    @property
    def next_missing_field(self) -> str | None:
        if not self.species:  return "species"
        if not self.symptoms: return "symptoms"
        if not self.duration: return "duration"
        return None


# ── System prompt ─────────────────────────────────────────────────────────────

_TRIAGE_SYSTEM_PROMPT = """\
You are a pet care assistant. An owner has described symptoms their pet is experiencing.

Here is what you know:
{state_summary}

{source_instruction}

Give a numbered list of 3–5 possible causes. For each, write one clear sentence \
describing what it is and why it fits the symptoms.{cite_inline}

After the numbered list, output the urgency level on its own line using EXACTLY one \
of these labels (include the emoji):
  🟢 MONITOR — symptoms are mild; watch for changes at home
  🟡 VET SOON — a vet visit is warranted within the next 24–48 hours
  🔴 VET NOW — symptoms are urgent; seek veterinary care immediately

Then, on its own final line, write:
  "Please consult your veterinarian for a proper diagnosis and treatment plan."

Do NOT diagnose the pet. Do NOT recommend specific treatments or medications. \
Do NOT speculate beyond what fits the symptoms described. Keep the tone calm and factual.
"""

# ── Symptom keywords ──────────────────────────────────────────────────────────
# Broad list so the keyword fallback captures common lay terms.

_SYMPTOM_KEYWORDS = [
    "vomit", "puke", "puking", "throw up", "threw up", "sick",
    "diarrhea", "diarrhoea", "loose stool", "runny stool",
    "limp", "limping", "letharg", "not eating", "won't eat", "refuses to eat",
    "scratch", "scratching", "bleed", "bleeding", "blood",
    "seizure", "convuls", "cough", "coughing", "sneez", "swell", "swollen",
    "shak", "trembl", "trembling", "collaps", "pale gum",
    "whimper", "cry", "yelp", "discharge", "loss of appetite",
    "not drinking", "won't drink", "pant", "panting",
    "laboured breath", "labored breath", "difficulty breathing",
    "weak", "tired", "lump", "mass", "bump", "rash",
    "itch", "itching", "hair loss", "bald", "wound", "injury", "hurt", "pain",
    "yellow", "green", "mucus", "foam", "foaming",
]

# ── Duration and weight regexes ───────────────────────────────────────────────

_DURATION_RE = re.compile(
    r"""
    (?:
        \d+\s*(?:minute|hour|day|week|month|year)s?   # "3 days", "2 hours"
      | since\s+\S+(?:\s+\S+)?                         # "since yesterday"
      | for\s+a(?:n)?\s+\w+                            # "for a day"
      | just\s+now
      | yesterday
      | this\s+(?:morning|afternoon|evening|night)
      | last\s+(?:night|week)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Follow-up questions ───────────────────────────────────────────────────────

# IMPORTANT: the marker strings in _TRIAGE_FOLLOWUP_MARKERS must be substrings
# of the actual question text below. is_triage_in_progress() relies on this.
_FOLLOW_UP_QUESTIONS: dict[str, str] = {
    "species":  "Is this your dog or your cat?",
    "symptoms": "What symptoms are you noticing exactly?",
    "duration": "How long has this been going on?",
}

# Substrings of the actual questions above — used by is_triage_in_progress().
_TRIAGE_FOLLOWUP_MARKERS = (
    "is this your dog or your cat",
    "what symptoms are you noticing",
    "how long has this been going on",
)


def is_triage_in_progress(conversation_history: list[dict] | None) -> bool:
    """
    Return True if the last assistant turn was a triage follow-up question.
    Used by agent.py to keep intent sticky across multi-turn triage flows.
    """
    if not conversation_history:
        return False
    for turn in reversed(conversation_history):
        if turn.get("role") == "assistant":
            content = (turn.get("content") or "").lower()
            return any(marker in content for marker in _TRIAGE_FOLLOWUP_MARKERS)
    return False


# ── State reconstruction ──────────────────────────────────────────────────────

def _reconstruct_state(
    current_query: str,
    history: list[dict],
    pet_context: dict | None,
) -> TriageState:
    """
    Rebuild TriageState from conversation history.

    Uses two passes:
      1. Marker-based: when we see an assistant question, the next user
         message is the answer to that field (same pattern as pet_profile.py).
      2. Keyword/regex fallback for fields still missing after pass 1.
    """
    state = TriageState()

    # Pre-populate from pet profile (most reliable source).
    if pet_context:
        state.species = pet_context.get("species")

    all_turns = list(history) + [{"role": "user", "content": current_query}]

    # ── Pass 1: marker-based extraction ──────────────────────────────────────
    pending_field: str | None = None

    for turn in all_turns:
        role    = turn.get("role", "")
        content = (turn.get("content") or "").strip()
        content_lower = content.lower()

        if role == "assistant":
            if "is this your dog or your cat" in content_lower:
                pending_field = "species"
            elif "what symptoms are you noticing" in content_lower:
                pending_field = "symptoms"
            elif "how long has this been going on" in content_lower:
                pending_field = "duration"

        elif role == "user" and pending_field and content:
            if pending_field == "species":
                low = content_lower
                if any(w in low for w in ["dog", "puppy", "canine"]):
                    state.species = state.species or "dog"
                elif any(w in low for w in ["cat", "kitten", "feline"]):
                    state.species = state.species or "cat"
            elif pending_field == "symptoms" and not state.symptoms:
                state.symptoms = content
            elif pending_field == "duration" and not state.duration:
                state.duration = content
            pending_field = None

    # ── Pass 2: fallback heuristics for fields still missing ─────────────────
    all_user_msgs = [t["content"] for t in all_turns if t.get("role") == "user"]
    full_text = " ".join(all_user_msgs)

    # Species fallback
    if not state.species:
        fl = full_text.lower()
        if any(w in fl for w in ["dog", "puppy", "canine"]):
            state.species = "dog"
        elif any(w in fl for w in ["cat", "kitten", "feline"]):
            state.species = "cat"

    # Symptoms fallback — collect any user message containing a symptom keyword
    if not state.symptoms:
        symptom_msgs = [
            msg.strip() for msg in all_user_msgs
            if any(kw in msg.lower() for kw in _SYMPTOM_KEYWORDS)
        ]
        if symptom_msgs:
            state.symptoms = "; ".join(dict.fromkeys(symptom_msgs))

    # Duration fallback — regex over full user text
    if not state.duration:
        m = _DURATION_RE.search(full_text)
        if m:
            state.duration = m.group(0).strip()

    return state


# ── Follow-up question generator ──────────────────────────────────────────────

def _ask_for_field(field: str, state: TriageState) -> str:
    """Return the next follow-up question."""
    base = _FOLLOW_UP_QUESTIONS.get(field, "Could you tell me a bit more?")
    return base


# ── Public handler ────────────────────────────────────────────────────────────

def handle_symptom_triage(
    query: str,
    conversation_history: list[dict],
    pet_context: dict | None,
) -> dict:
    """
    Multi-turn symptom handler. Collects species, symptoms, and duration,
    then lists possible causes and directs the owner to consult a vet.
    """
    # If we're NOT mid-triage (i.e. this is the opening turn of a new triage
    # flow, or the previous triage already completed), reconstruct state from
    # the current query only — don't bleed old symptoms/duration from history.
    if is_triage_in_progress(conversation_history):
        history_for_state = conversation_history
    else:
        history_for_state = []

    state = _reconstruct_state(query, history_for_state, pet_context)

    if not state.is_complete:
        follow_up = _ask_for_field(state.next_missing_field, state)
        return {
            "response": follow_up,
            "intent":   "symptom_triage",
            "sources":  [],
            "error":    None,
        }

    return _run_triage_assessment(state, conversation_history)


# ── Final assessment ──────────────────────────────────────────────────────────

def _run_triage_assessment(
    state: TriageState,
    conversation_history: list[dict],
) -> dict:
    """
    All fields collected. Retrieve relevant chunks and ask the LLM to list
    possible causes. No diagnosis, no urgency labels.
    """
    triage_query = f"{state.species} symptoms: {state.symptoms}. Duration: {state.duration}."

    retrieval_error: str | None = None
    try:
        chunks = retrieve(
            query=triage_query,
            top_k=5,
            species=state.species,
            unique_sources=True,
        )
        # Fallback: if species filter yields nothing, retry without it.
        if not chunks:
            chunks = retrieve(
                query=triage_query,
                top_k=5,
                unique_sources=True,
            )
    except Exception as e:
        chunks = []
        retrieval_error = f"retrieval_failed:{e}"

    context_block = format_context_for_prompt(chunks)

    state_summary = (
        f"  Species:  {state.species}\n"
        f"  Symptoms: {state.symptoms}\n"
        f"  Duration: {state.duration}"
    )

    if context_block:
        source_instruction = (
            "Retrieved sources are provided below. "
            "Cite them inline as [Source N] — weave citations naturally into sentences."
        )
        cite_inline = " Cite sources inline as [Source N] where relevant."
    else:
        source_instruction = (
            "No retrieved sources are available. "
            "Answer from general veterinary knowledge only. "
            "Do NOT use any [Source N] citations."
        )
        cite_inline = ""

    system_content = _TRIAGE_SYSTEM_PROMPT.format(
        state_summary=state_summary,
        source_instruction=source_instruction,
        cite_inline=cite_inline,
    )

    messages = [SystemMessage(content=system_content)]

    if context_block:
        messages.append(HumanMessage(
            content=f"Use these sources to inform your answer:\n\n{context_block}"
        ))

    for turn in conversation_history[-6:]:
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(
        content=(
            f"/no_think\n"
            f"My {state.species} has been showing: {state.symptoms} "
            f"for {state.duration}. What are the possible causes?"
        )
    ))

    try:
        response = _llm.invoke(messages)
        answer = response.content.strip()
    except Exception as e:
        return {
            "response": (
                "I ran into a problem generating the assessment. "
                "Please contact your veterinarian, especially if symptoms are severe."
            ),
            "intent": "symptom_triage",
            "sources": [],
            "error": f"LLM call failed: {e}",
        }

    return {
        "response": answer,
        "intent":   "symptom_triage",
        "sources":  [{"title": c.title, "url": c.url, "score": c.score} for c in chunks],
        "error":    retrieval_error,
    }
