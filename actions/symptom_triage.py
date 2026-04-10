"""
actions/symptom_triage.py
-------------------------
Multi-turn symptom triage handler with a lightweight state machine.

The triage flow collects four pieces of information before generating
a final urgency assessment:

  Step 1 — species   ("Is this for a dog or a cat?")
  Step 2 — symptoms  ("Can you describe the symptoms?")
  Step 3 — duration  ("How long has this been going on?")
  Step 4 — weight    ("Approximately how much does your pet weigh?")

Once all four are collected, the handler retrieves relevant chunks and
asks the LLM to classify urgency as:
  🟢 MONITOR | 🟡 VET SOON | 🔴 VET NOW

State is reconstructed from conversation_history on every call — no
external storage needed. This keeps the handler a pure function and
makes it straightforward to unit-test.

Public API
----------
    handle_symptom_triage(query, conversation_history, pet_context) -> dict
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from retriever import retrieve, format_context_for_prompt, RetrievedChunk

# ── LLM ───────────────────────────────────────────────────────────────────────

def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen3-30b-a3b-fp8",
        base_url="https://rsm-8430-finalproject.bjlkeng.io/v1",
        api_key=os.environ.get("RSM_API_KEY", "no-key"),
        temperature=0.2,
        max_tokens=600,
    )

_llm = _build_llm()

# ── Triage state ───────────────────────────────────────────────────────────────

@dataclass
class TriageState:
    """
    Represents what the triage handler has collected so far in a session.

    Fields are collected in order: species → symptoms → duration → weight.
    The state machine only progresses to the next field once the current
    one is non-None.
    """
    species:  str | None = None
    symptoms: str | None = None
    duration: str | None = None
    weight:   str | None = None

    @property
    def is_complete(self) -> bool:
        """True when all four fields have been collected."""
        return all([self.species, self.symptoms, self.duration, self.weight])

    @property
    def next_missing_field(self) -> str | None:
        """Return the name of the first field still needed, in collection order."""
        if not self.species:  return "species"
        if not self.symptoms: return "symptoms"
        if not self.duration: return "duration"
        if not self.weight:   return "weight"
        return None


# ── System prompt (used only for the final assessment turn) ───────────────────

_TRIAGE_SYSTEM_PROMPT = """\
You are a symptom triage specialist for a pet care assistant.

You have collected the following information about the pet:
{state_summary}

Based ONLY on the provided source context, classify the urgency as exactly one of:
  🟢 MONITOR — normal variation; watch at home, no immediate action needed
  🟡 VET SOON — concerning; schedule a vet visit within 24–48 hours
  🔴 VET NOW — potentially serious; seek emergency veterinary care immediately

Response format (follow this exactly):
1. Urgency label on the first line (e.g. "🔴 VET NOW").
2. Two to four sentences explaining your reasoning, citing [Source N] for each claim.
3. One to two specific things the owner should do or watch for right now.
4. Close with:
   "⚠️ This is general information only. Please consult a licensed veterinarian
   for medical advice."

If the context does not contain enough information to assess the specific
symptoms described, say so honestly and default to recommending a vet visit.
"""

# ── Public handler ─────────────────────────────────────────────────────────────

def handle_symptom_triage(
    query: str,
    conversation_history: list[dict],
    pet_context: dict | None,
) -> dict:
    """
    Multi-turn symptom triage. Prompts for species, symptoms, duration, and weight
    one at a time, then generates a grounded urgency classification.

    Args:
        query:                The user's latest message.
        conversation_history: All prior turns in the session.
        pet_context:          Optional pet profile dict (name, species, breed, age, weight).
                              If present, pre-populates species and weight.

    Returns:
        Standard agent response dict. While collecting info, "response" is a
        follow-up question. Once all fields are collected, it is a full triage report.
    """
    # Rebuild state from the full history + current query on every call.
    # This is O(n) in history length but trivially fast; the payoff is
    # a stateless, easily-testable handler with no external session store.
    state = _reconstruct_state(query, conversation_history, pet_context)

    if not state.is_complete:
        follow_up = _ask_for_field(state.next_missing_field, state)
        return {
            "response": follow_up,
            "intent": "symptom_triage",
            "sources": [],
            "error": None,
        }

    return _run_triage_assessment(state, conversation_history)


# ── State reconstruction ───────────────────────────────────────────────────────

# Keywords that strongly suggest a symptom is being described.
_SYMPTOM_KEYWORDS = [
    "vomit", "diarrhea", "diarrhoea", "limp", "letharg", "not eating",
    "won't eat", "refuses to eat", "scratch", "bleed", "seizure", "convuls",
    "cough", "sneez", "swell", "shak", "trembl", "collaps", "pale gum",
    "whimper", "cry", "yelp", "discharge", "loss of appetite", "drink",
    "urinat", "pant", "laboured breath", "labored breath", "weak", "tired",
    "lump", "mass", "bump", "rash", "itch", "hair loss", "bald", "wound",
]

def _reconstruct_state(
    current_query: str,
    history: list[dict],
    pet_context: dict | None,
) -> TriageState:
    """
    Rebuild the TriageState by scanning all user messages in the session.

    Uses regex and keyword heuristics rather than an LLM call — keeps
    latency low since this runs on every turn before the main LLM call.
    """
    state = TriageState()

    # ── Pre-populate from pet profile (most reliable source) ─────────────────
    if pet_context:
        state.species = pet_context.get("species")  # "dog" or "cat"
        raw_weight = str(pet_context.get("weight", "") or "")
        state.weight = _normalise_weight(raw_weight)

    # Collect all user messages (history + current query) for scanning.
    all_user_msgs = [
        t["content"] for t in history if t.get("role") == "user"
    ] + [current_query]

    full_text = " ".join(all_user_msgs)

    # ── Species ───────────────────────────────────────────────────────────────
    if not state.species:
        fl = full_text.lower()
        if any(w in fl for w in ["dog", "puppy", "canine"]):
            state.species = "dog"
        elif any(w in fl for w in ["cat", "kitten", "feline"]):
            state.species = "cat"

    # ── Symptoms ──────────────────────────────────────────────────────────────
    # Collect every user message that mentions a symptom keyword, then join
    # them so the final triage query is as complete as possible.
    symptom_msgs = []
    for msg in all_user_msgs:
        msg_lower = msg.lower()
        if any(kw in msg_lower for kw in _SYMPTOM_KEYWORDS):
            symptom_msgs.append(msg.strip())

    if symptom_msgs:
        # De-duplicate while preserving order.
        seen = set()
        unique = []
        for m in symptom_msgs:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        state.symptoms = "; ".join(unique)

    # ── Duration ─────────────────────────────────────────────────────────────
    # Matches expressions like "2 days", "since yesterday", "for a week", etc.
    duration_re = re.compile(
        r"""
        (?:
            \d+\s*(?:minute|hour|day|week|month|year)s?  # "3 days", "2 hours"
          | since\s+\S+(?:\s+\S+)?                        # "since yesterday", "since this morning"
          | for\s+a(?:n)?\s+\w+                           # "for a day", "for an hour"
          | just\s+now                                     # "just now"
          | yesterday                                      # "yesterday"
          | this\s+(?:morning|afternoon|evening|night)    # "this morning"
          | last\s+(?:night|week)                         # "last night"
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )
    match = duration_re.search(full_text)
    if match:
        state.duration = match.group(0).strip()

    # ── Weight ────────────────────────────────────────────────────────────────
    if not state.weight:
        weight_re = re.compile(
            r"\d+(?:\.\d+)?\s*(?:lb|lbs|pound|pounds|kg|kilogram|kilograms)",
            re.IGNORECASE,
        )
        match = weight_re.search(full_text)
        if match:
            state.weight = match.group(0).strip()

    return state


def _normalise_weight(raw: str) -> str | None:
    """
    Coerce a weight value from a pet profile to a display string.
    Returns None if the raw value is empty or clearly unparseable.
    """
    raw = raw.strip()
    if not raw or raw.lower() in ("none", "unknown", ""):
        return None
    # If it's a bare number (no unit), assume lbs.
    if re.match(r"^\d+(\.\d+)?$", raw):
        return f"{raw} lbs"
    return raw


# ── Follow-up question generation ─────────────────────────────────────────────

_FOLLOW_UP_QUESTIONS: dict[str, str] = {
    "species":  "To help assess the situation — is this for a dog or a cat?",
    "symptoms": "Can you describe the symptoms you're seeing in a bit more detail?",
    "duration": "How long has your pet been showing these symptoms?",
    "weight":   "Approximately how much does your pet weigh? (This helps assess severity.)",
}

# Distinctive phrases the triage handler produces in its follow-up questions.
# If the last assistant turn contains one of these, the next user message is
# almost certainly a reply to an in-flight triage flow — even if it reads like
# "about 20 lbs" on its own and the classifier would otherwise send it to
# general_qa or out_of_scope. Used by `is_triage_in_progress()`.
_TRIAGE_FOLLOWUP_MARKERS = (
    "is this for a dog or a cat",
    "describe the symptoms",
    "how long has your pet",
    "how much does your pet weigh",
)


def is_triage_in_progress(conversation_history: list[dict] | None) -> bool:
    """
    Return True if the most recent assistant turn was a triage follow-up
    question — i.e. the agent is mid-triage and waiting for a slot answer.

    Why this exists: the intent classifier looks at each query in near-isolation
    and often mis-routes bare follow-up answers like "about 20 lbs" or
    "for 2 days" to general_qa or out_of_scope. The agent loop consults this
    helper to keep the routing sticky across a triage conversation.
    """
    if not conversation_history:
        return False
    for turn in reversed(conversation_history):
        if turn.get("role") == "assistant":
            content = (turn.get("content") or "").lower()
            return any(marker in content for marker in _TRIAGE_FOLLOWUP_MARKERS)
    return False

def _ask_for_field(field: str, state: TriageState) -> str:
    """
    Generate a contextual follow-up question for the next missing field.
    Includes a brief recap of what's already been collected so the
    multi-turn conversation feels natural rather than robotic.
    """
    base_question = _FOLLOW_UP_QUESTIONS.get(field, "Could you tell me a bit more?")

    recap_parts = []
    if state.species:
        recap_parts.append(f"I see this is for your {state.species}")
    if state.symptoms:
        # Truncate long symptom descriptions in the recap.
        short_symptoms = state.symptoms[:60]
        if len(state.symptoms) > 60:
            short_symptoms += "..."
        recap_parts.append(f"who is experiencing {short_symptoms}")

    recap = (", ".join(recap_parts) + ". ") if recap_parts else ""
    return f"{recap}{base_question}"


# ── Final triage assessment ────────────────────────────────────────────────────

def _run_triage_assessment(
    state: TriageState,
    conversation_history: list[dict],
) -> dict:
    """
    All four fields have been collected. Retrieve relevant chunks and
    ask the LLM for a grounded urgency classification.
    """
    # Build a rich retrieval query from all collected fields.
    triage_query = (
        f"{state.species} symptoms: {state.symptoms}. "
        f"Duration: {state.duration}. Weight: {state.weight}."
    )

    chunks = retrieve(
        query=triage_query,
        top_k=5,
        species=state.species,
        unique_sources=True,
    )
    context_block = format_context_for_prompt(chunks)

    state_summary = (
        f"  Species:  {state.species}\n"
        f"  Symptoms: {state.symptoms}\n"
        f"  Duration: {state.duration}\n"
        f"  Weight:   {state.weight}"
    )

    messages = [SystemMessage(content=_TRIAGE_SYSTEM_PROMPT.format(state_summary=state_summary))]

    if context_block:
        messages.append(HumanMessage(
            content=f"Use the following sources to inform your assessment:\n\n{context_block}"
        ))

    # Inject recent history so the LLM has the full conversational context.
    for turn in conversation_history[-6:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Final assessment request that summarises all collected fields.
    messages.append(HumanMessage(
        content=(
            f"Please assess urgency for my {state.species} "
            f"showing: {state.symptoms} "
            f"for {state.duration}. "
            f"Weight: {state.weight}."
        )
    ))

    try:
        response = _llm.invoke(messages)
        answer = response.content.strip()
    except Exception as e:
        return {
            "response": (
                "I ran into a problem generating the triage assessment. "
                "If this is an emergency, please contact a veterinarian or "
                "emergency animal hospital immediately."
            ),
            "intent": "symptom_triage",
            "sources": [],
            "error": f"LLM call failed: {e}",
        }

    return {
        "response": answer,
        "intent": "symptom_triage",
        "sources": [{"title": c.title, "url": c.url, "score": c.score} for c in chunks],
        "error": None,
    }
