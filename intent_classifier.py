"""
intent_classifier.py
--------------------
Classifies a user query into one of five intents so the agent loop
can route it to the right handler.

Intents
-------
  food_safety     — "can my dog eat grapes?", "is chocolate safe for cats?"
  symptom_triage  — "my dog is limping", "my cat won't eat and is lethargic"
  care_routine    — "how often should I groom a golden retriever?"
  general_qa      — any other in-scope pet care question
  out_of_scope    — nothing to do with pet care

Design notes
------------
- Uses a few-shot prompt so the LLM has concrete examples to pattern-match
  against, rather than relying on the model's interpretation of vague labels.
- The LLM is asked to return ONLY the label — no explanation, no punctuation.
  This makes parsing trivial and eliminates ambiguity.
- The classifier is a pure function: same input → same output (modulo LLM
  sampling variance). No side effects, no state. Easy to unit-test.
- We accept `conversation_history` so the classifier can resolve pronouns
  ("is that safe for him?") by seeing what was said before.
"""
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import SystemMessage, HumanMessage
from llm_config import build_llm

# ── Constants ─────────────────────────────────────────────────────────────────

# The five labels the classifier may return. Anything else is treated as a
# parse failure and falls back to "general_qa" so the agent never hard-crashes.
VALID_INTENTS = {
    "food_safety",
    "symptom_triage",
    "care_routine",
    "general_qa",
    "out_of_scope",
}

# Few-shot examples teach the model the boundary between intents.
# Each example is a (query, label) pair written in plain English so the model
# can generalize to paraphrases it has never seen.
_FEW_SHOT_EXAMPLES = """
Examples — study these carefully before classifying:

Query: can my dog eat grapes?
Label: food_safety

Query: is onion powder toxic to cats?
Label: food_safety

Query: my dog ate some chocolate an hour ago
Label: food_safety

Query: my cat has been vomiting for two days
Label: symptom_triage

Query: my dog is limping on his back leg
Label: symptom_triage

Query: she seems lethargic and won't drink water
Label: symptom_triage

Query: how often should I bathe a golden retriever?
Label: care_routine

Query: what vaccinations does my kitten need?
Label: care_routine

Query: how much should a 10-pound cat eat per day?
Label: care_routine

Query: what is the average lifespan of a labrador?
Label: general_qa

Query: do cats need annual vet checkups?
Label: general_qa

Query: hi
Label: general_qa

Query: hello there
Label: general_qa

Query: hey
Label: general_qa

Query: thanks
Label: general_qa

Query: what is the capital of France?
Label: out_of_scope

Query: write me a poem about my dog
Label: out_of_scope

Query: how do I fix a leaky faucet?
Label: out_of_scope
""".strip()

_SYSTEM_PROMPT = f"""You are an intent classifier for a pet care support agent.

Your job is to read a user query and return EXACTLY ONE label from this list:
  food_safety | symptom_triage | care_routine | general_qa | out_of_scope

Rules:
- Return only the label. No explanation, no punctuation, no extra words.
- If the query mentions food, ingredients, or substances a pet ate or might eat → food_safety
- If the query describes a symptom, illness, injury, or unusual behaviour → symptom_triage
- If the query asks about grooming, feeding schedules, exercise, or preventive care → care_routine
- If the query is about pets but does not fit the above → general_qa
- Greetings (hi, hello, hey, thanks, etc.) and short social messages → general_qa
- If the query has nothing to do with pet care and is not a greeting → out_of_scope

{_FEW_SHOT_EXAMPLES}
"""


# ── LLM setup ─────────────────────────────────────────────────────────────────

# temperature=0 for deterministic classification; max_tokens=10 — label only.
_llm = build_llm(temperature=0, max_tokens=10)


# ── Public API ────────────────────────────────────────────────────────────────

def classify(query: str, conversation_history: list[dict] | None = None) -> str:
    """
    Classify a user query into one of the five intents.

    Args:
        query:                The user's latest message.
        conversation_history: Optional list of prior turns, each a dict with
                              keys "role" ("user" | "assistant") and "content".
                              Passed so the model can resolve pronouns like
                              "is that safe for him?" using prior context.

    Returns:
        One of: "food_safety", "symptom_triage", "care_routine",
                "general_qa", "out_of_scope".
        Never raises — falls back to "general_qa" on any failure.

    Why does this never raise?
        The agent loop must always route somewhere. A classification failure
        is not a reason to crash the agent; "general_qa" is a safe fallback
        that will attempt RAG retrieval and return a grounded answer.
    """
    if not query or not query.strip():
        # Empty query cannot be classified — treat as out_of_scope so the
        # agent prompts the user to ask a real question.
        return "out_of_scope"

    # Build the message list for the LLM.
    # System message contains the rules + few-shot examples.
    messages: list = [SystemMessage(content=_SYSTEM_PROMPT)]

    # Optionally prepend recent conversation history so the classifier can
    # resolve references to previously mentioned pets or substances.
    # We cap at the last 4 turns to keep the context window small — this is
    # a classifier, not a conversation engine.
    if conversation_history:
        for turn in conversation_history[-4:]:
            role    = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            # We skip assistant turns here — the classifier only needs to
            # see what the user said to resolve pronouns.

    messages.append(HumanMessage(content=f"Query: {query.strip()}\nLabel:"))

    try:
        response = _llm.invoke(messages)
        # Strip whitespace and lowercase so "Food_Safety\n" → "food_safety".
        label = response.content.strip().lower()

        if label not in VALID_INTENTS:
            # The model returned something unexpected (e.g. a sentence).
            # Log it so you can improve the prompt, then fall back gracefully.
            print(
                f"[intent_classifier] Unexpected label {label!r} for query "
                f"{query!r}. Falling back to 'general_qa'."
            )
            return "general_qa"

        return label

    except Exception as e:
        # ⚠️ Never let a classifier failure crash the agent.
        # Log the error so you can diagnose it, but keep the agent alive.
        print(f"[intent_classifier] LLM call failed: {e}. Falling back to 'general_qa'.")
        return "general_qa"
