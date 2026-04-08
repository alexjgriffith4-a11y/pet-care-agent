"""
actions/food_safety.py
----------------------
Single-turn food and toxin safety handler.

Given a species and a food/substance, returns:
  - A structured risk level:  🟢 SAFE | 🟡 CAUTION | 🔴 TOXIC | ❓ UNKNOWN
  - A grounded explanation with [Source N] citations
  - A vet/poison-control call-to-action when the item is toxic or unknown

This replaces the _handle_food_safety stub in agent.py.

Public API
----------
    handle_food_safety(query, conversation_history, pet_context) -> dict

Return shape (matches agent.py contract):
    {
        "response": str,
        "intent":   "food_safety",
        "sources":  list[dict],   # [{title, url, score}, ...]
        "error":    str | None,
    }
"""

from __future__ import annotations

import os

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
        temperature=0.1,   # near-deterministic for safety-critical output
        max_tokens=512,
    )

_llm = _build_llm()

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a food and toxin safety specialist for a pet care assistant.
Your job is to assess whether a food, plant, or household substance is safe for a dog or cat.

Response format — always use this exact structure:

1. Risk label on its own line (choose exactly one):
   🟢 SAFE — generally safe in normal amounts
   🟡 CAUTION — safe only in small amounts, or with important caveats
   🔴 TOXIC — dangerous; avoid completely
   ❓ UNKNOWN — no reliable information available in the provided sources

2. Two to four sentences explaining the risk or safety, citing [Source N] for each claim.

3. If the label is 🟡 CAUTION, 🔴 TOXIC, or ❓ UNKNOWN, end with:
   "If your pet has already consumed this, contact your veterinarian or
   ASPCA Animal Poison Control at 888-426-4435 immediately."

Strict rules:
- Never claim to be a veterinarian.
- Never invent safety facts not present in the retrieved sources.
- If the sources do not cover the specific item asked about, use ❓ UNKNOWN
  and recommend a vet — do not extrapolate from similar items.
- If the species is not specified, answer for both dogs and cats separately.
"""

# ── Public handler ────────────────────────────────────────────────────────────

def handle_food_safety(
    query: str,
    conversation_history: list[dict],
    pet_context: dict | None,
) -> dict:
    """
    Handle a food or toxin safety question.

    Args:
        query:                The user's latest message.
        conversation_history: All prior turns, each {"role": ..., "content": ...}.
        pet_context:          Optional pet profile dict (name, species, breed, age).
                              If present, species is used to filter retrieval.

    Returns:
        {"response": str, "intent": "food_safety", "sources": list[dict], "error": str|None}
    """
    species_filter = _detect_species(query, pet_context)

    # No topic filter — food safety questions span both "nutrition" and "toxins"
    # topic categories in the vector store. Filtering to just one risks missing
    # the most relevant chunks (e.g. "is ibuprofen safe?" is in toxins, not nutrition).
    chunks = retrieve(
        query=query,
        top_k=5,
        species=species_filter,
        unique_sources=True,
    )

    context_block = format_context_for_prompt(chunks)
    messages = _build_messages(query, context_block, conversation_history)

    try:
        response = _llm.invoke(messages)
        answer = response.content.strip()
    except Exception as e:
        return _error_response(f"LLM call failed: {e}")

    return {
        "response": answer,
        "intent": "food_safety",
        "sources": _to_source_list(chunks),
        "error": None,
    }

# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_species(query: str, pet_context: dict | None) -> str | None:
    """Return 'dog', 'cat', or None (no filter) based on profile then query keywords."""
    if pet_context and "species" in pet_context:
        return pet_context["species"]
    q = query.lower()
    if any(w in q for w in ["dog", "puppy", "canine"]):
        return "dog"
    if any(w in q for w in ["cat", "kitten", "feline"]):
        return "cat"
    return None


def _build_messages(
    query: str,
    context_block: str,
    conversation_history: list[dict],
) -> list:
    """Assemble the LLM message list: system → context → history → query."""
    messages = [SystemMessage(content=_SYSTEM_PROMPT)]

    if context_block:
        messages.append(HumanMessage(
            content=f"Use the following sources to answer the question:\n\n{context_block}"
        ))

    # Cap at 6 turns (3 exchanges) to keep the context window predictable.
    for turn in conversation_history[-6:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=query.strip()))
    return messages


def _to_source_list(chunks: list[RetrievedChunk]) -> list[dict]:
    return [{"title": c.title, "url": c.url, "score": c.score} for c in chunks]


def _error_response(error_message: str) -> dict:
    return {
        "response": (
            "I ran into a problem checking food safety information. "
            "If this is urgent, please contact your veterinarian or "
            "ASPCA Animal Poison Control (888-426-4435) directly."
        ),
        "intent": "food_safety",
        "sources": [],
        "error": error_message,
    }
