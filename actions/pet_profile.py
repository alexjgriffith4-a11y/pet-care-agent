"""
actions/pet_profile.py
----------------------
Multi-turn pet profile collection and personalized care routine handler.

The profile flow collects four fields in order:
  Step 1 — name    ("What is your pet's name?")
  Step 2 — species ("Is [name] a dog or a cat?")
  Step 3 — age     ("How old is [name]?")
  Step 4 — breed   ("What breed is [name]?")

Once all four are collected the profile is saved to actions/profiles.json
and the handler delivers personalized care-routine advice using RAG.

On return visits (profile already exists for this profile_id) the handler
skips collection and answers care-routine questions with the stored profile
as context, personalising the retrieval and the LLM prompt.

Public API
----------
    handle_profile_turn(query, conversation_history, profile_id, session_state) -> dict

Return shape (adds to the standard agent.py contract):
    {
        "response":      str,
        "intent":        "care_routine",
        "sources":       list[dict],
        "error":         str | None,
        "profile":       dict,   # current profile dict (may be incomplete)
        "profile_saved": bool,   # True if the profile was written to disk this turn
    }
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from retriever import retrieve, format_context_for_prompt, RetrievedChunk
from llm_config import build_llm

# ── Config ────────────────────────────────────────────────────────────────────

_PROFILES_PATH = Path(__file__).parent / "profiles.json"

# ── LLM ───────────────────────────────────────────────────────────────────────
_llm = build_llm(temperature=0.3, max_tokens=512)

# ── System prompt ─────────────────────────────────────────────────────────────

_CARE_SYSTEM_PROMPT = """\
You are a personalized pet care advisor.

You are helping the owner of {pet_description}.

Using ONLY the provided source context, give practical, friendly care-routine
advice tailored to this specific pet. Cite sources using [Source N] notation.

Rules:
- Never claim to be a veterinarian.
- Never invent facts not present in the retrieved sources.
- Keep advice relevant to the pet's species, breed, and age where possible.
- If the context does not cover the question, say so honestly.
"""

# ── Profile I/O ───────────────────────────────────────────────────────────────

def _load_profiles() -> dict:
    """Load the full profiles store. Returns empty dict on any error."""
    try:
        if _PROFILES_PATH.exists():
            with _PROFILES_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_profile(profile_id: str, profile: dict) -> bool:
    """
    Persist a single profile into the profiles store.
    Returns True on success, False on failure.
    """
    try:
        store = _load_profiles()
        store[profile_id] = {
            **profile,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        _PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PROFILES_PATH.open("w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def load_profile(profile_id: str) -> dict | None:
    """
    Return the stored profile for profile_id, or None if it does not exist.
    Strips internal keys (_readme, _example) before returning.
    """
    store = _load_profiles()
    profile = store.get(profile_id)
    if not profile or not isinstance(profile, dict):
        return None
    # Skip placeholder/meta keys
    if not profile.get("species"):
        return None
    return profile

# ── Profile state ─────────────────────────────────────────────────────────────

# Marker phrases the handler uses in its follow-up questions.
# Used to detect where in the collection flow the conversation currently is.
_ASKING_NAME_MARKERS    = ("what is your pet's name", "what's your pet's name")
_ASKING_SPECIES_MARKERS = ("is", "a dog or a cat", "dog or cat")
_ASKING_AGE_MARKERS     = ("how old is", "what is", "age")
_ASKING_BREED_MARKERS   = ("what breed", "breed is")


def _reconstruct_profile_from_history(
    current_query: str,
    history: list[dict],
) -> dict:
    """
    Scan conversation history to rebuild the in-progress profile dict.

    Tracks which question the assistant most recently asked and uses the
    following user turn as that field's answer. This is robust to the
    classifier occasionally mis-routing follow-up answers.
    """
    profile: dict = {}

    all_turns = list(history) + [{"role": "user", "content": current_query}]

    # Walk through pairs: when we see an assistant question, the next user
    # message is the answer to that question.
    pending_field: str | None = None

    for turn in all_turns:
        role    = turn.get("role", "")
        content = (turn.get("content") or "").lower().strip()

        if role == "assistant":
            if any(m in content for m in _ASKING_NAME_MARKERS):
                pending_field = "name"
            elif any(m in content for m in _ASKING_BREED_MARKERS):
                pending_field = "breed"
            elif any(m in content for m in _ASKING_AGE_MARKERS):
                pending_field = "age"
            elif "dog or cat" in content or "a dog or a cat" in content:
                pending_field = "species"

        elif role == "user" and pending_field and content:
            raw = turn.get("content", "").strip()
            if pending_field == "species":
                low = raw.lower()
                if any(w in low for w in ["dog", "puppy", "canine"]):
                    profile["species"] = "dog"
                elif any(w in low for w in ["cat", "kitten", "feline"]):
                    profile["species"] = "cat"
                else:
                    profile["species"] = raw  # store as-is; will re-ask if needed
            elif pending_field == "age":
                profile["age"] = _parse_age(raw)
            elif pending_field in ("name", "breed"):
                profile[pending_field] = raw.strip(" .,!?\"'")
            pending_field = None

    # Fallback: scan all user text for species keywords if still missing
    if "species" not in profile:
        full = " ".join(t.get("content", "") for t in all_turns if t.get("role") == "user")
        fl = full.lower()
        if any(w in fl for w in ["dog", "puppy", "canine"]):
            profile["species"] = "dog"
        elif any(w in fl for w in ["cat", "kitten", "feline"]):
            profile["species"] = "cat"

    return profile


def _parse_age(raw: str) -> str:
    """Normalise an age answer to a display string, e.g. '3' → '3 years'."""
    raw = raw.strip()
    if re.match(r"^\d+(\.\d+)?$", raw):
        return f"{raw} years"
    return raw


def _next_missing_field(profile: dict) -> str | None:
    """Return the next field to collect, in order. None if complete."""
    for field in ("name", "species", "age", "breed"):
        if not profile.get(field):
            return field
    return None


def _is_profile_complete(profile: dict) -> bool:
    return _next_missing_field(profile) is None

# ── Follow-up questions ───────────────────────────────────────────────────────

def _ask_for_field(field: str, profile: dict) -> str:
    name = profile.get("name", "your pet")
    species = profile.get("species", "pet")

    questions = {
        "name":    "I'd love to personalize your experience! What is your pet's name?",
        "species": f"Is {name} a dog or a cat?",
        "age":     f"How old is {name}?",
        "breed":   f"What breed is {name}?",
    }
    return questions.get(field, "Could you tell me a bit more about your pet?")


# Distinctive phrases that signal the profile flow is active.
# Used by agent.py to keep routing sticky.
PROFILE_FOLLOWUP_MARKERS = (
    "what is your pet's name",
    "what's your pet's name",
    "a dog or a cat",
    "how old is",
    "what breed is",
)


def is_profile_in_progress(conversation_history: list[dict] | None) -> bool:
    """
    Return True if the most recent assistant turn was a profile follow-up
    question — i.e. the agent is mid-collection.
    """
    if not conversation_history:
        return False
    for turn in reversed(conversation_history):
        if turn.get("role") == "assistant":
            content = (turn.get("content") or "").lower()
            return any(marker in content for marker in PROFILE_FOLLOWUP_MARKERS)
    return False

# ── Care-routine RAG response ─────────────────────────────────────────────────

def _build_pet_description(profile: dict) -> str:
    parts = []
    if profile.get("name"):
        parts.append(profile["name"])
    if profile.get("breed") and profile.get("species"):
        parts.append(f"a {profile['breed']} {profile['species']}")
    elif profile.get("species"):
        parts.append(f"a {profile['species']}")
    if profile.get("age"):
        parts.append(f"aged {profile['age']}")
    return " — ".join(parts) if parts else "your pet"


def _run_care_routine_response(
    query: str,
    conversation_history: list[dict],
    profile: dict,
) -> dict:
    """
    All profile fields collected. Retrieve relevant chunks and generate
    personalized care routine advice.
    """
    species = profile.get("species")
    breed   = profile.get("breed", "")
    name    = profile.get("name", "your pet")

    # Build a rich query that includes breed/age context for better retrieval
    rich_query = query
    if breed and species:
        rich_query = f"{breed} {species}: {query}"
    elif species:
        rich_query = f"{species}: {query}"

    try:
        chunks = retrieve(
            query=rich_query,
            top_k=5,
            species=species,
            unique_sources=True,
        )
    except Exception:
        chunks = []

    context_block = format_context_for_prompt(chunks)
    pet_description = _build_pet_description(profile)

    system_content = _CARE_SYSTEM_PROMPT.format(pet_description=pet_description)
    if not context_block:
        system_content += (
            "\n\nNo retrieved sources are available right now. "
            "Answer from your general pet care knowledge, but be honest about uncertainty "
            "and always recommend consulting a vet for medical decisions."
        )

    messages = [
        SystemMessage(content=system_content)
    ]

    if context_block:
        messages.append(HumanMessage(
            content=f"Use the following sources to answer the question:\n\n{context_block}"
        ))

    for turn in conversation_history[-6:]:
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=query.strip()))

    try:
        response = _llm.invoke(messages)
        answer   = response.content.strip()
    except Exception as e:
        return _error_response(f"LLM call failed: {e}")

    return {
        "response":      answer,
        "intent":        "care_routine",
        "sources":       [{"title": c.title, "url": c.url, "score": c.score} for c in chunks],
        "error":         None,
        "profile":       profile,
        "profile_saved": False,
    }

# ── Public handler ────────────────────────────────────────────────────────────

def handle_profile_turn(
    query: str,
    conversation_history: list[dict],
    profile_id: str = "default",
    session_state: dict | None = None,
) -> dict:
    """
    Handle one turn of the care-routine / pet-profile flow.

    On the first visit: collects name, species, age, and breed one field at a
    time, then saves the profile and delivers the first personalized answer.

    On return visits: loads the existing profile and answers the care-routine
    question immediately using RAG, with the pet's details in the prompt.

    Args:
        query:                The user's latest message.
        conversation_history: All prior turns in the session.
        profile_id:           Key for loading/saving in profiles.json.
        session_state:        Reserved for future use (ignored currently).

    Returns:
        Standard response dict plus "profile" and "profile_saved" keys.
    """
    # ── Return visit: profile already exists ──────────────────────────────────
    existing = load_profile(profile_id)
    if existing:
        result = _run_care_routine_response(query, conversation_history, existing)
        result["profile"]       = existing
        result["profile_saved"] = False
        return result

    # ── First visit: collect profile fields ───────────────────────────────────
    profile = _reconstruct_profile_from_history(query, conversation_history)

    if not _is_profile_complete(profile):
        next_field = _next_missing_field(profile)
        follow_up  = _ask_for_field(next_field, profile)
        return {
            "response":      follow_up,
            "intent":        "care_routine",
            "sources":       [],
            "error":         None,
            "profile":       profile,
            "profile_saved": False,
        }

    # ── All fields collected — save profile and answer ────────────────────────
    saved = _save_profile(profile_id, profile)

    # Deliver the first personalized response immediately
    result = _run_care_routine_response(query, conversation_history, profile)
    result["profile"]       = profile
    result["profile_saved"] = saved
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _error_response(error_message: str) -> dict:
    return {
        "response": (
            "I ran into a problem retrieving care-routine information. "
            "Please try again in a moment."
        ),
        "intent":        "care_routine",
        "sources":       [],
        "error":         error_message,
        "profile":       {},
        "profile_saved": False,
    }
