"""
app.py
------
Streamlit chat UI for the Pet Care Support Agent.

Run with:
    streamlit run app.py

Session state keys:
    user                   — dict | None  (logged-in user: username, display_name)
    conversation_history   — list[{"role": str, "content": str}]
    triage_state           — dict | None  (symptom triage in-flight state)
    profile_session_state  — dict | None  (profile collection in-flight state)
    profile_id             — str          (which profile to load/save — set to username)
    pet_context            — dict | None  (loaded profile for the LLM prompt)
"""

import streamlit as st
from auth import register, login
from agent import run_turn
from actions.pet_profile import load_profile

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pet Care Assistant",
    page_icon="🐾",
    layout="wide",
)

# ── Auth gate ─────────────────────────────────────────────────────────────────

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("🐾 Pet Care Assistant")
    st.caption("Your personal AI companion for dog and cat care.")
    st.divider()

    tab_login, tab_signup = st.tabs(["Login", "Create Account"])

    with tab_login:
        with st.form("login_form"):
            st.subheader("Welcome back")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                if not username or not password:
                    st.error("Please enter your username and password.")
                else:
                    user, error = login(username, password)
                    if user:
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error(error)

    with tab_signup:
        with st.form("signup_form"):
            st.subheader("Create your account")
            display_name = st.text_input("Display Name", placeholder="e.g. Alex")
            new_username = st.text_input("Username", placeholder="e.g. alex123")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    ok, error = register(new_username, display_name, new_password)
                    if ok:
                        user, _ = login(new_username, new_password)
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error(error)

    st.stop()

# ── Session state defaults (after login) ─────────────────────────────────────

user = st.session_state.user

if "profile_id" not in st.session_state:
    st.session_state.profile_id = user["username"]
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "triage_state" not in st.session_state:
    st.session_state.triage_state = None
if "profile_session_state" not in st.session_state:
    st.session_state.profile_session_state = None
if "pet_context" not in st.session_state:
    st.session_state.pet_context = load_profile(st.session_state.profile_id)

# ── Helpers ───────────────────────────────────────────────────────────────────

_RISK_COLORS = {
    "SAFE":    ("🟢", "green"),
    "CAUTION": ("🟡", "orange"),
    "TOXIC":   ("🔴", "red"),
    "UNKNOWN": ("❓", "gray"),
}

_URGENCY_COLORS = {
    "MONITOR":  ("🟢", "green"),
    "VET SOON": ("🟡", "orange"),
    "VET NOW":  ("🔴", "red"),
}

_INTENT_LABELS = {
    "food_safety":    "Food Safety",
    "symptom_triage": "Symptom Triage",
    "care_routine":   "Care Routine",
    "general_qa":     "General Q&A",
    "out_of_scope":   "Out of Scope",
}


def _extract_risk_level(response_text: str) -> str | None:
    for label in ("TOXIC", "CAUTION", "SAFE", "UNKNOWN"):
        if label in response_text.upper():
            return label
    return None


def _extract_urgency(response_text: str) -> str | None:
    for label in ("VET NOW", "VET SOON", "MONITOR"):
        if label in response_text.upper():
            return label
    return None


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, 1):
            title = src.get("title", "Unknown")
            url   = src.get("url", "")
            score = src.get("score")
            score_str = f"  —  relevance: {score:.2f}" if score is not None else ""
            if url:
                st.markdown(f"**[{i}]** [{title}]({url}){score_str}")
            else:
                st.markdown(f"**[{i}]** {title}{score_str}")


def _render_intent_badge(intent: str) -> None:
    label = _INTENT_LABELS.get(intent, intent)
    st.caption(f"Intent: **{label}**")


def _render_risk_badge(response_text: str, intent: str) -> None:
    if intent == "food_safety":
        level = _extract_risk_level(response_text)
        if level:
            emoji, color = _RISK_COLORS.get(level, ("❓", "gray"))
            st.markdown(
                f"<span style='color:{color};font-weight:bold'>{emoji} {level}</span>",
                unsafe_allow_html=True,
            )
    elif intent == "symptom_triage":
        urgency = _extract_urgency(response_text)
        if urgency:
            emoji, color = _URGENCY_COLORS.get(urgency, ("❓", "gray"))
            st.markdown(
                f"<span style='color:{color};font-weight:bold'>{emoji} {urgency}</span>",
                unsafe_allow_html=True,
            )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🐾 Pet Profile")

    profile = st.session_state.pet_context
    if profile:
        name    = profile.get("name", "Your pet")
        species = profile.get("species", "").capitalize()
        breed   = profile.get("breed", "")
        age     = profile.get("age", "")

        st.success(f"**{name}**")
        cols = st.columns(2)
        cols[0].metric("Species", species or "—")
        cols[1].metric("Age", age or "—")
        if breed:
            st.caption(f"Breed: {breed}")

        if st.button("Reset profile", use_container_width=True):
            st.session_state.pet_context            = None
            st.session_state.profile_session_state  = None
            st.session_state.conversation_history   = []
            st.session_state.triage_state           = None
            try:
                from actions.pet_profile import _PROFILES_PATH
                import json
                if _PROFILES_PATH.exists():
                    data = json.loads(_PROFILES_PATH.read_text())
                    data.pop(st.session_state.profile_id, None)
                    _PROFILES_PATH.write_text(json.dumps(data, indent=2))
            except Exception:
                pass
            st.rerun()
    else:
        st.info(
            "No pet profile yet.\n\n"
            "Ask a care routine question (e.g. *how often should I groom my dog?*) "
            "and I'll collect your pet's details."
        )

    st.divider()

    # ── User info & logout ────────────────────────────────────────────────────
    display_name = user.get("display_name") or user.get("username", "")
    st.caption(f"Logged in as **{display_name}**")

    if st.button("Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.conversation_history  = []
        st.session_state.triage_state          = None
        st.session_state.profile_session_state = None
        st.rerun()

    st.caption("Powered by Qwen3 + ChromaDB")

# ── Main chat area ────────────────────────────────────────────────────────────

display_name = user.get("display_name") or user.get("username", "")
st.title("Pet Care Assistant")
st.caption(
    f"Hi **{display_name}**! Ask about food safety, symptoms, or care routines "
    "for your dog or cat. I'm not a vet — always consult a professional for medical concerns."
)

# Render existing conversation history
for turn in st.session_state.conversation_history:
    role    = turn["role"]
    content = turn["content"]
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant":
            sources = turn.get("sources", [])
            intent  = turn.get("intent", "")
            _render_risk_badge(content, intent)
            _render_sources(sources)
            _render_intent_badge(intent)

# Chat input
user_input = st.chat_input("Ask about your pet...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_turn(
                query=user_input,
                conversation_history=st.session_state.conversation_history,
                pet_context=st.session_state.pet_context,
                triage_state=st.session_state.triage_state,
                profile_id=st.session_state.profile_id,
                profile_session_state=st.session_state.profile_session_state,
            )

        response_text = result.get("response", "")
        intent        = result.get("intent", "general_qa")
        sources       = result.get("sources", [])
        error         = result.get("error")

        st.markdown(response_text)
        _render_risk_badge(response_text, intent)
        _render_sources(sources)
        _render_intent_badge(intent)

        if error and not error.startswith("guardrail_block"):
            st.caption(f"⚠️ Internal note: `{error}`")

    # Update session state
    st.session_state.conversation_history.append(
        {"role": "user", "content": user_input}
    )
    st.session_state.conversation_history.append(
        {
            "role":    "assistant",
            "content": response_text,
            "sources": sources,
            "intent":  intent,
        }
    )

    if "triage_state" in result:
        st.session_state.triage_state = result["triage_state"]

    if intent == "care_routine":
        st.session_state.profile_session_state = result.get("profile")
        if result.get("profile_saved") or result.get("profile"):
            new_profile = result.get("profile") or {}
            if new_profile.get("species"):
                st.session_state.pet_context = new_profile
                st.rerun()
