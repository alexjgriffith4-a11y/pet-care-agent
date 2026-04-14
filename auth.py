"""
auth.py
-------
Lightweight user-account system for the Pet Care Assistant.

Stores accounts in users.json (next to this file). Passwords are hashed
with PBKDF2-HMAC-SHA256 + a random salt — never stored in plaintext.

Public API
----------
    register(username, display_name, password) -> (ok: bool, error: str)
    login(username, password)                  -> (user: dict | None, error: str)
    get_user(username)                         -> dict | None
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path

_USERS_PATH = Path(__file__).parent / "users.json"
_ITERATIONS = 260_000  # NIST recommendation for PBKDF2-SHA256


# ── Storage helpers ───────────────────────────────────────────────────────────

def _load() -> dict:
    try:
        if _USERS_PATH.exists():
            return json.loads(_USERS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save(data: dict) -> None:
    _USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _USERS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Password hashing ──────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Return (hash_hex, salt_hex). Generate a new salt if not provided."""
    if salt is None:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _ITERATIONS,
    )
    return key.hex(), salt


def _verify_password(password: str, hash_hex: str, salt: str) -> bool:
    computed, _ = _hash_password(password, salt)
    return secrets.compare_digest(computed, hash_hex)


# ── Public API ────────────────────────────────────────────────────────────────

def register(username: str, display_name: str, password: str) -> tuple[bool, str]:
    """
    Create a new account.

    Returns:
        (True, "")                 on success.
        (False, error_message)     on validation failure or duplicate username.
    """
    username = username.strip().lower()
    display_name = display_name.strip()

    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not username.replace("_", "").replace("-", "").isalnum():
        return False, "Username may only contain letters, numbers, hyphens, and underscores."
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters."
    if not display_name:
        display_name = username

    users = _load()
    if username in users:
        return False, "That username is already taken."

    hash_hex, salt = _hash_password(password)
    users[username] = {
        "username":     username,
        "display_name": display_name,
        "hash":         hash_hex,
        "salt":         salt,
    }
    _save(users)
    return True, ""


def login(username: str, password: str) -> tuple[dict | None, str]:
    """
    Verify credentials.

    Returns:
        (user_dict, "")           on success.
        (None, error_message)     on failure.
    """
    username = username.strip().lower()
    users = _load()
    user = users.get(username)

    if not user:
        return None, "No account found with that username."
    if not _verify_password(password, user["hash"], user["salt"]):
        return None, "Incorrect password."

    # Return safe subset (no hash/salt)
    return {
        "username":     user["username"],
        "display_name": user["display_name"],
    }, ""


def get_user(username: str) -> dict | None:
    """Return the public user dict for a username, or None."""
    users = _load()
    user = users.get(username.strip().lower())
    if not user:
        return None
    return {"username": user["username"], "display_name": user["display_name"]}
