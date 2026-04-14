"""
llm_config.py
-------------
Central LLM factory for all modules.

Reads from environment variables so every module uses the same endpoint
and model without duplicating configuration.

Environment variables (set in .env):
    RSM_API_KEY  — course endpoint API key
    LLM_BASE_URL — defaults to https://rsm-8430-finalproject.bjlkeng.io/v1
    LLM_MODEL    — defaults to qwen3-30b-a3b-fp8
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

_BASE_URL = os.environ.get("LLM_BASE_URL", "https://rsm-8430-finalproject.bjlkeng.io/v1")
_MODEL    = os.environ.get("LLM_MODEL",    "qwen3-30b-a3b-fp8")


def build_llm(temperature: float = 0.3, max_tokens: int = 512) -> ChatOpenAI:
    """
    Return a configured ChatOpenAI client pointed at the Qwen3 course endpoint.
    """
    api_key = os.environ.get("RSM_API_KEY") or "no-key"
    return ChatOpenAI(
        model=_MODEL,
        base_url=_BASE_URL,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
