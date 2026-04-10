"""
embedder_client.py
------------------
Single place that owns all communication with the A2 embeddings endpoint.

Both the embedder notebook and the retriever import from here.
That way, if the endpoint URL or model name ever changes, you fix it
in one file — not in two notebooks.

Endpoint: https://rsm-8430-a2.bjlkeng.io/v1/embeddings
Model:     bge-base-en-v1.5
Format:    OpenAI Embeddings API (identical request/response shape)

Response shape from the endpoint:
    {
        "data": [
            {"embedding": [0.123, -0.456, ...], "index": 0},
            {"embedding": [...], "index": 1},
            ...
        ]
    }
"""

import os

import requests
from dotenv import load_dotenv

# Make sure RSM_API_KEY is loaded from .env for callers that import this
# module directly (e.g. the retriever and the evaluation script).
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

EMBEDDING_ENDPOINT = "https://rsm-8430-a2.bjlkeng.io/v1/embeddings"
EMBEDDING_MODEL    = "bge-base-en-v1.5"

# bge-base-en-v1.5 produces 768-dimensional vectors.
# This is documented here so anyone reading the code knows what shape
# to expect from embed_texts() and embed_query() without having to call
# the endpoint to find out.
EMBEDDING_DIM = 768

# How many texts to send in one API call.
# The A2 endpoint accepts batches. 64 is conservative and safe.
# If you see timeout errors, lower this to 32.
DEFAULT_BATCH_SIZE = 64


# ── Core function ──────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
    """
    Embed a list of texts using the A2 endpoint.

    Sends texts in batches to avoid overwhelming the endpoint.
    Returns one embedding vector per input text, in the same order.

    Args:
        texts:      List of strings to embed. Must not be empty.
        batch_size: How many texts to send per API call.

    Returns:
        List of embedding vectors. Each vector is a list of 768 floats.
        The i-th vector corresponds to texts[i].

    Raises:
        ValueError:   if texts is empty.
        RuntimeError: if the endpoint returns a non-200 status or
                      the response is malformed.
    """
    if not texts:
        raise ValueError("texts must be a non-empty list")

    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]

        try:
            response = requests.post(
                EMBEDDING_ENDPOINT,
                # The endpoint speaks the OpenAI embeddings API format.
                # "model" tells it which embedding model to use.
                # "input" is the list of texts to embed.
                json={"model": EMBEDDING_MODEL, "input": batch},
                headers={
                    "Content-Type": "application/json",
                    # The A2 endpoint now requires a Bearer token. We reuse
                    # RSM_API_KEY (the same credential the Qwen3 endpoint uses)
                    # because the course provides a single key for both.
                    "Authorization": f"Bearer {os.environ.get('RSM_API_KEY', '')}",
                },
                timeout=60,  # seconds — batch calls can be slow on first request
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Network error calling embedding endpoint: {e}"
            ) from e

        if response.status_code != 200:
            raise RuntimeError(
                f"Embedding endpoint returned HTTP {response.status_code}. "
                f"Body: {response.text[:300]}"
            )

        try:
            payload = response.json()
            # The endpoint returns results in the order we sent them,
            # but we sort by index to be safe — the spec allows out-of-order.
            items = sorted(payload["data"], key=lambda x: x["index"])
            batch_embeddings = [item["embedding"] for item in items]
        except (KeyError, TypeError) as e:
            raise RuntimeError(
                f"Unexpected response shape from embedding endpoint: {e}. "
                f"Raw response: {response.text[:300]}"
            ) from e

        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.

    This is a convenience wrapper around embed_texts() for the retriever,
    which always embeds exactly one string at a time.

    Returns:
        A single embedding vector (list of 768 floats).

    Raises:
        ValueError:   if query is empty.
        RuntimeError: if the endpoint call fails.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    embeddings = embed_texts([query.strip()])
    return embeddings[0]
