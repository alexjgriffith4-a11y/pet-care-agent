"""
retriever.py
------------
Clean interface for querying the ChromaDB vector store.

This is the handoff boundary between the data pipeline (Phase 1) and the
agent logic (Phase 2+). Everything above this file is data engineering;
everything below is agent logic.

The embedder notebook populates the vector store. This module reads from it.
Both must use the same embedding model (bge-base-en-v1.5 via A2 endpoint)
or retrieval will silently return wrong results.
"""

from dataclasses import dataclass
from typing import Optional

import chromadb
from embedder_api import embed_query  # A2 endpoint — must match the embedder

# ── Configuration ─────────────────────────────────────────────────────────────
VECTORSTORE_PATH = "data/vectorstore"
COLLECTION_NAME  = "pet_care"
DEFAULT_TOP_K    = 5


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its source metadata."""
    text:    str
    title:   str
    url:     str
    source:  str
    species: str
    topic:   str
    score:   float  # cosine similarity, 0–1. Higher = more relevant.


# ── Core functions ────────────────────────────────────────────────────────────

def retrieve(
    query:          str,
    top_k:          int = DEFAULT_TOP_K,
    species:        Optional[str] = None,
    topic:          Optional[str] = None,
    unique_sources: bool = False,
) -> list[RetrievedChunk]:
    """
    Return the top-k most relevant chunks for a query.

    Args:
        query:          Natural-language question.
        top_k:          Number of chunks to return.
        species:        Optional filter — "dog", "cat", or None for both.
        topic:          Optional filter — e.g. "nutrition", "toxins", or None for all.
        unique_sources: If True, return at most one chunk per source URL.
                        Prevents one article from dominating all top-k slots.

    Raises:
        ValueError:   if query is empty or top_k is invalid.
        RuntimeError: if ChromaDB is unreachable or the query fails.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got: {top_k!r}")

    # Build filter only for the dimensions the caller actually specified.
    # ChromaDB applies this before scoring, so it's faster than post-filtering.
    active_filters = []
    if species:
        active_filters.append({"species": {"$eq": species}})
    if topic:
        active_filters.append({"topic": {"$eq": topic}})

    if len(active_filters) == 0:
        where_clause = None
    elif len(active_filters) == 1:
        where_clause = active_filters[0]
    else:
        where_clause = {"$and": active_filters}

    try:
        client     = chromadb.PersistentClient(path=VECTORSTORE_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        # get_collection (not get_or_create) — an empty collection returning
        # 0 results is harder to debug than a clear error here.
    except Exception as e:
        raise RuntimeError(
            f"Could not open ChromaDB collection '{COLLECTION_NAME}' "
            f"at '{VECTORSTORE_PATH}'. Has the embedder run? Error: {e}"
        ) from e

    # Embed the query via the A2 endpoint — same model used in the embedder.
    try:
        query_vector = embed_query(query.strip())
    except RuntimeError as e:
        raise RuntimeError(f"Failed to embed query: {e}") from e

    query_kwargs: dict = dict(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if where_clause:
        query_kwargs["where"] = where_clause

    try:
        results = collection.query(**query_kwargs)
    except Exception as e:
        raise RuntimeError(f"ChromaDB query failed: {e}") from e

    # ChromaDB batches results even for single queries — unwrap the outer list.
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []
    for text, meta, distance in zip(documents, metadatas, distances):
        # ChromaDB returns cosine distance; convert to similarity (1 - distance).
        chunks.append(RetrievedChunk(
            text    = text,
            title   = meta.get("title",   "Unknown"),
            url     = meta.get("url",     ""),
            source  = meta.get("source",  "unknown"),
            species = meta.get("species", "general"),
            topic   = meta.get("topic",   "general"),
            score   = round(1.0 - distance, 4),
        ))

    if unique_sources:
        seen_urls = set()
        deduped   = []
        for chunk in chunks:
            if chunk.url not in seen_urls:
                seen_urls.add(chunk.url)
                deduped.append(chunk)
        chunks = deduped

    return chunks


def format_context_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a numbered context block for an LLM prompt.
    Centralised here so all actions produce consistent source attribution.
    """
    if not chunks:
        return ""

    sections = []
    for i, chunk in enumerate(chunks, start=1):
        sections.append(
            f"[Source {i}] {chunk.title}\n"
            f"URL: {chunk.url}\n"
            f"{chunk.text}"
        )

    return "\n\n---\n\n".join(sections)