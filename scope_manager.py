#!/usr/bin/env python3
"""Scope Manager TUI — browse OpenAlex taxonomy, view fetch/embed status, launch jobs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pyalex
import pyarrow.parquet as pq

import bookgap.openalex_ingest as _ingest_mod
from bookgap.embedder import SentenceTransformerEmbedder as Embedder
from bookgap.openalex_ingest import ResolvedScope, _scope_parquet_path

# ---------------------------------------------------------------------------
# Hierarchy resolution (PyAlex queries — returns (id, display_name) pairs)
# ---------------------------------------------------------------------------

def _suffix(raw_id: str) -> str:
    return raw_id.rsplit("/", 1)[-1]


def get_domain_fields(domain_id: str) -> list[tuple[str, str]]:
    """Return (field_id, display_name) pairs for a domain, sorted by name."""
    data: Any = pyalex.Domains()[domain_id]
    fields = data.get("fields", [])
    result = [(_suffix(f["id"]), f["display_name"]) for f in fields]
    return sorted(result, key=lambda x: x[1])


def get_field_subfields(field_id: str) -> list[tuple[str, str]]:
    """Return (subfield_id, display_name) pairs for a field, sorted by name."""
    data: Any = pyalex.Fields()[field_id]
    subfields = data.get("subfields", [])
    result = [(_suffix(sf["id"]), sf["display_name"]) for sf in subfields]
    return sorted(result, key=lambda x: x[1])


def get_subfield_topics_named(subfield_id: str) -> list[tuple[str, str]]:
    """Return (topic_id, display_name) pairs for a subfield, sorted by name, deduplicated."""
    result: list[tuple[str, str]] = []
    for page in pyalex.Topics().filter(**{"subfield.id": subfield_id}).paginate(per_page=200):
        for topic in page:
            tid = topic.get("id")
            name = topic.get("display_name", "")
            if isinstance(tid, str):
                result.append((_suffix(tid), name))
    # Deduplicate by ID, sort by name
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for tid, name in sorted(result, key=lambda x: x[1]):
        if tid not in seen:
            seen.add(tid)
            unique.append((tid, name))
    return unique


# ---------------------------------------------------------------------------
# Status queries (filesystem only — no API calls)
# ---------------------------------------------------------------------------

_MODEL_PATHS = {
    "gist": Embedder.ModelName.GIST_SMALL.value,   # "avsolatorio/GIST-small-Embedding-v0"
    "gemma": Embedder.ModelName.GEMMA_300M.value,   # "google/embeddinggemma-300m"
}


def _embedding_dir(scope: ResolvedScope, model_value: str) -> Path:
    """Return the embedding cache directory for a scope + model.

    Uses the same CACHE_DIR as bookgap.openalex_ingest so that test
    monkeypatches on that module-level variable are respected.
    """
    cache_dir: Path = _ingest_mod.CACHE_DIR
    return cache_dir / "embeddings" / model_value / scope.field / scope.subfield / scope.topic / scope.work_type


def get_scope_status(scope: ResolvedScope) -> dict[str, Any]:
    """Check fetch and embed status for a single scope. Filesystem only."""
    parquet_path = _scope_parquet_path(scope)
    fetched = parquet_path.exists()
    work_count = 0
    if fetched:
        try:
            work_count = pq.read_metadata(parquet_path).num_rows
        except Exception:
            work_count = -1  # corrupt file

    gist_dir = _embedding_dir(scope, _MODEL_PATHS["gist"])
    gemma_dir = _embedding_dir(scope, _MODEL_PATHS["gemma"])

    return {
        "fetched": fetched,
        "work_count": work_count,
        "gist_embedded": gist_dir.exists() and any(gist_dir.glob("batch_*_embeddings.npy")),
        "gemma_embedded": gemma_dir.exists() and any(gemma_dir.glob("batch_*_embeddings.npy")),
    }
