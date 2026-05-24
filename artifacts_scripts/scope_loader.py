"""Lightweight scope loader for exploratory and analysis scripts.

Reads a single scope's works + GIST-small embeddings straight from the
per-scope parquet/embeddings caches, bypassing the full ``Cloud``
machinery. Useful for any script that wants raw text + embeddings for
a field, subfield, or topic without paying the cost of constructing a
full denoised ``Cloud``.

Scope-key formats accepted by :func:`load_scope`:

- ``"<field>"``                        — all (subfield, topic) under it
- ``"<field>/<subfield>"``             — all topics under it
- ``"<field>/<subfield>/<topic>"``     — single topic

Embeddings are assumed to exist under
``cache/embeddings/<model>/<scope>/<work_type>/batch_*_embeddings.npy``;
if not, run the fetch/embed pipeline for that scope first.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from bookgap.embedder import SentenceTransformerEmbedder as Embedder
from bookgap.util import CACHE_DIR


def _load_scope_shard(
    parquet_path: Path, emb_dir: Path, with_refs: bool = False,
) -> Tuple[np.ndarray, List[str], List[str], Optional[List[Optional[List[str]]]]]:
    """Return (embeddings, ids, titles, refs_or_None) for one shard.

    Aligns embeddings to parquet row order via the per-batch id manifests.
    Topic identity is derived from the parquet path
    (``<field>/<subfield>/<topic>/<type>``) by the caller — the path is
    the canonical source of truth for the per-scope layout, so we don't
    bother reading the parquet's primary_topic_id column.
    """
    columns = ["id", "display_name"]
    if with_refs:
        columns.append("referenced_works")
    t = pq.read_table(parquet_path, columns=columns)
    ids = t.column("id").to_pylist()
    titles = t.column("display_name").to_pylist()
    refs = t.column("referenced_works").to_pylist() if with_refs else None

    batch_embs: List[np.ndarray] = []
    batch_ids: List[str] = []
    for i in range(100_000):
        eb = emb_dir / f"batch_{i}_embeddings.npy"
        ib = emb_dir / f"batch_{i}_ids.json"
        if not eb.exists():
            break
        batch_embs.append(np.load(eb))
        batch_ids.extend(json.loads(ib.read_text()))
    if not batch_embs:
        raise FileNotFoundError(f"no embedding batches under {emb_dir}")

    emb = np.concatenate(batch_embs, axis=0).astype(np.float32)
    if batch_ids != ids:
        id_to_row = {wid: i for i, wid in enumerate(batch_ids)}
        reorder = np.array([id_to_row[i] for i in ids], dtype=np.int64)
        emb = emb[reorder]
    return emb, ids, titles, refs


def load_scope(scope_key: str, with_refs: bool = False):
    """Load embeddings + metadata for a scope.

    Returns ``(embeddings, ids, types, titles, topic_ids, refs_or_None)``
    where ``embeddings`` is a single ``(n, d)`` float32 array concatenated
    across all (topic, work_type) shards in the scope, and the other
    lists are parallel to its row axis.
    """
    parts = scope_key.split("/")
    model_name = Embedder.ModelName.GIST_SMALL.value
    parquet_root = CACHE_DIR / "works_parquet"
    emb_root = CACHE_DIR / "embeddings" / model_name
    base = parquet_root.joinpath(*parts)
    emb_base = emb_root.joinpath(*parts)

    if len(parts) == 1:
        # Field: aggregate every (subfield, topic) under it
        topic_dirs = sorted(
            t for sf in base.iterdir() if sf.is_dir()
            for t in sf.iterdir() if t.is_dir()
        )
        emb_dirs = [
            emb_base / t.relative_to(base).parts[0] / t.name
            for t in topic_dirs
        ]
    elif len(parts) == 2:
        topic_dirs = sorted(d for d in base.iterdir() if d.is_dir())
        emb_dirs = [emb_base / d.name for d in topic_dirs]
    elif len(parts) == 3:
        topic_dirs = [base]
        emb_dirs = [emb_base]
    else:
        raise ValueError(f"scope_key must be F, F/SF, or F/SF/T, got {scope_key!r}")

    all_emb: List[np.ndarray] = []
    all_ids: List[str] = []
    all_types: List[str] = []
    all_titles: List[str] = []
    all_topic_ids: List[str] = []
    all_refs: List[Optional[List[str]]] = []
    for topic_dir, emb_topic_dir in zip(topic_dirs, emb_dirs):
        topic_label = topic_dir.name  # e.g. "T10181"; path is authoritative
        for wt_parquet in sorted(topic_dir.glob("*.parquet")):
            wt = wt_parquet.stem
            if pq.ParquetFile(wt_parquet).metadata.num_rows == 0:
                continue
            emb_dir = emb_topic_dir / wt
            if not emb_dir.exists() or not any(emb_dir.glob("batch_*_embeddings.npy")):
                print(f"  WARN: no embeddings under {emb_dir}; skipping shard")
                continue
            emb, ids, titles, refs = _load_scope_shard(wt_parquet, emb_dir, with_refs=with_refs)
            all_emb.append(emb)
            all_ids.extend(ids)
            all_types.extend([wt] * len(ids))
            all_titles.extend(titles)
            all_topic_ids.extend([topic_label] * len(ids))
            if refs is not None:
                all_refs.extend(refs)

    embeddings = np.concatenate(all_emb, axis=0)
    return embeddings, all_ids, all_types, all_titles, all_topic_ids, (all_refs if with_refs else None)
