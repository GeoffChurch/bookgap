# bookgap/wikipedia/corpus.py
"""``WikipediaCorpus`` — a Wikipedia leaf corpus implementing the ``Corpus``
protocol, backed by a store dir built by ``scripts/embed_wikipedia.py``.

Store dir layout::

    <store>/articles.parquet    (page_id:int64, title:str, abstract:str)
    <store>/embeddings.parquet   row-aligned float32 (N, D), column "embedding"
    <store>/manifest.json        provenance, incl. embedder_fingerprint

Shares bookgap's GIST-small embedding space (the embeddings were produced with
the same ``embed_texts`` join as OpenAlex works), so a ``WikipediaCorpus`` and
an OpenAlex ``SubCloud`` with the same ``embedder_fingerprint`` can be combined
in a ``bookgap.corpus.UnionCorpus``.

``work_ids()`` returns native page ids as ``str`` (NOT namespaced) — the union
applies the ``wikipedia:`` source prefix.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Hashable

import faiss
import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from bookgap.index import FlatL2, build_index


@dataclass(frozen=True)
class WikipediaCorpus:
    """Frozen ``Corpus`` over a Wikipedia embedding store directory."""

    store_dir: Path

    def __post_init__(self) -> None:
        # Normalize to a Path (accept str) without breaking frozen semantics.
        if not isinstance(self.store_dir, Path):
            object.__setattr__(self, "store_dir", Path(self.store_dir))

    # --- store paths -----------------------------------------------------
    @property
    def _articles_path(self) -> Path:
        return self.store_dir / "articles.parquet"

    @property
    def _embeddings_path(self) -> Path:
        return self.store_dir / "embeddings.parquet"

    @property
    def _manifest_path(self) -> Path:
        return self.store_dir / "manifest.json"

    # --- Corpus protocol -------------------------------------------------
    def embeddings(self) -> NDArray[np.float32]:
        return _load_embeddings(self.store_dir)

    def index(self) -> faiss.Index:
        # FlatL2 exact index, matching SubCloud.index()'s build_index(...) path
        # but pinned to deterministic exact search for a freshly-built corpus.
        return build_index(FlatL2(), self.embeddings())

    def work_ids(self) -> list[str]:
        return [str(pid) for pid in _load_page_ids(self.store_dir)]

    def work_titles(self) -> list[str]:
        return _load_str_column(self.store_dir, "title")

    def work_abstracts(self) -> list[str]:
        return _load_str_column(self.store_dir, "abstract")

    def name(self) -> str:
        return "wikipedia"

    @property
    def embedder_fingerprint(self) -> str:
        return _load_manifest(self.store_dir)["embedder"]["embedder_fingerprint"]

    @property
    def identity(self) -> Hashable:
        return ("wikipedia", str(self.store_dir.resolve()))


# Module-level cached loaders keyed by store dir (str). The corpus is frozen and
# its data on disk is immutable for a given store, so caching is safe and keeps
# repeated embeddings()/index() calls cheap.
@lru_cache(maxsize=8)
def _load_embeddings(store_dir: Path) -> NDArray[np.float32]:
    table = pq.read_table(store_dir / "embeddings.parquet", columns=["embedding"])
    meta = table.schema.metadata or {}
    n = table.num_rows
    if n == 0:
        d = int(meta.get(b"bookgap:embedding_dim", b"0").decode())
        return np.empty((0, d), dtype=np.float32)
    col = table.column("embedding").combine_chunks()
    # FixedSizeListArray -> (n, d) float32
    d = col.type.list_size
    flat = np.asarray(col.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return np.ascontiguousarray(flat.reshape(n, d))


@lru_cache(maxsize=8)
def _load_page_ids(store_dir: Path) -> list[int]:
    table = pq.read_table(store_dir / "articles.parquet", columns=["page_id"])
    return table.column("page_id").to_pylist()


@lru_cache(maxsize=16)
def _load_str_column(store_dir: Path, column: str) -> list[str]:
    table = pq.read_table(store_dir / "articles.parquet", columns=[column])
    return [v or "" for v in table.column(column).to_pylist()]


@lru_cache(maxsize=8)
def _load_manifest(store_dir: Path) -> dict:
    return json.loads((store_dir / "manifest.json").read_text())
