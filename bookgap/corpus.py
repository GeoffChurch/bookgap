# bookgap/corpus.py
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Hashable, Protocol, runtime_checkable

import faiss
import numpy as np
from numpy.typing import NDArray


def embedder_fingerprint(embedder_cfg: Any) -> str:
    """Stable 16-hex-char sha256 of an embedder config.

    Identical to the embedder_fp computation in
    bookgap.cloud._embeddings_store_path so the embeddings
    store path is byte-for-byte unchanged when that callsite reuses this.
    """
    return hashlib.sha256(
        json.dumps(asdict(embedder_cfg), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


@runtime_checkable
class Corpus(Protocol):
    """Read-only view of a set of entities sharing one embedding space.

    The only thing gap detection (and, later, curriculum) consume. No scope,
    work_type, topic, or category leaks here — those are source-private.
    """

    def embeddings(self) -> NDArray[np.float32]: ...
    def index(self) -> faiss.Index: ...
    def work_ids(self) -> list[str]: ...
    def work_titles(self) -> list[str]: ...
    def work_abstracts(self) -> list[str]: ...
    def name(self) -> str: ...

    @property
    def embedder_fingerprint(self) -> str: ...

    @property
    def identity(self) -> Hashable: ...


def _source_prefix(child: "Corpus") -> str:
    return child.name().split("[", 1)[0] or "corpus"


@dataclass(frozen=True)
class UnionCorpus:
    """Composite Corpus: a union of child corpora sharing one embedding space."""

    children: tuple["Corpus", ...]

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("UnionCorpus requires at least one child")
        fps = {c.embedder_fingerprint for c in self.children}
        if len(fps) != 1:
            raise ValueError(f"UnionCorpus children must share one embedder; got {fps}")

    def embeddings(self) -> NDArray[np.float32]:
        return np.vstack([c.embeddings() for c in self.children]).astype(np.float32)

    def index(self) -> faiss.Index:
        # A single flat index over the concatenated embeddings. Its ids are the
        # successive global ranges aligned with the embeddings()/work_ids()
        # concatenation order — identically to the brute-force concat contract.
        #
        # Why not faiss.IndexShards(successive_ids=True)? It satisfies .search()
        # equivalently, but does NOT implement .range_search() (faiss raises
        # "range search not implemented"), which get_kball_coverage requires when
        # a UnionCorpus is the *ambient* of gap detection. A FlatL2 over the
        # concat supports both search() and range_search() and keeps the same
        # successive-id geometry, so it is the range-searchable equivalent.
        emb = self.embeddings()
        from bookgap.index import FlatL2, build_index
        return build_index(FlatL2(), np.ascontiguousarray(emb, dtype=np.float32))

    def work_ids(self) -> list[str]:
        out: list[str] = []
        for c in self.children:
            p = _source_prefix(c)
            out.extend(f"{p}:{wid}" for wid in c.work_ids())
        return out

    def work_titles(self) -> list[str]:
        return [t for c in self.children for t in c.work_titles()]

    def work_abstracts(self) -> list[str]:
        return [a for c in self.children for a in c.work_abstracts()]

    def name(self) -> str:
        return "union(" + ",".join(c.name() for c in self.children) + ")"

    @property
    def embedder_fingerprint(self) -> str:
        return self.children[0].embedder_fingerprint

    @property
    def identity(self):
        return ("union",) + tuple(c.identity for c in self.children)
