# bookgap/corpus.py
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Hashable, Protocol, runtime_checkable

import faiss
import numpy as np
from numpy.typing import NDArray


def embedder_fingerprint(embedder_cfg: Any) -> str:
    """Stable 16-hex-char sha256 of an embedder config.

    Identical to the embedder_fp computation in
    bookgap.cloud._embeddings_store_path (cloud.py:65-67) so the embeddings
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
