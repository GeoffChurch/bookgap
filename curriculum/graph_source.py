"""The :class:`GraphSource` protocol — the single input the engine consumes.

The curriculum engine only ever needs a ``(sparse graph, [optional node
embeddings], seed nodes, per-node text)`` quadruple. Everything domain-specific
(OpenAlex citations, Wikipedia links, a Lean proof-dependency graph, a Rust
call graph) is a thin adapter that implements this protocol; the engine itself
never imports any data backend.

Conventions
-----------
- ``adjacency()`` returns a directed ``scipy.sparse`` CSR matrix where
  ``A[i, j] > 0`` means "node ``i`` points to node ``j``". For citation/link
  graphs the convention is "``i`` cites/links ``j``", so forward PageRank mass
  flows ``i -> j`` and accumulates on foundational (heavily-pointed-to) nodes.
  For a Lean proof DAG, ``i -> j`` means "declaration ``i`` depends on premise
  ``j``", which is exactly the same prerequisite semantics.
- ``node_embeddings()`` is optional. Return ``None`` to run the pure-graph
  (Lean / Phase-0-bis) path, where node geometry is derived from the adjacency
  alone (``embeddings_mode='spectral_from_graph'`` or ``'community_detect'``).
- ``node_text()`` supplies per-node text for keyword extraction and the LLM
  judge. Empty strings are fine for nodes without text.
- ``seed_nodes()`` returns the integer indices (into the ``ids()`` order) of
  the target/seed nodes that personalize the PageRank teleport vector.

``GraphSource`` is a ``typing.Protocol`` (structural): any object with the
right methods qualifies. :class:`InMemoryGraphSource` is a frozen-dataclass
implementation handy for tests, the parity gate, and the Lean adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


@runtime_checkable
class GraphSource(Protocol):
    """Structural protocol for the engine's single input.

    Implementations are expected to be cheap to call repeatedly (cache heavy
    work internally); the engine may call ``adjacency()`` and
    ``node_embeddings()`` more than once.
    """

    def adjacency(self) -> sp.csr_matrix:
        """Directed CSR adjacency; ``A[i, j] > 0`` means ``i`` points to ``j``."""
        ...

    def node_embeddings(self) -> Optional[NDArray[np.floating]]:
        """``(n_nodes, dim)`` dense embeddings, or ``None`` for pure-graph mode."""
        ...

    def node_text(self) -> List[str]:
        """Per-node text (length ``n_nodes``) for keywords / the judge."""
        ...

    def ids(self) -> List[str]:
        """Stable per-node identifiers (length ``n_nodes``), index-aligned."""
        ...

    def seed_nodes(self) -> NDArray[np.int64]:
        """Integer node indices to seed (personalize) the PageRank teleport."""
        ...


@dataclass(frozen=True)
class InMemoryGraphSource:
    """A frozen :class:`GraphSource` over in-memory arrays.

    The reference implementation: holds a CSR adjacency, an optional embedding
    matrix, per-node text, ids, and seed indices. Used by the parity gate (the
    cached wiki subgraph loads straight into one of these) and is the natural
    shape for the Lean adapter to emit.
    """

    adj: sp.csr_matrix
    id_list: Sequence[str]
    text: Sequence[str]
    seeds: NDArray[np.int64]
    embeddings: Optional[NDArray[np.floating]] = None
    _index: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        n = self.adj.shape[0]
        if self.adj.shape[0] != self.adj.shape[1]:
            raise ValueError(f"adjacency must be square, got {self.adj.shape}")
        if len(self.id_list) != n:
            raise ValueError(
                f"id_list length {len(self.id_list)} != n_nodes {n}"
            )
        if len(self.text) != n:
            raise ValueError(f"text length {len(self.text)} != n_nodes {n}")
        if self.embeddings is not None and self.embeddings.shape[0] != n:
            raise ValueError(
                f"embeddings rows {self.embeddings.shape[0]} != n_nodes {n}"
            )
        # Build the id->index map once (object is frozen; mutate via the dict).
        object.__setattr__(
            self, "_index", {str(w): i for i, w in enumerate(self.id_list)}
        )

    def adjacency(self) -> sp.csr_matrix:
        return self.adj.tocsr()

    def node_embeddings(self) -> Optional[NDArray[np.floating]]:
        return self.embeddings

    def node_text(self) -> List[str]:
        return list(self.text)

    def ids(self) -> List[str]:
        return [str(w) for w in self.id_list]

    def seed_nodes(self) -> NDArray[np.int64]:
        return np.asarray(self.seeds, dtype=np.int64)

    # --- id <-> index convenience (not part of the protocol) ---------------

    def index_of(self, node_id: str) -> int:
        """Return the integer index of ``node_id`` (KeyError if absent)."""
        return self._index[str(node_id)]

    def seeds_from_ids(self, node_ids: Sequence[str]) -> NDArray[np.int64]:
        """Resolve a list of string ids to seed indices (skips unknown ids)."""
        return np.array(
            [self._index[str(w)] for w in node_ids if str(w) in self._index],
            dtype=np.int64,
        )
