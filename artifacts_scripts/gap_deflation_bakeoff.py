"""Gap-deflation bake-off: swappable deflation strategies on the
byte-stable K-ball harness. See
docs/superpowers/specs/2026-05-17-gap-deflation-bakeoff-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bookgap.index import kballs_with
from artifacts_scripts.kball_gap_probe import _build_topic_index


@dataclass
class Candidate:
    idx: int                 # article-local index of the ball's seed
    members: np.ndarray      # article-local indices in the k-ball (int64)
    centroid: np.ndarray     # mean embedding of members (float32, (d,))


def build_kball_candidates(
    article_emb: np.ndarray,
    book_emb: np.ndarray,
    radius_fraction: float = 0.75,
    k: int = 1,
) -> list[Candidate]:
    """Strategy-independent K-ball candidate pool.

    Mirrors the candidate construction inside
    kball_gap_probe._greedy_kball_gaps_inline (same indices, same
    kballs_with call) but stops before the greedy selection so any
    deflation strategy can consume the raw pool. members are
    article_emb-local indices.
    """
    book_index = _build_topic_index(book_emb)
    article_index = _build_topic_index(article_emb)
    coverage: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(article_emb)

    def _kernel(qi: int, dists: np.ndarray, ids: np.ndarray) -> None:
        coverage[qi] = (dists.copy(), ids.copy())

    kballs_with(
        ambient=article_index, landmark=book_index, query=article_emb,
        radius_fraction=radius_fraction, k=k, kernel_fn=_kernel,
        show_progress=False,
    )
    cands: list[Candidate] = []
    for i, cov in enumerate(coverage):
        if cov is None:
            members = np.array([], dtype=np.int64)
        else:
            members = cov[1].astype(np.int64)
        if members.size:
            centroid = article_emb[members].mean(axis=0).astype(np.float32)
        else:
            centroid = article_emb[i].astype(np.float32)
        cands.append(Candidate(idx=i, members=members, centroid=centroid))
    return cands
