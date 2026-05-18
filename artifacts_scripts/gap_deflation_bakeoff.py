"""Gap-deflation bake-off: swappable deflation strategies on the
byte-stable K-ball harness. See
docs/superpowers/specs/2026-05-17-gap-deflation-bakeoff-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from bookgap.index import kballs_with
from artifacts_scripts.kball_gap_probe import _build_topic_index


@dataclass(frozen=True)
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
            # np.mean upcasts float32→float64; cast back to match article_emb dtype
            centroid = article_emb[members].mean(axis=0).astype(np.float32)
        else:
            centroid = article_emb[i].astype(np.float32)
        cands.append(Candidate(idx=i, members=members, centroid=centroid))
    return cands


# ---------------------------------------------------------------------------
# Deflation strategies
# ---------------------------------------------------------------------------

def _greedy_hard(
    cands: list[Candidate],
    article_emb: np.ndarray,
    *,
    num_gaps: int,
    min_gap_size: int,
    sigma=None,
    **params,
) -> list[tuple[int, int, np.ndarray]]:
    """Greedy-prune deflation — byte-anchored to _greedy_kball_gaps_inline.

    The oracle iterates `for gap_idx in range(num_gaps)` and emits
    `(gap_idx, gap_size, best_ids.copy())` where `gap_idx` is the loop
    counter (0, 1, 2, …), NOT the argmax index.  That is matched here.

    Covered is maintained as a list of mutable id arrays, one per candidate,
    mirroring the oracle's list of (dists, ids) pairs — but since greedy
    ranking uses only `len(ids)`, the dists are irrelevant and we drop them.
    """
    # Mutable copy of each candidate's member set; ordering matches cands list
    covered = [c.members.copy() for c in cands]
    gaps: list[tuple[int, int, np.ndarray]] = []
    for gap_idx in range(num_gaps):
        if not any(len(ids) > 0 for ids in covered):
            break
        best = int(np.argmax([len(ids) for ids in covered]))
        best_ids = covered[best]
        if len(best_ids) == 0:
            break
        gap_size = int(len(best_ids))
        if min_gap_size > 0 and gap_size < min_gap_size:
            break
        gaps.append((gap_idx, gap_size, best_ids.copy()))
        pruned_set = set(best_ids.tolist())
        for i, ids in enumerate(covered):
            if ids.size:  # micro-opt vs oracle (omits this); result identical on empty arrays
                m = np.array([x not in pruned_set for x in ids.tolist()], dtype=bool)
                covered[i] = ids[m]
    return gaps


def _greedy_geom(cands, article_emb, *, num_gaps, min_gap_size,
                 sigma=None, **params):
    """Greedy, but every pick is taken on the RESIDUAL (unclaimed)
    article set: after each emit, drop all globally-claimed ids from
    every surviving candidate before the next argmax. v1 keeps the
    kball radius fixed (radius-vs-nearest-unclaimed-book recompute is
    intentionally deferred — residual-restricted pick is the behavior
    this arm isolates)."""
    alive = [c.members.copy() for c in cands]
    claimed: set[int] = set()
    gaps = []
    emitted = 0
    for _ in range(num_gaps):
        sizes = [len(ids) for ids in alive]
        if not any(sizes):
            break
        best = int(np.argmax(sizes))
        best_ids = alive[best]
        if len(best_ids) == 0 or (min_gap_size > 0 and len(best_ids) < min_gap_size):
            break
        # element-0 = sequential emission rank (uniform across arms; matches greedy_hard/oracle convention)
        gaps.append((emitted, int(len(best_ids)), best_ids.copy()))
        emitted += 1
        claimed |= set(best_ids.tolist())
        for i in range(len(alive)):
            if alive[i].size:
                m = np.array([x not in claimed for x in alive[i].tolist()],
                             dtype=bool)
                alive[i] = alive[i][m]
    return gaps


# Every strategy: (cands, article_emb, *, num_gaps, min_gap_size, sigma, **params) -> list[(seed_idx, size, member_ids)]
DeflateFn = Callable[..., list]
_STRATEGIES: dict[str, DeflateFn] = {
    "greedy_hard": _greedy_hard,
    "greedy_geom": _greedy_geom,
}


def deflate(
    cands: list[Candidate],
    strategy: str,
    article_emb: np.ndarray,
    *,
    num_gaps: int = 1000,
    min_gap_size: int = 0,
    sigma: float | None = None,
    **params,
) -> list[tuple[int, int, np.ndarray]]:
    """Dispatch to a named deflation strategy.

    Args:
        cands:        Candidate pool from build_kball_candidates.
        strategy:     One of the registered strategy names (e.g. "greedy_hard").
        article_emb:  Article embeddings array (passed to soft strategies).
        num_gaps:     Maximum gaps to emit.
        min_gap_size: Stop when the largest remaining ball < this (0 = off).
        sigma:        Smoothing bandwidth (soft strategies only).
        **params:     Extra keyword arguments forwarded to the strategy fn.

    Returns:
        List of (gap_idx, gap_size, member_ids) tuples.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown deflation strategy {strategy!r}")
    return _STRATEGIES[strategy](
        cands, article_emb, num_gaps=num_gaps,
        min_gap_size=min_gap_size, sigma=sigma, **params)
