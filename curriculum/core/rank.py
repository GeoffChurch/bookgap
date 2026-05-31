"""Rank (seeded / bg-norm PageRank) -> select (top-N) — the minimal core.

Reproduces ``wiki_curriculum_demo``'s selection logic and
``build_curriculum``'s background-normalized scoring exactly, decoupled from any
data backend. Works with no embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from curriculum._logging import get_logger
from curriculum.config import CurriculumConfig, RankMethod, SelectionMode
from curriculum.core.pagerank import compute_pagerank, two_direction_pagerank

LOGGER = get_logger(__name__)


@dataclass
class RankResult:
    """Output of :func:`compute_relevance`."""

    masses: NDArray[np.floating]
    """Raw forward-PR relevance per node (the cluster-weight signal)."""

    selection_score: NDArray[np.floating]
    """Score used to pick the top-N (== ``masses`` unless bg-norm is on)."""

    background: Optional[NDArray[np.floating]]
    """Uniform-teleport background PR ``g`` (only when bg-norm is on)."""


def compute_relevance(
    adj: sp.csr_matrix,
    seed_indices: NDArray[np.int64],
    cfg: CurriculumConfig,
    *,
    source_indices: Optional[NDArray[np.int64]] = None,
) -> RankResult:
    """Seeded (optionally bidirectional) PageRank + selection score.

    The cluster weights stay RAW forward-PR mass; background normalization is a
    selection-only re-rank (matches both the OpenAlex and wiki pipelines).
    """
    n = adj.shape[0]
    if len(seed_indices) == 0:
        raise ValueError("no seed nodes provided")

    target_seeds = np.zeros(n, dtype=np.float64)
    target_seeds[seed_indices] = 1.0 / len(seed_indices)

    rev_seeds = np.zeros(n, dtype=np.float64)
    if cfg.method == RankMethod.BIDIRECTIONAL_PAGE_RANK and source_indices is not None \
            and len(source_indices) > 0:
        rev_seeds[source_indices] = 1.0 / len(source_indices)

    result = two_direction_pagerank(
        adj,
        fwd_seeds=target_seeds,
        rev_seeds=rev_seeds,
        alpha=cfg.pagerank.alpha,
        tol=cfg.pagerank.tol,
        max_iter=cfg.pagerank.max_iter,
        combine="product",
    )
    masses = result.relevance

    selection_score = masses
    background: Optional[NDArray[np.floating]] = None
    if cfg.selection.mode == SelectionMode.BACKGROUND_NORMALIZED:
        bg = cfg.selection.background
        a_exp = float(bg.exponent)
        alpha_bg = cfg.background_alpha()
        eps = float(bg.eps)
        LOGGER.info(
            "Background normalization: a=%.3f, alpha_bg=%.3f, eps=%.1e",
            a_exp, alpha_bg, eps,
        )
        uniform_seeds = np.full(n, 1.0 / n, dtype=np.float64)
        background = compute_pagerank(
            adj, uniform_seeds, alpha=alpha_bg,
            tol=cfg.pagerank.tol, max_iter=cfg.pagerank.max_iter, normalize=True,
        )
        p_fg = np.maximum(masses, 0.0)
        selection_score = np.power(p_fg, a_exp) * np.log((p_fg + eps) / (background + eps))

    return RankResult(masses=masses, selection_score=selection_score, background=background)


def select_top_n(
    selection_score: NDArray[np.floating],
    top_n: int,
    *,
    node_type: Optional[NDArray[np.int64]] = None,
    per_corpus: bool = False,
) -> NDArray[np.int64]:
    """Return indices of the top-``top_n`` nodes by ``selection_score`` (desc).

    With ``per_corpus`` and a ``node_type`` vector, the score is z-normalized
    WITHIN each corpus (node_type value) before the cut, so a smaller corpus is
    not starved by its lower absolute PageRank mass (the joint-graph fix).
    """
    score = selection_score
    if per_corpus and node_type is not None:
        score = score.astype(np.float64).copy()
        for c in np.unique(node_type):
            mask = node_type == c
            vals = score[mask]
            mu = float(vals.mean())
            sd = float(vals.std())
            score[mask] = (vals - mu) / sd if sd > 0 else 0.0
    k = min(top_n, len(score))
    return np.argsort(score)[-k:][::-1].astype(np.int64)
