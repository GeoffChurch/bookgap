"""Sparse PageRank for curriculum building (decoupled core).

Lifted from ``artifacts_scripts/curriculum/pagerank.py``; behaviour-preserving.
The only changes vs the bookgap original: the ``bookgap.util`` LOGGER/TIMER
singletons are replaced by stdlib logging, and the optional ``tqdm`` progress
bar degrades to a no-op when tqdm is absent.

Provides seeded / batched / two-direction PageRank plus the absorbing-PR
``compute_frontier_adjacency`` used for inter-topic dependency ordering.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.preprocessing import normalize as sk_normalize

from curriculum._logging import get_logger

LOGGER = get_logger(__name__)

try:  # tqdm is an optional convenience, not a hard dependency
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - exercised only when tqdm is missing

    class _tqdm:  # type: ignore[no-redef]
        """No-op stand-in matching the tqdm call surface used below."""

        def __init__(self, *_, **kwargs) -> None:
            self.total = kwargs.get("total")

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

        def update(self, *_):
            return None

        def set_postfix(self, *_a, **_k):
            return None


def compute_pagerank(
    adj: sp.csr_matrix,
    seeds: NDArray[np.floating],
    alpha: float | NDArray[np.floating] = 0.85,
    tol: float = 1e-9,
    max_iter: int = 100,
    normalize: bool = True,
) -> NDArray[np.floating]:
    """Compute (Personalized) PageRank via sparse power iteration.

    Supports single and batched (column-wise) personalization vectors, scalar
    or per-node ``alpha`` (the absorbing-PR mode used by
    :func:`compute_frontier_adjacency`).

    Iteration: ``v_{t+1} = alpha * P^T @ v_t + (1 - alpha) * seeds`` where ``P``
    is the row-normalized transition matrix. For ``A[i, j] = 1`` meaning
    "``i`` points to ``j``", mass flows ``i -> j`` so it accumulates on
    foundational (heavily-pointed-to) nodes.

    Returns scores with the same shape as ``seeds``.
    """
    n = adj.shape[0]

    squeeze_output = False
    if seeds.ndim == 1:
        seeds = seeds.reshape(-1, 1)
        squeeze_output = True
    elif seeds.ndim != 2:
        raise ValueError("seeds must be 1D or 2D array")

    n_batches = seeds.shape[1]

    if np.ndim(alpha) != 0:
        alpha = np.asarray(alpha, dtype=np.float64)
        if alpha.ndim == 1:
            alpha = alpha.reshape(-1, 1)  # (n, 1) broadcastable against (n, batches)

    seeds = seeds.astype(np.float64)
    col_sums = seeds.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    seeds_norm = seeds / col_sums

    if normalize:
        P = sk_normalize(adj.astype(np.float64), norm="l1", axis=1, copy=False)
    else:
        P = adj.astype(np.float64)

    row_sums = np.asarray(P.sum(axis=1)).flatten()
    sinks = row_sums == 0
    n_sinks = sinks.sum()

    LOGGER.debug(
        "PageRank: n=%d, n_batches=%d, alpha=%s, n_sinks=%d, normalize=%s, tol=%s",
        n, n_batches, alpha if np.ndim(alpha) == 0 else "<array>", n_sinks,
        normalize, tol,
    )

    v = seeds_norm.copy()

    with _tqdm(
        total=max_iter,
        desc=f"PageRank (n={n}, batches={n_batches})",
        leave=False,
    ) as pbar:
        prev_diff = None
        final_diff = None
        for iteration in range(max_iter):
            if np.ndim(alpha) == 0:
                sink_mass = v[sinks].sum(axis=0) if n_sinks > 0 else 0.0
                v_next = alpha * (P.T @ v) + (alpha * sink_mass + (1 - alpha)) * seeds_norm
            else:
                mass_flowing = v * alpha
                mass_teleport = v * (1 - alpha)
                sink_mass_flowing = mass_flowing[sinks].sum(axis=0) if n_sinks > 0 else 0.0
                total_restart_mass = mass_teleport.sum(axis=0) + sink_mass_flowing
                v_next = P.T @ mass_flowing + total_restart_mass * seeds_norm

            diff = np.abs(v_next - v).max()
            final_diff = diff

            pbar.update(1)
            if prev_diff is not None and prev_diff > 0 and diff > 0:
                decay_rate = diff / prev_diff
                if 0 < decay_rate < 1:
                    try:
                        est_remaining = np.log(tol / diff) / np.log(decay_rate)
                        est_total = iteration + 1 + max(0, int(np.ceil(est_remaining)))
                        pbar.total = min(est_total, max_iter)
                    except Exception:
                        pass
            pbar.set_postfix({"diff/tol": f"{(diff / tol):.2e}"})

            if diff < tol:
                LOGGER.debug("PageRank converged in %d iterations", iteration + 1)
                v = v_next
                break

            v = v_next
            prev_diff = diff
        else:
            LOGGER.warning(
                "PageRank did not converge after %d iterations (final diff: %.2e)",
                max_iter, final_diff if final_diff is not None else float("nan"),
            )

    if squeeze_output:
        return v.squeeze(axis=1)
    return v


class TwoDirectionPageRankResult(NamedTuple):
    """Result of :func:`two_direction_pagerank`.

    ``forward_mass`` / ``reverse_mass`` are ``None`` when the corresponding seed
    vector summed to zero (no information for that direction).
    """

    relevance: NDArray[np.floating]
    forward_mass: Optional[NDArray[np.floating]]
    reverse_mass: Optional[NDArray[np.floating]]


def two_direction_pagerank(
    adj: sp.csr_matrix,
    fwd_seeds: NDArray[np.floating],
    rev_seeds: NDArray[np.floating],
    alpha: float = 0.85,
    tol: float = 1e-9,
    max_iter: int = 100,
    combine: str = "product",
    linear_theta: float = 0.5,
) -> TwoDirectionPageRankResult:
    """Personalized PageRank with two seed vectors (one per edge direction).

    Subsumes plain personalized PR (only ``fwd_seeds`` populated) and
    bidirectional PR (both populated). An all-zero seed vector means "no
    preference for that direction" — that pass is skipped and relevance falls
    back to the surviving direction; both zero gives a uniform relevance.
    """
    n = adj.shape[0]
    has_fwd = float(np.asarray(fwd_seeds).sum()) > 0
    has_rev = float(np.asarray(rev_seeds).sum()) > 0

    if not has_fwd and not has_rev:
        v = np.full(n, 1.0 / n, dtype=np.float64)
        return TwoDirectionPageRankResult(relevance=v, forward_mass=None, reverse_mass=None)

    P = sk_normalize(adj.astype(np.float64), norm="l1", axis=1, copy=False)

    v_fwd: Optional[NDArray[np.floating]] = None
    if has_fwd:
        v_fwd = compute_pagerank(
            P, fwd_seeds, alpha=alpha, tol=tol, max_iter=max_iter, normalize=False,
        )

    v_rev: Optional[NDArray[np.floating]] = None
    if has_rev:
        v_rev = compute_pagerank(
            P.T, rev_seeds, alpha=alpha, tol=tol, max_iter=max_iter, normalize=False,
        )

    if v_fwd is None:
        assert v_rev is not None  # has_rev was True (one side is non-None)
        rel = v_rev
    elif v_rev is None:
        rel = v_fwd
    elif combine == "product":
        rel = v_fwd * v_rev
    elif combine == "linear":
        rel = linear_theta * v_fwd + (1.0 - linear_theta) * v_rev
    else:
        raise ValueError(f"combine must be 'product' or 'linear', got {combine!r}")

    s = float(rel.sum())
    if s > 0:
        rel = rel / s

    return TwoDirectionPageRankResult(
        relevance=rel, forward_mass=v_fwd, reverse_mass=v_rev,
    )


def build_batched_pagerank_seeds(
    n_samples: int,
    num_topics: int,
    leaf_to_cluster: NDArray[np.int64],
) -> NDArray[np.floating]:
    """Build a batched personalization matrix: column ``i`` is uniform over the
    leaves belonging to frontier topic ``i`` (``-1`` = unassigned)."""
    seeds_batched = np.zeros((n_samples, num_topics), dtype=np.float64)
    for topic_idx in range(num_topics):
        topic_leaves = np.where(leaf_to_cluster == topic_idx)[0]
        if len(topic_leaves) > 0:
            seeds_batched[topic_leaves, topic_idx] = 1.0 / len(topic_leaves)
    return seeds_batched


def compute_frontier_adjacency(
    adj: sp.csr_matrix,
    frontier_nodes: NDArray[np.int64],
    leaf_to_cluster: NDArray[np.int64],
    alpha: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 100,
    forward_weights: Optional[NDArray[np.floating]] = None,
    reverse_weights: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """Inter-topic dependency matrix via batched absorbing PageRank.

    For each frontier topic, runs PageRank seeded from that topic's leaves with
    all OTHER topic nodes made absorbing (``alpha=0``), then sums the resulting
    mass on each other topic's leaves. Entry ``[i, j]`` = how much topic ``i``
    depends on topic ``j`` (``j`` is a prerequisite for ``i``). Absorbing logic
    suppresses transitive ``A->B->C`` flow so only direct ``A->B`` is counted.

    Returns a ``(num_topics, num_topics)`` matrix.
    """
    n_samples = adj.shape[0]
    num_topics = len(frontier_nodes)

    LOGGER.info(
        "Computing frontier adjacency for %d topics with Absorbing PageRank...",
        num_topics,
    )

    seeds_batched = build_batched_pagerank_seeds(n_samples, num_topics, leaf_to_cluster)

    alpha_matrix = np.full((n_samples, num_topics), alpha, dtype=np.float64)
    assigned_indices = np.where(leaf_to_cluster != -1)[0]
    alpha_matrix[assigned_indices, :] = 0.0
    for topic_idx in range(num_topics):
        topic_leaves = np.where(leaf_to_cluster == topic_idx)[0]
        if len(topic_leaves) > 0:
            alpha_matrix[topic_leaves, topic_idx] = alpha

    leaf_masses_batched = compute_pagerank(
        adj, seeds_batched, alpha=alpha_matrix, tol=tol, max_iter=max_iter,
        normalize=True,
    )

    adj_frontier = np.zeros((num_topics, num_topics), dtype=np.float64)
    for source_idx in range(num_topics):
        for target_idx in range(num_topics):
            if target_idx == source_idx:
                continue
            target_leaves = np.where(leaf_to_cluster == target_idx)[0]
            if len(target_leaves) > 0:
                adj_frontier[source_idx, target_idx] = leaf_masses_batched[
                    target_leaves, source_idx
                ].sum()
    return adj_frontier
