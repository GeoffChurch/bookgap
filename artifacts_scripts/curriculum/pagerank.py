"""Sparse PageRank implementation for curriculum building.

Provides efficient PageRank computation using sparse matrices, with support for
both single and batched personalization vectors.
"""

from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from tqdm import tqdm

from bookgap.util import LOGGER, TIMER
from bookgap.util import build_citation_adjacency_from_ids
from sklearn.preprocessing import normalize as sk_normalize


@TIMER.time(formatters=dict(
    adj=lambda a: f"<sparse {a.shape}, nnz={a.nnz}>",
    seeds=lambda s: f"<shape={s.shape}>",
))
def compute_pagerank(
    adj: sp.csr_matrix,
    seeds: NDArray[np.floating],
    alpha: float = 0.85,
    tol: float = 1e-9,
    max_iter: int = 100,
) -> NDArray[np.floating]:
    """Compute (Personalized) PageRank using sparse power iteration.
    
    Supports both single and batched personalization vectors.
    
    The PageRank equation is:
        v_{t+1} = alpha * P^T @ v_t + (1 - alpha) * seeds
    where P is the row-normalized transition matrix.
    
    For citation graphs where A[i,j] = 1 means i cites j, flow goes from
    citer (i) to cited (j), so mass accumulates on foundational works.
    
    Args:
        adj: Sparse adjacency matrix (A[i,j] = 1 means i cites j).
        seeds: Personalization vector(s), shape (n,) or (n, n_batches).
               Will be normalized to sum to 1 (per column if batched).
        alpha: Damping factor (probability of following a citation).
        tol: Convergence tolerance (max absolute change in v).
        max_iter: Maximum number of iterations.
        
    Returns:
        PageRank scores, same shape as seeds.
    """
    n = adj.shape[0]
    
    # Ensure seeds is 2D (column-wise batches) for uniform handling
    squeeze_output = False
    if seeds.ndim == 1:
        seeds = seeds.reshape(-1, 1)
        squeeze_output = True
    elif seeds.ndim == 2:
        # Keep as-is
        pass
    else:
        raise ValueError("seeds must be 1D or 2D array")
    
    n_batches = seeds.shape[1]
    
    # Normalize seeds (each column sums to 1)
    seeds = seeds.astype(np.float64)
    col_sums = seeds.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0  # Avoid division by zero
    seeds_norm = seeds / col_sums
    
    # Build transition matrix P (row-normalized adjacency)
    P = sk_normalize(adj.astype(np.float64), norm="l1", axis=1, copy=False)
    
    # Identify sink nodes (no outgoing edges)
    row_sums = np.asarray(adj.sum(axis=1)).flatten()
    sinks = (row_sums == 0)
    n_sinks = sinks.sum()
    
    LOGGER.debug(
        f"PageRank: n={n}, n_batches={n_batches}, alpha={alpha}, "
        f"n_sinks={n_sinks}, tol={tol}"
    )
    
    # Initialize v to seeds
    v = seeds_norm.copy()
    
    # Power iteration
    with tqdm(
        total=max_iter,
        desc=f"PageRank (n={n}, batches={n_batches})",
        leave=False,
    ) as pbar:
        prev_diff = None
        final_diff = None
        for iteration in range(max_iter):
            # Mass from sink nodes gets redistributed according to seeds
            sink_mass = v[sinks].sum(axis=0) if n_sinks > 0 else 0.0

            # v_next = alpha * P^T @ v + (alpha * sink_mass + (1-alpha)) * seeds
            v_next = alpha * (P.T @ v) + (alpha * sink_mass + (1 - alpha)) * seeds_norm

            # Check convergence
            diff = np.abs(v_next - v).max()
            final_diff = diff

            # Update progress bar
            pbar.update(1)

            # Dynamic estimation of remaining iterations based on diff decay
            if prev_diff is not None and prev_diff > 0 and diff > 0:
                decay_rate = diff / prev_diff
                if 0 < decay_rate < 1:
                    try:
                        est_remaining = np.log(tol / diff) / np.log(decay_rate)
                        est_total = iteration + 1 + max(0, int(np.ceil(est_remaining)))
                        pbar.total = min(est_total, max_iter)
                    except Exception:
                        # Fallback: keep existing total
                        pass
            pbar.set_postfix({"diff/tol": f"{(diff / tol):.2e}"})

            if diff < tol:
                LOGGER.debug(f"PageRank converged in {iteration + 1} iterations")
                v = v_next
                break

            v = v_next
            prev_diff = diff
        else:
            LOGGER.warning(
                f"PageRank did not converge after {max_iter} iterations "
                f"(final diff: {final_diff:.2e})"
            )
    
    if squeeze_output:
        return v.squeeze(axis=1)
    return v


def build_batched_pagerank_seeds(
    n_samples: int,
    num_topics: int,
    leaf_to_cluster: NDArray[np.int64],
) -> NDArray[np.floating]:
    """Build batched personalization matrix for PageRank.
    
    Creates a matrix where each column is a uniform personalization vector
    over the leaves belonging to that frontier topic.
    
    Args:
        n_samples: Number of leaf samples (works).
        num_topics: Number of frontier topics.
        leaf_to_cluster: Array mapping leaf indices to cluster IDs.
            Values of -1 indicate unassigned nodes (not in any topic).
        
    Returns:
        Array of shape (n_samples, num_topics) where column i is the
        personalization vector for topic i.
    """
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
) -> NDArray[np.floating]:
    """Compute inter-topic dependency matrix via batched PageRank.
    
    For each frontier topic, runs PageRank seeded from that topic's leaves,
    then sums the resulting mass on each other topic's leaves. This produces
    a |T| x |T| matrix where entry [i,j] represents how much topic i depends
    on topic j (i.e., j is a prerequisite for i).
    
    When using PPR-first mode, the adjacency matrix may be the full citation
    graph while only a subset of nodes have cluster assignments. Nodes with
    cluster assignment -1 are "unassigned" - they don't belong to any topic
    but can serve as intermediaries for PageRank flow between topics. This
    allows detecting indirect dependencies through papers outside the top-k.
    
    Args:
        adj: Sparse citation adjacency matrix.
        frontier_nodes: Array of frontier node indices.
        leaf_to_cluster: Array mapping leaf indices to cluster IDs.
            Values of -1 indicate unassigned nodes (intermediaries only).
        alpha: PageRank damping factor.
        tol: PageRank convergence tolerance.
        max_iter: Maximum PageRank iterations.
        
    Returns:
        Frontier adjacency matrix of shape (num_topics, num_topics).
    """
    n_samples = adj.shape[0]
    num_topics = len(frontier_nodes)
    
    LOGGER.info(f"Computing frontier adjacency for {num_topics} topics...")
    
    # Build batched seeds
    seeds_batched = build_batched_pagerank_seeds(n_samples, num_topics, leaf_to_cluster)
    
    # Run batched PageRank
    leaf_masses_batched = compute_pagerank(
        adj, seeds_batched, alpha=alpha, tol=tol, max_iter=max_iter
    )
    
    # Project onto frontier clusters
    adj_frontier = np.zeros((num_topics, num_topics), dtype=np.float64)
    
    for source_idx in range(num_topics):
        for target_idx in range(num_topics):
            if target_idx == source_idx:
                continue
            # Get leaves belonging to target cluster
            target_leaves = np.where(leaf_to_cluster == target_idx)[0]
            if len(target_leaves) > 0:
                # Sum mass that landed on target cluster from source's PageRank
                # This represents how much source depends on target
                adj_frontier[source_idx, target_idx] = leaf_masses_batched[
                    target_leaves, source_idx
                ].sum()
    
    return adj_frontier
