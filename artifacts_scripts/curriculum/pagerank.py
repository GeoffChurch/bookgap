"""Sparse PageRank implementation for curriculum building.

Provides efficient PageRank computation using sparse matrices, with support for
both single and batched personalization vectors.
"""

from __future__ import annotations
from typing import NamedTuple, Optional

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from tqdm import tqdm

from bookgap.util import LOGGER, TIMER
from sklearn.preprocessing import normalize as sk_normalize


@TIMER.time(formatters=dict(
    adj=lambda a: f"<sparse {a.shape}, nnz={a.nnz}>",
    seeds=lambda s: f"<shape={s.shape}>",
))
def compute_pagerank(
    adj: sp.csr_matrix,
    seeds: NDArray[np.floating],
    alpha: float | NDArray[np.floating] = 0.85,
    tol: float = 1e-9,
    max_iter: int = 100,
    normalize: bool = True,
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
               Can be a scalar or a vector/array of shape broadcastable with v (n, n_batches).
               If vector, alpha[i] is the probability node i follows a link (vs teleporting).
        tol: Convergence tolerance (max absolute change in v).
        max_iter: Maximum number of iterations.
        normalize: If True, row-normalize the adjacency matrix to build P.
                   If False, use adj as P directly (assumes it is already
                   row-normalized or the desired transition matrix).
        
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
    
    # Handle scalar vs array alpha
    if np.ndim(alpha) == 0:
        # Scalar alpha (common case)
        # Keep as scalar for efficiency unless needed
        pass
    else:
        # Normalize alpha to array form if needed
        alpha = np.asarray(alpha, dtype=np.float64)
        if alpha.ndim == 1:
            alpha = alpha.reshape(-1, 1) # (n, 1) broadcastable against (n, batches)
    
    # Normalize seeds (each column sums to 1)
    seeds = seeds.astype(np.float64)
    col_sums = seeds.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0  # Avoid division by zero
    seeds_norm = seeds / col_sums
    
    # Build transition matrix P
    if normalize:
        P = sk_normalize(adj.astype(np.float64), norm="l1", axis=1, copy=False)
    else:
        P = adj.astype(np.float64)
    
    # Identify sink nodes (no outgoing edges)
    # Mass is lost if the row sum of P is zero.
    row_sums = np.asarray(P.sum(axis=1)).flatten()
    sinks = (row_sums == 0)
    n_sinks = sinks.sum()
    
    LOGGER.debug(
        f"PageRank: n={n}, n_batches={n_batches}, alpha={alpha if np.ndim(alpha)==0 else '<array>'}, "
        f"n_sinks={n_sinks}, normalize={normalize}, tol={tol}"
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
            # Compute next step mass distribution
            if np.ndim(alpha) == 0:
                # Optimized scalar case
                
                # Mass from sink nodes gets redistributed according to seeds
                sink_mass = v[sinks].sum(axis=0) if n_sinks > 0 else 0.0

                # v_next = alpha * P^T @ v + (alpha * sink_mass + (1-alpha)) * seeds
                v_next = alpha * (P.T @ v) + (alpha * sink_mass + (1 - alpha)) * seeds_norm
            else:
                # Vector/Array alpha case
                # v * alpha -> mass that successfully flows from each node
                mass_flowing = v * alpha
                
                # Mass that teleports (locally damped)
                mass_teleport = v * (1 - alpha)
                
                # Sinks redistribute ALL their mass (both flowing and teleport parts)
                # because they have nowhere to flow. Effectively they restart.
                # Mass from sinks that 'tried' to flow but couldn't:
                sink_mass_flowing = mass_flowing[sinks].sum(axis=0) if n_sinks > 0 else 0.0
                
                # Total restart mass = standard teleport + sink redistribution
                total_restart_mass = mass_teleport.sum(axis=0) + sink_mass_flowing
                
                # Flow step + Restart step
                v_next = P.T @ mass_flowing + total_restart_mass * seeds_norm

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


class BidirectionalPageRankResult(NamedTuple):
    """Container for bidirectional PageRank results."""
    relevance: NDArray[np.floating]
    forward_mass: NDArray[np.floating]
    reverse_mass: NDArray[np.floating]


def compute_bidirectional_pagerank(
    adj: sp.csr_matrix,
    source_seeds: NDArray[np.floating],
    target_seeds: NDArray[np.floating],
    alpha: float = 0.85,
    tol: float = 1e-9,
    max_iter: int = 100,
    return_components: bool = False,
) -> NDArray[np.floating] | BidirectionalPageRankResult:
    """Compute bidirectional PageRank relevance.
    
    The relevance of a node is the product of:
    1. Forward mass from target (following citations -> prerequisites)
    2. Reverse mass from source (following citations backwards -> descendants)
    
    This highlights nodes on paths from source to target.
    
    Args:
        adj: Sparse adjacency matrix (A[i,j] = 1 means i cites j).
        source_seeds: Seed vector for the reverse pass (source nodes).
        target_seeds: Seed vector for the forward pass (target nodes).
        alpha: Damping factor.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.
        return_components: If True, returns a NamedTuple with relevance, forward_mass, and reverse_mass.
        
    Returns:
        Bidirectional relevance scores, shape (n,), or BidirectionalPageRankResult if return_components is True.
    """
    LOGGER.info("Computing bidirectional PageRank...")
    
    # 1. Pre-normalize once: P[i,j] = A[i,j] / out_degree(i)
    # Mass flows citer (i) -> cited (j)
    P = sk_normalize(adj.astype(np.float64), norm="l1", axis=1, copy=False)
    
    # 2. Forward pass from target: mass flows i -> j (prerequisites)
    # Operator is P.T
    v_forward = compute_pagerank(
        P, target_seeds, alpha=alpha, tol=tol, max_iter=max_iter, normalize=False
    )
    
    # 3. Reverse pass from source: mass flows j -> i (descendants)
    # Operator is P.T.T = P
    # We pass P.T to compute_pagerank(normalize=False), which will then use
    # (P.T).T = P as its transition matrix.
    v_reverse = compute_pagerank(
        P.T, source_seeds, alpha=alpha, tol=tol, max_iter=max_iter, normalize=False
    )
    
    # 4. Product is proportional to path probability
    relevance = v_forward * v_reverse
    
    # Normalize relevance so it sums to 1 (optional, but good for consistency)
    total = relevance.sum()
    if total > 0:
        relevance /= total
        
    if return_components:
        return BidirectionalPageRankResult(
            relevance=relevance,
            forward_mass=v_forward,
            reverse_mass=v_reverse,
        )
        
    return relevance


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
    forward_weights: Optional[NDArray[np.floating]] = None,
    reverse_weights: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """Compute inter-topic dependency matrix via batched PageRank.
    
    For each frontier topic, runs PageRank seeded from that topic's leaves,
    then sums the resulting mass on each other topic's leaves. This produces
    a |T| x |T| matrix where entry [i,j] represents how much topic i depends
    on topic j (i.e., j is a prerequisite for i).
    
    Uses "Absorbing PageRank" via per-node alpha:
    - Nodes belonging to ANY topic (other than the source topic's own leaves)
      are set to be absorbing (alpha=0).
    - This allows measuring direct flow A->B while suppressing transitive
      flow A->B->C (since flow is absorbed at B).
    
    Args:
        adj: Sparse citation adjacency matrix.
        frontier_nodes: Array of frontier node indices.
        leaf_to_cluster: Array mapping leaf indices to cluster IDs.
            Values of -1 indicate unassigned nodes (intermediaries only).
        alpha: PageRank damping factor (global).
        tol: PageRank convergence tolerance.
        max_iter: Maximum PageRank iterations.
        forward_weights: Optional global relevance vector (ignored in absorbing mode).
        reverse_weights: Optional global relevance vector (ignored in absorbing mode).
        
    Returns:
        Frontier adjacency matrix of shape (num_topics, num_topics).
    """
    n_samples = adj.shape[0]
    num_topics = len(frontier_nodes)
    
    LOGGER.info(f"Computing frontier adjacency for {num_topics} topics with Absorbing PageRank...")
    
    # 1. Build batched seeds
    seeds_batched = build_batched_pagerank_seeds(n_samples, num_topics, leaf_to_cluster)
    
    # 2. Construct Matrix Alpha for Absorbing Logic
    # Default alpha everywhere
    alpha_matrix = np.full((n_samples, num_topics), alpha, dtype=np.float64)
    
    # Identify all nodes that belong to any cluster
    # We want these to be absorbing (alpha=0) unless they are the source.
    assigned_indices = np.where(leaf_to_cluster != -1)[0]
    
    # Set alpha=0 for all assigned nodes across all batches first
    # This makes all topics absorbing by default
    alpha_matrix[assigned_indices, :] = 0.0
    
    # Now restore alpha for the source nodes in each batch
    # For batch i (Topic i), the leaves of Topic i should NOT be absorbing (they are sources)
    for topic_idx in range(num_topics):
        topic_leaves = np.where(leaf_to_cluster == topic_idx)[0]
        if len(topic_leaves) > 0:
            alpha_matrix[topic_leaves, topic_idx] = alpha
            
    # 3. Run batched PageRank with matrix alpha
    # Use standard adjacency (or weighted if we wanted, but absorbing usually replaces weighting need)
    # We ignore forward_weights/reverse_weights for now to keep the logic clean as per "cleaner matrix" request,
    # or apply them to adj if desired. Assuming pure absorbing logic for now.
    
    leaf_masses_batched = compute_pagerank(
        adj, # Use original adjacency
        seeds_batched, 
        alpha=alpha_matrix, 
        tol=tol, 
        max_iter=max_iter,
        normalize=True # Standard normalization
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
