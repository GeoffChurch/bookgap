"""Utilities for building embedding-similarity graphs for CircuitRank."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from bookgap.cloud import Cloud
from bookgap.cloud_index import knn
from bookgap.util import LOGGER, TIMER


def cosine_from_sq_l2(sq_dists: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert squared L2 distances between unit vectors to cosine similarity.

    For normalized vectors: ||x - y||^2 = 2 - 2 cos(x, y)
    => cos = 1 - 0.5 * ||x - y||^2
    """
    return 1.0 - 0.5 * sq_dists


def _drop_self_neighbors(
    neighbor_ids: NDArray[np.int64],
    neighbor_sq_dists: NDArray[np.floating],
) -> Tuple[NDArray[np.int64], NDArray[np.floating]]:
    n, k = neighbor_ids.shape
    rows = np.arange(n, dtype=np.int64)[:, None]
    mask = neighbor_ids != rows

    keep_counts = mask.sum(axis=1)
    num_zero = int((keep_counts == 0).sum())
    if num_zero > 0:
        # # If a row has all neighbors filtered (pathological), keep the first one.
        # mask[keep_counts == 0, 0] = True
        msg = f"{num_zero} rows have all neighbors filtered when dropping self-neighbors"
        LOGGER.fatal(msg)
        raise ValueError(msg)

    # Compact to (n, k-1) if possible, else keep original k and mask later.
    max_keep = int(mask.sum(axis=1).max())
    kept_ids = np.full((n, max_keep), -1, dtype=np.int64)
    kept_dists = np.full((n, max_keep), np.inf, dtype=np.float64)

    for i in range(n):
        row_ids = neighbor_ids[i][mask[i]]
        row_d = neighbor_sq_dists[i][mask[i]]
        kept_ids[i, : len(row_ids)] = row_ids
        kept_dists[i, : len(row_d)] = row_d

    return kept_ids, kept_dists


def build_similarity_adjacency_from_knn(
    neighbor_ids: NDArray[np.int64],
    neighbor_sq_dists: NDArray[np.floating],
    *,
    n_nodes: int,
    sharpness: float = 8.0,
    row_normalize: bool = True,
    density_correction_alpha: float = 0.0,
    include_self: bool = False,
) -> sp.csr_matrix:
    """Build a sparse similarity adjacency from kNN results.

    Args:
        neighbor_ids: (n, k) neighbor indices for each node.
        neighbor_sq_dists: (n, k) squared L2 distances to neighbors.
        n_nodes: Total number of nodes.
        sharpness: Weight sharpness for exp(sharpness * cos).
        row_normalize: If True, normalize each row to sum to 1.
        include_self: If False, drop self-neighbors.

    Returns:
        CSR adjacency with rows corresponding to source nodes (directed kNN).
    """
    if neighbor_ids.shape != neighbor_sq_dists.shape:
        raise ValueError("neighbor_ids and neighbor_sq_dists must have the same shape")

    ids = neighbor_ids
    dists = neighbor_sq_dists.astype(np.float64, copy=False)
    if not include_self:
        ids, dists = _drop_self_neighbors(ids, dists)

    n, k = ids.shape
    if n != n_nodes:
        raise ValueError(f"n_nodes={n_nodes} but neighbor_ids has n={n}")

    # Flatten
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = ids.reshape(-1)
    vals = dists.reshape(-1)

    valid = cols >= 0
    rows = rows[valid]
    cols = cols[valid]
    vals = vals[valid]

    cos = cosine_from_sq_l2(vals)
    weights = np.exp(float(sharpness) * (cos - 1.0))

    alpha = float(density_correction_alpha)
    if alpha > 0:
        rho = np.bincount(rows, weights=weights, minlength=n).astype(np.float64, copy=False)
        rho_safe = np.where(rho > 0, rho, 1.0)
        inv_rho_alpha = np.power(rho_safe, -alpha)
        weights = weights * inv_rho_alpha[rows] * inv_rho_alpha[cols]

    if row_normalize:
        row_sums = np.bincount(rows, weights=weights, minlength=n).astype(np.float64, copy=False)
        denom = row_sums[rows]
        nz = denom > 0
        weights = np.where(nz, weights / denom, 0.0)

    adj = sp.csr_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))
    return adj


@TIMER.time()
def build_similarity_adjacency_from_cloud(
    cloud: Cloud,
    *,
    k: int,
    sharpness: float = 8.0,
    row_normalize: bool = True,
    density_correction_alpha: float = 0.0,
    include_self: bool = False,
) -> sp.csr_matrix:
    """Build a directed kNN similarity adjacency for a Cloud using FAISS."""
    if k <= 0:
        raise ValueError("k must be positive")

    # k+1 to allow dropping self for intra-cloud queries
    k_search = k + 1 if not include_self else k
    res = knn(cloud, cloud, k=k_search)

    neighbor_ids = res.I
    neighbor_sq_dists = res.sq_dists

    LOGGER.info(
        "Similarity kNN: n=%d, k=%d, include_self=%s",
        int(neighbor_ids.shape[0]),
        int(neighbor_ids.shape[1]),
        include_self,
    )

    # If include_self is False, dropping self will reduce to <=k neighbors per row.
    return build_similarity_adjacency_from_knn(
        neighbor_ids,
        neighbor_sq_dists,
        n_nodes=int(neighbor_ids.shape[0]),
        sharpness=sharpness,
        row_normalize=row_normalize,
        density_correction_alpha=density_correction_alpha,
        include_self=include_self,
    )


def mix_adjacency_matrices(
    adj_citation: sp.csr_matrix,
    adj_similarity: sp.csr_matrix,
    *,
    mix_weight: float,
    mix_mode: str = "convex",
) -> sp.csr_matrix:
    """Combine citation and similarity adjacencies.

    Modes:
        - "convex": (1-w) * A_cite + w * A_sim
        - "additive_global_scale": A_cite + w * s * A_sim
          where s matches total edge mass of A_cite.
    """
    if mix_weight < 0:
        raise ValueError("mix_weight must be >= 0")

    if mix_mode == "convex":
        w = float(mix_weight)
        return (adj_citation * (1.0 - w)) + (adj_similarity * w)

    if mix_mode == "additive_global_scale":
        sum_cite = float(adj_citation.sum())
        sum_sim = float(adj_similarity.sum())
        scale = sum_cite / (sum_sim + 1e-12) if sum_sim > 0 else 0.0
        return adj_citation + (adj_similarity * float(mix_weight) * scale)

    raise ValueError(f"Unknown mix_mode={mix_mode!r}")
