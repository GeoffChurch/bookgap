"""Graph community detection (the pure-graph ``COMMUNITY_DETECT`` path).

Clusters the adjacency directly into exactly ``n_clusters`` communities via
spectral clustering on the symmetrized affinity — no embeddings, no dendrogram.
Lifted from ``build_curriculum._cluster_with_spectral``; the optional Leiden
path is provided too (lazy-imports ``igraph``/``leidenalg`` only if requested).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from curriculum._logging import get_logger

LOGGER = get_logger(__name__)


def cluster_spectral(
    adj: sp.spmatrix,
    *,
    n_clusters: int,
    seed: int = 42,
) -> NDArray[np.intp]:
    """Spectral clustering on a (possibly directed) sparse graph.

    Symmetrizes the adjacency, treats it as a precomputed affinity, and runs
    sklearn ``SpectralClustering`` to assign each node to one of ``n_clusters``
    communities. Exactly ``n_clusters`` by construction (unlike Leiden's
    resolution-driven count).
    """
    from sklearn.cluster import SpectralClustering

    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.intp)
    if n_clusters >= n:
        return np.arange(n, dtype=np.intp)

    sym = (adj + adj.T) * 0.5
    sym = sym + sp.eye(n, dtype=sym.dtype, format="csr") * 1e-9
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=seed,
        assign_labels="kmeans",
        n_init=10,
    )
    return sc.fit_predict(sym).astype(np.intp)


def cluster_leiden(
    adj: sp.spmatrix,
    *,
    resolution: float = 1.0,
    seed: int = 42,
) -> Tuple[NDArray[np.intp], int]:
    """Leiden community detection on a (possibly directed) sparse graph.

    Symmetrizes, then runs the Reichardt-Bornholdt configuration null model.
    Lazy-imports ``igraph`` / ``leidenalg`` (optional deps). Returns
    ``(membership, n_communities)``.
    """
    import igraph
    import leidenalg

    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.intp), 0
    sym = (adj + adj.T) * 0.5
    sym_coo = sym.tocoo()
    edges = list(zip(sym_coo.row.tolist(), sym_coo.col.tolist()))
    weights = sym_coo.data.tolist()
    g = igraph.Graph(n=n, edges=edges, directed=False)
    if weights:
        g.es["weight"] = weights
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution, weights="weight", seed=seed,
        )
    else:
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution, seed=seed,
        )
    membership = np.array(partition.membership, dtype=np.intp)
    n_communities = int(membership.max()) + 1 if len(membership) > 0 else 0
    return membership, n_communities
