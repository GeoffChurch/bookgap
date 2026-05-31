"""Order (linearize) topics: prerequisite -> frontier.

Two pure-graph orderings plus the mass ordering, matching
``wiki_curriculum_demo``'s ``--order-by`` semantics. The default
frontier-adjacency order builds the inter-topic dependency matrix over the RAW
forward-PR flow (decoupled from any bg-norm selection) and sorts by
(in-degree - out-degree) so foundations land first.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from curriculum._logging import get_logger
from curriculum.core.pagerank import compute_frontier_adjacency

LOGGER = get_logger(__name__)


def linearization_order_from_adjacency(
    adj_matrix: NDArray[np.floating],
) -> NDArray[np.int64]:
    """Order topics by citation in-degree minus out-degree (prerequisites first).

    ``adj_matrix[i, j]`` = how much topic ``i`` depends on topic ``j``. A topic
    that is depended-on by many (high in-degree) and depends-on few (low
    out-degree) is a prerequisite and sorts first.
    """
    n = adj_matrix.shape[0]
    scores = {}
    for k in range(n):
        cited_by_others = adj_matrix[:, k].sum()
        cites_others = adj_matrix[k, :].sum()
        scores[k] = cited_by_others - cites_others
    sorted_indices = sorted(range(n), key=lambda k: scores[k], reverse=True)
    return np.array(sorted_indices, dtype=np.int64)


def order_by_mass(
    point_to_cluster: NDArray[np.intp],
    masses: NDArray[np.floating],
    n_topics: int,
) -> Tuple[NDArray[np.int64], NDArray[np.floating]]:
    """Order topics by total raw forward-PR mass (descending).

    Returns ``(order, topic_mass)``. Reproduces ``--order-by mass``: foundations
    accrue forward-PR mass, so they sort first.
    """
    topic_mass = np.array(
        [masses[point_to_cluster == t].sum() for t in range(n_topics)]
    )
    order = np.argsort(-topic_mass)
    return order.astype(np.int64), topic_mass


def order_by_frontier_adjacency(
    adj: sp.csr_matrix,
    point_to_cluster: NDArray[np.intp],
    top_idx: NDArray[np.int64],
    n_topics: int,
    *,
    n_nodes: int,
    alpha: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> Tuple[NDArray[np.int64], NDArray[np.floating], NDArray[np.floating]]:
    """Directional DAG order via absorbing PageRank over the RAW forward flow.

    Maps each clustered node back to the full graph (``-1`` for unclustered),
    runs ``compute_frontier_adjacency`` to get the inter-topic dependency
    matrix, then sorts by (in-degree - out-degree). Returns
    ``(order, in_minus_out, adj_frontier)``.
    """
    leaf_to_cluster_full = np.full(n_nodes, -1, dtype=np.int64)
    leaf_to_cluster_full[top_idx] = point_to_cluster.astype(np.int64)
    frontier_nodes = np.arange(n_topics, dtype=np.int64)
    adj_frontier = compute_frontier_adjacency(
        adj, frontier_nodes, leaf_to_cluster_full,
        alpha=alpha, tol=tol, max_iter=max_iter,
    )
    inflow = np.array([adj_frontier[:, k].sum() for k in range(n_topics)])
    outflow = np.array([adj_frontier[k, :].sum() for k in range(n_topics)])
    in_minus_out = inflow - outflow
    order = linearization_order_from_adjacency(adj_frontier)
    return order, in_minus_out, adj_frontier
