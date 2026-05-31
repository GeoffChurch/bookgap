"""Scalable bisecting-KMeans dendrogram + mass-threshold frontier cut.

Lifts ``artifacts_scripts/curriculum/scalable_clustering.py`` (bisecting-KMeans
linkage) and ``dendrogram_ops.py`` (mass propagation + frontier cut) into one
decoupled module. Behaviour-preserving; the only change is the LOGGER swap.

The frontier cut turns a dendrogram into ~``desired_n_topics`` clusters by
finding a per-node mass threshold and reading off the nodes whose mass exceeds
it but whose children do not.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans, MiniBatchKMeans

from curriculum._logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class BisectingKMeansResult:
    """Scipy-compatible linkage + leaf-cluster mapping from bisecting KMeans."""

    linkage: NDArray[np.floating]
    leaf_cluster_assignments: NDArray[np.intp]
    n_leaf_clusters: int


def bisecting_kmeans_linkage(
    embeddings: NDArray[np.floating],
    *,
    sample_weights: Optional[NDArray[np.floating]] = None,
    min_cluster_size: int = 1,
    min_cluster_weight: float = 0.0,
    max_depth: Optional[int] = None,
    random_state: Optional[int] = 42,
    use_minibatch: bool = True,
    batch_size: int = 10000,
    verbose: bool = False,
) -> BisectingKMeansResult:
    """Hierarchical clustering via recursive bisecting K-Means (O(n) memory).

    With ``sample_weights`` the bisection respects per-point relevance mass
    (high-relevance points anchor centroids instead of being peeled off as
    singletons). Returns a scipy-compatible linkage matrix.
    """
    if min_cluster_weight < 0:
        raise ValueError("min_cluster_weight must be >= 0")
    if min_cluster_weight > 0 and sample_weights is None:
        raise ValueError(
            "min_cluster_weight > 0 requires sample_weights "
            "(weight-based threshold is undefined without per-point weights)"
        )
    n_samples = len(embeddings)
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if sample_weights.shape != (n_samples,):
            raise ValueError(
                f"sample_weights shape {sample_weights.shape} must match "
                f"embeddings n_samples ({n_samples},)"
            )
        if (sample_weights < 0).any():
            raise ValueError("sample_weights must be non-negative")
    if n_samples <= 1:
        assignments = np.zeros(n_samples, dtype=np.intp)
        return BisectingKMeansResult(
            linkage=np.zeros((0, 4), dtype=np.float64),
            leaf_cluster_assignments=assignments,
            n_leaf_clusters=max(1, n_samples),
        )
    if n_samples <= min_cluster_size or (
        sample_weights is not None
        and min_cluster_weight > 0
        and float(sample_weights.sum()) <= min_cluster_weight
    ):
        assignments = np.zeros(n_samples, dtype=np.intp)
        return BisectingKMeansResult(
            linkage=np.zeros((0, 4), dtype=np.float64),
            leaf_cluster_assignments=assignments,
            n_leaf_clusters=1,
        )

    if verbose:
        LOGGER.info(
            "Bisecting K-Means: %d points, min_cluster_size=%d, max_depth=%s%s",
            n_samples, min_cluster_size, max_depth,
            "" if sample_weights is None else " (weighted)",
        )

    old_limit = sys.getrecursionlimit()
    needed_limit = max(old_limit, 10000)
    if needed_limit > old_limit:
        sys.setrecursionlimit(needed_limit)

    try:
        return _bisecting_kmeans_impl(
            embeddings,
            sample_weights=sample_weights,
            n_samples=n_samples,
            min_cluster_size=min_cluster_size,
            min_cluster_weight=min_cluster_weight,
            max_depth=max_depth,
            random_state=random_state,
            use_minibatch=use_minibatch,
            batch_size=batch_size,
            verbose=verbose,
        )
    finally:
        sys.setrecursionlimit(old_limit)


def _bisecting_kmeans_impl(
    embeddings: NDArray[np.floating],
    *,
    sample_weights: Optional[NDArray[np.floating]],
    n_samples: int,
    min_cluster_size: int,
    min_cluster_weight: float,
    max_depth: Optional[int],
    random_state: Optional[int],
    use_minibatch: bool,
    batch_size: int,
    verbose: bool,
) -> BisectingKMeansResult:
    splits_done = [0]
    log_interval = max(1, n_samples // (min_cluster_size * 20))

    leaf_cluster_assignments = np.full(n_samples, -1, dtype=np.intp)
    next_leaf_cluster_id = [0]

    def build_tree(indices: NDArray[np.intp], depth: int = 0) -> Tuple:
        size = len(indices)
        if size == 1:
            return ("leaf", int(indices[0]))
        if size <= min_cluster_size:
            return ("leaf_cluster", indices)
        if (
            sample_weights is not None
            and min_cluster_weight > 0
            and float(sample_weights[indices].sum()) <= min_cluster_weight
        ):
            return ("leaf_cluster", indices)
        if max_depth is not None and depth >= max_depth:
            return ("leaf_cluster", indices)
        if size == 2:
            dist = float(np.linalg.norm(embeddings[indices[0]] - embeddings[indices[1]]))
            return ("internal", ("leaf", int(indices[0])), ("leaf", int(indices[1])), dist, 2)

        cluster_embeddings = embeddings[indices]
        cluster_weights = sample_weights[indices] if sample_weights is not None else None
        if cluster_weights is not None and float(cluster_weights.sum()) == 0:
            cluster_weights = None

        if use_minibatch and size > batch_size:
            kmeans = MiniBatchKMeans(
                n_clusters=2, max_iter=100, n_init=1,
                random_state=random_state, batch_size=min(batch_size, size),
            )
        else:
            kmeans = KMeans(
                n_clusters=2, max_iter=100, n_init=1, algorithm="lloyd",
                random_state=random_state,
            )

        labels = kmeans.fit_predict(cluster_embeddings, sample_weight=cluster_weights)
        left_indices = indices[labels == 0]
        right_indices = indices[labels == 1]
        if len(left_indices) == 0 or len(right_indices) == 0:
            mid = size // 2
            left_indices = indices[:mid]
            right_indices = indices[mid:]

        if sample_weights is not None:
            w_left = sample_weights[left_indices]
            w_right = sample_weights[right_indices]
            wsum_left = float(w_left.sum())
            wsum_right = float(w_right.sum())
            if wsum_left > 0:
                left_centroid = (embeddings[left_indices] * w_left[:, None]).sum(axis=0) / wsum_left
            else:
                left_centroid = embeddings[left_indices].mean(axis=0)
                wsum_left = float(len(left_indices))
            if wsum_right > 0:
                right_centroid = (embeddings[right_indices] * w_right[:, None]).sum(axis=0) / wsum_right
            else:
                right_centroid = embeddings[right_indices].mean(axis=0)
                wsum_right = float(len(right_indices))
            n_left = wsum_left
            n_right = wsum_right
        else:
            left_centroid = embeddings[left_indices].mean(axis=0)
            right_centroid = embeddings[right_indices].mean(axis=0)
            n_left, n_right = len(left_indices), len(right_indices)
        dist = float(
            np.sqrt(2 * n_left * n_right / (n_left + n_right))
            * np.linalg.norm(left_centroid - right_centroid)
        )

        splits_done[0] += 1
        if verbose and splits_done[0] % log_interval == 0:
            LOGGER.info("  Bisecting K-Means: %d splits done", splits_done[0])

        left_tree = build_tree(left_indices, depth + 1)
        right_tree = build_tree(right_indices, depth + 1)
        return ("internal", left_tree, right_tree, dist, size)

    tree = build_tree(np.arange(n_samples))

    linkage_rows: List[List[float]] = []
    leaf_cluster_to_id: Dict[int, int] = {}

    def count_leaves(node, depth: int = 0) -> int:
        node_type = node[0]
        if node_type == "leaf":
            point_idx = node[1]
            if id(node) not in leaf_cluster_to_id:
                leaf_id = next_leaf_cluster_id[0]
                leaf_cluster_to_id[id(node)] = leaf_id
                leaf_cluster_assignments[point_idx] = leaf_id
                next_leaf_cluster_id[0] += 1
            return 1
        elif node_type == "leaf_cluster":
            indices = node[1]
            if id(node) not in leaf_cluster_to_id:
                leaf_id = next_leaf_cluster_id[0]
                leaf_cluster_to_id[id(node)] = leaf_id
                leaf_cluster_assignments[indices] = leaf_id
                next_leaf_cluster_id[0] += 1
            return 1
        else:
            _, left, right, _, _ = node
            return count_leaves(left, depth + 1) + count_leaves(right, depth + 1)

    count_leaves(tree)
    n_leaf_clusters = next_leaf_cluster_id[0]

    next_internal_id = [n_leaf_clusters]

    def get_node_id(node) -> int:
        node_type = node[0]
        if node_type in ("leaf", "leaf_cluster"):
            return leaf_cluster_to_id[id(node)]
        _, left, right, dist, size = node
        left_id = get_node_id(left)
        right_id = get_node_id(right)
        my_id = next_internal_id[0]
        next_internal_id[0] += 1
        linkage_rows.append([float(left_id), float(right_id), dist, float(size)])
        return my_id

    get_node_id(tree)

    if not linkage_rows:
        return BisectingKMeansResult(
            linkage=np.zeros((0, 4), dtype=np.float64),
            leaf_cluster_assignments=leaf_cluster_assignments,
            n_leaf_clusters=n_leaf_clusters,
        )

    linkage = np.array(linkage_rows, dtype=np.float64)
    return BisectingKMeansResult(
        linkage=linkage,
        leaf_cluster_assignments=leaf_cluster_assignments,
        n_leaf_clusters=n_leaf_clusters,
    )


# ---------------------------------------------------------------------------
# Dendrogram mass propagation + frontier cut (from dendrogram_ops.py)
# ---------------------------------------------------------------------------

def compute_dendrogram_masses(
    linked: NDArray[np.floating],
    leaf_masses: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Propagate leaf masses up the dendrogram (internal = sum of children)."""
    n_samples = len(leaf_masses)
    n_nodes = 2 * n_samples - 1
    node_masses = np.zeros(n_nodes, dtype=np.float64)
    node_masses[:n_samples] = leaf_masses
    for i in range(len(linked)):
        idx1 = int(linked[i, 0])
        idx2 = int(linked[i, 1])
        parent_idx = n_samples + i
        node_masses[parent_idx] = node_masses[idx1] + node_masses[idx2]
    return node_masses


def get_frontier_nodes(
    linked: NDArray[np.floating],
    node_masses: NDArray[np.floating],
    threshold: float,
) -> NDArray[np.int64]:
    """Frontier nodes: mass > threshold, and either a leaf or all children below."""
    n_samples = (len(node_masses) + 1) // 2
    frontier: List[int] = []
    root = len(node_masses) - 1

    def traverse(node_idx: int) -> None:
        if node_masses[node_idx] <= threshold:
            return
        if node_idx < n_samples:
            frontier.append(node_idx)
            return
        row_idx = node_idx - n_samples
        child1 = int(linked[row_idx, 0])
        child2 = int(linked[row_idx, 1])
        c1_heavy = node_masses[child1] > threshold
        c2_heavy = node_masses[child2] > threshold
        if not c1_heavy and not c2_heavy:
            frontier.append(node_idx)
        else:
            if c1_heavy:
                traverse(child1)
            if c2_heavy:
                traverse(child2)

    traverse(root)
    return np.array(frontier, dtype=np.int64)


def find_mass_threshold_for_n_topics(
    linked: NDArray[np.floating],
    node_masses: NDArray[np.floating],
    n_topics_desired: int,
) -> float:
    """Binary-search a mass threshold yielding ~``n_topics_desired`` frontier nodes."""
    unique_masses = np.sort(np.unique(node_masses))
    unique_masses = unique_masses[unique_masses > 1e-9]
    if len(unique_masses) == 0:
        LOGGER.warning("No non-zero masses found, returning 0.0 threshold")
        return 0.0

    low = 0
    high = len(unique_masses) - 1
    best_threshold = unique_masses[len(unique_masses) // 2]
    min_diff = float("inf")

    while low <= high:
        mid_idx = (low + high) // 2
        threshold = unique_masses[mid_idx]
        k = len(get_frontier_nodes(linked, node_masses, threshold))
        if abs(k - n_topics_desired) < min_diff:
            min_diff = abs(k - n_topics_desired)
            best_threshold = threshold
        if k < n_topics_desired:
            high = mid_idx - 1
        else:
            low = mid_idx + 1

    final_k = len(get_frontier_nodes(linked, node_masses, best_threshold))
    LOGGER.info(
        "Found mass threshold %.4e yielding %d topics (desired: %d)",
        best_threshold, final_k, n_topics_desired,
    )
    return best_threshold


def map_leaves_to_frontier_clusters(
    linked: NDArray[np.floating],
    frontier_nodes: NDArray[np.int64],
    n_samples: int,
) -> NDArray[np.int64]:
    """Map each leaf to its containing frontier cluster id (-1 if none)."""
    leaf_to_cluster = np.full(n_samples, -1, dtype=np.int64)

    def map_cluster_recursive(node_idx: int, cluster_id: int) -> None:
        if node_idx < n_samples:
            leaf_to_cluster[node_idx] = cluster_id
            return
        row_idx = node_idx - n_samples
        child1 = int(linked[row_idx, 0])
        child2 = int(linked[row_idx, 1])
        map_cluster_recursive(child1, cluster_id)
        map_cluster_recursive(child2, cluster_id)

    for cid, f_node in enumerate(frontier_nodes):
        map_cluster_recursive(int(f_node), cid)
    return leaf_to_cluster


def cluster_by_mass_frontier(
    embeddings: NDArray[np.floating],
    masses: NDArray[np.floating],
    desired_n_topics: int,
    *,
    sample_weights: Optional[NDArray[np.floating]] = None,
    min_cluster_size: int = 1,
    random_state: int = 42,
) -> NDArray[np.intp]:
    """Bisecting-KMeans dendrogram + mass-threshold frontier cut -> labels.

    Mirrors ``wiki_curriculum_demo.cluster_embeddings`` exactly: bisecting
    K-Means with ``sample_weights`` (None reproduces the wiki demo), masses
    aggregated to leaves, threshold chosen for ~``desired_n_topics`` topics.
    Returns a per-point cluster label array.
    """
    res = bisecting_kmeans_linkage(
        embeddings,
        sample_weights=sample_weights,
        min_cluster_size=min_cluster_size,
        random_state=random_state,
        verbose=False,
    )
    linked = res.linkage
    leaf_assign = res.leaf_cluster_assignments
    n_leaf = res.n_leaf_clusters
    leaf_masses = np.zeros(n_leaf, dtype=np.float64)
    np.add.at(leaf_masses, leaf_assign, masses)
    node_masses = compute_dendrogram_masses(linked, leaf_masses)
    threshold = find_mass_threshold_for_n_topics(linked, node_masses, desired_n_topics)
    frontier_nodes = get_frontier_nodes(linked, node_masses, threshold)
    leaf_to_frontier = map_leaves_to_frontier_clusters(linked, frontier_nodes, n_leaf)
    return leaf_to_frontier[leaf_assign]
