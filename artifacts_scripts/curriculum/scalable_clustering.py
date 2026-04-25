"""Scalable clustering methods for large datasets.

Ward linkage requires O(n²) memory which is prohibitive for 1M+ points.
This module provides scalable alternatives that produce scipy-compatible
linkage matrices.
"""

import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans, MiniBatchKMeans

from bookgap.util import LOGGER

@dataclass
class BisectingKMeansResult:
    """Result of bisecting K-Means clustering.
    
    Attributes:
        linkage: Scipy-compatible linkage matrix of shape (n_merges, 4).
        leaf_cluster_assignments: Array of shape (n_samples,) mapping each original
            point to its leaf cluster ID. Leaf cluster IDs are in range [0, n_leaf_clusters).
            These IDs correspond to the "leaves" of the dendrogram.
        n_leaf_clusters: Number of leaf clusters (= n_merges + 1 when there are merges).
    """
    linkage: NDArray[np.floating]
    leaf_cluster_assignments: NDArray[np.intp]
    n_leaf_clusters: int


def bisecting_kmeans_linkage(
    embeddings: NDArray[np.floating],
    *,
    sample_weights: Optional[NDArray[np.floating]] = None,
    min_cluster_size: int = 1,
    max_depth: Optional[int] = None,
    random_state: Optional[int] = 42,
    use_minibatch: bool = True,
    batch_size: int = 10000,
    verbose: bool = True,
) -> BisectingKMeansResult:
    """Build a hierarchical clustering via recursive bisecting K-Means.

    This produces a scipy-compatible linkage matrix with O(n) memory,
    suitable for datasets with millions of points.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim).
        sample_weights: Optional per-point weights of shape (n_samples,).
            When provided, ``KMeans``/``MiniBatchKMeans`` are called with
            ``sample_weight=`` so centroid placement and the bisection
            choice both respect the weights, and the Ward-like distance
            uses weighted centroids and weighted sizes. Useful for
            curriculum building, where post-PageRank relevance weights
            let high-relevance works act as cluster anchors instead of
            getting peeled off as singletons by relevance-blind
            geometry. ``None`` (default) is unweighted (backwards-
            compatible behaviour).
        min_cluster_size: Stop splitting clusters smaller than this size.
            Larger values = faster but coarser hierarchy.
            Default 1 means split down to individual points.
        max_depth: Maximum tree depth. None means no limit.
            Useful for very large datasets to limit recursion.
        random_state: Random seed for reproducibility.
        use_minibatch: Use MiniBatchKMeans for speed (recommended for large n).
        batch_size: Batch size for MiniBatchKMeans.
        verbose: Log progress.

    Returns:
        BisectingKMeansResult containing:
        - linkage: Linkage matrix compatible with scipy.cluster.hierarchy.
          When min_cluster_size=1 and max_depth=None: shape is (n_samples - 1, 4).
          Otherwise: shape is (n_merges, 4) where n_merges < n_samples - 1.
          Columns are: [left_child, right_child, distance, n_points_in_cluster].
        - leaf_cluster_assignments: Array mapping each point to its leaf cluster.
        - n_leaf_clusters: Number of leaf clusters in the tree.

    Note:
        When min_cluster_size > 1 or max_depth is set, the "leaves" of the tree
        may contain multiple points. Use leaf_cluster_assignments to map points
        to leaf cluster IDs, then aggregate masses before using dendrogram_ops.
    """
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
        # Single point or empty - one leaf cluster containing all points
        assignments = np.zeros(n_samples, dtype=np.intp)
        return BisectingKMeansResult(
            linkage=np.zeros((0, 4), dtype=np.float64),
            leaf_cluster_assignments=assignments,
            n_leaf_clusters=max(1, n_samples),
        )
    if n_samples <= min_cluster_size:
        # Entire dataset is one leaf cluster - no merges needed
        assignments = np.zeros(n_samples, dtype=np.intp)
        return BisectingKMeansResult(
            linkage=np.zeros((0, 4), dtype=np.float64),
            leaf_cluster_assignments=assignments,
            n_leaf_clusters=1,
        )

    if verbose:
        weight_note = "" if sample_weights is None else " (weighted)"
        LOGGER.info(f"Bisecting K-Means: {n_samples} points, min_cluster_size={min_cluster_size}, max_depth={max_depth}{weight_note}")
    
    # Increase recursion limit for deep trees
    old_limit = sys.getrecursionlimit()
    needed_limit = max(old_limit, 10000)  # Reasonable limit for tree depth
    if needed_limit > old_limit:
        sys.setrecursionlimit(needed_limit)
    
    try:
        return _bisecting_kmeans_impl(
            embeddings,
            sample_weights=sample_weights,
            n_samples=n_samples,
            min_cluster_size=min_cluster_size,
            max_depth=max_depth,
            random_state=random_state,
            use_minibatch=use_minibatch,
            batch_size=batch_size,
            verbose=verbose,
        )
    finally:
        # Restore recursion limit
        sys.setrecursionlimit(old_limit)


def _bisecting_kmeans_impl(
    embeddings: NDArray[np.floating],
    *,
    sample_weights: Optional[NDArray[np.floating]],
    n_samples: int,
    min_cluster_size: int,
    max_depth: Optional[int],
    random_state: Optional[int],
    use_minibatch: bool,
    batch_size: int,
    verbose: bool,
) -> BisectingKMeansResult:
    """Implementation of bisecting K-Means."""
    
    # Track progress for logging
    splits_done = [0]
    log_interval = max(1, n_samples // (min_cluster_size * 20))  # Log ~20 times
    
    # Track which points belong to which leaf cluster
    # This will be filled in during tree-to-linkage conversion
    leaf_cluster_assignments = np.full(n_samples, -1, dtype=np.intp)
    next_leaf_cluster_id = [0]  # Mutable counter for leaf clusters
    
    def build_tree(indices: NDArray[np.intp], depth: int = 0) -> Tuple:
        """Recursively build tree.
        
        Returns:
            ('leaf', point_index) for single-point leaf
            ('leaf_cluster', indices_array) for multi-point leaf (when stopping early)
            ('internal', left_subtree, right_subtree, dist, size) for internal node
        """
        size = len(indices)
        
        # Stop conditions
        if size == 1:
            return ('leaf', int(indices[0]))
        
        if size <= min_cluster_size:
            return ('leaf_cluster', indices)
        
        if max_depth is not None and depth >= max_depth:
            return ('leaf_cluster', indices)
        
        # Special case: exactly 2 points
        if size == 2:
            dist = float(np.linalg.norm(embeddings[indices[0]] - embeddings[indices[1]]))
            return ('internal',
                    ('leaf', int(indices[0])),
                    ('leaf', int(indices[1])),
                    dist, 2)
        
        # Split using K-Means (optionally weighted: high-relevance points
        # pull the centroid harder, so the bisection respects importance).
        cluster_embeddings = embeddings[indices]
        cluster_weights = sample_weights[indices] if sample_weights is not None else None

        # sklearn's KMeans crashes on all-zero sample weights (divides
        # by sample_weight.sum() during seeding). Drop to unweighted
        # for that subset; the post-split weighted-centroid path already
        # falls back to unit weights when a side's mass is zero.
        if cluster_weights is not None and float(cluster_weights.sum()) == 0:
            cluster_weights = None

        if use_minibatch and size > batch_size:
            kmeans = MiniBatchKMeans(
                n_clusters=2,
                max_iter=100,
                n_init=1,
                random_state=random_state,
                batch_size=min(batch_size, size),
            )
        else:
            kmeans = KMeans(
                n_clusters=2,
                max_iter=100,
                n_init=1,
                algorithm="lloyd",
                random_state=random_state,
            )

        labels = kmeans.fit_predict(cluster_embeddings, sample_weight=cluster_weights)

        left_indices = indices[labels == 0]
        right_indices = indices[labels == 1]

        # Handle degenerate case (all points in one cluster)
        if len(left_indices) == 0 or len(right_indices) == 0:
            mid = size // 2
            left_indices = indices[:mid]
            right_indices = indices[mid:]

        # Compute Ward-like distance. With weights, use weighted centroids
        # and weighted "size" so the merge cost reflects the relevance mass
        # being unified at this dendrogram height.
        if sample_weights is not None:
            w_left = sample_weights[left_indices]
            w_right = sample_weights[right_indices]
            wsum_left = float(w_left.sum())
            wsum_right = float(w_right.sum())
            # Fall back to unit weights if a side's mass is zero (avoid /0).
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
        dist = float(np.sqrt(2 * n_left * n_right / (n_left + n_right)) *
                     np.linalg.norm(left_centroid - right_centroid))
        
        # Progress logging
        splits_done[0] += 1
        if verbose and splits_done[0] % log_interval == 0:
            LOGGER.info(f"  Bisecting K-Means: {splits_done[0]} splits done")
        
        # Recurse
        left_tree = build_tree(left_indices, depth + 1)
        right_tree = build_tree(right_indices, depth + 1)
        
        return ('internal', left_tree, right_tree, dist, size)
    
    # Build the tree
    tree = build_tree(np.arange(n_samples))
    
    if verbose:
        LOGGER.info(f"  Bisecting K-Means: {splits_done[0]} total splits, converting to linkage")
    
    # Convert tree to linkage matrix
    # The "leaves" of the dendrogram are either:
    # 1. Single points (when min_cluster_size=1)
    # 2. Clusters of points (when min_cluster_size > 1)
    # We need to:
    # - Assign each leaf cluster a sequential ID (0, 1, 2, ...)
    # - Record which points belong to which leaf cluster
    # - Build linkage matrix with internal node IDs starting at n_leaf_clusters
    
    linkage_rows: List[List[float]] = []
    
    # First pass: count leaf clusters and assign them sequential IDs
    leaf_cluster_to_id: Dict[int, int] = {}  # id(indices_array) -> leaf_cluster_id
    # Record depth of each leaf cluster (depth=0 for root)
    leaf_depths: List[int] = []
    
    def count_leaves(node, depth: int = 0) -> int:
        """Count and assign IDs to leaf clusters and record leaf depths.

        Args:
            node: tree node tuple
            depth: depth of this node (root=0)
        """
        node_type = node[0]
        if node_type == 'leaf':
            # Single point leaf
            point_idx = node[1]
            if id(node) not in leaf_cluster_to_id:
                leaf_id = next_leaf_cluster_id[0]
                leaf_cluster_to_id[id(node)] = leaf_id
                leaf_cluster_assignments[point_idx] = leaf_id
                next_leaf_cluster_id[0] += 1
                leaf_depths.append(depth)
            return 1
        elif node_type == 'leaf_cluster':
            # Multi-point leaf cluster
            indices = node[1]
            if id(node) not in leaf_cluster_to_id:
                leaf_id = next_leaf_cluster_id[0]
                leaf_cluster_to_id[id(node)] = leaf_id
                leaf_cluster_assignments[indices] = leaf_id
                next_leaf_cluster_id[0] += 1
                leaf_depths.append(depth)
            return 1
        else:  # 'internal'
            _, left, right, _, _ = node
            return count_leaves(left, depth + 1) + count_leaves(right, depth + 1)
    
    count_leaves(tree)
    n_leaf_clusters = next_leaf_cluster_id[0]
    
    if verbose:
        LOGGER.info(f"  Bisecting K-Means: {n_leaf_clusters} leaf clusters")
    # Compute and log leaf depth distribution statistics for diagnostics.
    if leaf_depths:
        depths_arr = np.array(leaf_depths, dtype=np.intp)
        mean_depth = float(depths_arr.mean())
        std_depth = float(depths_arr.std())
        deciles = np.percentile(depths_arr, np.linspace(0, 100, 11)).tolist()
        # Format deciles for compact logging
        deciles_str = ", ".join(f"{d:.0f}" for d in deciles)
        if verbose:
            LOGGER.info(
                f"  Leaf depth stats — mean={mean_depth:.3f}, std={std_depth:.3f}, deciles=[{deciles_str}]"
            )
    
    # Second pass: build linkage matrix
    # Internal node IDs start at n_leaf_clusters
    next_internal_id = [n_leaf_clusters]
    
    def get_node_id(node) -> int:
        """Get scipy-compatible node ID, creating linkage rows as needed."""
        node_type = node[0]
        
        if node_type == 'leaf' or node_type == 'leaf_cluster':
            # Return the pre-assigned leaf cluster ID
            return leaf_cluster_to_id[id(node)]
        
        else:  # 'internal'
            _, left, right, dist, size = node
            
            left_id = get_node_id(left)
            right_id = get_node_id(right)
            
            # Create merge row
            my_id = next_internal_id[0]
            next_internal_id[0] += 1
            linkage_rows.append([float(left_id), float(right_id), dist, float(size)])
            
            return my_id
    
    get_node_id(tree)
    
    if not linkage_rows:
        # Edge case: no merges (single leaf cluster)
        return BisectingKMeansResult(
            linkage=np.zeros((0, 4), dtype=np.float64),
            leaf_cluster_assignments=leaf_cluster_assignments,
            n_leaf_clusters=n_leaf_clusters,
        )
    
    linkage = np.array(linkage_rows, dtype=np.float64)
    
    if verbose:
        LOGGER.info(f"Bisecting K-Means: linkage matrix shape {linkage.shape}")
    
    return BisectingKMeansResult(
        linkage=linkage,
        leaf_cluster_assignments=leaf_cluster_assignments,
        n_leaf_clusters=n_leaf_clusters,
    )
