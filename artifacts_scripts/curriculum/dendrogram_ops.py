"""Dendrogram operations for curriculum building.

Functions for manipulating hierarchical clustering results including:
- Propagating masses from leaves to internal nodes
- Finding frontier nodes based on mass thresholds
- Gathering leaves under dendrogram nodes
"""

from __future__ import annotations

from typing import List, Set

import numpy as np
from numpy.typing import NDArray

from bookgap.util import LOGGER


def compute_dendrogram_masses(
    linked: NDArray[np.floating],
    leaf_masses: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Propagate masses from leaves up to the root of the dendrogram.
    
    Args:
        linked: Linkage matrix from scipy.cluster.hierarchy.linkage, shape (n_samples-1, 4).
                Each row [idx1, idx2, dist, count] represents a merge.
        leaf_masses: Array of masses for each leaf node, shape (n_samples,).
        
    Returns:
        Array of masses for all nodes (leaves + internal), shape (2*n_samples - 1,).
        Indices 0..n_samples-1 are leaves, n_samples..2*n_samples-2 are internal nodes.
    """
    n_samples = len(leaf_masses)
    n_nodes = 2 * n_samples - 1
    node_masses = np.zeros(n_nodes, dtype=np.float64)
    
    # Set leaf masses
    node_masses[:n_samples] = leaf_masses
    
    # Propagate up the tree: each internal node's mass = sum of children's masses
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
    """Find frontier nodes in the dendrogram based on mass threshold.
    
    A frontier node is one where:
    - Its mass > threshold, AND
    - Either it's a leaf, OR all its children have mass <= threshold
    
    This identifies the "right" level of granularity for the curriculum.
    
    Args:
        linked: Linkage matrix from scipy.cluster.hierarchy.linkage.
        node_masses: Masses for all nodes (from compute_dendrogram_masses).
        threshold: Mass threshold for frontier identification.
        
    Returns:
        Array of frontier node indices.
    """
    n_samples = (len(node_masses) + 1) // 2
    frontier: List[int] = []
    root = len(node_masses) - 1
    
    def traverse(node_idx: int) -> None:
        # Skip nodes below threshold
        if node_masses[node_idx] <= threshold:
            return
        
        # Leaf check - leaves above threshold are frontier nodes
        if node_idx < n_samples:
            frontier.append(node_idx)
            return
        
        # Internal node - check children
        row_idx = node_idx - n_samples
        child1 = int(linked[row_idx, 0])
        child2 = int(linked[row_idx, 1])
        
        c1_heavy = node_masses[child1] > threshold
        c2_heavy = node_masses[child2] > threshold
        
        if not c1_heavy and not c2_heavy:
            # Both children below threshold - this node is frontier
            frontier.append(node_idx)
        else:
            # Recurse into heavy children
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
    """Find a mass threshold that yields approximately n_topics_desired frontier nodes.
    
    Uses binary search over unique mass values to find the threshold that produces
    a frontier set closest to the desired size.
    
    Args:
        linked: Linkage matrix from scipy.cluster.hierarchy.linkage.
        node_masses: Masses for all nodes (from compute_dendrogram_masses).
        n_topics_desired: Target number of frontier topics.
        
    Returns:
        Mass threshold value.
    """
    # Possible thresholds are the masses of the nodes
    unique_masses = np.sort(np.unique(node_masses))
    # Remove extremely small masses (zeros) to speed up
    unique_masses = unique_masses[unique_masses > 1e-9]
    
    if len(unique_masses) == 0:
        LOGGER.warning("No non-zero masses found, returning 0.0 threshold")
        return 0.0
    
    low = 0
    high = len(unique_masses) - 1
    best_threshold = unique_masses[len(unique_masses) // 2]
    min_diff = float('inf')
    
    # Binary search (approximately monotonic: lower threshold -> more clusters)
    while low <= high:
        mid_idx = (low + high) // 2
        threshold = unique_masses[mid_idx]
        frontier = get_frontier_nodes(linked, node_masses, threshold)
        k = len(frontier)
        
        if abs(k - n_topics_desired) < min_diff:
            min_diff = abs(k - n_topics_desired)
            best_threshold = threshold
        
        if k < n_topics_desired:
            # Too few clusters -> need lower threshold (go deeper)
            high = mid_idx - 1
        else:
            # Too many clusters -> need higher threshold
            low = mid_idx + 1
    
    final_k = len(get_frontier_nodes(linked, node_masses, best_threshold))
    LOGGER.info(
        f"Found mass threshold {best_threshold:.4e} yielding {final_k} topics "
        f"(desired: {n_topics_desired})"
    )
    return best_threshold


def gather_leaves(
    node_idx: int,
    linked: NDArray[np.floating],
    n_samples: int,
) -> List[int]:
    """Recursively gather all leaf indices under a dendrogram node.
    
    Args:
        node_idx: Index of the dendrogram node (leaf or internal).
        linked: Linkage matrix from scipy.cluster.hierarchy.linkage.
        n_samples: Number of leaf samples.
        
    Returns:
        List of leaf indices under this node.
    """
    leaves: List[int] = []
    
    def _gather(idx: int) -> None:
        if idx < n_samples:
            # Leaf node
            leaves.append(idx)
        else:
            # Internal node - recurse into children
            row_idx = idx - n_samples
            child1 = int(linked[row_idx, 0])
            child2 = int(linked[row_idx, 1])
            _gather(child1)
            _gather(child2)
    
    _gather(node_idx)
    return leaves


def get_children(
    node_idx: int,
    linked: NDArray[np.floating],
    n_samples: int,
) -> tuple[int, int]:
    """Get the two children of an internal dendrogram node.
    
    Args:
        node_idx: Index of an internal node (must be >= n_samples).
        linked: Linkage matrix from scipy.cluster.hierarchy.linkage.
        n_samples: Number of leaf samples.
        
    Returns:
        Tuple of (child1_idx, child2_idx).
        
    Raises:
        ValueError: If node_idx is a leaf node.
    """
    if node_idx < n_samples:
        raise ValueError(f"Node {node_idx} is a leaf, not an internal node")
    
    row_idx = node_idx - n_samples
    child1 = int(linked[row_idx, 0])
    child2 = int(linked[row_idx, 1])
    return child1, child2


def map_leaves_to_frontier_clusters(
    linked: NDArray[np.floating],
    frontier_nodes: NDArray[np.int64],
    n_samples: int,
) -> NDArray[np.int64]:
    """Map each leaf to its containing frontier cluster.
    
    Args:
        linked: Linkage matrix from scipy.cluster.hierarchy.linkage.
        frontier_nodes: Array of frontier node indices.
        n_samples: Number of leaf samples.
        
    Returns:
        Array of shape (n_samples,) where entry i is the cluster ID (index into
        frontier_nodes) that leaf i belongs to. Leaves not in any frontier cluster
        have value -1.
    """
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
        map_cluster_recursive(f_node, cid)
    
    return leaf_to_cluster
