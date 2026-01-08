"""Compute statistics for mass distributions in curriculum building.

This module provides functions to compute diagnostic statistics about
the point mass distributions produced by PageRank and CircuitRank,
helping to identify issues with parameter tuning (e.g., leakage too high).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class MassDistributionStats:
    """Statistics for a mass distribution.
    
    Attributes:
        total_mass: Sum of all masses.
        min_mass: Minimum mass value.
        max_mass: Maximum mass value.
        mean_mass: Mean mass value.
        std_mass: Standard deviation of mass values.
        n_nonzero: Number of non-zero masses.
        n_above_1e8: Number of masses > 1e-8.
        n_above_1e6: Number of masses > 1e-6.
        n_above_1e4: Number of masses > 1e-4.
        entropy: Entropy of normalized mass distribution.
        max_entropy: Maximum possible entropy (log(n)).
        entropy_ratio: Ratio of entropy to max entropy.
        effective_n: Effective number of works (exp(entropy)).
        gini: Gini coefficient (0 = equal, 1 = concentrated).
        percentiles: Dictionary of percentile values.
    """
    total_mass: float
    min_mass: float
    max_mass: float
    mean_mass: float
    std_mass: float
    n_nonzero: int
    n_above_1e8: int
    n_above_1e6: int
    n_above_1e4: int
    entropy: float
    max_entropy: float
    entropy_ratio: float
    effective_n: float
    gini: float
    percentiles: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ClusterSizeStats:
    """Statistics for cluster sizes.
    
    Attributes:
        n_clusters: Number of clusters.
        min_size: Minimum cluster size.
        max_size: Maximum cluster size.
        mean_size: Mean cluster size.
        std_size: Standard deviation of cluster sizes.
        n_singletons: Number of singleton clusters (size 1).
        sizes: List of cluster sizes.
    """
    n_clusters: int
    min_size: int
    max_size: int
    mean_size: float
    std_size: float
    n_singletons: int
    sizes: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def compute_mass_distribution_stats(
    masses: NDArray[np.floating],
    percentile_values: List[float] = [50, 90, 95, 99, 99.9, 99.99],
) -> MassDistributionStats:
    """Compute comprehensive statistics for a mass distribution.
    
    Args:
        masses: Array of mass values (e.g., PageRank or CircuitRank scores).
        percentile_values: List of percentile values to compute.
        
    Returns:
        MassDistributionStats with computed statistics.
    """
    n = len(masses)
    
    # Basic statistics
    total_mass = float(masses.sum())
    min_mass = float(masses.min())
    max_mass = float(masses.max())
    mean_mass = float(masses.mean())
    std_mass = float(masses.std())
    
    # Count statistics
    n_nonzero = int((masses > 0).sum())
    n_above_1e8 = int((masses > 1e-8).sum())
    n_above_1e6 = int((masses > 1e-6).sum())
    n_above_1e4 = int((masses > 1e-4).sum())
    
    # Entropy (using normalized distribution)
    if total_mass > 0:
        p = masses / total_mass
        p_nonzero = p[p > 0]
        entropy = float(-np.sum(p_nonzero * np.log(p_nonzero)))
    else:
        entropy = 0.0
    
    max_entropy = float(np.log(n)) if n > 0 else 0.0
    entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.0
    effective_n = float(np.exp(entropy))
    
    # Gini coefficient
    sorted_masses = np.sort(masses)
    cumsum = np.cumsum(sorted_masses)
    if cumsum[-1] > 0:
        gini = float(1 - 2 * np.sum(cumsum) / (n * cumsum[-1]) + 1/n)
    else:
        gini = 0.0
    
    # Percentiles
    percentiles = {
        f"p{p}": float(np.percentile(masses, p))
        for p in percentile_values
    }
    
    return MassDistributionStats(
        total_mass=total_mass,
        min_mass=min_mass,
        max_mass=max_mass,
        mean_mass=mean_mass,
        std_mass=std_mass,
        n_nonzero=n_nonzero,
        n_above_1e8=n_above_1e8,
        n_above_1e6=n_above_1e6,
        n_above_1e4=n_above_1e4,
        entropy=entropy,
        max_entropy=max_entropy,
        entropy_ratio=entropy_ratio,
        effective_n=effective_n,
        gini=gini,
        percentiles=percentiles,
    )


def compute_cluster_size_stats(
    point_to_cluster: NDArray[np.intp],
    n_clusters: int,
) -> ClusterSizeStats:
    """Compute statistics for cluster sizes.
    
    Args:
        point_to_cluster: Array mapping point indices to cluster IDs.
        n_clusters: Number of clusters.
        
    Returns:
        ClusterSizeStats with computed statistics.
    """
    sizes = []
    for c in range(n_clusters):
        size = int((point_to_cluster == c).sum())
        sizes.append(size)
    
    sizes_array = np.array(sizes)
    
    return ClusterSizeStats(
        n_clusters=n_clusters,
        min_size=int(sizes_array.min()) if len(sizes) > 0 else 0,
        max_size=int(sizes_array.max()) if len(sizes) > 0 else 0,
        mean_size=float(sizes_array.mean()) if len(sizes) > 0 else 0.0,
        std_size=float(sizes_array.std()) if len(sizes) > 0 else 0.0,
        n_singletons=int((sizes_array == 1).sum()),
        sizes=sizes,
    )


def compute_adjacency_stats(adj_frontier: NDArray[np.floating]) -> Dict[str, Any]:
    """Compute statistics for the inter-topic adjacency matrix.
    
    Args:
        adj_frontier: Inter-topic adjacency matrix.
        
    Returns:
        Dictionary with adjacency statistics.
    """
    n = adj_frontier.shape[0]
    
    # Mask out diagonal
    off_diag = adj_frontier[~np.eye(n, dtype=bool)]
    
    return {
        "shape": list(adj_frontier.shape),
        "min": float(adj_frontier.min()),
        "max": float(adj_frontier.max()),
        "mean": float(adj_frontier.mean()),
        "off_diag_sum": float(off_diag.sum()),
        "off_diag_mean": float(off_diag.mean()) if len(off_diag) > 0 else 0.0,
        "n_nonzero_off_diag": int((off_diag > 0).sum()),
        "is_all_zeros": bool(np.allclose(adj_frontier, 0)),
    }
