#!/usr/bin/env python3
"""Build curriculum from a target topic using citation-based prerequisite mining.

This module provides functions to build a curriculum for a target topic
by mining prerequisite relationships using personalized PageRank on the citation graph.

The workflow:
1. Load cloud and embeddings
2. Build citation graph from works
3. Build dendrogram via agglomerative clustering on embeddings
4. Run Personalized PageRank from target works → identify foundational content
5. Propagate masses up dendrogram → find frontier set of curriculum topics
6. Compute inter-topic dependency matrix via batched PageRank
7. Save outputs for downstream analysis and visualization

Use via the main.py unified entry point or import functions directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from numpy.typing import NDArray
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage

from bookgap.cloud import Cloud, DendrogramResult
from bookgap.configs.config_store import register_configs
from bookgap.util import LOGGER, TIMER

from .config import CurriculumConfig, get_curriculum_output_dir
from .dendrogram_ops import (
    compute_dendrogram_masses,
    find_mass_threshold_for_n_topics,
    get_frontier_nodes,
    map_leaves_to_frontier_clusters,
)
from .pagerank import (
    compute_frontier_adjacency,
    compute_pagerank,
)

# Logging is initialized by the pipeline entry point (main.py)

# Register configs
register_configs()

# Register curriculum config
cs = ConfigStore.instance()
cs.store(name="curriculum", node=CurriculumConfig())


def _find_work_indices(
    work_ids: List[str],
    target_ids: List[str],
) -> List[int]:
    """Find indices of target works in the work_ids list.
    
    Args:
        work_ids: List of work IDs.
        target_ids: List of OpenAlex work IDs to find.
        
    Returns:
        List of indices for found works.
    """
    id_to_idx = {wid: i for i, wid in enumerate(work_ids)}
    
    found_indices: List[int] = []
    missing_ids: List[str] = []
    
    for target_id in target_ids:
        if target_id in id_to_idx:
            found_indices.append(id_to_idx[target_id])
        else:
            missing_ids.append(target_id)
    
    if missing_ids:
        LOGGER.warning(
            f"Could not find {len(missing_ids)} target works in cloud: "
            f"{missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}"
        )
    
    LOGGER.info(f"Found {len(found_indices)}/{len(target_ids)} target works in cloud")
    return found_indices


# Scipy agglomerative methods
SCIPY_LINKAGE_METHODS = {"ward", "average", "complete", "single"}


@TIMER.time(formatters=dict(embeddings=lambda e: f"<shape={e.shape}>"))
# @MEMORY.cache
def _build_dendrogram(
    embeddings: NDArray[np.floating],
    method: str = "ward",
    min_cluster_size: int = 1,
    max_depth: Optional[int] = None,
) -> DendrogramResult:
    """Build dendrogram via hierarchical clustering on embeddings.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim).
        method: Clustering method:
            - 'ward', 'average', 'complete', 'single': scipy agglomerative (O(n²) memory)
            - 'bisecting_kmeans': recursive bisecting K-Means (O(n) memory)
        min_cluster_size: For bisecting_kmeans, stop splitting clusters smaller than this.
        max_depth: For bisecting_kmeans, maximum tree depth (None = no limit).
        
    Returns:
        DendrogramResult containing linkage matrix and leaf cluster mappings.
    """
    n_samples = len(embeddings)
    
    if method in SCIPY_LINKAGE_METHODS:
        LOGGER.info(f"Building dendrogram with {method} linkage on {len(embeddings)} embeddings...")
        linked = linkage(embeddings, method=method)
        LOGGER.info(f"Dendrogram built: {len(linked)} merges")
        # Standard scipy linkage: each point is its own leaf cluster
        return DendrogramResult(
            linkage=linked,
            leaf_cluster_assignments=np.arange(n_samples, dtype=np.intp),
            n_leaf_clusters=n_samples,
        )
    elif method == "bisecting_kmeans":
        from artifacts_scripts.curriculum.scalable_clustering import bisecting_kmeans_linkage
        LOGGER.info(f"Building dendrogram with bisecting K-Means on {len(embeddings)} embeddings...")
        result = bisecting_kmeans_linkage(
            embeddings,
            min_cluster_size=min_cluster_size,
            max_depth=max_depth,
            verbose=True,
        )
        LOGGER.info(f"Dendrogram built: {len(result.linkage)} merges, {result.n_leaf_clusters} leaf clusters")
        return DendrogramResult(
            linkage=result.linkage,
            leaf_cluster_assignments=result.leaf_cluster_assignments,
            n_leaf_clusters=result.n_leaf_clusters,
        )
    else:
        raise ValueError(f"Unknown linkage method: {method}. "
                        f"Options: {SCIPY_LINKAGE_METHODS | {'bisecting_kmeans'}}")


def _compute_linearization_order(adj_matrix: NDArray[np.floating]) -> NDArray[np.int64]:
    """Compute linearization order based on citation in-degree minus out-degree.
    
    Topics with high in-degree (cited by many other topics) and low out-degree
    (cite few other topics) are prerequisites and should come first.
    
    Args:
        adj_matrix: Adjacency matrix where adj[i,j] is how much topic i depends on topic j.
        
    Returns:
        Array of indices sorted by linearization score (prerequisites first).
    """
    n = adj_matrix.shape[0]
    scores = {}
    
    for k in range(n):
        # How much other topics depend on k (k is a prerequisite for others)
        cited_by_others = adj_matrix[:, k].sum()
        # How much k depends on other topics
        cites_others = adj_matrix[k, :].sum()
        # High score = prerequisite (cited much, cites little)
        scores[k] = cited_by_others - cites_others
    
    # Sort descending (prerequisites first)
    sorted_indices = sorted(range(n), key=lambda k: scores[k], reverse=True)
    return np.array(sorted_indices, dtype=np.int64)


def run_curriculum_building(
    cfg: CurriculumConfig
) -> Tuple[Cloud, NDArray[np.floating], Optional[NDArray[np.intp]]]:
    """Execute the curriculum building pipeline.
    
    When cfg.ppr_first is True (default), the pipeline is:
    1. Build citation graph on all works
    2. Run PPR from target works on full graph
    3. Select top-k works by PPR mass
    4. Build dendrogram only on filtered works (Ward's is now tractable!)
    5. Lift PPR masses to dendrogram → find frontier → linearize
    
    When cfg.ppr_first is False (legacy), the pipeline is:
    1. Build citation graph and dendrogram on all works
    2. Run PPR → lift to dendrogram → find frontier
    
    Args:
        cfg: CurriculumConfig object.
        
    Returns:
        Tuple of:
        - cloud: The Cloud object
        - embeddings_for_clustering: The embeddings used for clustering
        - filtered_indices: Array of original indices if ppr_first=True, else None
    """
    # Setup output directory
    output_dir = get_curriculum_output_dir(cfg.output_dir)
    
    # Log startup
    LOGGER.info("=" * 60)
    LOGGER.info("Curriculum Building")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Target work IDs: {len(cfg.target_work_ids)} works")
    LOGGER.info(f"Desired topics: {cfg.desired_n_topics}")
    LOGGER.info(f"PageRank alpha: {cfg.pagerank_alpha}")
    LOGGER.info(f"PPR-first mode: {cfg.ppr_first}")
    if cfg.ppr_first:
        LOGGER.info(f"PPR top-k: {cfg.ppr_top_k}")
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info("")
    
    # Build the cloud
    LOGGER.info("Loading cloud...")
    cloud: Cloud = instantiate(cfg.experiment.cloud)
    LOGGER.info(f"Cloud: {cloud.name()}")
    LOGGER.info(f"Embedder: {cloud.embedder_model.value}")
    LOGGER.info(f"Denoiser: {cloud.denoiser}")
    
    work_ids = cloud.work_ids()
    embeddings = cloud.embeddings()
    n_total = len(work_ids)
    
    LOGGER.info(f"Loaded {n_total} works with embeddings of shape {embeddings.shape}")
    
    # Find target work indices
    if not cfg.target_work_ids:
        raise ValueError("No target_work_ids specified in configuration")
    
    seed_indices = _find_work_indices(work_ids, cfg.target_work_ids)
    if not seed_indices:
        raise ValueError("No target works found in cloud - cannot build curriculum")
    
    # Build citation graph on ALL works
    LOGGER.info("Building citation adjacency matrix...")
    adj = cloud.citation_graph(normalize=False)
    
    # Create seed vector for personalized PageRank
    seeds = np.zeros(n_total, dtype=np.float64)
    seeds[seed_indices] = 1.0 / len(seed_indices)
    
    # Run Personalized PageRank on FULL graph
    LOGGER.info("Running Personalized PageRank from target topic...")
    full_point_masses = compute_pagerank(
        adj,
        seeds,
        alpha=cfg.pagerank_alpha,
        tol=cfg.pagerank_tol,
        max_iter=cfg.pagerank_max_iter,
    )
    
    # Log mass distribution
    LOGGER.info(
        f"PageRank mass distribution: min={full_point_masses.min():.4e}, "
        f"max={full_point_masses.max():.4e}, sum={full_point_masses.sum():.4f}"
    )
    
    if cfg.ppr_first:
        # PPR-FIRST PIPELINE: Filter to top-k works, then cluster
        top_k = min(cfg.ppr_top_k, n_total)
        LOGGER.info(f"Selecting top {top_k} works by PPR mass...")
        
        # Get indices of top-k works by mass
        top_k_indices = np.argsort(full_point_masses)[-top_k:][::-1]  # Descending order
        
        # Create filtered data structures
        filtered_work_ids = [work_ids[i] for i in top_k_indices]
        filtered_embeddings = embeddings[top_k_indices]
        filtered_point_masses = full_point_masses[top_k_indices]
        
        # Map from original index to filtered index
        original_to_filtered = {orig: filt for filt, orig in enumerate(top_k_indices)}
        
        # Re-map seed indices to filtered space (if they're in top-k)
        filtered_seed_indices = [
            original_to_filtered[i] for i in seed_indices if i in original_to_filtered
        ]
        if not filtered_seed_indices:
            # Seeds should always be in top-k since they have high PPR mass
            LOGGER.warning("No seed works in top-k - using all top-k works")
        
        n_samples = len(filtered_work_ids)
        point_masses = filtered_point_masses
        work_ids_for_clustering = filtered_work_ids
        embeddings_for_clustering = filtered_embeddings
        
        # For inter-topic computation, we use the FULL citation graph to capture
        # indirect dependencies through papers outside the top-k. We'll create a
        # cluster assignment array for the full graph where only top-k works have
        # cluster assignments (the rest will be -1, meaning unassigned).
        # This is set up after clustering below.
        
        LOGGER.info(f"Filtered to {n_samples} works (top {100*n_samples/n_total:.1f}% by PPR mass)")
        LOGGER.info(f"Mass retained: {point_masses.sum():.4f} ({100*point_masses.sum()/full_point_masses.sum():.1f}%)")
    else:
        # LEGACY PIPELINE: Cluster all works
        n_samples = n_total
        point_masses = full_point_masses
        work_ids_for_clustering = work_ids
        embeddings_for_clustering = embeddings
        top_k_indices = np.arange(n_total)  # Identity mapping
    
    # Build dendrogram on (possibly filtered) embeddings
    LOGGER.info("Building dendrogram...")
    if (not cfg.ppr_first) and cfg.linkage_method == "bisecting_kmeans":
        # Full-cloud + bisecting_kmeans is expensive: leverage Cloud-level caching/subsumption.
        dendrogram_result = cloud.dendrogram(
            "bisecting_kmeans",
            min_cluster_size=cfg.min_cluster_size,
            max_depth=cfg.max_cluster_depth,
        )
    else:
        dendrogram_result = _build_dendrogram(
            embeddings_for_clustering,
            method=cfg.linkage_method,
            min_cluster_size=cfg.min_cluster_size,
            max_depth=cfg.max_cluster_depth,
        )
    linked = dendrogram_result.linkage
    leaf_cluster_assignments = dendrogram_result.leaf_cluster_assignments
    n_leaf_clusters = dendrogram_result.n_leaf_clusters
    
    # Aggregate point masses to leaf cluster masses
    # When n_leaf_clusters < n_samples, multiple points belong to each leaf cluster
    LOGGER.info(f"Aggregating masses: {n_samples} points -> {n_leaf_clusters} leaf clusters")
    leaf_masses = np.zeros(n_leaf_clusters, dtype=np.float64)
    np.add.at(leaf_masses, leaf_cluster_assignments, point_masses)
    
    # Propagate masses up dendrogram
    LOGGER.info("Propagating masses up dendrogram...")
    node_masses = compute_dendrogram_masses(linked, leaf_masses)
    
    # Find frontier set
    LOGGER.info(f"Finding frontier with ~{cfg.desired_n_topics} topics...")
    threshold = find_mass_threshold_for_n_topics(linked, node_masses, cfg.desired_n_topics)
    frontier_nodes = get_frontier_nodes(linked, node_masses, threshold)
    
    LOGGER.info(f"Found {len(frontier_nodes)} frontier topics")
    
    # Map dendrogram leaves to frontier clusters
    # Note: frontier_nodes are indices into the node_masses array (0..n_leaf_clusters-1 for leaves)
    leaf_to_frontier = map_leaves_to_frontier_clusters(linked, frontier_nodes, n_leaf_clusters)
    
    # Map original points to frontier clusters via the leaf cluster assignments
    point_to_cluster = leaf_to_frontier[leaf_cluster_assignments]
    
    # Log cluster sizes (counting original points, not leaf clusters)
    cluster_sizes = [
        int((point_to_cluster == cid).sum()) for cid in range(len(frontier_nodes))
    ]
    LOGGER.info(f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.1f}")
    
    # Compute inter-topic dependency matrix using the FULL citation graph
    # This captures indirect dependencies through papers outside the clustered set
    LOGGER.info("Computing inter-topic dependency matrix (using full citation graph)...")
    
    if cfg.ppr_first:
        # Create cluster assignment for full graph: only top-k works have assignments
        # Rest are -1 (unassigned) but can still serve as intermediaries in PageRank
        point_to_cluster_full = np.full(n_total, -1, dtype=np.int64)
        point_to_cluster_full[top_k_indices] = point_to_cluster
        cluster_mapping_for_frontier = point_to_cluster_full
    else:
        cluster_mapping_for_frontier = point_to_cluster
    
    adj_frontier = compute_frontier_adjacency(
        adj,  # Always use full citation graph
        frontier_nodes,
        cluster_mapping_for_frontier,
        alpha=cfg.pagerank_alpha,
        tol=cfg.pagerank_tol,
        max_iter=cfg.pagerank_max_iter,
    )
    
    # Compute linearization order
    LOGGER.info("Computing linearization order...")
    linearization_order = _compute_linearization_order(adj_frontier)
    LOGGER.info(f"Linearization order: {linearization_order.tolist()}")
    
    # Prepare metadata
    metadata = {
        "cloud_name": cloud.name(),
        "embedder": cloud.embedder_model.value,
        "denoiser": str(cloud.denoiser),
        "n_works_total": n_total,
        "n_works_clustered": n_samples,
        "ppr_first": cfg.ppr_first,
        "ppr_top_k": cfg.ppr_top_k if cfg.ppr_first else None,
        "n_target_works": len(seed_indices),
        "target_work_ids": list(cfg.target_work_ids),  # Convert from Hydra ListConfig
        "desired_n_topics": cfg.desired_n_topics,
        "actual_n_topics": len(frontier_nodes),
        "pagerank_alpha": cfg.pagerank_alpha,
        "linkage_method": cfg.linkage_method,
        "mass_threshold": float(threshold),
        "frontier_topic_ids": frontier_nodes.tolist(),
        "linearization_order": linearization_order.tolist(),
        "cluster_sizes": cluster_sizes,
    }
    
    # Save outputs
    LOGGER.info("Saving outputs...")
    
    # Frontier topics
    np.save(output_dir / "frontier_nodes.npy", frontier_nodes)
    LOGGER.info(f"  frontier_nodes.npy: {frontier_nodes.shape}")
    
    # Point to cluster mapping (original points -> frontier clusters)
    np.save(output_dir / "point_to_cluster.npy", point_to_cluster)
    LOGGER.info(f"  point_to_cluster.npy: {point_to_cluster.shape}")
    
    # Leaf cluster assignments (original points -> dendrogram leaves)
    np.save(output_dir / "leaf_cluster_assignments.npy", leaf_cluster_assignments)
    LOGGER.info(f"  leaf_cluster_assignments.npy: {leaf_cluster_assignments.shape}")
    
    # Inter-topic adjacency
    np.save(output_dir / "adj_frontier.npy", adj_frontier)
    LOGGER.info(f"  adj_frontier.npy: {adj_frontier.shape}")
    
    # Citation graph (for visualization)
    sp.save_npz(output_dir / "citation_graph.npz", adj)
    LOGGER.info(f"  citation_graph.npz: {adj.shape} (sparse)")
    
    # Node masses
    np.save(output_dir / "node_masses.npy", node_masses)
    LOGGER.info(f"  node_masses.npy: {node_masses.shape}")
    
    # Leaf masses (aggregated PageRank scores per leaf cluster)
    np.save(output_dir / "leaf_masses.npy", leaf_masses)
    LOGGER.info(f"  leaf_masses.npy: {leaf_masses.shape}")
    
    # Point masses (PageRank scores per original point)
    np.save(output_dir / "point_masses.npy", point_masses)
    LOGGER.info(f"  point_masses.npy: {point_masses.shape}")
    
    # Linkage matrix
    np.save(output_dir / "linked.npy", linked)
    LOGGER.info(f"  linked.npy: {linked.shape}")
    
    # Metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    LOGGER.info(f"  metadata.json")
    
    # Also save work IDs for each frontier topic
    # For each topic, find all original points that belong to it
    topic_work_ids_output: Dict[int, List[str]] = {}
    for topic_idx in range(len(frontier_nodes)):
        # Get original point indices for this topic
        point_indices = np.where(point_to_cluster == topic_idx)[0]
        topic_work_ids_output[topic_idx] = [work_ids_for_clustering[int(i)] for i in point_indices]
    
    with open(output_dir / "topic_work_ids.json", "w") as f:
        json.dump(topic_work_ids_output, f, indent=2)
    LOGGER.info(f"  topic_work_ids.json")
    
    # Save mapping from filtered indices to original work IDs (for ppr_first mode)
    if cfg.ppr_first:
        with open(output_dir / "filtered_work_ids.json", "w") as f:
            json.dump(work_ids_for_clustering, f, indent=2)
        LOGGER.info(f"  filtered_work_ids.json")
    
    # Print summary
    LOGGER.info("=" * 60)
    LOGGER.info("CURRICULUM BUILDING COMPLETED")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info(f"Frontier topics: {len(frontier_nodes)}")
    LOGGER.info(f"Linearization order (topic indices): {linearization_order.tolist()}")
    
    # Load titles for nice output (cached separately from full works)
    LOGGER.info("Loading titles for summary...")
    all_titles = cloud.work_titles()
    if cfg.ppr_first:
        titles_for_clustering = [all_titles[i] for i in top_k_indices]
    else:
        titles_for_clustering = all_titles
    
    # Print topic summary
    LOGGER.info("")
    LOGGER.info("Topic Summary (in linearization order):")
    for rank, topic_idx in enumerate(linearization_order):
        node_idx = frontier_nodes[topic_idx]
        mass = node_masses[node_idx]
        size = cluster_sizes[topic_idx]
        # Get a sample work title from this topic
        point_indices = np.where(point_to_cluster == topic_idx)[0]
        sample_title = titles_for_clustering[int(point_indices[0])][:60] if len(point_indices) > 0 else "N/A"
        LOGGER.info(
            f"  {rank + 1}. Topic {topic_idx} (node {node_idx}): "
            f"mass={mass:.4f}, size={size}, sample='{sample_title}...'"
        )
    
    # Return cloud, embeddings, and filtered indices for downstream use
    filtered_indices_out = top_k_indices if cfg.ppr_first else None
    return cloud, embeddings_for_clustering, filtered_indices_out
