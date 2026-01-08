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
from pathlib import Path
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

from .config import CurriculumConfig, get_curriculum_output_dir, derive_output_dir_name, RankMethod
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
from .circuitrank import (
    compute_graph_potential,
    compute_frontier_adjacency_potential,
)
from .mass_statistics import (
    compute_mass_distribution_stats,
    compute_cluster_size_stats,
    compute_adjacency_stats,
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


def _compute_linearization_order_from_adjacency(adj_matrix: NDArray[np.floating]) -> NDArray[np.int64]:
    """Compute linearization order based on citation in-degree minus out-degree.
    
    DEPRECATED: This heuristic doesn't distinguish source from target topics.
    Use _compute_linearization_order_from_potentials instead when potentials
    are available.
    
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


def _compute_linearization_order_from_potentials(
    point_potentials: NDArray[np.floating],
    point_to_cluster: NDArray[np.intp],
    n_topics: int,
) -> NDArray[np.int64]:
    """Compute linearization order from work-level potentials.
    
    Uses voltage clamping potentials where source=1V, target=0V.
    Topics are sorted by average potential of their works:
    - High potential = close to source = learn early
    - Low potential = close to target = learn late
    
    Args:
        point_potentials: Potential (voltage) for each work from voltage clamping solve.
        point_to_cluster: Mapping from work index to topic index.
        n_topics: Number of topics.
        
    Returns:
        Array of topic indices sorted by average potential (descending = source first).
    """
    # Compute average potential per topic
    topic_potentials = np.zeros(n_topics, dtype=np.float64)
    topic_counts = np.zeros(n_topics, dtype=np.int64)
    
    for work_idx, topic_idx in enumerate(point_to_cluster):
        if topic_idx >= 0:  # Skip unassigned works
            topic_potentials[topic_idx] += point_potentials[work_idx]
            topic_counts[topic_idx] += 1
    
    # Avoid division by zero for empty topics
    topic_counts = np.maximum(topic_counts, 1)
    topic_avg_potentials = topic_potentials / topic_counts
    
    # Sort by descending average potential (high = source = first)
    sorted_indices = np.argsort(-topic_avg_potentials)
    return sorted_indices.astype(np.int64)


def run_curriculum_building(
    cfg: CurriculumConfig,
    output_dir: Optional[Path] = None,
) -> Tuple[Cloud, NDArray[np.floating], Optional[NDArray[np.intp]]]:
    """Execute the curriculum building pipeline.
    
    When cfg.rank_before_clustering is True, the pipeline is:
    1. Build citation graph on all works
    2. Run ranking (PageRank/CircuitRank) from target works on full graph
    3. Select top-k works by relevance mass
    4. Build dendrogram only on filtered works (Ward's is now tractable!)
    5. Lift masses to dendrogram → find frontier → linearize
    
    When cfg.rank_before_clustering is False (default), the pipeline is:
    1. Build citation graph and dendrogram on all works
    2. Run ranking → lift to dendrogram → find frontier
    
    Args:
        cfg: CurriculumConfig object.
        output_dir: Pre-resolved output directory Path. If None, will be derived and created.
        
    Returns:
        Tuple of:
        - cloud: The Cloud object
        - embeddings_for_clustering: The embeddings used for clustering
        - filtered_indices: Array of original indices if rank_before_clustering=True, else None
    """
    # Use provided output_dir or derive/create it
    if output_dir is None:
        auto_name = cfg.output_dir in (None, "")
        if auto_name:
            output_dir_name = derive_output_dir_name(cfg)
        else:
            output_dir_name = str(cfg.output_dir)
        output_dir = get_curriculum_output_dir(output_dir_name, auto_derived=auto_name)
    
    # Log startup
    LOGGER.info("=" * 60)
    LOGGER.info("Curriculum Building")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Target work IDs: {len(cfg.target_work_ids)} works")
    LOGGER.info(f"Desired topics: {cfg.desired_n_topics}")
    LOGGER.info(f"Ranking method: {cfg.method.value}")
    if cfg.method == RankMethod.PAGE_RANK:
        LOGGER.info(f"PageRank alpha: {cfg.pagerank.alpha}")
    elif cfg.method == RankMethod.CIRCUIT_RANK:
        LOGGER.info(f"CircuitRank leakage: {cfg.circuitrank.leakage}")
        if cfg.circuitrank.beta is not None:
            LOGGER.info(f"CircuitRank beta: {cfg.circuitrank.beta}")
        if cfg.circuitrank.normalize:
            LOGGER.info(f"CircuitRank normalize: True (PageRank mode)")
            # Log equivalent alpha for reference
            equiv_alpha = 1.0 / (1.0 + cfg.circuitrank.leakage)
            LOGGER.info(f"CircuitRank equivalent alpha: {equiv_alpha:.4f}")
    
    LOGGER.info(f"Rank-before-clustering: {cfg.rank_before_clustering}")
    if cfg.rank_before_clustering:
        LOGGER.info(f"Ranking top-k: {cfg.ranking_top_k}")
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
    
    # Run Ranking (PageRank or CircuitRank) on FULL graph
    full_point_potentials: Optional[NDArray[np.floating]] = None  # For potential-based linearization
    if cfg.method == RankMethod.PAGE_RANK:
        LOGGER.info("Running Personalized PageRank from target topic...")
        # Create seed vector for personalized PageRank
        seeds = np.zeros(n_total, dtype=np.float64)
        seeds[seed_indices] = 1.0 / len(seed_indices)
        
        full_point_masses = compute_pagerank(
            adj,
            seeds,
            alpha=cfg.pagerank.alpha,
            tol=cfg.pagerank.tol,
            max_iter=cfg.pagerank.max_iter,
        )
    elif cfg.method == RankMethod.CIRCUIT_RANK:
        if cfg.circuitrank.use_voltage_clamping:
            LOGGER.info("Running CircuitRank (Voltage Clamping)...")
            # Fix target nodes at φ=0 (ground - things to learn)
            # Fix known works at φ=1 (source - things already known)
            # Current flows from known (1V) toward targets (0V)
            # Curriculum order: start from known, end at targets
            fixed_potentials = {idx: 0.0 for idx in seed_indices}
            
            # If known works are specified, clamp them to 1V (source)
            if cfg.source_work_ids:
                source_indices = _find_work_indices(work_ids, cfg.source_work_ids)
                if source_indices:
                    for idx in source_indices:
                        fixed_potentials[idx] = 1.0
                    LOGGER.info(
                        f"Voltage clamping: {len(source_indices)} known works at 1V (source), "
                        f"{len(seed_indices)} targets at 0V (ground)"
                    )
                else:
                    LOGGER.warning(
                        f"No source works found in cloud - using targets only. "
                        f"Requested: {cfg.source_work_ids}"
                    )
                    LOGGER.info(f"Voltage clamping: {len(seed_indices)} targets at 0V (leakage from 1V)")
            else:
                LOGGER.info(f"Voltage clamping: {len(seed_indices)} targets at 0V (leakage from 1V)")
            
            full_point_masses = compute_graph_potential(
                adj,
                fixed_potentials=fixed_potentials,
                leakage=cfg.circuitrank.leakage,
                beta=cfg.circuitrank.beta,
                normalize=cfg.circuitrank.normalize,
                gmres_tol=cfg.circuitrank.gmres_tol,
                gmres_maxiter=cfg.circuitrank.gmres_maxiter,
                gmres_preconditioner=cfg.circuitrank.gmres_preconditioner,
                gmres_log_every=cfg.circuitrank.gmres_log_every,
            )
        else:
            LOGGER.info("Running CircuitRank (Current Injection) from target topic...")
            
            # Build current sources vector
            # Targets: inject current (+1 total, distributed among targets)
            # Sources (known works): suction current (-1 total, distributed among sources)
            current_sources = np.zeros(n_total, dtype=np.float64)
            current_sources[seed_indices] = 1.0 / len(seed_indices)
            
            # Find source (known) work indices if provided
            source_indices: list[int] = []
            if cfg.source_work_ids:
                source_indices = _find_work_indices(work_ids, cfg.source_work_ids)
                if source_indices:
                    current_sources[source_indices] = -1.0 / len(source_indices)
                    LOGGER.info(
                        f"Current injection: +1.0 into {len(seed_indices)} targets, "
                        f"-1.0 from {len(source_indices)} known works"
                    )
                else:
                    LOGGER.warning(
                        f"No source works found in cloud - using targets only. "
                        f"Requested: {cfg.source_work_ids}"
                    )
                    LOGGER.info(f"Current injection: +1.0 into {len(seed_indices)} targets only")
            else:
                LOGGER.info(f"Current injection: +1.0 into {len(seed_indices)} targets (no known works)")
            
            # Solve for potentials with no fixed nodes (all nodes free, leakage to ground)
            potentials = compute_graph_potential(
                adj,
                current_sources=current_sources,
                leakage=cfg.circuitrank.leakage,
                beta=cfg.circuitrank.beta,
                normalize=cfg.circuitrank.normalize,
                gmres_tol=cfg.circuitrank.gmres_tol,
                gmres_maxiter=cfg.circuitrank.gmres_maxiter,
                gmres_preconditioner=cfg.circuitrank.gmres_preconditioner,
                gmres_log_every=cfg.circuitrank.gmres_log_every,
            )
            
            # Compute current flow through each node - this is symmetric wrt source/target
            # Nodes with high current flow are "on the path" between source and target
            from artifacts_scripts.curriculum.circuitrank import compute_current_flow
            full_point_masses = compute_current_flow(
                adj, potentials, normalize=cfg.circuitrank.normalize
            )
            LOGGER.info(
                f"Current flow: min={full_point_masses.min():.4e}, "
                f"max={full_point_masses.max():.4e}, sum={full_point_masses.sum():.4f}"
            )
            
            # Also compute potentials via voltage clamping for linearization
            # This gives us directional information: source=1V (if specified), target=0V
            # Even without source works, target-only clamping gives useful "distance to target" ordering
            LOGGER.info("Computing potentials via voltage clamping for topic ordering...")
            voltage_clamp_potentials = {idx: 0.0 for idx in seed_indices}  # targets at 0V
            
            if cfg.source_work_ids and source_indices:
                for idx in source_indices:
                    voltage_clamp_potentials[idx] = 1.0  # sources at 1V
                LOGGER.info(
                    f"Voltage clamping: {len(source_indices)} sources at 1V, "
                    f"{len(seed_indices)} targets at 0V"
                )
            else:
                LOGGER.info(
                    f"Voltage clamping: {len(seed_indices)} targets at 0V only "
                    "(no source works, ordering by distance from target)"
                )
            
            full_point_potentials = compute_graph_potential(
                adj,
                fixed_potentials=voltage_clamp_potentials,
                leakage=cfg.circuitrank.leakage,
                beta=cfg.circuitrank.beta,
                normalize=cfg.circuitrank.normalize,
                gmres_tol=cfg.circuitrank.gmres_tol,
                gmres_maxiter=cfg.circuitrank.gmres_maxiter,
                gmres_preconditioner=cfg.circuitrank.gmres_preconditioner,
                gmres_log_every=cfg.circuitrank.gmres_log_every,
            )
            LOGGER.info(
                f"Voltage clamping potentials: min={full_point_potentials.min():.4f}, "
                f"max={full_point_potentials.max():.4f}, mean={full_point_potentials.mean():.4f}"
            )
    else:
        raise ValueError(f"Unknown ranking method: {cfg.method}")
    
    # Log mass distribution with detailed statistics
    LOGGER.info(
        f"Mass distribution: min={full_point_masses.min():.4e}, "
        f"max={full_point_masses.max():.4e}, sum={full_point_masses.sum():.4f}"
    )
    
    # Compute detailed mass statistics
    full_mass_stats = compute_mass_distribution_stats(full_point_masses)
    LOGGER.info(
        f"Mass entropy: {full_mass_stats.entropy:.4f} "
        f"({100*full_mass_stats.entropy_ratio:.1f}% of max), "
        f"effective_n={full_mass_stats.effective_n:.0f}"
    )
    
    if cfg.rank_before_clustering:
        # RANK-FIRST PIPELINE: Filter to top-k works, then cluster
        top_k = min(cfg.ranking_top_k, n_total)
        LOGGER.info(f"Selecting top {top_k} works by PPR mass...")
        
        # Get indices of top-k works by mass
        top_k_indices = np.argsort(full_point_masses)[-top_k:][::-1]  # Descending order
        
        # Create filtered data structures
        filtered_work_ids = [work_ids[i] for i in top_k_indices]
        filtered_embeddings = embeddings[top_k_indices]
        filtered_point_masses = full_point_masses[top_k_indices]
        filtered_point_potentials = full_point_potentials[top_k_indices] if full_point_potentials is not None else None
        
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
        point_potentials = filtered_point_potentials
        work_ids_for_clustering = filtered_work_ids
        embeddings_for_clustering = filtered_embeddings
        
        # For inter-topic computation, we use the FULL citation graph to capture
        # indirect dependencies through papers outside the top-k. We'll create a
        # cluster assignment array for the full graph where only top-k works have
        # cluster assignments (the rest will be -1, meaning unassigned).
        # This is set up after clustering below.
        
        LOGGER.info(f"Filtered to {n_samples} works (top {100*n_samples/n_total:.1f}% by relevance)")
        LOGGER.info(f"Mass retained: {point_masses.sum():.4f} ({100*point_masses.sum()/full_point_masses.sum():.1f}%)")
    else:
        # FULL PIPELINE: Cluster all works
        n_samples = n_total
        point_masses = full_point_masses
        point_potentials = full_point_potentials
        work_ids_for_clustering = work_ids
        embeddings_for_clustering = embeddings
        top_k_indices = np.arange(n_total)  # Identity mapping
    
    # Build dendrogram on (possibly filtered) embeddings
    LOGGER.info("Building dendrogram...")
    if (not cfg.rank_before_clustering) and cfg.linkage_method == "bisecting_kmeans":
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
    
    if cfg.rank_before_clustering:
        # Create cluster assignment for full graph: only top-k works have assignments
        # Rest are -1 (unassigned) but can still serve as intermediaries in ranking
        point_to_cluster_full = np.full(n_total, -1, dtype=np.int64)
        point_to_cluster_full[top_k_indices] = point_to_cluster
        cluster_mapping_for_frontier = point_to_cluster_full
    else:
        cluster_mapping_for_frontier = point_to_cluster
    
    if cfg.method == RankMethod.PAGE_RANK:
        adj_frontier = compute_frontier_adjacency(
            adj,  # Always use full citation graph
            frontier_nodes,
            cluster_mapping_for_frontier,
            alpha=cfg.pagerank.alpha,
            tol=cfg.pagerank.tol,
            max_iter=cfg.pagerank.max_iter,
        )
    elif cfg.method == RankMethod.CIRCUIT_RANK:
        adj_frontier = compute_frontier_adjacency_potential(
            adj,
            frontier_nodes,
            cluster_mapping_for_frontier,
            leakage=cfg.circuitrank.leakage,
            beta=cfg.circuitrank.beta,
            normalize=cfg.circuitrank.normalize,
            gmres_tol=cfg.circuitrank.gmres_tol,
            gmres_maxiter=cfg.circuitrank.gmres_maxiter,
            gmres_preconditioner=cfg.circuitrank.gmres_preconditioner,
            gmres_log_every=cfg.circuitrank.gmres_log_every,
        )
    else:
        raise ValueError(f"Unknown ranking method: {cfg.method}")
    
    # Log inter-topic adjacency statistics
    adj_stats = compute_adjacency_stats(adj_frontier)
    LOGGER.info(
        f"Inter-topic adjacency: off_diag_sum={adj_stats['off_diag_sum']:.4f}, "
        f"n_nonzero={adj_stats['n_nonzero_off_diag']}"
    )
    if adj_stats['is_all_zeros']:
        LOGGER.warning(
            "Inter-topic adjacency matrix is all zeros! "
            "This may indicate a bug or extreme parameter settings."
        )
    
    # Compute linearization order and make it canonical (topic 0 = first to cover)
    LOGGER.info("Computing linearization order...")
    n_topics = len(frontier_nodes)
    
    if point_potentials is not None:
        # Use potential-based ordering: high potential = close to source = learn first
        LOGGER.info("Using potential-based linearization (source→target ordering)")
        linearization_order = _compute_linearization_order_from_potentials(
            point_potentials, point_to_cluster, n_topics
        )
    else:
        # Fall back to adjacency-based heuristic (less accurate)
        LOGGER.info("Using adjacency-based linearization (no source works available)")
        linearization_order = _compute_linearization_order_from_adjacency(adj_frontier)

    # Relabel topics so that new topic 0 is the first in the linearization order.
    perm = linearization_order  # old_idx -> new position order
    inverse_perm = np.empty_like(perm)
    inverse_perm[perm] = np.arange(n_topics)

    # Reorder frontier nodes and derived structures
    frontier_nodes = frontier_nodes[perm]
    adj_frontier = adj_frontier[np.ix_(perm, perm)]
    cluster_sizes = [cluster_sizes[i] for i in perm]
    point_to_cluster = inverse_perm[point_to_cluster]

    # The canonical linearization is now 0..n_topics-1
    linearization_order = np.arange(n_topics, dtype=np.int64)
    
    # Compute cluster size statistics
    cluster_stats = compute_cluster_size_stats(point_to_cluster, n_topics)
    LOGGER.info(
        f"Cluster stats: n_singletons={cluster_stats.n_singletons}, "
        f"size_std={cluster_stats.std_size:.1f}"
    )
    
    # Prepare metadata
    metadata = {
        "cloud_name": cloud.name(),
        "embedder": cloud.embedder_model.value,
        "denoiser": str(cloud.denoiser),
        "n_works_total": n_total,
        "n_works_clustered": n_samples,
        "rank_before_clustering": cfg.rank_before_clustering,
        "ranking_top_k": cfg.ranking_top_k if cfg.rank_before_clustering else None,
        "n_target_works": len(seed_indices),
        "target_work_ids": list(cfg.target_work_ids),  # Convert from Hydra ListConfig
        "desired_n_topics": cfg.desired_n_topics,
        "actual_n_topics": len(frontier_nodes),
        "ranking_method": cfg.method.value,
        "pagerank_alpha": cfg.pagerank.alpha if cfg.method == RankMethod.PAGE_RANK else None,
        "circuitrank_leakage": cfg.circuitrank.leakage if cfg.method == RankMethod.CIRCUIT_RANK else None,
        "linkage_method": cfg.linkage_method,
        "mass_threshold": float(threshold),
        "frontier_topic_ids": frontier_nodes.tolist(),
        "cluster_sizes": cluster_sizes,
        # Add statistics
        "mass_distribution_stats": full_mass_stats.to_dict(),
        "cluster_size_stats": cluster_stats.to_dict(),
        "adjacency_stats": adj_stats,
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
    # For each topic, find all original points that belong to it, sorted by mass
    topic_work_ids_output: Dict[int, List[str]] = {}
    for topic_idx in range(len(frontier_nodes)):
        # Get original point indices for this topic
        point_indices = np.where(point_to_cluster == topic_idx)[0]
        
        # Get masses for these points
        topic_masses = point_masses[point_indices]
        
        # Sort by mass descending
        sorted_local_indices = np.argsort(topic_masses)[::-1]
        sorted_point_indices = point_indices[sorted_local_indices]
        sorted_masses = topic_masses[sorted_local_indices]
        
        # Filter to top works accounting for threshold fraction of mass
        total_mass = sorted_masses.sum()
        if total_mass > 0:
            cumulative_mass = np.cumsum(sorted_masses)
            cutoff_mass = total_mass * cfg.topic_work_ids_mass_threshold
            # Find first index where cumulative mass >= cutoff_mass
            cutoff_idx = np.searchsorted(cumulative_mass, cutoff_mass)
            
            # Apply max count constraint if set
            if cfg.topic_work_ids_max_count is not None:
                # We want at most max_count items, so index should be at most max_count - 1
                max_idx = cfg.topic_work_ids_max_count - 1
                cutoff_idx = min(cutoff_idx, max_idx)
                
            # Include the item at cutoff_idx
            selected_indices = sorted_point_indices[:cutoff_idx + 1]
        else:
            selected_indices = sorted_point_indices
            if cfg.topic_work_ids_max_count is not None:
                selected_indices = selected_indices[:cfg.topic_work_ids_max_count]
            
        topic_work_ids_output[topic_idx] = [work_ids_for_clustering[int(i)] for i in selected_indices]
    
    with open(output_dir / "topic_work_ids.json", "w") as f:
        json.dump(topic_work_ids_output, f, indent=2)
    LOGGER.info(f"  topic_work_ids.json")
    
    # Save mapping from filtered indices to original work IDs (for rank_before_clustering mode)
    if cfg.rank_before_clustering:
        with open(output_dir / "filtered_work_ids.json", "w") as f:
            json.dump(work_ids_for_clustering, f, indent=2)
        LOGGER.info(f"  filtered_work_ids.json")
    
    # Print summary
    LOGGER.info("=" * 60)
    LOGGER.info("CURRICULUM BUILDING COMPLETED")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info(f"Frontier topics: {len(frontier_nodes)}")
    
    # Load titles for nice output (cached separately from full works)
    LOGGER.info("Loading titles for summary...")
    all_titles = cloud.work_titles()
    if cfg.rank_before_clustering:
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
    filtered_indices_out = top_k_indices if cfg.rank_before_clustering else None
    return cloud, embeddings_for_clustering, filtered_indices_out
