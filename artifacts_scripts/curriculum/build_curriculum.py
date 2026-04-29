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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from numpy.typing import NDArray
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage

from artifacts_scripts.curriculum.scalable_clustering import bisecting_kmeans_linkage
from bookgap.cloud import Cloud, DendrogramResult
from bookgap.configs.config_store import register_configs
from bookgap.util import LOGGER, TIMER

from .config import (
    CircuitRankSolveMode,
    CurriculumConfig,
    get_curriculum_output_dir,
    derive_output_dir_name,
    RankMethod,
)
from .dendrogram_ops import (
    compute_dendrogram_masses,
    find_mass_threshold_for_n_topics,
    get_frontier_nodes,
    map_leaves_to_frontier_clusters,
)
from .pagerank import (
    compute_frontier_adjacency,
    two_direction_pagerank,
)
from .circuitrank import (
    build_conductance_adjacency,
    compute_current_flow,
    compute_curriculum_ordering_potentials,
    compute_graph_potential,
    compute_graph_potential_galerkin,
    compute_frontier_adjacency_potential,
    transform_node_masses,
)
from .similarity_graph import (
    build_similarity_adjacency_from_cloud,
    mix_adjacency_matrices,
)

from .mass_statistics import (
    compute_mass_distribution_stats,
    compute_cluster_size_stats,
    compute_adjacency_stats,
)

# Register configs
register_configs()

# Register curriculum config
cs = ConfigStore.instance()
cs.store(name="curriculum", node=CurriculumConfig())


def _citation_jaccard_distance_matrix(
    sub_adj: sp.spmatrix,
    *,
    mode: str = "binary",
) -> NDArray[np.floating]:
    """Pairwise Jaccard / Ruzicka distance between citation neighbourhoods.

    Three modes selectable via ``mode``:

    - ``"binary"`` (default; original behaviour): treat each work's
      in+out neighbourhood (cited + citer set) as an unweighted set.
      ``J = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|``.

    - ``"weighted_unified"``: weight each neighbour by its edge weight
      in the row-normalised random-walk transition matrix
      ``P[i, j] = A[i, j] / out_deg(i)``. Each work's neighbourhood
      vector is the elementwise max of its row in P and its column in
      P (i.e. ``W = max(P, P^T)``), so an out-edge ``u → k`` contributes
      ``1/out_deg(u)`` and an in-edge ``v → u`` contributes
      ``1/out_deg(v)``. Down-weights citation-promiscuous reviewers
      (their per-edge contribution is diluted across their large
      reference list). Generalised Jaccard (a.k.a. Ruzicka):
      ``J = Σ min(w_u, w_v) / Σ max(w_u, w_v)``. Reduces to ``"binary"``
      when all degrees are equal.

    - ``"weighted_per_direction"``: same idea, but separately
      ℓ1-normalise the out-distribution and the in-distribution per
      node (out-distribution = row-normalised P; in-distribution =
      column-normalised A so each citee's column sums to 1). Concatenate
      the two as a single 2n-feature vector per node and apply the same
      Ruzicka similarity. Decouples out-degree noise from in-degree
      noise — the two directions can't drown each other out.

    For both weighted modes we exploit the identity
      ``Σ min(a, b) = (‖a‖_1 + ‖b‖_1 − ‖a − b‖_1) / 2``,
    so Ruzicka similarity is computable from row ℓ1-norms plus a
    pairwise L1 (cityblock) distance matrix:
      ``J = (R_u + R_v − L1) / (R_u + R_v + L1)``.
    """
    n = sub_adj.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    if mode == "binary":
        sym = (sub_adj + sub_adj.T) > 0  # union of cited + citer relationships
        sym = sym.astype(np.uint8).tocsr()
        # Each row's nonzero pattern = the neighbourhood set.
        # Jaccard(i, j) = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|.
        # |N(i) ∩ N(j)| = (sym @ sym.T)[i, j] when both are 0/1 indicators.
        # |N(i)| = row sum.
        intersections = (sym @ sym.T).toarray().astype(np.float64)
        sizes = np.asarray(sym.sum(axis=1)).flatten().astype(np.float64)
        unions = sizes[:, None] + sizes[None, :] - intersections
        # Avoid 0/0 for empty-neighbourhood rows: define Jaccard = 0 there
        # (-> max distance = 1).
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(unions > 0, intersections / unions, 0.0)
    elif mode in ("weighted_unified", "weighted_per_direction"):
        from scipy.spatial.distance import cdist

        sub_csr = sub_adj.tocsr().astype(np.float64)
        out_deg = np.asarray(sub_csr.sum(axis=1)).flatten()
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_out = np.where(out_deg > 0, 1.0 / out_deg, 0.0)
        # P[i, j] = A[i, j] / out_deg(i): random-walk transition matrix.
        P = sp.diags(inv_out) @ sub_csr

        if mode == "weighted_unified":
            # Weighted union neighbourhood: W[u, k] = max(P[u, k], P[k, u]).
            W = P.maximum(P.T).toarray()
        else:  # weighted_per_direction
            # In-distribution: column-normalise A so each citee's column
            # sums to 1; column k of Q is then citers(k)'s distribution.
            in_deg = np.asarray(sub_csr.sum(axis=0)).flatten()
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_in = np.where(in_deg > 0, 1.0 / in_deg, 0.0)
            Q = sub_csr @ sp.diags(inv_in)
            # Each row of W is [P[u, :] ; Q^T[u, :]] — out then in.
            W = sp.hstack([P, Q.T], format="csr").toarray()

        row_l1 = W.sum(axis=1)
        L1 = cdist(W, W, metric="cityblock")
        sums = row_l1[:, None] + row_l1[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = sums + L1
            sim = np.where(denom > 0, (sums - L1) / denom, 0.0)
    else:
        raise ValueError(
            f"Unknown citation_jaccard_mode: {mode!r}. "
            f"Expected 'binary', 'weighted_unified', or 'weighted_per_direction'."
        )

    dist = 1.0 - sim
    # Squash diagonal to 0 (distance to self).
    np.fill_diagonal(dist, 0.0)
    return dist


def _cluster_with_graph_bisect(
    adj: sp.spmatrix,
    *,
    n_clusters: int,
    seed: int = 42,
) -> NDArray[np.intp]:
    """Recursive Fiedler-based spectral bisection on a sparse graph.

    At each step, picks the largest current cluster, computes the
    Fiedler vector (second-smallest eigenvector) of its induced
    subgraph's normalised Laplacian, and splits by the median of that
    vector for balance. Recurses until ``n_clusters`` clusters exist.

    Different bias from global spectral clustering: each bisection
    adapts to the local subgraph structure rather than relying on a
    single global eigenvector decomposition. Produces a clean
    recursive hierarchy (you could read off a binary tree from the
    bisection sequence). The pick-largest heuristic mirrors bisecting
    K-means; an alternative is pick-by-Fiedler-value (bisect the
    cluster with the clearest cut next).

    Returns ``membership`` of length ``adj.shape[0]``, values in
    ``[0, n_clusters)``.
    """
    import heapq
    from scipy.sparse.linalg import eigsh

    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.intp)
    if n_clusters <= 1:
        return np.zeros(n, dtype=np.intp)
    if n_clusters >= n:
        return np.arange(n, dtype=np.intp)

    sym = ((adj + adj.T) * 0.5).tocsr()

    cluster_ids = np.zeros(n, dtype=np.intp)
    # heap entries: (-size, tiebreak, cluster_id, indices)
    # The tiebreak counter prevents heapq from comparing numpy arrays.
    heap: list = []
    counter = 0
    heapq.heappush(heap, (-n, counter, 0, np.arange(n, dtype=np.intp)))
    counter += 1
    next_cid = 1
    rng = np.random.RandomState(seed)

    def _fiedler_split(sub: sp.spmatrix, size: int) -> np.ndarray:
        """Return a boolean mask splitting ``sub`` via its Fiedler vector
        (median split). Falls back to a random 50/50 split on degenerate
        cases (no edges, eigsh failure, all-equal eigenvector)."""
        deg = np.asarray(sub.sum(axis=1)).flatten()
        if deg.max() <= 0:
            return rng.rand(size) >= 0.5
        with np.errstate(divide="ignore"):
            deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        D_inv_sqrt = sp.diags(deg_inv_sqrt)
        L_norm = sp.eye(size) - D_inv_sqrt @ sub @ D_inv_sqrt
        try:
            if size <= 200:
                # Dense path for small subgraphs — more reliable.
                eigvals, eigvecs = np.linalg.eigh(L_norm.toarray())
            else:
                k = 3  # ask for 3, take the second-smallest non-zero
                eigvals, eigvecs = eigsh(
                    L_norm, k=k, sigma=0, which="LM",
                )
            order = np.argsort(eigvals)
            eigvecs = eigvecs[:, order]
            # Skip the trivial all-constant eigenvector (eigval ~ 0);
            # prefer the next non-degenerate one as Fiedler. If a
            # subgraph has multiple connected components the second
            # eigenvalue is also ~0 and we use the corresponding
            # eigenvector as a (still-meaningful) component indicator.
            fiedler = eigvecs[:, 1]
        except Exception:
            return rng.rand(size) >= 0.5
        # Sign split (threshold at 0) — respects natural community
        # boundaries rather than forcing balance. Falls back to median
        # split if sign split is fully one-sided (the eigenvector's
        # sign is uniform, common for short eigenvectors with numeric
        # bias), and to random if even median is degenerate.
        mask = fiedler > 0
        if mask.all() or (~mask).all():
            med = float(np.median(fiedler))
            mask = fiedler > med
        if mask.all() or (~mask).all():
            return rng.rand(size) >= 0.5
        return mask

    while heap and next_cid < n_clusters:
        neg_size, _t, cid, indices = heapq.heappop(heap)
        size = -neg_size
        if size <= 1:
            heapq.heappush(heap, (neg_size, _t, cid, indices))
            break
        sub = sym[indices][:, indices]
        split_mask = _fiedler_split(sub, size)
        idx_a = indices[split_mask]
        idx_b = indices[~split_mask]
        new_cid = next_cid
        next_cid += 1
        cluster_ids[idx_b] = new_cid
        # idx_a keeps cid
        heapq.heappush(heap, (-len(idx_a), counter, cid, idx_a))
        counter += 1
        heapq.heappush(heap, (-len(idx_b), counter, new_cid, idx_b))
        counter += 1

    return cluster_ids


def _cluster_with_spectral(
    adj: sp.spmatrix,
    *,
    n_clusters: int,
    seed: int = 42,
) -> NDArray[np.intp]:
    """Run spectral clustering on a (possibly directed) sparse graph.

    Symmetrises the adjacency, treats it as a precomputed affinity
    matrix, computes the *K* smallest eigenvectors of the normalised
    Laplacian, and runs K-means in that embedding to assign each node
    to one of *K* communities. Used when ``cfg.linkage_method == 'spectral'``.

    Unlike Leiden's resolution-driven count (which is data-dependent and
    hard to control on tightly-knit corpora), spectral clustering takes
    *K* as a direct parameter — exactly *K* clusters by construction.
    Plays well with citation subgraphs where the underlying community
    structure is well-defined; quality on noisy / weakly-connected
    graphs is generally close to Leiden's but not strictly better.

    Returns: ``membership`` of length ``adj.shape[0]``, values in
    ``[0, n_clusters)``.
    """
    from sklearn.cluster import SpectralClustering  # type: ignore

    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.intp)
    if n_clusters >= n:
        # Each node gets its own cluster.
        return np.arange(n, dtype=np.intp)

    # Symmetrise — spectral clustering needs an undirected affinity.
    sym = (adj + adj.T) * 0.5
    # Add small self-loops so isolated nodes don't break Laplacian
    # normalisation (avoids divide-by-zero in degree matrix inverse).
    sym = sym + sp.eye(n, dtype=sym.dtype, format="csr") * 1e-9

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=seed,
        assign_labels="kmeans",
        n_init=10,
    )
    membership = sc.fit_predict(sym).astype(np.intp)
    return membership


def _cluster_with_leiden(
    adj: sp.spmatrix,
    *,
    resolution: float = 1.0,
    seed: int = 42,
) -> Tuple[NDArray[np.intp], int]:
    """Run Leiden community detection on a (possibly directed) sparse graph.

    Used when ``cfg.linkage_method == 'leiden'``. The graph is symmetrised
    (Leiden expects undirected), and the Reichardt-Bornholdt
    configuration null model is used with ``resolution`` as the
    resolution parameter — higher values yield more, smaller communities.

    Citations are the semantic relation curricula *means* (one paper
    cites another because it builds on it), so community detection on
    the citation subgraph of the top-k relevance-filtered works produces
    a topic structure based on the underlying relation rather than on
    embedding distance. See ``CLUSTERING_NOTES.md`` for context on why
    embedding-based clustering can produce singleton-blob failure modes
    on heterogeneous corpora.

    Returns: ``(membership, n_communities)``.
    """
    import igraph  # type: ignore[import-untyped]
    import leidenalg  # type: ignore[import-untyped]

    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.intp), 0
    # Symmetrise — Leiden treats edges as undirected.
    sym = (adj + adj.T) * 0.5
    sym_coo = sym.tocoo()
    edges = list(zip(sym_coo.row.tolist(), sym_coo.col.tolist()))
    weights = sym_coo.data.tolist()
    g = igraph.Graph(n=n, edges=edges, directed=False)
    if weights:
        g.es["weight"] = weights
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights="weight",
            seed=seed,
        )
    else:
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=seed,
        )
    membership = np.array(partition.membership, dtype=np.intp)
    n_communities = int(membership.max()) + 1 if len(membership) > 0 else 0
    return membership, n_communities


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
    min_cluster_weight: float = 0.0,
    max_depth: Optional[int] = None,
    sample_weights: Optional[NDArray[np.floating]] = None,
) -> DendrogramResult:
    """Build dendrogram via hierarchical clustering on embeddings.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim).
        method: Clustering method:
            - 'ward', 'average', 'complete', 'single': scipy agglomerative (O(n²) memory)
            - 'bisecting_kmeans': recursive bisecting K-Means (O(n) memory, unweighted)
            - 'weighted_bisecting_kmeans': recursive bisecting K-Means with
              ``sample_weights`` plumbed through KMeans + the Ward-like merge
              distance. Use for relevance-weighted clustering of a top-k
              subset where high-relevance works should anchor clusters.
        min_cluster_size: For bisecting_kmeans, stop splitting clusters smaller than this.
        min_cluster_weight: For weighted_bisecting_kmeans, stop splitting
            clusters whose total ``sample_weights`` sum is at or below this
            threshold. Acts in addition to ``min_cluster_size``.
        max_depth: For bisecting_kmeans, maximum tree depth (None = no limit).
        sample_weights: Optional per-point weights, only used by
            'weighted_bisecting_kmeans'. Must be shape (n_samples,) and
            non-negative.

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
    elif method in ("bisecting_kmeans", "weighted_bisecting_kmeans"):
        weighted = method == "weighted_bisecting_kmeans"
        LOGGER.info(
            f"Building dendrogram with {'weighted ' if weighted else ''}"
            f"bisecting K-Means on {len(embeddings)} embeddings..."
        )
        if weighted and sample_weights is None:
            raise ValueError(
                "weighted_bisecting_kmeans requires sample_weights; "
                "got None"
            )
        if min_cluster_weight > 0 and not weighted:
            raise ValueError(
                "min_cluster_weight > 0 requires weighted_bisecting_kmeans; "
                f"got method={method!r}"
            )
        result = bisecting_kmeans_linkage(
            embeddings,
            sample_weights=sample_weights if weighted else None,
            min_cluster_size=min_cluster_size,
            min_cluster_weight=min_cluster_weight if weighted else 0.0,
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
                        f"Options: {SCIPY_LINKAGE_METHODS | {'bisecting_kmeans', 'weighted_bisecting_kmeans'}}")


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
        LOGGER.info(f"CircuitRank solve mode: {cfg.circuitrank.solve_mode.value}")
        if cfg.circuitrank.normalize:
            if cfg.circuitrank.edge_weighting == "pagerank":
                LOGGER.info("CircuitRank normalize: True (PageRank mode)")
                # Log equivalent alpha for reference
                equiv_alpha = 1.0 / (1.0 + cfg.circuitrank.leakage)
                LOGGER.info(f"CircuitRank equivalent alpha: {equiv_alpha:.4f}")
            else:
                LOGGER.info(
                    "CircuitRank normalize: True (note: edge_weighting is non-PageRank; "
                    "solver uses general circuit Laplacian, not PageRank-equivalent form)"
                )
        LOGGER.info(
            "CircuitRank edge weighting: "
            f"{cfg.circuitrank.edge_weighting} (gamma={cfg.circuitrank.indegree_discount_gamma}, "
            f"eps={cfg.circuitrank.degree_epsilon}, r={cfg.circuitrank.edge_series_resistance})"
        )
        if cfg.circuitrank.mass_transform != "none":
            LOGGER.info(
                "CircuitRank mass transform: "
                f"{cfg.circuitrank.mass_transform} (power={cfg.circuitrank.mass_power}, renorm={cfg.circuitrank.mass_transform_renormalize})"
            )
        if cfg.circuitrank.solve_mode != CircuitRankSolveMode.NODE_UNSMOOTHED:
            sim_cfg = cfg.similarity
            LOGGER.info(
                "Similarity graph: "
                f"k={sim_cfg.knn_k}, sharpness={sim_cfg.sharpness}, row_normalize={sim_cfg.row_normalize}, "
                f"mix_weight={sim_cfg.mix_weight}, mix_mode={sim_cfg.mix_mode}, include_self={sim_cfg.include_self}"
            )
    
    LOGGER.info(f"Rank-before-clustering: {cfg.rank_before_clustering}")
    if cfg.rank_before_clustering:
        LOGGER.info(f"Ranking top-k: {cfg.ranking_top_k}")
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info("")
    
    # Build the cloud
    LOGGER.info("Loading cloud...")
    cloud: Cloud = instantiate(cfg.experiment.cloud)
    LOGGER.info(f"Cloud: {cloud.name()}")
    LOGGER.info(f"Embedder: {cloud.embedder}")
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
    adj_citation = cloud.citation_graph(normalize=False)

    adj_for_solver = adj_citation
    adj_for_frontier = adj_citation
    adj_for_solver = adj_citation

    # Determine if similarity smoothing is requested.
    # We enable it if explicitly configured OR if legacy CircuitRank modes expect it.
    use_similarity = cfg.similarity.enabled
    if cfg.method == RankMethod.CIRCUIT_RANK and cfg.circuitrank.solve_mode != CircuitRankSolveMode.NODE_UNSMOOTHED:
        use_similarity = True

    if use_similarity:
        sim_cfg = cfg.similarity
        LOGGER.info(
            "Building similarity adjacency (kNN): "
            f"k={sim_cfg.knn_k}, sharpness={sim_cfg.sharpness}, row_normalize={sim_cfg.row_normalize}, "
            f"density_alpha={sim_cfg.density_correction_alpha}"
        )
        adj_sim = build_similarity_adjacency_from_cloud(
            cloud,
            k=sim_cfg.knn_k,
            sharpness=sim_cfg.sharpness,
            row_normalize=sim_cfg.row_normalize,
            density_correction_alpha=sim_cfg.density_correction_alpha,
            include_self=sim_cfg.include_self,
        )

        LOGGER.info(
            "Mixing citation + similarity graphs: "
            f"mode={sim_cfg.mix_mode}, mix_weight={sim_cfg.mix_weight}"
        )
        adj_for_solver = mix_adjacency_matrices(
            adj_citation,
            adj_sim,
            mix_weight=sim_cfg.mix_weight,
            mix_mode=sim_cfg.mix_mode,
        )
        adj_for_frontier = adj_for_solver

        LOGGER.info(
            "Adjacency stats: citation nnz=%d, similarity nnz=%d, mixed nnz=%d",
            int(adj_citation.nnz),
            int(adj_sim.nnz),
            int(adj_for_solver.nnz),
        )
    
    # Run Ranking (PageRank or CircuitRank) on FULL graph
    full_point_potentials: Optional[NDArray[np.floating]] = None  # For potential-based linearization
    
    # Store intermediate relevance vectors for frontier adjacency calculation
    fwd_relevance: Optional[NDArray[np.floating]] = None
    rev_relevance: Optional[NDArray[np.floating]] = None
    
    if cfg.method in (RankMethod.PAGE_RANK, RankMethod.BIDIRECTIONAL_PAGE_RANK):
        # Unified path: one personalised-PR call with two seed vectors
        # (forward = targets → prerequisites, reverse = sources → descendants).
        # Either seed vector may sum to zero; the corresponding pass is
        # skipped and the relevance falls back to whichever direction has
        # signal. This subsumes both:
        #   PAGE_RANK = fwd_seeds populated, rev_seeds=0
        #   BIDIRECTIONAL_PAGE_RANK = both populated; missing source_work_ids
        #     no longer raises — it gracefully degrades to forward-only PR.
        target_seeds = np.zeros(n_total, dtype=np.float64)
        target_seeds[seed_indices] = 1.0 / len(seed_indices)

        source_seeds = np.zeros(n_total, dtype=np.float64)
        if cfg.source_work_ids:
            source_indices = _find_work_indices(work_ids, cfg.source_work_ids)
            if source_indices:
                source_seeds[source_indices] = 1.0 / len(source_indices)
            else:
                LOGGER.warning(
                    "No source works found in cloud — running forward-only PR. "
                    f"Requested: {cfg.source_work_ids}"
                )

        # When method=PAGE_RANK we deliberately ignore source_work_ids
        # even if provided (preserves the historical contract that
        # PAGE_RANK is a forward-only ranking).
        if cfg.method == RankMethod.PAGE_RANK:
            source_seeds = np.zeros(n_total, dtype=np.float64)
            LOGGER.info("Running Personalized PageRank (forward only)...")
        else:
            n_src = int((source_seeds > 0).sum())
            n_tgt = int((target_seeds > 0).sum())
            LOGGER.info(
                "Running two-direction PageRank: %d target seed(s), "
                "%d source seed(s), combine=product",
                n_tgt, n_src,
            )

        result = two_direction_pagerank(
            adj_for_solver,
            fwd_seeds=target_seeds,
            rev_seeds=source_seeds,
            alpha=cfg.pagerank.alpha,
            tol=cfg.pagerank.tol,
            max_iter=cfg.pagerank.max_iter,
            combine="product",
        )
        full_point_masses = result.relevance
        fwd_relevance = result.forward_mass
        rev_relevance = result.reverse_mass


    elif cfg.method == RankMethod.CIRCUIT_RANK:
        if cfg.circuitrank.use_voltage_clamping:
            LOGGER.info("Running CircuitRank (Voltage Clamping)...")
            # Voltage clamping (Dirichlet BC): solve for potentials, then derive relevance
            # from through-current on edges.
            source_indices: list[int] = []
            if cfg.source_work_ids:
                source_indices = _find_work_indices(work_ids, cfg.source_work_ids)
                if not source_indices:
                    LOGGER.warning(
                        f"No source works found in cloud - using targets only. "
                        f"Requested: {cfg.source_work_ids}"
                    )

            full_point_potentials, _, sources_found = compute_curriculum_ordering_potentials(
                adj_for_solver,
                target_indices=list(seed_indices),
                source_indices=list(source_indices) if source_indices else None,
                leakage=cfg.circuitrank.leakage,
                beta=cfg.circuitrank.beta,
                normalize=cfg.circuitrank.normalize,
                edge_weighting=cfg.circuitrank.edge_weighting,
                indegree_discount_gamma=cfg.circuitrank.indegree_discount_gamma,
                degree_epsilon=cfg.circuitrank.degree_epsilon,
                edge_series_resistance=cfg.circuitrank.edge_series_resistance,
                gmres_tol=cfg.circuitrank.gmres_tol,
                gmres_maxiter=cfg.circuitrank.gmres_maxiter,
                gmres_preconditioner=cfg.circuitrank.gmres_preconditioner,
                gmres_log_every=cfg.circuitrank.gmres_log_every,
            )

            if sources_found:
                LOGGER.info(
                    f"Voltage clamping: {len(source_indices)} known works at 1V (source), "
                    f"{len(seed_indices)} targets at 0V (ground)"
                )
            else:
                LOGGER.info(
                    f"Voltage clamping (targets-only): {len(seed_indices)} targets at 1V (leakage to ground)"
                )

            # Relevance: through-current magnitude under the same conductances used by the solver.
            full_point_masses_raw = compute_current_flow(
                adj_for_solver,
                full_point_potentials,
                normalize=cfg.circuitrank.normalize,
                edge_weighting=cfg.circuitrank.edge_weighting,
                indegree_discount_gamma=cfg.circuitrank.indegree_discount_gamma,
                degree_epsilon=cfg.circuitrank.degree_epsilon,
                edge_series_resistance=cfg.circuitrank.edge_series_resistance,
            )

            full_point_masses = transform_node_masses(
                full_point_masses_raw,
                transform=cfg.circuitrank.mass_transform,
                power=cfg.circuitrank.mass_power,
                renormalize=cfg.circuitrank.mass_transform_renormalize,
            )

            LOGGER.info(
                f"Potentials: min={full_point_potentials.min():.4f}, "
                f"max={full_point_potentials.max():.4f}, mean={full_point_potentials.mean():.4f}"
            )
            if cfg.circuitrank.mass_transform != "none":
                LOGGER.info(
                    f"Through-current (raw): min={full_point_masses_raw.min():.4e}, "
                    f"max={full_point_masses_raw.max():.4e}, sum={full_point_masses_raw.sum():.4f}"
                )
                LOGGER.info(
                    f"Through-current (transformed): min={full_point_masses.min():.4e}, "
                    f"max={full_point_masses.max():.4e}, sum={full_point_masses.sum():.4f}"
                )
            else:
                LOGGER.info(
                    f"Through-current: min={full_point_masses.min():.4e}, "
                    f"max={full_point_masses.max():.4e}, sum={full_point_masses.sum():.4f}"
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
            if cfg.circuitrank.solve_mode == CircuitRankSolveMode.GALERKIN:
                potentials = compute_graph_potential_galerkin(
                    adj_for_solver,
                    embeddings,
                    current_sources=current_sources,
                    leakage=cfg.circuitrank.leakage,
                    beta=cfg.circuitrank.beta,
                    normalize=cfg.circuitrank.normalize,
                    edge_weighting=cfg.circuitrank.edge_weighting,
                    indegree_discount_gamma=cfg.circuitrank.indegree_discount_gamma,
                    degree_epsilon=cfg.circuitrank.degree_epsilon,
                    edge_series_resistance=cfg.circuitrank.edge_series_resistance,
                )
            else:
                potentials = compute_graph_potential(
                    adj_for_solver,
                    current_sources=current_sources,
                    leakage=cfg.circuitrank.leakage,
                    beta=cfg.circuitrank.beta,
                    normalize=cfg.circuitrank.normalize,
                    edge_weighting=cfg.circuitrank.edge_weighting,
                    indegree_discount_gamma=cfg.circuitrank.indegree_discount_gamma,
                    degree_epsilon=cfg.circuitrank.degree_epsilon,
                    edge_series_resistance=cfg.circuitrank.edge_series_resistance,
                    gmres_tol=cfg.circuitrank.gmres_tol,
                    gmres_maxiter=cfg.circuitrank.gmres_maxiter,
                    gmres_preconditioner=cfg.circuitrank.gmres_preconditioner,
                    gmres_log_every=cfg.circuitrank.gmres_log_every,
                )
            
            # Compute current flow through each node - this is symmetric wrt source/target
            # Nodes with high current flow are "on the path" between source and target
            full_point_masses_raw = compute_current_flow(
                adj_for_solver,
                potentials,
                normalize=cfg.circuitrank.normalize,
                edge_weighting=cfg.circuitrank.edge_weighting,
                indegree_discount_gamma=cfg.circuitrank.indegree_discount_gamma,
                degree_epsilon=cfg.circuitrank.degree_epsilon,
                edge_series_resistance=cfg.circuitrank.edge_series_resistance,
            )

            full_point_masses = transform_node_masses(
                full_point_masses_raw,
                transform=cfg.circuitrank.mass_transform,
                power=cfg.circuitrank.mass_power,
                renormalize=cfg.circuitrank.mass_transform_renormalize,
            )
            if cfg.circuitrank.mass_transform != "none":
                LOGGER.info(
                    f"Current flow (raw): min={full_point_masses_raw.min():.4e}, "
                    f"max={full_point_masses_raw.max():.4e}, sum={full_point_masses_raw.sum():.4f}"
                )
                LOGGER.info(
                    f"Current flow (transformed): min={full_point_masses.min():.4e}, "
                    f"max={full_point_masses.max():.4e}, sum={full_point_masses.sum():.4f}"
                )
            else:
                LOGGER.info(
                    f"Current flow: min={full_point_masses.min():.4e}, "
                    f"max={full_point_masses.max():.4e}, sum={full_point_masses.sum():.4f}"
                )
            
            # Also compute potentials via voltage clamping for linearization
            # This gives us directional information: source=1V (if specified), target=0V
            # If no source works are available, we use targets-only clamping (targets=1V) and invert
            # so potentials still satisfy: high=early, low=late.
            if cfg.circuitrank.solve_mode == CircuitRankSolveMode.GALERKIN:
                if cfg.circuitrank.ordering_voltage_clamp:
                    LOGGER.info(
                        "Computing ordering potentials via node-space voltage clamping (Galerkin relevance)."
                    )
                    ordering_source_indices = list(source_indices) if (cfg.source_work_ids and source_indices) else None
                    full_point_potentials, _, ordering_sources_found = compute_curriculum_ordering_potentials(
                        adj_for_solver,
                        target_indices=list(seed_indices),
                        source_indices=ordering_source_indices,
                        leakage=cfg.circuitrank.leakage,
                        beta=cfg.circuitrank.beta,
                        normalize=cfg.circuitrank.normalize,
                        edge_weighting=cfg.circuitrank.edge_weighting,
                        indegree_discount_gamma=cfg.circuitrank.indegree_discount_gamma,
                        degree_epsilon=cfg.circuitrank.degree_epsilon,
                        edge_series_resistance=cfg.circuitrank.edge_series_resistance,
                        gmres_tol=cfg.circuitrank.gmres_tol,
                        gmres_maxiter=cfg.circuitrank.gmres_maxiter,
                        gmres_preconditioner=cfg.circuitrank.gmres_preconditioner,
                        gmres_log_every=cfg.circuitrank.gmres_log_every,
                    )

                    if ordering_sources_found:
                        LOGGER.info(
                            f"Voltage clamping: {len(source_indices)} sources at 1V, "
                            f"{len(seed_indices)} targets at 0V"
                        )
                    else:
                        LOGGER.info(
                            f"Voltage clamping (targets-only): {len(seed_indices)} targets at 1V (inverted potentials for ordering)"
                        )
                    LOGGER.info(
                        f"Voltage clamping potentials: min={full_point_potentials.min():.4f}, "
                        f"max={full_point_potentials.max():.4f}, mean={full_point_potentials.mean():.4f}"
                    )
                else:
                    LOGGER.info(
                        "Skipping voltage clamping in Galerkin mode; using current-injection potentials for ordering."
                    )
                    full_point_potentials = potentials
            else:
                LOGGER.info("Computing potentials via voltage clamping for topic ordering...")

                ordering_source_indices = list(source_indices) if (cfg.source_work_ids and source_indices) else None
                full_point_potentials, _, ordering_sources_found = compute_curriculum_ordering_potentials(
                    adj_for_solver,
                    target_indices=list(seed_indices),
                    source_indices=ordering_source_indices,
                    leakage=cfg.circuitrank.leakage,
                    beta=cfg.circuitrank.beta,
                    normalize=cfg.circuitrank.normalize,
                    edge_weighting=cfg.circuitrank.edge_weighting,
                    indegree_discount_gamma=cfg.circuitrank.indegree_discount_gamma,
                    degree_epsilon=cfg.circuitrank.degree_epsilon,
                    edge_series_resistance=cfg.circuitrank.edge_series_resistance,
                    gmres_tol=cfg.circuitrank.gmres_tol,
                    gmres_maxiter=cfg.circuitrank.gmres_maxiter,
                    gmres_preconditioner=cfg.circuitrank.gmres_preconditioner,
                    gmres_log_every=cfg.circuitrank.gmres_log_every,
                )

                if ordering_sources_found:
                    LOGGER.info(
                        f"Voltage clamping: {len(source_indices)} sources at 1V, "
                        f"{len(seed_indices)} targets at 0V"
                    )
                else:
                    LOGGER.info(
                        f"Voltage clamping (targets-only): {len(seed_indices)} targets at 1V (inverted potentials for ordering)"
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
    if full_mass_stats.top_k_shares:
        top_k_str = ", ".join(
            [f"{k}={v:.3%}" for k, v in full_mass_stats.top_k_shares.items()]
        )
        LOGGER.info(f"Mass top-k shares: {top_k_str}")
    if full_mass_stats.top_pct_shares:
        top_pct_str = ", ".join(
            [f"{k}={v:.3%}" for k, v in full_mass_stats.top_pct_shares.items()]
        )
        LOGGER.info(f"Mass top-pct shares: {top_pct_str}")
    
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
    
    # Cluster the (possibly filtered) works. Two paths:
    #   1. ``leiden``: community detection on the citation subgraph of the
    #      top-k filtered works (no embedding-based dendrogram). Citations
    #      are the semantic relation curricula *mean*; Leiden communities
    #      directly serve as frontier topics, no mass-threshold cut needed.
    #   2. all other ``linkage_method``s: embedding-based hierarchical
    #      clustering, then mass-threshold cut to find frontier topics.
    if cfg.linkage_method == "citation_avg_linkage":
        if not cfg.rank_before_clustering:
            raise ValueError(
                "citation_avg_linkage requires rank_before_clustering=True so the "
                "citation subgraph is restricted to the top-k filtered works"
            )
        LOGGER.info(
            f"Building citation-Jaccard distance matrix on top-{n_samples} "
            f"(mode={cfg.citation_jaccard_mode})..."
        )
        sub_adj = adj_citation[top_k_indices][:, top_k_indices]
        dist = _citation_jaccard_distance_matrix(
            sub_adj, mode=cfg.citation_jaccard_mode,
        )
        LOGGER.info("Running scipy avg-linkage on the citation distance matrix...")
        # ``linkage`` takes a condensed (upper-triangular flattened) form
        # of the distance matrix.
        from scipy.spatial.distance import squareform
        condensed = squareform(dist, checks=False)
        linked = linkage(condensed, method="average")
        # Standard scipy linkage: each point is its own leaf cluster.
        leaf_cluster_assignments = np.arange(n_samples, dtype=np.intp)
        n_leaf_clusters = n_samples
        LOGGER.info(
            f"Citation avg-linkage built: linkage shape {linked.shape}; "
            f"applying mass-threshold cut for ~{cfg.desired_n_topics} topics..."
        )

        # Aggregate masses to leaves (1:1 here since each work is a leaf).
        leaf_masses = np.zeros(n_leaf_clusters, dtype=np.float64)
        np.add.at(leaf_masses, leaf_cluster_assignments, point_masses)
        node_masses = compute_dendrogram_masses(linked, leaf_masses)
        threshold = find_mass_threshold_for_n_topics(
            linked, node_masses, cfg.desired_n_topics,
        )
        frontier_nodes = get_frontier_nodes(linked, node_masses, threshold)
        LOGGER.info(f"Found {len(frontier_nodes)} frontier topics")
        leaf_to_frontier = map_leaves_to_frontier_clusters(
            linked, frontier_nodes, n_leaf_clusters,
        )
        point_to_cluster = leaf_to_frontier[leaf_cluster_assignments]
    elif cfg.linkage_method in ("spectral", "graph_bisect", "multi_view_spectral"):
        if not cfg.rank_before_clustering:
            raise ValueError(
                f"{cfg.linkage_method} requires rank_before_clustering=True so the "
                "citation subgraph is restricted to the top-k filtered works"
            )
        sub_adj = adj_citation[top_k_indices][:, top_k_indices]
        if cfg.linkage_method == "spectral":
            LOGGER.info(
                f"Running spectral clustering on citation subgraph of "
                f"top-{n_samples} works (n_clusters={cfg.desired_n_topics})..."
            )
            membership = _cluster_with_spectral(
                sub_adj, n_clusters=cfg.desired_n_topics,
            )
            method_label = "Spectral"
        elif cfg.linkage_method == "graph_bisect":
            LOGGER.info(
                f"Running recursive Fiedler bisection on citation subgraph "
                f"of top-{n_samples} works (n_clusters={cfg.desired_n_topics})..."
            )
            membership = _cluster_with_graph_bisect(
                sub_adj, n_clusters=cfg.desired_n_topics,
            )
            method_label = "Graph-bisect"
        else:  # multi_view_spectral
            from artifacts_scripts.curriculum.multi_view_spectral import (
                build_citation_similarity_view,
                build_embedding_similarity_view,
                cluster_with_multi_view_spectral,
            )
            lam_display = (
                f"auto (scale={cfg.multi_view_spectral_auto_lambda_scale})"
                if cfg.multi_view_spectral_co_reg_lambda is None
                else f"{cfg.multi_view_spectral_co_reg_lambda}"
            )
            LOGGER.info(
                f"Running multi-view spectral on top-{n_samples} works "
                f"(n_clusters={cfg.desired_n_topics}, "
                f"λ={lam_display}, "
                f"csls_k={cfg.multi_view_spectral_csls_k}, "
                f"citation_mode={cfg.multi_view_spectral_citation_mode})..."
            )
            W_emb = build_embedding_similarity_view(
                embeddings_for_clustering,
                csls_k=cfg.multi_view_spectral_csls_k,
            )
            W_cite = build_citation_similarity_view(
                sub_adj,
                csls_k=cfg.multi_view_spectral_csls_k,
                mode=cfg.multi_view_spectral_citation_mode,
            )
            membership = cluster_with_multi_view_spectral(
                W_emb, W_cite,
                n_clusters=cfg.desired_n_topics,
                co_reg_lambda=cfg.multi_view_spectral_co_reg_lambda,
                auto_lambda_scale=cfg.multi_view_spectral_auto_lambda_scale,
                max_iters=cfg.multi_view_spectral_max_iters,
                tol=cfg.multi_view_spectral_tol,
            )
            method_label = "Multi-view-spectral"
        n_communities = int(membership.max()) + 1 if len(membership) > 0 else 0
        LOGGER.info(
            f"{method_label} produced {n_communities} communities. "
            f"Top-10 sizes: "
            f"{sorted(np.bincount(membership).tolist(), reverse=True)[:10]}"
        )
        # Both methods always produce exactly n_clusters communities (unless
        # n_samples < n_clusters), so no truncation step.
        leaf_cluster_assignments = membership
        n_leaf_clusters = n_communities
        linked = np.zeros((0, 4), dtype=np.float64)
        leaf_masses = np.zeros(n_leaf_clusters, dtype=np.float64)
        np.add.at(leaf_masses, leaf_cluster_assignments, point_masses)
        node_masses = leaf_masses.copy()
        threshold = 0.0
        frontier_nodes = np.arange(n_communities, dtype=np.int64)
        point_to_cluster = leaf_cluster_assignments.copy()
    elif cfg.linkage_method == "leiden":
        if not cfg.rank_before_clustering:
            raise ValueError(
                "leiden requires rank_before_clustering=True so the "
                "citation subgraph is restricted to the top-k filtered works"
            )
        LOGGER.info(
            f"Running Leiden on citation subgraph of top-{n_samples} works "
            f"(resolution={cfg.leiden_resolution})..."
        )
        sub_adj = adj_citation[top_k_indices][:, top_k_indices]
        raw_membership, n_raw = _cluster_with_leiden(
            sub_adj, resolution=cfg.leiden_resolution,
        )
        # Leiden typically produces *many* communities (one giant +
        # hundreds of singletons / tiny tail) regardless of resolution,
        # because the citation subgraph has lots of weakly-connected
        # nodes. Truncate to the top ``desired_n_topics`` by size and
        # mark all other works as unassigned (-1) — matches the
        # rank_before_clustering convention where works outside top-k
        # are also -1.
        sizes = np.bincount(raw_membership, minlength=n_raw)
        N = cfg.desired_n_topics
        # argsort gives ascending; reverse and take first N to get the
        # largest N by size. Stable so ties break on community-ID order.
        top_n_old_ids = np.argsort(-sizes, kind="stable")[:N]
        remap = np.full(n_raw, -1, dtype=np.intp)
        for new_id, old_id in enumerate(top_n_old_ids):
            remap[old_id] = new_id
        membership = remap[raw_membership]
        n_communities = N
        LOGGER.info(
            f"Leiden produced {n_raw} raw communities; truncated to "
            f"top-{N} by size (covering {(membership >= 0).sum()} of "
            f"{n_samples} top-k works; rest unassigned). "
            f"Top-{min(10, N)} sizes: "
            f"{sorted(sizes.tolist(), reverse=True)[:min(10, N)]}"
        )

        # Communities are leaf clusters AND frontier topics — no
        # hierarchical merging or mass-threshold cut. Trivial linkage
        # so saved outputs stay schema-compatible with other methods.
        leaf_cluster_assignments = membership  # may contain -1s for unassigned
        n_leaf_clusters = n_communities
        linked = np.zeros((0, 4), dtype=np.float64)
        # Aggregate masses only over assigned works.
        leaf_masses = np.zeros(n_leaf_clusters, dtype=np.float64)
        assigned = membership >= 0
        np.add.at(
            leaf_masses, membership[assigned], point_masses[assigned],
        )
        # No dendrogram → no internal-node mass propagation. Save the
        # leaf masses themselves as ``node_masses`` for output schema.
        node_masses = leaf_masses.copy()
        threshold = 0.0
        frontier_nodes = np.arange(n_communities, dtype=np.int64)
        point_to_cluster = membership.copy()
    else:
        LOGGER.info("Building dendrogram...")
        # Weights are only meaningful for the rank_before_clustering path
        # (where ``point_masses`` is the relevance over the top-k subset).
        # The full-cloud cached path uses uniform weights — appropriate
        # because no PR step has produced a per-work mass yet at that point.
        weighted = cfg.linkage_method == "weighted_bisecting_kmeans"
        if (not cfg.rank_before_clustering) and cfg.linkage_method == "bisecting_kmeans":
            # Full-cloud + bisecting_kmeans is expensive: leverage Cloud-level caching/subsumption.
            dendrogram_result = cloud.dendrogram(
                "bisecting_kmeans",
                min_cluster_size=cfg.min_cluster_size,
                max_depth=cfg.max_cluster_depth,
            )
        else:
            if weighted and not cfg.rank_before_clustering:
                raise ValueError(
                    "weighted_bisecting_kmeans requires rank_before_clustering=True "
                    "(weights come from the PageRank pass on the top-k subset)."
                )
            dendrogram_result = _build_dendrogram(
                embeddings_for_clustering,
                method=cfg.linkage_method,
                min_cluster_size=cfg.min_cluster_size,
                min_cluster_weight=cfg.min_cluster_weight,
                max_depth=cfg.max_cluster_depth,
                sample_weights=point_masses if weighted else None,
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
    
    if cfg.method in (RankMethod.PAGE_RANK, RankMethod.BIDIRECTIONAL_PAGE_RANK):
        adj_frontier = compute_frontier_adjacency(
            adj_citation,  # Always use full citation graph
            frontier_nodes,
            cluster_mapping_for_frontier,
            alpha=cfg.pagerank.alpha,
            tol=cfg.pagerank.tol,
            max_iter=cfg.pagerank.max_iter,
            forward_weights=fwd_relevance,
            reverse_weights=rev_relevance,
        )
    elif cfg.method == RankMethod.CIRCUIT_RANK:
        adj_frontier = compute_frontier_adjacency_potential(
            adj_for_frontier,
            frontier_nodes,
            cluster_mapping_for_frontier,
            leakage=cfg.circuitrank.leakage,
            beta=cfg.circuitrank.beta,
            normalize=cfg.circuitrank.normalize,
            edge_weighting=cfg.circuitrank.edge_weighting,
            indegree_discount_gamma=cfg.circuitrank.indegree_discount_gamma,
            degree_epsilon=cfg.circuitrank.degree_epsilon,
            edge_series_resistance=cfg.circuitrank.edge_series_resistance,
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
    # Apply the inverse permutation to assigned (>=0) entries only.
    # ``leiden`` may produce -1 entries for works in dropped communities;
    # naive ``inverse_perm[point_to_cluster]`` would wrap those -1s to
    # the last permutation entry. Other linkage methods produce no -1s
    # and the masked branch is a no-op for them.
    assigned_mask = point_to_cluster >= 0
    if assigned_mask.all():
        point_to_cluster = inverse_perm[point_to_cluster]
    else:
        new_pc = np.full_like(point_to_cluster, -1)
        new_pc[assigned_mask] = inverse_perm[point_to_cluster[assigned_mask]]
        point_to_cluster = new_pc

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
        "embedder": str(cloud.embedder),
        "denoiser": str(cloud.denoiser),
        "n_works_total": n_total,
        "n_works_clustered": n_samples,
        "rank_before_clustering": cfg.rank_before_clustering,
        "ranking_top_k": cfg.ranking_top_k if cfg.rank_before_clustering else None,
        "n_target_works": len(seed_indices),
        "target_work_ids": list(cfg.target_work_ids),  # Convert from Hydra ListConfig
        "source_work_ids": list(cfg.source_work_ids),
        "desired_n_topics": cfg.desired_n_topics,
        "actual_n_topics": len(frontier_nodes),
        "ranking_method": cfg.method.value,
        "pagerank_alpha": cfg.pagerank.alpha if cfg.method == RankMethod.PAGE_RANK else None,
        "circuitrank_leakage": cfg.circuitrank.leakage if cfg.method == RankMethod.CIRCUIT_RANK else None,
        "linkage_method": cfg.linkage_method,
        "citation_jaccard_mode": (
            cfg.citation_jaccard_mode
            if cfg.linkage_method == "citation_avg_linkage" else None
        ),
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
    sp.save_npz(output_dir / "citation_graph.npz", adj_citation)
    LOGGER.info(f"  citation_graph.npz: {adj_citation.shape} (sparse)")
    
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
    
    # Load titles for the topic summary. We avoid cloud.work_titles() on the
    # full retained corpus: at 41M works its combined display_name bytes
    # exceed pyarrow's 32-bit StringArray offset limit (~2 GB) and table.take
    # overflows. In rank_before_clustering mode the ~top_k titles fit
    # comfortably; otherwise we fetch only the single sample title per topic.
    LOGGER.info("Loading titles for summary...")
    if cfg.rank_before_clustering:
        titles_for_clustering = cloud.work_titles_subset(top_k_indices.astype(np.uint64))
        def _sample_title(point_idx: int) -> str:
            return titles_for_clustering[point_idx]
    else:
        sample_point_idx_by_topic = {}
        for topic_idx in linearization_order:
            point_indices = np.where(point_to_cluster == topic_idx)[0]
            if len(point_indices) > 0:
                sample_point_idx_by_topic[topic_idx] = int(point_indices[0])
        unique_point_idxs = sorted(set(sample_point_idx_by_topic.values()))
        if unique_point_idxs:
            unique_titles = cloud.work_titles_subset(
                np.array(unique_point_idxs, dtype=np.uint64)
            )
            point_idx_to_title = dict(zip(unique_point_idxs, unique_titles))
        else:
            point_idx_to_title = {}
        def _sample_title(point_idx: int) -> str:
            return point_idx_to_title[point_idx]

    # Print topic summary
    LOGGER.info("")
    LOGGER.info("Topic Summary (in linearization order):")
    for rank, topic_idx in enumerate(linearization_order):
        node_idx = frontier_nodes[topic_idx]
        mass = node_masses[node_idx]
        size = cluster_sizes[topic_idx]
        # Get a sample work title from this topic
        point_indices = np.where(point_to_cluster == topic_idx)[0]
        sample_title = _sample_title(int(point_indices[0]))[:60] if len(point_indices) > 0 else "N/A"
        LOGGER.info(
            f"  {rank + 1}. Topic {topic_idx} (node {node_idx}): "
            f"mass={mass:.4f}, size={size}, sample='{sample_title}...'"
        )
    
    # Return cloud, embeddings, and filtered indices for downstream use
    filtered_indices_out = top_k_indices if cfg.rank_before_clustering else None
    return cloud, embeddings_for_clustering, filtered_indices_out


def _run_clustering_for_test(
    *,
    sub_adj: sp.spmatrix,
    embeddings: NDArray[np.floating],
    n_clusters: int,
    co_reg_lambda: float = 0.5,
    csls_k: int = 10,
    citation_mode: str = "weighted_unified",
    max_iters: int = 30,
    tol: float = 1e-5,
) -> NDArray[np.intp]:
    """Test entry point: run multi-view spectral on raw inputs.

    Bypasses the Hydra/Cloud machinery so unit tests can exercise the
    clustering kernel without standing up a full curriculum config.
    Defaults match the corresponding ``CurriculumConfig`` fields so the
    entry point is suitable for convergence regression tests.
    """
    from artifacts_scripts.curriculum.multi_view_spectral import (
        build_citation_similarity_view,
        build_embedding_similarity_view,
        cluster_with_multi_view_spectral,
    )

    W_emb = build_embedding_similarity_view(embeddings, csls_k=csls_k)
    W_cite = build_citation_similarity_view(sub_adj, csls_k=csls_k, mode=citation_mode)
    return cluster_with_multi_view_spectral(
        W_emb, W_cite,
        n_clusters=n_clusters,
        co_reg_lambda=co_reg_lambda,
        max_iters=max_iters,
        tol=tol,
    )
