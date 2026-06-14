"""``build_curriculum(graph_source, config) -> Curriculum`` — the orchestrator.

Wires the four engine axes (rank -> select -> cluster -> order) over a
:class:`~curriculum.graph_source.GraphSource`, pulling in optional layers
(keywords / quality / judge / diffusion / spectral geometry) only when the
config toggles them on. Works with NO embeddings (the Lean / pure-graph path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from curriculum._logging import get_logger
from curriculum.config import (
    CurriculumConfig,
    EmbeddingsMode,
    OrderingMode,
)
from curriculum.core.dendrogram import cluster_by_mass_frontier
from curriculum.core.order import (
    order_by_frontier_adjacency,
    order_by_mass,
)
from curriculum.core.rank import compute_relevance, select_top_n
from curriculum.graph_source import GraphSource

LOGGER = get_logger(__name__)


@dataclass
class Curriculum:
    """The built curriculum.

    ``topic_order`` lists topic ids in prerequisite -> frontier order.
    ``point_to_cluster`` maps each SELECTED node (index into ``selected_indices``)
    to its topic id. ``selected_indices`` are indices into the original
    ``GraphSource`` node order.
    """

    topic_order: NDArray[np.int64]
    point_to_cluster: NDArray[np.intp]
    selected_indices: NDArray[np.int64]
    n_topics: int
    masses: NDArray[np.floating]
    """Raw forward-PR mass over ALL nodes."""
    selection_score: NDArray[np.floating]
    """Selection score over ALL nodes (== masses unless bg-norm)."""
    topic_mass: NDArray[np.floating]
    """Per-topic score used for the report's score column."""
    topic_keywords: Dict[int, List[Tuple[str, float]]] = field(default_factory=dict)
    topic_member_ids: Dict[int, List[str]] = field(default_factory=dict)
    frontier_in_out: Optional[NDArray[np.floating]] = None
    topic_adj: Optional[NDArray[np.floating]] = None
    quality: Optional[Dict[str, float]] = None
    judge: Optional[Dict[str, Any]] = None

    def topic_member_indices(self, topic_id: int) -> NDArray[np.int64]:
        """Original-graph node indices belonging to ``topic_id`` (mass-sorted)."""
        local = np.where(self.point_to_cluster == topic_id)[0]
        orig = self.selected_indices[local]
        return orig[np.argsort(self.selection_score[orig])[::-1]]


def _resolve_embeddings(
    graph: GraphSource,
    adj: sp.csr_matrix,
    cfg: CurriculumConfig,
) -> Optional[NDArray[np.floating]]:
    """Produce node embeddings per ``embeddings_mode`` (None => no embeddings)."""
    mode = cfg.embeddings_mode
    if mode == EmbeddingsMode.PROVIDED:
        emb = graph.node_embeddings()
        if emb is None:
            raise ValueError(
                "embeddings_mode=PROVIDED but GraphSource.node_embeddings() is None"
            )
        return np.asarray(emb)
    if mode == EmbeddingsMode.CALLABLE:
        fn = cfg.embeddings_callable
        if fn is None or not callable(fn):
            raise ValueError("embeddings_mode=CALLABLE requires a callable embeddings_callable")
        return np.asarray(fn(graph))
    if mode == EmbeddingsMode.SPECTRAL_FROM_GRAPH:
        from curriculum.layers.lpp import spectral_embedding_from_graph
        return spectral_embedding_from_graph(
            adj, n_components=cfg.spectral.n_components,
            hub_suppress=cfg.spectral.hub_suppress,
        )
    if mode == EmbeddingsMode.COMMUNITY_DETECT:
        return None  # clustering happens directly on the graph
    raise ValueError(f"unknown embeddings_mode {mode!r}")


def _cluster(
    adj: sp.csr_matrix,
    top_idx: NDArray[np.int64],
    masses: NDArray[np.floating],
    embeddings: Optional[NDArray[np.floating]],
    cfg: CurriculumConfig,
) -> NDArray[np.intp]:
    """Cluster the selected nodes into topics; returns per-(selected-node) labels."""
    n_sel = len(top_idx)
    if not cfg.clustering.enabled:
        # Minimal core with clustering off: every node is its own topic.
        return np.arange(n_sel, dtype=np.intp)

    if cfg.embeddings_mode == EmbeddingsMode.COMMUNITY_DETECT:
        from curriculum.layers.clustering import cluster_spectral
        n_clusters = cfg.clustering.community_n_clusters or cfg.clustering.desired_n_topics
        sub_adj = adj[top_idx][:, top_idx].tocsr()
        return cluster_spectral(sub_adj, n_clusters=n_clusters, seed=cfg.random_state)

    assert embeddings is not None
    top_emb = np.ascontiguousarray(embeddings[top_idx], dtype=np.float32)
    top_masses = masses[top_idx]
    sample_weights = top_masses if cfg.clustering.weighted else None
    return cluster_by_mass_frontier(
        top_emb, top_masses, cfg.clustering.desired_n_topics,
        sample_weights=sample_weights,
        min_cluster_size=cfg.clustering.min_cluster_size,
        random_state=cfg.random_state,
    )


def build_curriculum(
    graph: GraphSource,
    cfg: CurriculumConfig,
    *,
    source_indices: Optional[NDArray[np.int64]] = None,
) -> Curriculum:
    """Build a curriculum from a graph source + config (the public entry point).

    ``source_indices`` (optional) are extra reverse-direction seed node indices
    threaded straight to :func:`compute_relevance`. They are consumed ONLY by the
    ``RankMethod.BIDIRECTIONAL_PAGE_RANK`` branch (reverse-walk seeds); for every
    other method, and whenever ``None``, the call is byte-identical to the default
    forward-PR path (``compute_relevance(adj, seed_indices, cfg)``).
    """
    adj = graph.adjacency().tocsr()
    n_nodes = adj.shape[0]

    if cfg.diffusion.enabled:
        from curriculum.layers.diffusion import diffuse_adjacency
        LOGGER.info("Applying GDC diffusion densification to the adjacency")
        adj = diffuse_adjacency(
            adj, kernel=cfg.diffusion.kernel.value, t=cfg.diffusion.t,
            ppr_alpha=cfg.diffusion.ppr_alpha, top_m=cfg.diffusion.top_m,
            series_terms=cfg.diffusion.series_terms, cap_m=cfg.diffusion.cap_m,
        )

    seed_indices = np.asarray(graph.seed_nodes(), dtype=np.int64)
    if len(seed_indices) == 0:
        raise ValueError("GraphSource.seed_nodes() returned no seeds")

    LOGGER.info("Ranking %d nodes from %d seeds (method=%s)",
                n_nodes, len(seed_indices), cfg.method.value)
    rank = compute_relevance(adj, seed_indices, cfg, source_indices=source_indices)

    top_idx = select_top_n(
        rank.selection_score, cfg.selection.top_n,
        per_corpus=cfg.selection.per_corpus,
    )
    LOGGER.info("Selected top-%d nodes (retains %.1f%% PR mass)",
                len(top_idx), 100.0 * float(rank.masses[top_idx].sum()))

    embeddings = _resolve_embeddings(graph, adj, cfg)
    point_to_cluster = _cluster(adj, top_idx, rank.masses, embeddings, cfg)
    n_topics = int(point_to_cluster.max()) + 1 if len(point_to_cluster) else 0

    # --- order ---
    top_masses = rank.masses[top_idx]
    frontier_in_out: Optional[NDArray[np.floating]] = None
    topic_adj: Optional[NDArray[np.floating]] = None
    if cfg.ordering == OrderingMode.MASS:
        order, topic_mass = order_by_mass(point_to_cluster, top_masses, n_topics)
    else:  # FRONTIER_ADJACENCY / IN_OUT_DEGREE
        order, frontier_in_out, topic_adj = order_by_frontier_adjacency(
            adj, point_to_cluster, top_idx, n_topics,
            n_nodes=n_nodes, alpha=cfg.pagerank.alpha,
        )
        _, topic_mass = order_by_mass(point_to_cluster, top_masses, n_topics)

    curr = Curriculum(
        topic_order=order,
        point_to_cluster=point_to_cluster,
        selected_indices=top_idx,
        n_topics=n_topics,
        masses=rank.masses,
        selection_score=rank.selection_score,
        topic_mass=topic_mass,
        frontier_in_out=frontier_in_out,
        topic_adj=topic_adj,
    )

    # member ids per topic (mass-sorted), always populated
    ids = graph.ids()
    for t in range(n_topics):
        members = curr.topic_member_indices(t)
        curr.topic_member_ids[t] = [ids[int(i)] for i in members]

    # --- optional keyword layer ---
    if cfg.keywords.enabled:
        from curriculum.layers.keywords import keywords_for_topics
        texts = graph.node_text()
        top_texts = [texts[int(i)] for i in top_idx]
        kw = keywords_for_topics(
            top_texts, point_to_cluster, top_masses, n_topics,
            samples_per_topic=cfg.keywords.samples_per_topic,
            top_k=cfg.keywords.top_k,
            bigrams=cfg.keywords.bigrams,
            bigram_iterations=cfg.keywords.bigram_iterations,
            npmi_threshold=cfg.keywords.npmi_threshold,
            min_boundary_be=cfg.keywords.min_boundary_be,
            count_floor=cfg.keywords.count_floor,
            max_phrases=cfg.keywords.max_phrases,
        )
        curr.topic_keywords = {t: kw[t] for t in range(n_topics)}

    # --- optional quality layer ---
    if cfg.quality_enabled:
        from curriculum.layers.quality import CurriculumInputs, score_curriculum
        sub_adj = adj[top_idx][:, top_idx].tocsr()
        sizes = np.bincount(point_to_cluster[point_to_cluster >= 0], minlength=n_topics).tolist()
        tk = {t: [w for w, _ in curr.topic_keywords.get(t, [])] for t in range(n_topics)}
        curr.quality = score_curriculum(CurriculumInputs(
            topic_order=list(order),
            point_to_cluster=point_to_cluster,
            n_topics=n_topics,
            cluster_sizes=sizes,
            topic_keywords=tk,
            graph=sub_adj,
            topic_adj=topic_adj,
            keyword_top_k=cfg.keywords.top_k,
        ))

    # --- optional LLM judge layer ---
    if cfg.judge.enabled:
        from pathlib import Path
        from curriculum.layers.judge import TopicCard, judge_curriculum
        cards: List[TopicCard] = []
        for rank_pos, t in enumerate(order):
            t = int(t)
            members = curr.topic_member_indices(t)
            sizes_t = int((point_to_cluster == t).sum())
            cards.append(TopicCard(
                rank=rank_pos + 1,
                size=sizes_t,
                keywords=[w for w, _ in curr.topic_keywords.get(t, [])],
                titles=[graph.node_text()[int(i)][:60] for i in members[:6]],
            ))
        curr.judge = judge_curriculum(
            cards, model=cfg.judge.model, repeats=cfg.judge.repeats,
            temperature=cfg.judge.temperature, num_ctx=cfg.judge.num_ctx,
            num_predict=cfg.judge.num_predict, n_order_pairs=cfg.judge.n_order_pairs,
            seed=cfg.judge.seed,
            rawfail_path=Path(cfg.judge.rawfail_path) if cfg.judge.rawfail_path else None,
        )

    return curr
