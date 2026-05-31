"""Phase-0-bis pure-graph tests: build a curriculum from (graph, seeds, text)
with NO embeddings, via spectral-from-graph and community-detect modes."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from curriculum import (
    CurriculumConfig,
    EmbeddingsMode,
    InMemoryGraphSource,
    OrderingMode,
    RankMethod,
    build_curriculum,
)
from curriculum.config import SelectionMode
from curriculum.layers.lpp import (
    hub_suppressed_link_affinity,
    spectral_embedding_from_graph,
)


def _two_community_graph(per: int = 25, seed: int = 0):
    """Two dense communities with a couple of bridge edges; directed."""
    rng = np.random.default_rng(seed)
    n = 2 * per
    rows, cols = [], []
    for block in (range(per), range(per, n)):
        nodes = list(block)
        for i in nodes:
            rows.append(i)
            cols.append(i)  # self loop
            for _ in range(6):
                j = int(rng.choice(nodes))
                if j != i:
                    rows.append(i)
                    cols.append(j)
    # a few bridges
    for _ in range(3):
        rows.append(int(rng.integers(0, per)))
        cols.append(int(rng.integers(per, n)))
    data = np.ones(len(rows), dtype=np.float64)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    text = [f"alpha token{i % 7}" if i < per else f"beta token{i % 7}" for i in range(n)]
    seeds = np.array([0, per], dtype=np.int64)
    return InMemoryGraphSource(
        adj=adj, id_list=[f"n{i}" for i in range(n)], text=text, seeds=seeds,
    )


def test_hub_suppressed_affinity_symmetric() -> None:
    adj = _two_community_graph().adjacency()
    W = hub_suppressed_link_affinity(adj)
    assert W.shape == adj.shape
    diff = (W - W.T)
    assert abs(float(diff.multiply(diff).sum())) < 1e-12  # symmetric


def test_spectral_embedding_shape_and_finite() -> None:
    adj = _two_community_graph().adjacency()
    Y = spectral_embedding_from_graph(adj, n_components=8)
    assert Y.shape[0] == adj.shape[0]
    assert Y.shape[1] <= 8
    assert np.isfinite(Y).all()


def test_build_spectral_from_graph_no_embeddings() -> None:
    gs = _two_community_graph()
    assert gs.node_embeddings() is None  # genuinely no embeddings
    cfg = CurriculumConfig(
        method=RankMethod.PAGE_RANK,
        embeddings_mode=EmbeddingsMode.SPECTRAL_FROM_GRAPH,
        ordering=OrderingMode.FRONTIER_ADJACENCY,
    )
    cfg.spectral.n_components = 8
    cfg.selection.mode = SelectionMode.TOP_N
    cfg.selection.top_n = 50
    cfg.clustering.desired_n_topics = 2
    cfg.keywords.enabled = True
    curr = build_curriculum(gs, cfg)
    assert curr.n_topics >= 1
    assert len(curr.topic_order) == curr.n_topics
    # keywords were produced for the topics
    assert any(curr.topic_keywords.get(t) for t in range(curr.n_topics))


def test_build_community_detect_no_embeddings() -> None:
    gs = _two_community_graph()
    cfg = CurriculumConfig(
        method=RankMethod.PAGE_RANK,
        embeddings_mode=EmbeddingsMode.COMMUNITY_DETECT,
        ordering=OrderingMode.MASS,
    )
    cfg.selection.mode = SelectionMode.TOP_N
    cfg.selection.top_n = 50
    cfg.clustering.community_n_clusters = 2
    cfg.keywords.enabled = False
    curr = build_curriculum(gs, cfg)
    assert curr.n_topics == 2
    # selected indices cover both communities' seeds
    assert len(curr.selected_indices) == 50


def test_minimal_core_clustering_disabled() -> None:
    """Rank -> select -> order with NO clustering and NO embeddings (Lean core)."""
    gs = _two_community_graph()
    cfg = CurriculumConfig(
        method=RankMethod.PAGE_RANK,
        embeddings_mode=EmbeddingsMode.COMMUNITY_DETECT,  # avoids needing embeddings
        ordering=OrderingMode.MASS,
    )
    cfg.clustering.enabled = False
    cfg.selection.mode = SelectionMode.TOP_N
    cfg.selection.top_n = 10
    cfg.keywords.enabled = False
    curr = build_curriculum(gs, cfg)
    # every selected node is its own topic
    assert curr.n_topics == 10
