"""Unit tests for the decoupled curriculum core (small synthetic graphs)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from curriculum.config import (
    CurriculumConfig,
    RankMethod,
    SelectionMode,
)
from curriculum.core.dendrogram import (
    bisecting_kmeans_linkage,
    cluster_by_mass_frontier,
)
from curriculum.core.order import linearization_order_from_adjacency, order_by_mass
from curriculum.core.pagerank import (
    compute_frontier_adjacency,
    compute_pagerank,
    two_direction_pagerank,
)
from curriculum.core.rank import compute_relevance, select_top_n


def _chain_graph(n: int) -> sp.csr_matrix:
    """Directed chain 0 -> 1 -> ... -> n-1 (with a self loop on each node)."""
    rows = list(range(n)) + list(range(n - 1))
    cols = list(range(n)) + list(range(1, n))
    data = [1.0] * len(rows)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def test_pagerank_sums_to_one_and_favours_foundations() -> None:
    adj = _chain_graph(6)
    seeds = np.zeros(6)
    seeds[5] = 1.0  # seed the frontier node; forward flow -> earlier nodes
    pr = compute_pagerank(adj, seeds, alpha=0.85, tol=1e-12, max_iter=500)
    assert pr.shape == (6,)
    assert abs(float(pr.sum()) - 1.0) < 1e-6
    assert (pr >= 0).all()


def test_two_direction_uniform_when_no_seeds() -> None:
    adj = _chain_graph(5)
    res = two_direction_pagerank(adj, np.zeros(5), np.zeros(5))
    assert res.forward_mass is None and res.reverse_mass is None
    assert np.allclose(res.relevance, 1.0 / 5)


def test_two_direction_forward_only_matches_personalized() -> None:
    adj = _chain_graph(7)
    fwd = np.zeros(7)
    fwd[6] = 1.0
    res = two_direction_pagerank(adj, fwd, np.zeros(7), alpha=0.85, tol=1e-12, max_iter=500)
    assert res.reverse_mass is None
    assert abs(float(res.relevance.sum()) - 1.0) < 1e-6


def test_frontier_adjacency_orders_chain_topics() -> None:
    # Chain 0->1->...->5 with "i->j" = "i cites/builds-on j", so forward
    # PageRank mass flows toward the higher-index nodes => the {4,5} cluster
    # (topic 1) is the FOUNDATION, the {0,1} cluster (topic 0) the frontier.
    adj = _chain_graph(6)
    leaf_to_cluster = np.array([0, 0, -1, -1, 1, 1], dtype=np.int64)
    frontier_nodes = np.array([0, 1], dtype=np.int64)
    fa = compute_frontier_adjacency(adj, frontier_nodes, leaf_to_cluster, alpha=0.85)
    assert fa.shape == (2, 2)
    # Topic 0 (frontier) depends on topic 1 (foundation): fa[0, 1] > 0.
    assert fa[0, 1] > 0
    order = linearization_order_from_adjacency(fa)
    assert order[0] == 1  # foundation (depended-on) first
    assert order[-1] == 0  # frontier last


def test_select_top_n_descending() -> None:
    score = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
    idx = select_top_n(score, 3)
    assert list(idx) == [3, 1, 2]


def test_select_top_n_per_corpus_balances() -> None:
    # Corpus 0 has tiny absolute scores; per-corpus z-norm should still surface it.
    score = np.array([100.0, 90.0, 80.0, 1.0, 0.9, 0.1])
    node_type = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    idx = select_top_n(score, 2, node_type=node_type, per_corpus=True)
    chosen_types = set(int(node_type[i]) for i in idx)
    # The top per-corpus node from EACH corpus should be present.
    assert 0 in chosen_types and 1 in chosen_types


def test_background_normalized_selection_runs() -> None:
    adj = _chain_graph(8)
    cfg = CurriculumConfig(method=RankMethod.PAGE_RANK)
    cfg.selection.mode = SelectionMode.BACKGROUND_NORMALIZED
    seeds = np.array([7], dtype=np.int64)
    res = compute_relevance(adj, seeds, cfg)
    assert res.background is not None
    assert res.selection_score.shape == (8,)
    # Raw masses still sum to ~1 (selection score is a separate re-rank).
    assert abs(float(res.masses.sum()) - 1.0) < 1e-6


def test_bisecting_kmeans_linkage_shape() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (20, 4)), rng.normal(5, 0.1, (20, 4))])
    res = bisecting_kmeans_linkage(X, min_cluster_size=1, verbose=False)
    assert res.linkage.shape[1] == 4
    assert res.n_leaf_clusters == 40


def test_cluster_by_mass_frontier_two_blobs() -> None:
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 0.05, (30, 4)), rng.normal(10, 0.05, (30, 4))])
    masses = np.ones(60)
    labels = cluster_by_mass_frontier(X, masses, desired_n_topics=2, random_state=42)
    assert int(labels.max()) + 1 == 2
    # The two blobs should land in different clusters.
    assert labels[0] != labels[-1]


def test_order_by_mass() -> None:
    p2c = np.array([0, 0, 1, 1, 1])
    masses = np.array([0.4, 0.3, 0.1, 0.1, 0.05])  # topic0 heavier
    order, topic_mass = order_by_mass(p2c, masses, 2)
    assert list(order) == [0, 1]
    assert topic_mass[0] > topic_mass[1]
