"""Unit tests for the optional layers (keywords, quality, judge-parse, diffusion)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from curriculum.layers.diffusion import diffuse_adjacency
from curriculum.layers.judge import TopicCard, extract_first_json_object
from curriculum.layers.keywords import keywords_for_topics
from curriculum.layers.quality import (
    CurriculumInputs,
    avg_pairwise_jaccard_distance,
    modularity_q,
    score_curriculum,
)


def test_keywords_for_topics_basic() -> None:
    texts = [
        "neural network deep learning",
        "neural network gradient descent",
        "syntax parsing grammar tree",
        "syntax parsing grammar rule",
    ]
    p2c = np.array([0, 0, 1, 1])
    masses = np.ones(4)
    kw = keywords_for_topics(texts, p2c, masses, 2, samples_per_topic=10, top_k=5)
    assert len(kw) == 2
    # topic 0 should surface a network/learning term; topic 1 a parsing term.
    t0 = {w for w, _ in kw[0]}
    t1 = {w for w, _ in kw[1]}
    assert any("neural" in w or "network" in w or "learning" in w for w in t0)
    assert any("parsing" in w or "grammar" in w or "syntax" in w for w in t1)


def test_judge_robust_parse_tolerates_preamble_and_trailing() -> None:
    content = 'Here is the JSON:\n{"rating": 4, "is_grab_bag": false, "reason": "{nested}"}\nThanks!'
    obj = extract_first_json_object(content)
    assert obj == {"rating": 4, "is_grab_bag": False, "reason": "{nested}"}


def test_judge_robust_parse_returns_none_on_garbage() -> None:
    assert extract_first_json_object("no json here") is None


def test_modularity_and_jaccard() -> None:
    # two disconnected triangles -> high modularity under the right partition
    rows = [0, 1, 2, 0, 3, 4, 5, 3]
    cols = [1, 2, 0, 2, 4, 5, 3, 5]
    adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(6, 6))
    membership = np.array([0, 0, 0, 1, 1, 1])
    Q = modularity_q(adj, membership)
    assert Q > 0.3

    kws = {0: ["a", "b", "c"], 1: ["d", "e", "f"]}
    assert avg_pairwise_jaccard_distance(kws, top_k=3) == 1.0  # fully distinct


def test_score_curriculum_smoke() -> None:
    rows = [0, 1, 2, 3]
    cols = [1, 2, 3, 0]
    adj = sp.csr_matrix((np.ones(4), (rows, cols)), shape=(4, 4))
    p2c = np.array([0, 0, 1, 1])
    inputs = CurriculumInputs(
        topic_order=[0, 1],
        point_to_cluster=p2c,
        n_topics=2,
        cluster_sizes=[2, 2],
        topic_keywords={0: ["a", "b"], 1: ["c", "d"]},
        graph=adj,
    )
    metrics = score_curriculum(inputs)
    assert "modularity_q" in metrics
    assert "size_gini" in metrics
    assert metrics["n_singletons"] == 0.0


def test_diffuse_adjacency_densifies() -> None:
    rows = [0, 1, 2, 3, 4]
    cols = [1, 2, 3, 4, 0]
    adj = sp.csr_matrix((np.ones(5), (rows, cols)), shape=(5, 5))
    block = diffuse_adjacency(adj, kernel="heat", t=2.0, top_m=4, series_terms=6, cap_m=8)
    assert block.shape == (5, 5)
    # diffusion should add multi-hop edges beyond the directed skeleton
    assert block.nnz >= adj.nnz
