from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

from artifacts_scripts.embedder_evaluation import citation_recall_at_k, citation_map


def _simple_graph_and_embeddings():
    """4 documents: 0<->1, 2<->3 are citation pairs. Embeddings place pairs close."""
    embeddings = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [5.0, 5.0],
        [5.1, 5.0],
    ], dtype=np.float32)
    row = [0, 1, 2, 3]
    col = [1, 0, 3, 2]
    data = [1, 1, 1, 1]
    A = sp.csr_matrix((data, (row, col)), shape=(4, 4))
    return embeddings, A


class TestCitationRecall:
    def test_perfect_recall(self):
        embeddings, A = _simple_graph_and_embeddings()
        recall = citation_recall_at_k(embeddings, A, k=1)
        assert_allclose(recall, 1.0, atol=1e-6)

    def test_recall_decreases_with_noise(self):
        embeddings, A = _simple_graph_and_embeddings()
        rng = np.random.default_rng(42)
        noisy = embeddings + rng.normal(scale=10.0, size=embeddings.shape).astype(np.float32)
        recall_clean = citation_recall_at_k(embeddings, A, k=1)
        recall_noisy = citation_recall_at_k(noisy, A, k=1)
        assert recall_clean >= recall_noisy

    def test_recall_at_higher_k(self):
        embeddings, A = _simple_graph_and_embeddings()
        recall_1 = citation_recall_at_k(embeddings, A, k=1)
        recall_3 = citation_recall_at_k(embeddings, A, k=3)
        assert recall_3 >= recall_1

    def test_isolated_docs_excluded(self):
        embeddings = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [99.0, 99.0],
        ], dtype=np.float32)
        A = sp.csr_matrix(([1, 1], ([0, 1], [1, 0])), shape=(3, 3))
        recall = citation_recall_at_k(embeddings, A, k=1)
        assert_allclose(recall, 1.0, atol=1e-6)


class TestCitationMAP:
    def test_perfect_map(self):
        embeddings, A = _simple_graph_and_embeddings()
        map_score = citation_map(embeddings, A)
        assert_allclose(map_score, 1.0, atol=1e-6)
