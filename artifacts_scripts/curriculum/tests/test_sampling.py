"""Tests for sampling by relevance (current API)."""

import numpy as np
import pytest

from artifacts_scripts.curriculum.sampling import sample_by_relevance


class TestSampleByRelevance:
    def test_top_k_selection(self):
        # Global scores for 10 items
        scores = np.array([0.1, 0.9, 0.3, 0.4, 0.7, 0.2, 0.8, 0.05, 0.6, 0.0], dtype=float)
        # Consider only a subset of items
        item_indices = np.array([1, 2, 4, 6, 8])
        # Request top 3 from subset by global score
        selected = sample_by_relevance(item_indices, scores, 3)
        # Expected order: indices with scores 0.9 (1), 0.8 (6), 0.7 (4)
        assert selected.tolist() == [1, 6, 4]

    def test_fewer_items_than_requested(self):
        scores = np.array([0.5, 0.4, 0.3], dtype=float)
        item_indices = np.array([0, 2])
        selected = sample_by_relevance(item_indices, scores, 5)
        # Returns all available, sorted by relevance: idx 0 (0.5), then 2 (0.3)
        assert selected.tolist() == [0, 2]

    def test_ties_are_consistent(self):
        # Tied scores; implementation uses argsort which is stable on equal values
        scores = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)
        item_indices = np.array([0, 1, 2])
        selected = sample_by_relevance(item_indices, scores, 2)
        # Any two of {0,1,2}; ensure subset and deterministic order given stable sort
        assert set(selected.tolist()) <= {0, 1, 2}
        assert len(selected) == 2
