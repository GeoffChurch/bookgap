"""Tests for multi-view spectral clustering with CSLS correction.

Components:
- CSLS hubness correction (Conneau et al. 2017).
- Per-view similarity builders (embedding cosine, citation Ruzicka).
- Co-regularized multi-view spectral solver (Kumar-Rai-Daumé 2011).
"""

import numpy as np
import pytest
import scipy.sparse as sp

from artifacts_scripts.curriculum.multi_view_spectral import csls_correct


class TestCSLSCorrect:
    def test_uniform_similarity_collapses_to_zero(self):
        # Uniform off-diagonal similarity → each row's r_K equals that
        # uniform value → CSLS off-diagonal = 2*sim - sim - sim = 0.
        n = 5
        W = np.full((n, n), 0.5)
        np.fill_diagonal(W, 1.0)
        result = csls_correct(W, k=2, clip=False)
        off_diag = result - np.diag(np.diag(result))
        assert np.allclose(off_diag, 0.0, atol=1e-9)

    def test_hub_penalty_differentiates(self):
        # Node 0 is similar (0.8) to all of 1..4; nodes 1..4 are mutually
        # dissimilar (0). Node 0 is a "hub". With k=2:
        #   r_K(0) = mean of top-2 of [0.8, 0.8, 0.8, 0.8] = 0.8
        #   r_K(1) = mean of top-2 of [0.8, 0, 0, 0] = 0.4
        # CSLS(0, 1) = 2 * 0.8 - 0.8 - 0.4 = 0.4 (less than raw 0.8 — hub
        # contribution discounted).
        n = 5
        W = np.zeros((n, n))
        W[0, 1:] = 0.8
        W[1:, 0] = 0.8
        np.fill_diagonal(W, 1.0)
        result = csls_correct(W, k=2, clip=False)
        assert result[0, 1] == pytest.approx(0.4, abs=1e-9)

    def test_symmetry_preserved(self):
        rng = np.random.default_rng(42)
        n = 10
        raw = rng.uniform(0, 1, size=(n, n))
        W = (raw + raw.T) / 2
        np.fill_diagonal(W, 1.0)
        result = csls_correct(W, k=3)
        assert np.allclose(result, result.T, atol=1e-9)

    def test_clip_at_zero(self):
        n = 5
        W = np.full((n, n), 0.1)
        np.fill_diagonal(W, 1.0)
        result_clipped = csls_correct(W, k=2, clip=True)
        result_unclipped = csls_correct(W, k=2, clip=False)
        assert (result_clipped >= 0).all()
        # Whatever was negative in unclipped is exactly zero in clipped.
        neg_mask = result_unclipped < 0
        assert np.allclose(result_clipped[neg_mask], 0.0)

    def test_excludes_self_in_neighbourhood(self):
        # Self-similarity = 1; off-diagonal = 0.1.
        # If self were included in top-k, r_K(i) would pull toward 1 and
        # over-correct. With self excluded, r_K(i) = 0.1; CSLS off-diagonal
        # = 2*0.1 - 0.1 - 0.1 = 0; CSLS diagonal = 2*1 - 0.1 - 0.1 = 1.8.
        n = 5
        W = np.full((n, n), 0.1)
        np.fill_diagonal(W, 1.0)
        result = csls_correct(W, k=2, clip=False)
        assert result[0, 1] == pytest.approx(0.0, abs=1e-9)
        assert result[0, 0] == pytest.approx(1.8, abs=1e-9)

    def test_k_too_large_raises(self):
        W = np.eye(3)
        with pytest.raises(ValueError, match="k=3 must be less than n=3"):
            csls_correct(W, k=3)
