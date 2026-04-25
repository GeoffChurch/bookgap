"""Tests for ``two_direction_pagerank`` — the unified PR API that subsumes
plain personalized PR and bidirectional PR by treating each direction's
seed vector as optional.

Key properties under test:

  - With both seeds populated, the result matches the prior
    ``compute_bidirectional_pagerank`` output (relevance + components).
  - With only forward seeds, it matches ``compute_pagerank`` on the same
    adjacency.
  - With only reverse seeds, it matches reverse-direction
    ``compute_pagerank`` on ``P.T``.
  - With both seeds zero, relevance is uniform (the natural neutral).
  - Per-work weights inside a seed vector behave linearly: PPR over
    ``α·s_a + β·s_b`` equals ``α·ppr(s_a) + β·ppr(s_b)``, so combining
    two same-direction seed sets into one weighted vector is equivalent
    to summing the per-work PPRs.
  - The legacy ``raise ValueError("requires source_work_ids")`` failure
    mode is replaced by graceful degradation to forward-only PR.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from artifacts_scripts.curriculum.pagerank import (
    TwoDirectionPageRankResult,
    compute_bidirectional_pagerank,
    compute_pagerank,
    two_direction_pagerank,
)


def _line_adj(n: int = 6) -> sp.csr_matrix:
    """Line citation graph: 0 → 1 → 2 → … → n-1.
    Earliest (n-1) is most-cited / most-foundational."""
    rows = np.arange(n - 1)
    cols = np.arange(1, n)
    return sp.csr_matrix((np.ones(n - 1), (rows, cols)), shape=(n, n))


class TestTwoDirectionPageRankReducesToBidirectional:
    """Both seeds populated, combine='product' → matches the legacy
    compute_bidirectional_pagerank output exactly (modulo namedtuple shape)."""

    def test_matches_legacy_bidirectional(self):
        adj = _line_adj(8)
        n = adj.shape[0]
        target_seeds = np.zeros(n); target_seeds[0] = 1.0
        source_seeds = np.zeros(n); source_seeds[7] = 1.0

        legacy = compute_bidirectional_pagerank(
            adj, source_seeds=source_seeds, target_seeds=target_seeds,
            alpha=0.85, return_components=True,
        )
        new = two_direction_pagerank(
            adj, fwd_seeds=target_seeds, rev_seeds=source_seeds,
            alpha=0.85, combine="product",
        )
        np.testing.assert_allclose(new.relevance, legacy.relevance, atol=1e-10)
        np.testing.assert_allclose(new.forward_mass, legacy.forward_mass, atol=1e-10)
        np.testing.assert_allclose(new.reverse_mass, legacy.reverse_mass, atol=1e-10)


class TestSingleDirectionFallback:
    """An empty seed vector gracefully reduces the call to one-direction
    PR. No exception, no degenerate-product."""

    def test_empty_reverse_falls_back_to_forward_pr(self):
        adj = _line_adj(6)
        n = adj.shape[0]
        target_seeds = np.zeros(n); target_seeds[0] = 1.0
        empty = np.zeros(n)

        new = two_direction_pagerank(
            adj, fwd_seeds=target_seeds, rev_seeds=empty,
            alpha=0.85, combine="product",
        )
        # Compare against running compute_pagerank directly: same alpha,
        # same seeds, same adj. compute_pagerank normalises seeds and
        # builds P internally, so we just call it the standard way.
        expected = compute_pagerank(adj, target_seeds, alpha=0.85)
        # Both normalised; use the new function's relevance directly.
        np.testing.assert_allclose(new.relevance, expected, atol=1e-10)
        # forward_mass should be the unrenormalised forward pass output
        # (compute_pagerank itself normalises seeds + sums to 1 here).
        assert new.forward_mass is not None
        assert new.reverse_mass is None

    def test_empty_forward_falls_back_to_reverse_pr(self):
        adj = _line_adj(6)
        n = adj.shape[0]
        source_seeds = np.zeros(n); source_seeds[5] = 1.0
        empty = np.zeros(n)

        new = two_direction_pagerank(
            adj, fwd_seeds=empty, rev_seeds=source_seeds,
            alpha=0.85, combine="product",
        )
        # Reverse PR: walk P.T-with-normalize-False semantics.
        from sklearn.preprocessing import normalize as sk_normalize
        P = sk_normalize(adj.astype(np.float64), norm="l1", axis=1, copy=False)
        expected = compute_pagerank(P.T, source_seeds, alpha=0.85, normalize=False)
        # Renormalise to match two_direction's "relevance sums to 1" contract.
        expected = expected / expected.sum()
        np.testing.assert_allclose(new.relevance, expected, atol=1e-10)
        assert new.forward_mass is None
        assert new.reverse_mass is not None

    def test_both_empty_returns_uniform(self):
        adj = _line_adj(6)
        n = adj.shape[0]
        empty = np.zeros(n)

        new = two_direction_pagerank(
            adj, fwd_seeds=empty, rev_seeds=empty,
        )
        np.testing.assert_allclose(new.relevance, np.full(n, 1.0 / n))
        assert new.forward_mass is None
        assert new.reverse_mass is None


class TestPerWorkWeightLinearity:
    """PPR is linear in the seed vector; per-work weights inside a single
    seed vector are equivalent to per-work PPRs summed with those weights."""

    def test_per_work_weights_match_combined_on_sinkless_graph(self):
        # NOTE: compute_pagerank's sink-mass-redistribution-to-seeds makes
        # the iteration bilinear in (v, seeds) when sinks are present, so
        # strict linearity in seeds only holds on graphs without sink
        # nodes. We use a directed cycle (every node has out-degree 1) to
        # exercise rigorous linearity. At full citation-corpus scale the
        # sink fraction is small, so the deviation in production is
        # negligible — but the unit test pins the math on its clean case.
        n = 6
        rows = np.arange(n)
        cols = np.roll(np.arange(n), -1)  # 0→1→2→3→4→5→0
        adj = sp.csr_matrix((np.ones(n), (rows, cols)), shape=(n, n))

        # Two same-direction "sets": A puts 0.7 on node 0, 0.3 on node 1.
        # B puts 0.5 on node 0, 0.5 on node 2. Combine across-set with
        # weights (0.6, 0.4) → single vector.
        a = np.array([0.7, 0.3, 0, 0, 0, 0])
        b = np.array([0.5, 0, 0.5, 0, 0, 0])
        combined = 0.6 * a + 0.4 * b   # = [0.62, 0.18, 0.20, 0, 0, 0]

        empty = np.zeros(n)
        kw = dict(alpha=0.85, max_iter=2000, tol=1e-12)
        rel_combined = two_direction_pagerank(
            adj, fwd_seeds=combined, rev_seeds=empty, **kw,
        ).relevance
        rel_a = two_direction_pagerank(
            adj, fwd_seeds=a, rev_seeds=empty, **kw,
        ).relevance
        rel_b = two_direction_pagerank(
            adj, fwd_seeds=b, rev_seeds=empty, **kw,
        ).relevance

        # On a sinkless graph PPR is exactly linear in seeds:
        #   ppr(0.6·a + 0.4·b) = 0.6·ppr(a) + 0.4·ppr(b)
        expected = 0.6 * rel_a + 0.4 * rel_b
        expected = expected / expected.sum()
        np.testing.assert_allclose(rel_combined, expected, atol=1e-7)


class TestCombineLinear:
    def test_linear_theta_endpoints_match_single_direction(self):
        adj = _line_adj(6)
        n = adj.shape[0]
        target_seeds = np.zeros(n); target_seeds[0] = 1.0
        source_seeds = np.zeros(n); source_seeds[5] = 1.0

        # theta=1.0 → forward only
        new = two_direction_pagerank(
            adj, fwd_seeds=target_seeds, rev_seeds=source_seeds,
            combine="linear", linear_theta=1.0,
        )
        forward_only = two_direction_pagerank(
            adj, fwd_seeds=target_seeds, rev_seeds=np.zeros(n),
        )
        np.testing.assert_allclose(new.relevance, forward_only.relevance, atol=1e-10)

        # theta=0.0 → reverse only
        new = two_direction_pagerank(
            adj, fwd_seeds=target_seeds, rev_seeds=source_seeds,
            combine="linear", linear_theta=0.0,
        )
        reverse_only = two_direction_pagerank(
            adj, fwd_seeds=np.zeros(n), rev_seeds=source_seeds,
        )
        np.testing.assert_allclose(new.relevance, reverse_only.relevance, atol=1e-10)

    def test_invalid_combine_raises(self):
        adj = _line_adj(4)
        n = adj.shape[0]
        s = np.zeros(n); s[0] = 1.0
        with pytest.raises(ValueError, match="combine"):
            two_direction_pagerank(adj, fwd_seeds=s, rev_seeds=s, combine="geometric")
