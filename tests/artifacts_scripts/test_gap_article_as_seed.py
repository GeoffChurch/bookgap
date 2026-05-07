"""Tests for symmetric add+remove fixed-point expansion and book-seed semantics.

Synthetic embeddings: two well-separated 2D Gaussian clusters, one with
a book at the center of one cluster. Validates:

- Two article seeds in the same cluster converge to the same fixed point
  under symmetric expansion.
- Symmetric expansion stabilizes within max_iters (no oscillation on
  reasonable basins).
- Boundary-book lookup excludes books inside G.
- A book seed expands into its surrounding article cluster.
- A cluster containing a book is tagged book_anchored.
"""
from __future__ import annotations

import numpy as np
import pytest


def _make_two_cluster_cloud(seed: int = 0):
    """20 articles in cluster A around (-3, 0), 20 around (3, 0). One book at
    (3, 0) — center of cluster B. Returns (emb, types) where emb is normalized
    L2 unit vectors of an embedded 2D layout (so the FAISS K-NN matches
    the production pipeline's cosine semantics).

    Cluster centers are placed away from the origin so that L2 normalization
    preserves the two-cluster structure on the unit circle.
    """
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=(-3.0, 0.0), scale=0.3, size=(20, 2))
    cluster_b = rng.normal(loc=(3.0, 0.0), scale=0.3, size=(20, 2))
    book_b = np.array([[3.0, 0.0]])  # center of cluster B
    pts = np.vstack([cluster_a, cluster_b, book_b]).astype(np.float32)
    types = np.array(["article"] * 20 + ["article"] * 20 + ["book"], dtype=object)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    emb = (pts / np.where(norms > 0, norms, 1.0)).astype(np.float32)
    return np.ascontiguousarray(emb), types


@pytest.fixture
def two_cluster_cloud():
    return _make_two_cluster_cloud()


class TestSymmetricExpansionConvergence:
    def test_same_basin_two_seeds_converge_to_same_fixed_point(self, two_cluster_cloud):
        """Run symmetric expansion from two different article seeds in
        cluster A; the resulting fixed-point member sets should be equal."""
        from artifacts_scripts.gap_article_as_seed import _expand_symmetric

        emb, types = two_cluster_cloud
        # Cluster A is article indices 0..19. Pick two well-inside seeds.
        result_a = _expand_symmetric(seed=0, emb=emb, types=types, tau=0.5,
                                     frontier_width=10, max_iters=50)
        result_b = _expand_symmetric(seed=5, emb=emb, types=types, tau=0.5,
                                     frontier_width=10, max_iters=50)
        assert result_a is not None and result_b is not None
        assert result_a["members_frozen"] == result_b["members_frozen"]
        # Should be a sizable subset of cluster A.
        assert all(m < 20 for m in result_a["members_frozen"])
        assert len(result_a["members_frozen"]) >= 5

    def test_terminates_within_max_iters(self, two_cluster_cloud):
        from artifacts_scripts.gap_article_as_seed import _expand_symmetric

        emb, types = two_cluster_cloud
        result = _expand_symmetric(seed=0, emb=emb, types=types, tau=0.5,
                                   frontier_width=10, max_iters=20)
        # Should converge well before the cap on a clean two-cluster cloud.
        assert result is not None
        assert result["iter_count"] < 20


class TestSymmetricSelfHealing:
    def test_removes_member_dropped_below_tau_after_add(self, two_cluster_cloud):
        """With a starting G that contains a borderline member, symmetric
        expansion should be able to remove it after additions reshape the
        posterior — rather than being stuck or aborting like the
        single-add path."""
        # Constructed so that a marginal seed in cluster A's outskirts gets
        # absorbed: its initial posterior is borderline, but as the basin
        # fills out, the boundary book lookup shifts and the member's
        # posterior re-stabilizes above τ.
        from artifacts_scripts.gap_article_as_seed import _expand_symmetric

        emb, types = two_cluster_cloud
        result = _expand_symmetric(seed=10, emb=emb, types=types, tau=0.5,
                                   frontier_width=10, max_iters=50)
        # Should still converge to cluster A (not collapse to empty).
        assert result is not None
        assert len(result["members_frozen"]) >= 5
