from __future__ import annotations

import numpy as np

from artifacts_scripts.curriculum.scalable_clustering import bisecting_kmeans_linkage
from bookgap.dendrogram import _truncate_bisecting_kmeans_dendrogram


def _canonical_clusters(labels: np.ndarray) -> list[tuple[int, ...]]:
    clusters: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels.tolist()):
        clusters.setdefault(int(lab), []).append(int(idx))
    return sorted(tuple(sorted(v)) for v in clusters.values())


def test_truncate_bisecting_kmeans_matches_direct_build():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(256, 8)).astype(np.float64)

    fine = bisecting_kmeans_linkage(
        embeddings,
        min_cluster_size=2,
        max_depth=None,
        random_state=0,
        use_minibatch=False,
        verbose=False,
    )

    requested_min = 25
    requested_max_depth = 4

    truncated = _truncate_bisecting_kmeans_dendrogram(
        linkage=fine.linkage,
        leaf_cluster_assignments=fine.leaf_cluster_assignments,
        n_leaf_clusters=fine.n_leaf_clusters,
        min_cluster_size=requested_min,
        max_depth=requested_max_depth,
    )

    direct = bisecting_kmeans_linkage(
        embeddings,
        min_cluster_size=requested_min,
        max_depth=requested_max_depth,
        random_state=0,
        use_minibatch=False,
        verbose=False,
    )

    assert truncated.n_leaf_clusters == direct.n_leaf_clusters
    assert _canonical_clusters(truncated.leaf_cluster_assignments) == _canonical_clusters(
        direct.leaf_cluster_assignments
    )


def test_truncate_bisecting_kmeans_no_depth_limit_matches_direct_build():
    rng = np.random.default_rng(123)
    embeddings = rng.normal(size=(200, 6)).astype(np.float64)

    fine = bisecting_kmeans_linkage(
        embeddings,
        min_cluster_size=1,
        max_depth=None,
        random_state=42,
        use_minibatch=False,
        verbose=False,
    )

    requested_min = 15

    truncated = _truncate_bisecting_kmeans_dendrogram(
        linkage=fine.linkage,
        leaf_cluster_assignments=fine.leaf_cluster_assignments,
        n_leaf_clusters=fine.n_leaf_clusters,
        min_cluster_size=requested_min,
        max_depth=None,
    )

    direct = bisecting_kmeans_linkage(
        embeddings,
        min_cluster_size=requested_min,
        max_depth=None,
        random_state=42,
        use_minibatch=False,
        verbose=False,
    )

    assert truncated.n_leaf_clusters == direct.n_leaf_clusters
    assert _canonical_clusters(truncated.leaf_cluster_assignments) == _canonical_clusters(
        direct.leaf_cluster_assignments
    )
