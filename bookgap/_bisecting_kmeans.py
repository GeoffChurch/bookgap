"""Bisecting k-means linkage, vendored from the sibling ``curriculum`` repo.

Provenance
----------
Copied verbatim (modulo the logger import) from ``curriculum/core/dendrogram.py``
at commit dee42ec, <https://github.com/GeoffChurch/curriculum>.

Why vendored rather than imported
---------------------------------
``curriculum`` is a private repo consumed as an editable path dependency
(``../curriculum``). That makes bookgap uninstallable for anyone without the
sibling checkout: ``[tool.uv.sources]`` is unconditional, so ``uv`` resolves the
path even when no selected extra needs it, and ``uv sync`` fails with exit 2 on a
fresh clone. Marking the dependency optional does *not* fix this -- ``uv lock``
resolves every extra. Since ``bookgap/`` used exactly one symbol from
``curriculum`` (this function), copying it is cheaper than either publishing that
repo or shipping something that will not install.

Keeping it in sync
------------------
This is a fork, not a mirror: it will drift, and nothing checks that it hasn't.
If ``curriculum`` is ever published, delete this module and restore the import in
``dendrogram.py``. ``tests/test_curriculum_api_contract.py`` guards the *other*
call sites (artifacts_scripts, curriculum_lean, tests) and does not cover this.

Licence
-------
Same-author code, and ``curriculum`` carries no LICENSE file at time of copying.
Both repos need one; see docs/backlog/open-source-release.md.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bookgap.util import LOGGER


@dataclass
class BisectingKMeansResult:
    """Scipy-compatible linkage + leaf-cluster mapping from bisecting KMeans."""

    linkage: NDArray[np.floating]
    leaf_cluster_assignments: NDArray[np.intp]
    n_leaf_clusters: int


def bisecting_kmeans_linkage(
    embeddings: NDArray[np.floating],
    *,
    sample_weights: NDArray[np.floating] | None = None,
    min_cluster_size: int = 1,
    min_cluster_weight: float = 0.0,
    max_depth: int | None = None,
    random_state: int | None = 42,
    use_minibatch: bool = True,
    batch_size: int = 10000,
    verbose: bool = False,
) -> BisectingKMeansResult:
    """Hierarchical clustering via recursive bisecting K-Means (O(n) memory).

    With ``sample_weights`` the bisection respects per-point relevance mass
    (high-relevance points anchor centroids instead of being peeled off as
    singletons). Returns a scipy-compatible linkage matrix.
    """
    if min_cluster_weight < 0:
        raise ValueError("min_cluster_weight must be >= 0")
    if min_cluster_weight > 0 and sample_weights is None:
        raise ValueError(
            "min_cluster_weight > 0 requires sample_weights "
            "(weight-based threshold is undefined without per-point weights)"
        )
    n_samples = len(embeddings)
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if sample_weights.shape != (n_samples,):
            raise ValueError(
                f"sample_weights shape {sample_weights.shape} must match "
                f"embeddings n_samples ({n_samples},)"
            )
        if (sample_weights < 0).any():
            raise ValueError("sample_weights must be non-negative")
    if n_samples <= 1:
        assignments = np.zeros(n_samples, dtype=np.intp)
        return BisectingKMeansResult(
            linkage=np.zeros((0, 4), dtype=np.float64),
            leaf_cluster_assignments=assignments,
            n_leaf_clusters=max(1, n_samples),
        )
    if n_samples <= min_cluster_size or (
        sample_weights is not None
        and min_cluster_weight > 0
        and float(sample_weights.sum()) <= min_cluster_weight
    ):
        assignments = np.zeros(n_samples, dtype=np.intp)
        return BisectingKMeansResult(
            linkage=np.zeros((0, 4), dtype=np.float64),
            leaf_cluster_assignments=assignments,
            n_leaf_clusters=1,
        )

    if verbose:
        LOGGER.info(
            "Bisecting K-Means: %d points, min_cluster_size=%d, max_depth=%s%s",
            n_samples, min_cluster_size, max_depth,
            "" if sample_weights is None else " (weighted)",
        )

    old_limit = sys.getrecursionlimit()
    needed_limit = max(old_limit, 10000)
    if needed_limit > old_limit:
        sys.setrecursionlimit(needed_limit)

    try:
        return _bisecting_kmeans_impl(
            embeddings,
            sample_weights=sample_weights,
            n_samples=n_samples,
            min_cluster_size=min_cluster_size,
            min_cluster_weight=min_cluster_weight,
            max_depth=max_depth,
            random_state=random_state,
            use_minibatch=use_minibatch,
            batch_size=batch_size,
            verbose=verbose,
        )
    finally:
        sys.setrecursionlimit(old_limit)


def _bisecting_kmeans_impl(
    embeddings: NDArray[np.floating],
    *,
    sample_weights: NDArray[np.floating] | None,
    n_samples: int,
    min_cluster_size: int,
    min_cluster_weight: float,
    max_depth: int | None,
    random_state: int | None,
    use_minibatch: bool,
    batch_size: int,
    verbose: bool,
) -> BisectingKMeansResult:
    # sklearn imported here, not at module scope: bookgap.cloud imports this
    # module transitively, and it must stay importable without sklearn installed.
    from sklearn.cluster import KMeans, MiniBatchKMeans

    splits_done = [0]
    log_interval = max(1, n_samples // (min_cluster_size * 20))

    leaf_cluster_assignments = np.full(n_samples, -1, dtype=np.intp)
    next_leaf_cluster_id = [0]

    def build_tree(indices: NDArray[np.intp], depth: int = 0) -> tuple:
        size = len(indices)
        if size == 1:
            return ("leaf", int(indices[0]))
        if size <= min_cluster_size:
            return ("leaf_cluster", indices)
        if (
            sample_weights is not None
            and min_cluster_weight > 0
            and float(sample_weights[indices].sum()) <= min_cluster_weight
        ):
            return ("leaf_cluster", indices)
        if max_depth is not None and depth >= max_depth:
            return ("leaf_cluster", indices)
        if size == 2:
            dist = float(np.linalg.norm(embeddings[indices[0]] - embeddings[indices[1]]))
            return ("internal", ("leaf", int(indices[0])), ("leaf", int(indices[1])), dist, 2)

        cluster_embeddings = embeddings[indices]
        cluster_weights = sample_weights[indices] if sample_weights is not None else None
        if cluster_weights is not None and float(cluster_weights.sum()) == 0:
            cluster_weights = None

        if use_minibatch and size > batch_size:
            kmeans = MiniBatchKMeans(
                n_clusters=2, max_iter=100, n_init=1,
                random_state=random_state, batch_size=min(batch_size, size),
            )
        else:
            kmeans = KMeans(
                n_clusters=2, max_iter=100, n_init=1, algorithm="lloyd",
                random_state=random_state,
            )

        labels = kmeans.fit_predict(cluster_embeddings, sample_weight=cluster_weights)
        left_indices = indices[labels == 0]
        right_indices = indices[labels == 1]
        if len(left_indices) == 0 or len(right_indices) == 0:
            mid = size // 2
            left_indices = indices[:mid]
            right_indices = indices[mid:]

        if sample_weights is not None:
            w_left = sample_weights[left_indices]
            w_right = sample_weights[right_indices]
            wsum_left = float(w_left.sum())
            wsum_right = float(w_right.sum())
            if wsum_left > 0:
                left_centroid = (embeddings[left_indices] * w_left[:, None]).sum(axis=0) / wsum_left
            else:
                left_centroid = embeddings[left_indices].mean(axis=0)
                wsum_left = float(len(left_indices))
            if wsum_right > 0:
                right_centroid = (embeddings[right_indices] * w_right[:, None]).sum(axis=0) / wsum_right
            else:
                right_centroid = embeddings[right_indices].mean(axis=0)
                wsum_right = float(len(right_indices))
            n_left = wsum_left
            n_right = wsum_right
        else:
            left_centroid = embeddings[left_indices].mean(axis=0)
            right_centroid = embeddings[right_indices].mean(axis=0)
            n_left, n_right = len(left_indices), len(right_indices)
        dist = float(
            np.sqrt(2 * n_left * n_right / (n_left + n_right))
            * np.linalg.norm(left_centroid - right_centroid)
        )

        splits_done[0] += 1
        if verbose and splits_done[0] % log_interval == 0:
            LOGGER.info("  Bisecting K-Means: %d splits done", splits_done[0])

        left_tree = build_tree(left_indices, depth + 1)
        right_tree = build_tree(right_indices, depth + 1)
        return ("internal", left_tree, right_tree, dist, size)

    tree = build_tree(np.arange(n_samples))

    linkage_rows: list[list[float]] = []
    leaf_cluster_to_id: dict[int, int] = {}

    def count_leaves(node, depth: int = 0) -> int:
        node_type = node[0]
        if node_type == "leaf":
            point_idx = node[1]
            if id(node) not in leaf_cluster_to_id:
                leaf_id = next_leaf_cluster_id[0]
                leaf_cluster_to_id[id(node)] = leaf_id
                leaf_cluster_assignments[point_idx] = leaf_id
                next_leaf_cluster_id[0] += 1
            return 1
        elif node_type == "leaf_cluster":
            indices = node[1]
            if id(node) not in leaf_cluster_to_id:
                leaf_id = next_leaf_cluster_id[0]
                leaf_cluster_to_id[id(node)] = leaf_id
                leaf_cluster_assignments[indices] = leaf_id
                next_leaf_cluster_id[0] += 1
            return 1
        else:
            _, left, right, _, _ = node
            return count_leaves(left, depth + 1) + count_leaves(right, depth + 1)

    count_leaves(tree)
    n_leaf_clusters = next_leaf_cluster_id[0]

    next_internal_id = [n_leaf_clusters]

    def get_node_id(node) -> int:
        node_type = node[0]
        if node_type in ("leaf", "leaf_cluster"):
            return leaf_cluster_to_id[id(node)]
        _, left, right, dist, size = node
        left_id = get_node_id(left)
        right_id = get_node_id(right)
        my_id = next_internal_id[0]
        next_internal_id[0] += 1
        linkage_rows.append([float(left_id), float(right_id), dist, float(size)])
        return my_id

    get_node_id(tree)

    if not linkage_rows:
        return BisectingKMeansResult(
            linkage=np.zeros((0, 4), dtype=np.float64),
            leaf_cluster_assignments=leaf_cluster_assignments,
            n_leaf_clusters=n_leaf_clusters,
        )

    linkage = np.array(linkage_rows, dtype=np.float64)
    return BisectingKMeansResult(
        linkage=linkage,
        leaf_cluster_assignments=leaf_cluster_assignments,
        n_leaf_clusters=n_leaf_clusters,
    )
