"""Maximal-bookless-cluster extraction from a single-linkage tree.

Given an HDBSCAN ``single_linkage_tree_`` (scipy-style merge matrix) and a
boolean mask of which leaf nodes are books, identify each article's
"maximal bookless cluster": the largest ancestor cluster whose subtree
contains no books. Each unique maximal-bookless cluster is a "gap" — a
region of articles bounded by where books are, defined hierarchically
rather than geometrically.

This is the HDBSCAN-native analog of K-ball gap detection:

- K-ball: ball of articles within radius_fraction × dist_to_nearest_book
- Maximal-bookless-cluster: walk up the dendrogram until adding a book

Both define "regions of articles bounded by where books are." K-ball does
this geometrically (Euclidean balls); this function does it hierarchically
(via the agglomerative merge tree).

The algorithm is O(n) given the precomputed tree:

1. Build the parent map from the merge tree (each merge creates an
   internal node whose children are the two merged subtrees).
2. Bottom-up propagate ``has_book[node] = OR over leaf descendants``.
3. For each article leaf, walk up to the largest ancestor whose subtree
   has no books — that ancestor IS the article's maximal bookless
   cluster.
4. Articles sharing the same maximal-bookless ancestor form a gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MaximalBooklessGaps:
    """Result of maximal-bookless-cluster extraction.

    Attributes
    ----------
    article_to_cluster
        Length-``n_articles`` array. ``article_to_cluster[i]`` is the tree
        node ID of article ``i``'s maximal-bookless ancestor, or ``-1`` if
        the article's leaf parent already contains a book (so the article
        forms a singleton "gap" of size 1 by itself, semantically the
        article's own leaf).
    gap_to_articles
        ``{cluster_id: array_of_article_indices}``. One entry per unique
        maximal-bookless cluster. Article indices are into the original
        article-leaf indexing, not the global tree-leaf indexing.
    gap_sizes
        ``{cluster_id: n_articles_in_gap}``. Same keys as
        ``gap_to_articles``; convenience for histogram analysis.
    """

    article_to_cluster: NDArray[np.int64]
    gap_to_articles: dict[int, NDArray[np.int64]]
    gap_sizes: dict[int, int]


def find_maximal_bookless_clusters(
    single_linkage_tree: NDArray[np.floating],
    is_book: NDArray[np.bool_],
) -> MaximalBooklessGaps:
    """Extract maximal-bookless clusters from an HDBSCAN single-linkage tree.

    Parameters
    ----------
    single_linkage_tree
        The HDBSCAN ``single_linkage_tree_`` (or ``_single_linkage_tree``
        for fast_hdbscan). Shape ``(n_points - 1, 4)`` with columns
        ``[left_child, right_child, distance, count]``. Leaf node IDs are
        ``0..n_points-1``; internal node ``n_points + i`` is created by
        merge row ``i``.
    is_book
        Length-``n_points`` boolean array; ``True`` for book leaves,
        ``False`` for article leaves. Leaves not in either category
        (e.g. preprints, book-chapters) should be passed as ``False`` —
        they don't gate the walk-up but they will be counted as articles
        for gap membership. Callers wanting strict article-only gaps
        should pre-filter their leaves.

    Returns
    -------
    MaximalBooklessGaps
        Per-article cluster assignments, gap-to-articles mapping, and
        gap sizes.

    Notes
    -----
    Singleton-gap semantics: if an article's parent in the merge tree
    already contains a book, the article's "maximal bookless cluster" is
    the leaf itself — a degenerate size-1 gap. We mark these with
    ``article_to_cluster[i] = -1`` rather than the leaf ID, so callers
    can easily filter them out. The size-1 leaf-gaps don't appear in
    ``gap_to_articles`` either; they're representable but typically not
    interesting.
    """
    n_points = is_book.shape[0]
    n_internal = single_linkage_tree.shape[0]
    if n_internal != n_points - 1:
        raise ValueError(
            f"single_linkage_tree has {n_internal} merges but is_book "
            f"has {n_points} points; expected {n_points - 1} merges."
        )
    n_total = 2 * n_points - 1

    # Build parent map: leaves and internal nodes.
    parent = np.full(n_total, -1, dtype=np.int64)
    for i in range(n_internal):
        left = int(single_linkage_tree[i, 0])
        right = int(single_linkage_tree[i, 1])
        new_node = n_points + i
        parent[left] = new_node
        parent[right] = new_node

    # Bottom-up: has_book[node] = OR over leaf descendants.
    has_book = np.zeros(n_total, dtype=bool)
    has_book[:n_points] = is_book
    for i in range(n_internal):
        left = int(single_linkage_tree[i, 0])
        right = int(single_linkage_tree[i, 1])
        new_node = n_points + i
        has_book[new_node] = has_book[left] or has_book[right]

    # Walk up from each article leaf to find its maximal bookless ancestor.
    article_indices = np.where(~is_book)[0]
    article_to_cluster = np.full(n_points, -1, dtype=np.int64)
    for idx in article_indices:
        p = parent[idx]
        if p == -1 or has_book[p]:
            # Either the article is the root (degenerate) or its
            # immediate parent already contains a book — singleton gap.
            article_to_cluster[idx] = -1
            continue
        current = p
        while True:
            grandparent = parent[current]
            if grandparent == -1:
                break  # reached root
            if has_book[grandparent]:
                break
            current = grandparent
        article_to_cluster[idx] = current

    # Group articles by their maximal-bookless cluster.
    gap_to_articles: dict[int, NDArray[np.int64]] = {}
    valid = article_to_cluster[article_indices] >= 0
    valid_articles = article_indices[valid]
    valid_clusters = article_to_cluster[valid_articles]
    for cluster_id in np.unique(valid_clusters):
        members = valid_articles[valid_clusters == cluster_id]
        gap_to_articles[int(cluster_id)] = members

    gap_sizes = {k: len(v) for k, v in gap_to_articles.items()}

    return MaximalBooklessGaps(
        article_to_cluster=article_to_cluster,
        gap_to_articles=gap_to_articles,
        gap_sizes=gap_sizes,
    )


def compute_subtree_leaves(
    single_linkage_tree: NDArray[np.floating],
    n_points: int,
) -> dict[int, NDArray[np.int64]]:
    """Compute the set of leaf descendants for each tree node.

    Returns ``{node_id: array_of_leaf_ids}`` for every internal node.
    Leaves themselves aren't included (they're trivially their own
    subtree). Uses iterative bottom-up accumulation; O(total subtree
    sizes) which is O(n log n) for a balanced tree.

    Useful for finding the books that "bound" a gap — i.e., the books
    in the lowest book-containing ancestor of a maximal-bookless cluster.
    """
    n_internal = single_linkage_tree.shape[0]
    if n_internal != n_points - 1:
        raise ValueError(
            f"single_linkage_tree has {n_internal} merges; expected "
            f"{n_points - 1} for {n_points} points."
        )
    subtree: dict[int, NDArray[np.int64]] = {}
    for i in range(n_internal):
        left = int(single_linkage_tree[i, 0])
        right = int(single_linkage_tree[i, 1])
        new_node = n_points + i
        left_leaves = subtree.get(left, np.array([left], dtype=np.int64))
        right_leaves = subtree.get(right, np.array([right], dtype=np.int64))
        subtree[new_node] = np.concatenate([left_leaves, right_leaves])
        # Free intermediate dict entries we no longer need to save memory.
        # (Each non-root internal node is only used as a child once.)
        if left >= n_points:
            subtree.pop(left, None)
        if right >= n_points:
            subtree.pop(right, None)
    return subtree


def find_boundary_books(
    single_linkage_tree: NDArray[np.floating],
    is_book: NDArray[np.bool_],
    cluster_id: int,
    children_map: dict[int, tuple[int, int]] | None = None,
    parent_map: NDArray[np.int64] | None = None,
) -> NDArray[np.int64]:
    """For a given maximal-bookless cluster, find the books in the lowest
    book-containing ancestor — i.e., the books that "stopped" the walk-up.

    These books form the natural boundary of the gap; they're the
    closest-related books to the gap's articles in the dendrogram.

    Parameters
    ----------
    single_linkage_tree, is_book : as in find_maximal_bookless_clusters.
    cluster_id : the maximal-bookless cluster ID whose boundary books we want.
    children_map, parent_map : optional precomputed maps (for callers
        querying many clusters; saves rebuilding per call). If omitted,
        constructed locally from ``single_linkage_tree``.
    """
    n_points = is_book.shape[0]
    n_internal = single_linkage_tree.shape[0]

    if parent_map is None:
        parent_map = np.full(2 * n_points - 1, -1, dtype=np.int64)
        for i in range(n_internal):
            left = int(single_linkage_tree[i, 0])
            right = int(single_linkage_tree[i, 1])
            parent_map[left] = n_points + i
            parent_map[right] = n_points + i

    bounding_ancestor = int(parent_map[cluster_id])
    if bounding_ancestor == -1:
        return np.array([], dtype=np.int64)

    if children_map is None:
        children_map = {}
        for i in range(n_internal):
            new_node = n_points + i
            children_map[new_node] = (
                int(single_linkage_tree[i, 0]),
                int(single_linkage_tree[i, 1]),
            )

    # Iterative descent from bounding_ancestor; collect book leaves.
    book_leaves: list[int] = []
    stack: list[int] = [bounding_ancestor]
    while stack:
        node = stack.pop()
        if node < n_points:
            if is_book[node]:
                book_leaves.append(node)
        else:
            children = children_map.get(node)
            if children is not None:
                stack.append(children[0])
                stack.append(children[1])
    return np.array(book_leaves, dtype=np.int64)
