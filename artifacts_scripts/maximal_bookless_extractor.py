"""Extract maximal-bookless clusters from a saved HDBSCAN run state.

Pairs with ``hdbscan_gap_probe.py --save-state-dir``: takes the dumped
single-linkage tree + labels + types + topic_ids and produces a CSV of
the maximal-bookless gaps (the HDBSCAN-native analog of K-ball gap
detection).

Output schema mirrors ``kball_gap_probe.py``'s CSV so downstream
quantitative-metrics comparison can load both algorithms' gaps with the
same column structure:

    artifacts/maximal_bookless/{state_dir_name}.csv

with per-gap rows::

    scope, algorithm, denoiser, drop_k, gap_id,
    n_articles, n_boundary_books, dominant_topic_id, dominant_topic_count,
    article_indices_json, boundary_book_indices_json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from bookgap.maximal_bookless import (
    find_maximal_bookless_clusters,
    find_boundary_books,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dir", required=True,
                   help="Directory produced by hdbscan_gap_probe --save-state-dir")
    p.add_argument("--output-csv", default=None,
                   help="Output CSV path. Default: "
                        "artifacts/maximal_bookless/{state_dir_name}.csv")
    p.add_argument("--include-singletons", action="store_true",
                   help="Include articles whose immediate parent already "
                        "contains a book (size-1 gaps). Default: skip them.")
    p.add_argument("--min-gap-size", type=int, default=1,
                   help="Filter gaps smaller than this (default: 1, no filter).")
    args = p.parse_args()

    state_dir = Path(args.state_dir)
    if not state_dir.is_dir():
        raise SystemExit(f"State dir not found: {state_dir}")

    print(f"Loading state from {state_dir}...")
    slt = np.load(state_dir / "single_linkage_tree.npy")
    with open(state_dir / "ids.json") as f:
        ids: list[str] = json.load(f)
    with open(state_dir / "types.json") as f:
        types: list[str] = json.load(f)
    with open(state_dir / "topic_ids.json") as f:
        topic_ids: list[str] = json.load(f)
    with open(state_dir / "metadata.json") as f:
        metadata: dict = json.load(f)

    n_points = len(ids)
    if slt.shape != (n_points - 1, 4):
        raise SystemExit(
            f"Tree shape {slt.shape} doesn't match n_points {n_points} "
            f"(expected ({n_points - 1}, 4))"
        )

    types_arr = np.array(types)
    is_book = types_arr == "book"
    is_article = types_arr == "article"
    print(f"  {n_points:,} works ({int(is_book.sum()):,} books, "
          f"{int(is_article.sum()):,} articles, "
          f"{int(((~is_book) & (~is_article)).sum()):,} other)")

    # We treat ONLY books as "bounding" the walk. Non-book non-article
    # works (preprints, book-chapters) are passed as is_book=False so they
    # don't gate the walk-up; they are not selected as gap members either
    # — only true articles count.
    print("Computing maximal-bookless clusters...")
    result = find_maximal_bookless_clusters(slt, is_book)

    gap_ids_sorted = sorted(result.gap_to_articles.keys(),
                            key=lambda k: -result.gap_sizes[k])
    n_singleton = int((result.article_to_cluster[is_article] == -1).sum())
    print(f"  unique gaps: {len(gap_ids_sorted)}")
    print(f"  size distribution: "
          f"min={min(result.gap_sizes.values()) if result.gap_sizes else 0}, "
          f"median={int(np.median(list(result.gap_sizes.values()))) if result.gap_sizes else 0}, "
          f"max={max(result.gap_sizes.values()) if result.gap_sizes else 0}")
    print(f"  singletons (article with book in immediate parent): {n_singleton}")

    # Restrict gap membership to actual articles (filter out non-article
    # leaves that may have been swept into a gap because they aren't
    # books; the gap definition is article-centric).
    filtered_gap_to_articles: dict[int, np.ndarray] = {}
    for gap_id, members in result.gap_to_articles.items():
        article_members = members[is_article[members]]
        if len(article_members) >= args.min_gap_size:
            filtered_gap_to_articles[gap_id] = article_members
    print(f"  gaps after filtering (min_size={args.min_gap_size}, articles only): "
          f"{len(filtered_gap_to_articles)}")

    # Optionally add singleton gaps (each article that didn't have a
    # bookless ancestor becomes its own size-1 gap).
    if args.include_singletons:
        singleton_articles = np.where(
            is_article & (result.article_to_cluster == -1)
        )[0]
        for art_idx in singleton_articles:
            filtered_gap_to_articles[-int(art_idx) - 1] = np.array([art_idx])
        print(f"  + {len(singleton_articles)} singleton gaps appended")

    output_csv = args.output_csv
    if output_csv is None:
        out_dir = Path("artifacts/maximal_bookless")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_csv = out_dir / f"{state_dir.name}.csv"
    output_csv = Path(output_csv)

    print(f"Writing {output_csv}...")
    fieldnames = [
        "scope", "algorithm", "denoiser", "drop_k", "gap_id",
        "n_articles", "n_boundary_books",
        "dominant_topic_id", "dominant_topic_count", "n_topics_spanned",
        "article_indices_json", "boundary_book_indices_json",
    ]
    # Precompute parent and children maps once so find_boundary_books is
    # O(subtree_size) per call instead of O(n) per call.
    print("Precomputing parent/children maps for boundary-book lookups...")
    parent_map = np.full(2 * n_points - 1, -1, dtype=np.int64)
    children_map: dict[int, tuple[int, int]] = {}
    for i in range(slt.shape[0]):
        left = int(slt[i, 0]); right = int(slt[i, 1])
        new_node = n_points + i
        parent_map[left] = new_node
        parent_map[right] = new_node
        children_map[new_node] = (left, right)

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for gap_id in sorted(filtered_gap_to_articles.keys(),
                             key=lambda k: -len(filtered_gap_to_articles[k])):
            members = filtered_gap_to_articles[gap_id]
            n_articles = len(members)
            # Topic distribution
            member_topics = [topic_ids[i] for i in members]
            topic_counts = Counter(member_topics)
            dominant_topic, dominant_count = topic_counts.most_common(1)[0]
            # Boundary books: only meaningful for non-singleton gaps
            if gap_id >= 0:
                boundary_books = find_boundary_books(
                    slt, is_book, gap_id,
                    children_map=children_map, parent_map=parent_map,
                )
            else:
                boundary_books = np.array([], dtype=np.int64)

            w.writerow({
                "scope": metadata["scope"],
                "algorithm": "maximal_bookless",
                "denoiser": metadata["denoiser"],
                "drop_k": metadata["drop_k"],
                "gap_id": int(gap_id),
                "n_articles": n_articles,
                "n_boundary_books": int(len(boundary_books)),
                "dominant_topic_id": dominant_topic,
                "dominant_topic_count": dominant_count,
                "n_topics_spanned": len(topic_counts),
                "article_indices_json": json.dumps([int(i) for i in members]),
                "boundary_book_indices_json": json.dumps(
                    [int(i) for i in boundary_books]
                ),
            })

    print(f"Wrote {output_csv} with {len(filtered_gap_to_articles)} gaps")


if __name__ == "__main__":
    main()
