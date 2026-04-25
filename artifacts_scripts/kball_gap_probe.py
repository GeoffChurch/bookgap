"""Stripped-down kball / Voronoi gap-detection probe.

Runs the existing ball-based or Voronoi-based gap algorithms over a
single scope and prints the resulting gaps per topic, skipping the LLM
proposal-generation pipeline that ``batch_generator.py`` would do
next. For comparing kball/Voronoi output against the HDBSCAN-based gap
probe (``artifacts_scripts.hdbscan_gap_probe``).

Usage::

    conda run -p .conda python -m artifacts_scripts.kball_gap_probe \\
        --scope 17/1702/T10181 \\
        --algorithm kball \\
        --radius-fraction 0.75 \\
        --num-gaps 5

Builds a Cloud at the project's standard defaults (SimHash dedup, GIST
embedder, NONE denoiser, etc.) — same defaults as the HDBSCAN probe
— then iterates topics in scope and runs gap detection per topic via
``bookgap.gaps._compute_gaps_for_topic``.
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np

from itertools import islice

from bookgap.cloud import Cloud
from bookgap.configs.cloud.cloud_config import CloudConfig
from bookgap.configs.util.filter_config import FilterConfig, FilterPairConfig
from bookgap.gaps import compute_gap_statistics
from bookgap.openalex_scopes import (
    get_field_topics, get_subfield_topics,
)
from bookgap.snapshot_ingest import load_taxonomy
from bookgap.subcloud import SubCloud
from bookgap.types import CellType
from bookgap.util import Filter, LOGGER, WorkType
from hydra.utils import instantiate


def build_cloud(scope_key: str, snapshot_date: str) -> Cloud:
    """Build a Cloud whose pre-filter is the given scope, post-filter empty.

    The gap-detection code creates per-topic SubClouds under this parent
    via ``get_subclouds_for_topic`` (post-filter on work_type), so the
    parent's pre-filter just needs to cover the universe — all 4 work
    types so books and articles are both in scope.
    """
    parts = scope_key.split("/")
    work_types = tuple(wt.value for wt in WorkType)
    if len(parts) == 1:
        pre = FilterConfig(field=parts[0], work_types=work_types)
    elif len(parts) == 2:
        pre = FilterConfig(field=parts[0], subfield=parts[1], work_types=work_types)
    elif len(parts) == 3:
        pre = FilterConfig(field=parts[0], subfield=parts[1], topic=parts[2],
                           work_types=work_types)
    else:
        raise ValueError(f"scope_key must be F, F/SF, or F/SF/T, got {scope_key!r}")

    fp_cfg = FilterPairConfig(pre=pre, post=FilterConfig())
    # Build a default CloudConfig and instantiate via Hydra utility so we
    # pick up the project defaults (SimHash dedup, GIST embedder, etc.).
    cloud_cfg = CloudConfig(filter_pair=fp_cfg, snapshot_date=snapshot_date)
    return instantiate(cloud_cfg)


def topics_in_scope(scope_key: str, snapshot_date: str) -> List[str]:
    parts = scope_key.split("/")
    if len(parts) == 3:
        return [parts[2]]
    if len(parts) == 2:
        return get_subfield_topics(parts[1], snapshot_date=snapshot_date)
    return get_field_topics(parts[0], snapshot_date=snapshot_date)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, help="F, F/SF, or F/SF/T")
    parser.add_argument("--algorithm", choices=["kball", "voronoi"], default="kball")
    parser.add_argument("--radius-fraction", type=float, default=0.75)
    parser.add_argument("--num-gaps", type=int, default=5)
    parser.add_argument("--snapshot-date", default="2025-11-06")
    parser.add_argument("--top-titles", type=int, default=5,
                        help="number of sample titles to print per gap")
    args = parser.parse_args()

    cell_type = CellType.KBALL if args.algorithm == "kball" else CellType.VORONOI
    print(f"Cell type:        {cell_type}")
    print(f"Radius fraction:  {args.radius_fraction}")
    print(f"Gaps per topic:   {args.num_gaps}")

    cloud = build_cloud(args.scope, args.snapshot_date)
    topics = topics_in_scope(args.scope, args.snapshot_date)
    print(f"Cloud:            {cloud.name()}")
    print(f"Topics in scope:  {len(topics)}")

    pinned_tax = load_taxonomy(args.snapshot_date)

    all_gaps = []
    for i, topic_id in enumerate(topics, 1):
        topic_rec = pinned_tax.topics.get(topic_id, {})
        topic_name = topic_rec.get("display_name", topic_id)
        print(f"\n[{i}/{len(topics)}] Topic {topic_id} — {topic_name}")

        # Split by work_type only — primary_topic_id isn't populated in
        # our parquets (would need a snapshot-ingest fix), but the parent
        # Cloud's pre-filter already restricts to this topic, so a
        # work_type-only post-filter gives us the books/articles split
        # the gap algorithm wants.
        books = SubCloud(parent=cloud, post_filter=Filter(work_types=("book",)))
        articles = SubCloud(parent=cloud, post_filter=Filter(work_types=("article",)))
        n_books = len(books.embeddings())
        n_articles = len(articles.embeddings())
        print(f"  {n_books} books, {n_articles} articles")
        if n_articles == 0:
            print(f"  (no articles)"); continue

        # Per-book gap sizes (used for percentile/z-score statistics)
        book_gap_sizes = cell_type.value.compute_gap_sizes_for_embeddings(
            ambient=articles, landmark=books,
            embeddings=books.embeddings(),
            radius_fraction=args.radius_fraction, k=1, is_intra_cloud=True,
        )

        # Article-query gaps
        gen = cell_type.value.find_gaps(
            ambient=articles, landmark=books, query=articles,
            radius_fraction=args.radius_fraction, k=1, is_intra_cloud=False,
        )
        article_titles = articles.work_titles()
        for gap_idx, gap in enumerate(islice(gen, args.num_gaps)):
            idxs = gap.get_dists_idxs().idxs
            gap_size = len(idxs)
            pctile, zscore, shapiro = compute_gap_statistics(gap_size, book_gap_sizes)
            all_gaps.append(gap)
            print(f"  Gap {gap_idx}: size={gap_size}, pctile={pctile:.1f}, "
                  f"z={zscore:.2f}, shapiro_log10p={shapiro:.2f}")
            for ii in list(idxs[:args.top_titles]):
                t = article_titles[int(ii)]
                print(f"    - {(t or '(no title)')[:90]}")

    print(f"\n=== Total gaps: {len(all_gaps)} across {len(topics)} topic(s) ===")


if __name__ == "__main__":
    main()
