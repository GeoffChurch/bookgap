"""Stripped-down kball / Voronoi gap-detection probe.

Runs the existing ball-based or Voronoi-based gap algorithms over a
single scope, prints the resulting gaps per topic, and writes a CSV of
the per-gap statistics (gap_size, percentile, z-score, Shapiro-Wilk
log10 p-value) for direct comparison with the HDBSCAN probe's CSV
output. Skips the LLM proposal-generation pipeline that
``batch_generator.py`` would do next.

Usage::

    conda run -p .conda python -m artifacts_scripts.kball_gap_probe \\
        --scope 17/1702/T10181 \\
        --algorithm kball \\
        --radius-fraction 0.75 \\
        --num-gaps 5

Builds a Cloud at the project's standard defaults (SimHash dedup, GIST
embedder, NONE denoiser, etc.) — same defaults as the HDBSCAN probe
— then iterates topics in scope and runs gap detection per topic via
``bookgap.gaps._compute_gaps_for_topic``, which is joblib-cached so
repeated runs over the same (cloud, topic, radius, num_gaps) hit cache.

CSV output lands at::

    artifacts/kball_gap_probe/{scope_safe}__{algorithm}__rf{radius_fraction}__num{num_gaps}.csv

with per-gap rows (one row per detected gap). HDBSCAN-style
binomial-p / FDR / effect-size scoring is *not* applied here: K-ball
gaps are by construction book-free spheres (the radius stops at the
nearest book), so the depletion test would always score them as
extreme. The percentile/z-score/Shapiro framework is K-ball's native
gap-significance signal; cross-algorithm comparison happens at the
"how many qualifying gaps does each find?" level rather than via a
shared per-gap p-value.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from bookgap.cloud import Cloud
from bookgap.configs.cloud.cloud_config import CloudConfig
from bookgap.configs.util.filter_config import FilterConfig, FilterPairConfig
from bookgap.gaps import _compute_gaps_for_topic
from bookgap.openalex_scopes import (
    get_field_topics, get_subfield_topics,
)
from bookgap.snapshot_ingest import load_taxonomy
from bookgap.types import CellType
from bookgap.util import WorkType
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
    csv_rows: List[dict] = []
    for i, topic_id in enumerate(topics, 1):
        topic_rec = pinned_tax.topics.get(topic_id, {})
        topic_name = topic_rec.get("display_name", topic_id)
        print(f"\n[{i}/{len(topics)}] Topic {topic_id} — {topic_name}")
        gaps = _compute_gaps_for_topic(
            cloud=cloud,
            topic_id=topic_id,
            topic_name=topic_name,
            radius_fraction=args.radius_fraction,
            cell_type=cell_type,
            num_gaps=args.num_gaps,
        )
        all_gaps.extend(gaps)
        if not gaps:
            print(f"  (no gaps found)")
            continue
        for g in gaps:
            print(f"  Gap {g.gap_idx}: size={g.gap_size}, "
                  f"pctile={g.gap_size_percentile:.1f}, "
                  f"z={g.gap_zscore:.2f}, "
                  f"shapiro_log10p={g.shapiro_log10_pval:.2f}")
            idxs = list(g.gap.get_dists_idxs().idxs[:args.top_titles])
            article_titles = g.articles.work_titles()
            for ii in idxs:
                t = article_titles[int(ii)]
                print(f"    - {(t or '(no title)')[:90]}")
            csv_rows.append({
                "scope": args.scope,
                "algorithm": args.algorithm,
                "radius_fraction": args.radius_fraction,
                "num_gaps_per_topic": args.num_gaps,
                "topic_id": topic_id,
                "topic_name": topic_name,
                "gap_idx": int(g.gap_idx),
                "gap_size": int(g.gap_size),
                "gap_size_percentile": float(g.gap_size_percentile),
                "gap_zscore": float(g.gap_zscore),
                "shapiro_log10_pval": float(g.shapiro_log10_pval),
            })

    print(f"\n=== Total gaps: {len(all_gaps)} across {len(topics)} topic(s) ===")

    # CSV output for cross-run aggregation.
    csv_dir = Path("artifacts/kball_gap_probe")
    csv_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = args.scope.replace("/", "-")
    csv_path = (
        csv_dir
        / f"{safe_scope}__{args.algorithm}__rf{args.radius_fraction}__num{args.num_gaps}.csv"
    )
    fieldnames = [
        "scope", "algorithm", "radius_fraction", "num_gaps_per_topic",
        "topic_id", "topic_name",
        "gap_idx", "gap_size",
        "gap_size_percentile", "gap_zscore", "shapiro_log10_pval",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in csv_rows:
            w.writerow(row)
    print(f"CSV: {csv_path} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
