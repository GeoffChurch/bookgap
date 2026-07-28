"""Build top-gap leaderboards from per-topic refined CSVs.

Two outputs:

1. Per-topic top-1: one row per topic, the highest-median-logit survivor.
   Columns: topic_id, topic_name, kball_gap_id, refined_n_articles,
   median_logit, centroid_title, keywords (top-5).

2. Cross-topic top-50: top-50 survivors across all topics, sorted by
   median_logit. Same columns as (1) plus a cross-topic affinity report:
   top1_topic, top1_fraction, top2_topic, top2_fraction. Affinity is
   measured as kernel-weighted mass from the gap centroid to each of
   the 76 topics' mean embeddings; the home topic typically tops this
   ranking, the 2nd-place topic surfaces the most-related neighboring
   topic.

Survival filter: only gaps with ``refined_n_articles >= --min-size`` are
considered (default 30, "textbook-relevant" floor).

Usage::

    uv run python -u -m artifacts_scripts.per_topic_leaderboard \\
        --subfield 17/1702 \\
        --refined-dir artifacts/gap_kball_llr_refine/per_topic \\
        --out-per-topic artifacts/leaderboards/17-1702_per_topic_top1.csv \\
        --out-cross-topic artifacts/leaderboards/17-1702_cross_topic_top50.csv \\
        --top-n-cross 50
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from artifacts_scripts.keyword_helpers import compute_gap_keywords
from artifacts_scripts.hdbscan_gap_probe import load_scope


def _load_topic_metadata(subfield: str, topic_id: str) -> dict:
    """Load a topic's full work corpus (embeddings + titles + abstracts +
    types + ids) in the SAME order as gap_static_seed_diagnostic.setup()
    so refined-CSV article_indices align with the returned arrays.

    Returns dict: emb (N×d), titles (N,), abstracts (N,), types (N,),
    ids (N,).
    """
    scope_key = f"{subfield}/{topic_id}"
    emb, ids, types, titles, _, _ = load_scope(scope_key, with_refs=False)
    # Load abstracts in the same order as load_scope walks the parquets:
    # for each work-type (sorted), read its parquet's abstract column.
    from bookgap.util import CACHE_DIR
    parts = scope_key.split("/")
    base = CACHE_DIR / "works_parquet"
    base = base.joinpath(*parts)
    abstracts: list[str] = []
    # Match load_scope's worktype iteration order
    seen_types: list[str] = []
    for wt_parquet in sorted(base.glob("*.parquet")):
        wt = wt_parquet.stem
        meta = pq.ParquetFile(wt_parquet).metadata
        if meta.num_rows == 0:
            continue
        seen_types.append(wt)
        table = pq.read_table(wt_parquet, columns=["abstract"])
        abstracts.extend(table.column("abstract").to_pylist())
    if len(abstracts) != len(titles):
        # load_scope may have dropped rows with missing embeddings — fall back
        # to truncating abstracts to match titles length (best-effort alignment).
        abstracts = abstracts[: len(titles)] + [""] * max(
            0, len(titles) - len(abstracts),
        )
    return {
        "emb": emb,
        "titles": titles,
        "abstracts": abstracts,
        "types": types,
        "ids": ids,
    }


def _gap_centroid(emb: np.ndarray, member_indices: list[int]) -> np.ndarray:
    """Normalized centroid of gap members in the topic-scope embedding space."""
    members = np.asarray(member_indices, dtype=np.int64)
    centroid = emb[members].mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm < 1e-12:
        return centroid
    return centroid / norm


def _centroid_title(
    emb: np.ndarray, titles: list[str], types: list[str],
    member_indices: list[int],
) -> str:
    """Find the ARTICLE (work-type) nearest to the gap centroid; return its title."""
    centroid = _gap_centroid(emb, member_indices)
    # Restrict to articles (typically what we want as a title proxy)
    article_mask = np.array([t == "article" for t in types])
    article_idxs = np.flatnonzero(article_mask)
    if len(article_idxs) == 0:
        article_idxs = np.arange(len(emb))
    d_sq = np.sum((emb[article_idxs] - centroid) ** 2, axis=1)
    nearest_local = int(article_idxs[np.argmin(d_sq)])
    return titles[nearest_local]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subfield", required=True, help="e.g. 17/1702")
    p.add_argument("--refined-dir", required=True, type=Path,
                   help="Directory of <topic>__llr_refined.csv files.")
    p.add_argument("--out-per-topic", required=True, type=Path)
    p.add_argument("--out-cross-topic", required=True, type=Path)
    p.add_argument("--top-n-cross", type=int, default=50)
    p.add_argument("--min-size", type=int, default=30,
                   help="Minimum refined_n_articles to consider (default 30 = "
                        "textbook-relevant floor).")
    p.add_argument("--cache-dir", type=Path, default=Path("cache"))
    p.add_argument("--top-keywords", type=int, default=5)
    p.add_argument("--per-topic-kw-context", type=int, default=20,
                   help="For the per-topic top-1 keyword extraction, use this "
                        "many of the topic's TOP survivors (by median_logit) "
                        "as contrastive context for log-odds. Default 20. "
                        "Smaller = faster but less stable log-odds; larger = "
                        "more accurate scores but slower.")
    args = p.parse_args()

    args.out_per_topic.parent.mkdir(parents=True, exist_ok=True)
    args.out_cross_topic.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: collect per-topic top-1 + cross-topic candidates
    refined_files = sorted(
        glob.glob(str(args.refined_dir / "T*__llr_refined.csv")),
    )
    print(f"[load] {len(refined_files)} refined CSVs in {args.refined_dir}",
          flush=True)
    per_topic_pick: dict[str, dict] = {}  # topic_id → top-1 row
    per_topic_survivors: dict[str, list[dict]] = {}  # all topic survivors (for KW context)
    cross_topic_pool: list[dict] = []     # all survivors with n >= min_size
    topic_name_lookup: dict[str, str] = {}
    for path in refined_files:
        topic = Path(path).stem.split("__")[0]
        rows = []
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["survivor"] != "1":
                    continue
                n_arts = int(r["refined_n_articles"])
                if n_arts < args.min_size:
                    continue
                rows.append({
                    "topic_id": topic,
                    "kball_gap_id": int(r["kball_gap_id"]),
                    "refined_n_articles": n_arts,
                    "refined_k_star": int(r["refined_k_star"]),
                    "median_logit": float(r["median_logit"]),
                    "min_logit": float(r["min_logit"]),
                    "mean_logit": float(r["mean_logit"]),
                    "members": json.loads(r["article_indices_json"]),
                })
        if not rows:
            print(f"  {topic}: 0 survivors at n>={args.min_size}", flush=True)
            continue
        rows.sort(key=lambda x: -x["median_logit"])
        per_topic_pick[topic] = rows[0]
        per_topic_survivors[topic] = rows  # full context for keyword scoring
        cross_topic_pool.extend(rows)
        # Topic name (look up from a kball CSV; cache)
        if topic not in topic_name_lookup:
            kball_csv = Path(
                f"artifacts/kball_gap_probe/{args.subfield.replace('/', '-')}-{topic}__pca__drop0.csv"
            )
            if kball_csv.exists():
                with open(kball_csv) as f:
                    first = next(csv.DictReader(f), None)
                    topic_name_lookup[topic] = first["topic_name"] if first else topic
            else:
                topic_name_lookup[topic] = topic
    print(f"[load] {len(per_topic_pick)} topics with ≥1 qualifying survivor; "
          f"{len(cross_topic_pool)} total qualifying gaps", flush=True)

    # Step 2: cross-topic top-N
    cross_topic_pool.sort(key=lambda x: -x["median_logit"])
    cross_top = cross_topic_pool[: args.top_n_cross]
    print(f"[load] cross-topic top-{args.top_n_cross}: median_logit range "
          f"[{cross_top[-1]['median_logit']:.2f}, {cross_top[0]['median_logit']:.2f}]",
          flush=True)

    # Step 3: compute per-topic centroids for cross-topic affinity report.
    # We need centroids for ALL 76 topics (so any gap can be compared against
    # all of them). Topics that ALSO appear in the per-topic pick or cross-topic
    # top-N get their metadata cached for keyword extraction; other topics
    # discard metadata after computing the centroid.
    print(f"[centroid] loading per-topic metadata (76 topics)...", flush=True)
    topic_meta_cache: dict[str, dict] = {}
    all_topics = sorted({Path(p).stem.split("__")[0] for p in refined_files})
    t0 = time.time()
    # Subfield-level concatenation: one big embedding array indexed by
    # (topic_id, in-topic-position). We use this for the gap→neighborhood
    # affinity report.
    subfield_emb_parts: list[np.ndarray] = []
    subfield_topic_ids: list[str] = []
    subfield_is_article: list[bool] = []
    for i, topic in enumerate(all_topics, 1):
        meta = _load_topic_metadata(args.subfield, topic)
        topic_meta_cache[topic] = meta
        subfield_emb_parts.append(meta["emb"])
        subfield_topic_ids.extend([topic] * len(meta["emb"]))
        subfield_is_article.extend(t == "article" for t in meta["types"])
        if i % 10 == 0:
            print(f"  loaded {i}/{len(all_topics)} topics "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
    subfield_emb = np.concatenate(subfield_emb_parts, axis=0).astype(np.float32)
    subfield_topic_ids_arr = np.array(subfield_topic_ids)
    subfield_article_mask = np.array(subfield_is_article)
    print(f"  loaded {len(all_topics)} topics in {time.time() - t0:.0f}s; "
          f"subfield_emb shape={subfield_emb.shape}, "
          f"{subfield_article_mask.sum()} articles",
          flush=True)

    def _get_meta(topic: str) -> dict:
        if topic not in topic_meta_cache:
            print(f"  loading topic metadata for {topic}...", flush=True)
            topic_meta_cache[topic] = _load_topic_metadata(args.subfield, topic)
        return topic_meta_cache[topic]

    # Step 5: write per-topic top-1 CSV. For keyword extraction we need
    # contrastive context — pass ALL of the topic's survivors (not just the
    # top-1) so log-odds has a real pooled-marginal background, then pick
    # out the top-1's keywords.
    print(f"[write] per-topic top-1 → {args.out_per_topic}", flush=True)
    per_topic_rows: list[dict] = []
    for topic in sorted(per_topic_pick):
        meta = _get_meta(topic)
        # Use the top-N survivors as contrastive context — keyword
        # log-odds is a comparison against pooled-marginal background, so
        # we need >1 gap. 20 is enough for stable scores without paying the
        # full all-survivors cost.
        context_gaps = per_topic_survivors[topic][: args.per_topic_kw_context]
        gaps_for_kw = [g["members"] for g in context_gaps]
        kw_results = compute_gap_keywords(
            gaps=gaps_for_kw,
            titles=meta["titles"], abstracts=meta["abstracts"],
            top_k=args.top_keywords, verbose=False,
        )
        top_gap = per_topic_pick[topic]
        # The top_gap is context_gaps[0] (sorted by -median_logit), so
        # kw_results[0] is its keyword set.
        kws = kw_results[0]
        ct_title = _centroid_title(
            meta["emb"], meta["titles"], meta["types"], top_gap["members"],
        )
        per_topic_rows.append({
            "topic_id": topic,
            "topic_name": topic_name_lookup.get(topic, topic),
            "kball_gap_id": top_gap["kball_gap_id"],
            "refined_n_articles": top_gap["refined_n_articles"],
            "refined_k_star": top_gap["refined_k_star"],
            "median_logit": f"{top_gap['median_logit']:+.3f}",
            "min_logit": f"{top_gap['min_logit']:+.3f}",
            "centroid_title": ct_title,
            "keywords": " | ".join(f"{k} ({s:.2f})" for k, s in kws),
        })

    fieldnames_pt = [
        "topic_id", "topic_name", "kball_gap_id",
        "refined_n_articles", "refined_k_star",
        "median_logit", "min_logit",
        "centroid_title", "keywords",
    ]
    per_topic_rows.sort(key=lambda r: r["topic_id"])
    with open(args.out_per_topic, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_pt)
        w.writeheader()
        for r in per_topic_rows:
            w.writerow(r)
    print(f"  {len(per_topic_rows)} rows written", flush=True)

    # Step 6: write cross-topic top-N CSV with affinity columns
    print(f"[write] cross-topic top-{args.top_n_cross} → {args.out_cross_topic}",
          flush=True)
    # Group cross-top gaps by topic for batched keyword extraction
    cross_by_topic: dict[str, list[dict]] = defaultdict(list)
    for gap in cross_top:
        cross_by_topic[gap["topic_id"]].append(gap)
    cross_kw_results: dict[tuple[str, int], list[tuple[str, float]]] = {}
    cross_centroid_titles: dict[tuple[str, int], str] = {}
    cross_affinity: dict[tuple[str, int], list[tuple[str, float]]] = {}
    for topic, gaps_in_topic in cross_by_topic.items():
        meta = _get_meta(topic)
        gaps_for_kw = [g["members"] for g in gaps_in_topic]
        kw_results = compute_gap_keywords(
            gaps=gaps_for_kw,
            titles=meta["titles"], abstracts=meta["abstracts"],
            top_k=args.top_keywords, verbose=False,
        )
        for gap, kws in zip(gaps_in_topic, kw_results):
            key = (topic, gap["kball_gap_id"])
            cross_kw_results[key] = kws
            cross_centroid_titles[key] = _centroid_title(
                meta["emb"], meta["titles"], meta["types"], gap["members"],
            )
            # Cross-topic affinity: kernel weight from gap centroid to each
            # topic centroid; normalize across topics.
            c_gap = _gap_centroid(meta["emb"], gap["members"]).astype(np.float32)
            # Cross-topic affinity = top-K subfield-NN of the gap centroid,
            # mass-weighted by primary_topic_id of each neighbor article.
            # Each neighbor contributes weight ∝ exp(τ·cosine), so close
            # neighbors dominate. K=500 captures the gap's effective
            # neighborhood without leaking into far topics.
            article_idxs = np.flatnonzero(subfield_article_mask)
            cos_to_subfield = subfield_emb[article_idxs] @ c_gap
            K_NEIGH = 500
            top_idx_into_articles = np.argpartition(
                -cos_to_subfield, K_NEIGH,
            )[:K_NEIGH]
            top_global_idx = article_idxs[top_idx_into_articles]
            top_cos = cos_to_subfield[top_idx_into_articles]
            tau = 20.0
            top_logits = tau * top_cos
            top_logits -= top_logits.max()
            top_w = np.exp(top_logits)
            top_topics = subfield_topic_ids_arr[top_global_idx]
            # Aggregate
            mass_by_topic: dict[str, float] = {}
            for tid, w in zip(top_topics, top_w):
                mass_by_topic[tid] = mass_by_topic.get(tid, 0.0) + float(w)
            total_mass = sum(mass_by_topic.values()) or 1.0
            sorted_topics = sorted(
                mass_by_topic.items(), key=lambda x: -x[1],
            )
            top2 = [
                (sorted_topics[i][0], sorted_topics[i][1] / total_mass)
                for i in range(min(2, len(sorted_topics)))
            ]
            while len(top2) < 2:
                top2.append(("", 0.0))
            cross_affinity[key] = top2

    fieldnames_xt = [
        "rank", "topic_id", "topic_name", "kball_gap_id",
        "refined_n_articles", "refined_k_star",
        "median_logit", "min_logit",
        "centroid_title", "keywords",
        "top1_topic", "top1_fraction",
        "top2_topic", "top2_fraction",
    ]
    with open(args.out_cross_topic, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_xt)
        w.writeheader()
        for rank, gap in enumerate(cross_top, 1):
            key = (gap["topic_id"], gap["kball_gap_id"])
            kws = cross_kw_results[key]
            ct_title = cross_centroid_titles[key]
            top1_id, top1_frac = cross_affinity[key][0]
            top2_id, top2_frac = cross_affinity[key][1]
            w.writerow({
                "rank": rank,
                "topic_id": gap["topic_id"],
                "topic_name": topic_name_lookup.get(gap["topic_id"], gap["topic_id"]),
                "kball_gap_id": gap["kball_gap_id"],
                "refined_n_articles": gap["refined_n_articles"],
                "refined_k_star": gap["refined_k_star"],
                "median_logit": f"{gap['median_logit']:+.3f}",
                "min_logit": f"{gap['min_logit']:+.3f}",
                "centroid_title": ct_title,
                "keywords": " | ".join(
                    f"{k} ({s:.2f})" for k, s in kws
                ),
                "top1_topic": topic_name_lookup.get(top1_id, top1_id),
                "top1_fraction": f"{top1_frac:.3f}",
                "top2_topic": topic_name_lookup.get(top2_id, top2_id),
                "top2_fraction": f"{top2_frac:.3f}",
            })
    print(f"  {len(cross_top)} rows written", flush=True)
    print(f"[done]", flush=True)


if __name__ == "__main__":
    main()
