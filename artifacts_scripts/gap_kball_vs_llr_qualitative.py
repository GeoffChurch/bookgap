"""Qualitative side-by-side: KBALL ranking vs LLR-refined ranking.

For each topic, print:
  • KBALL top-N gaps with their LLR fate (survivor? merged? pruned?).
  • LLR top-N gaps by median_logit (survivors only) with their KBALL rank.

For each gap, show a sample of article titles so we can eyeball what the
gap is "about" and whether refinement looks like it's tightening to a
real topic or wandering.

Usage::

    uv run python -u -m artifacts_scripts.gap_kball_vs_llr_qualitative \\
        --scope 17/1702/T11303 \\
        --kball-csv artifacts/kball_gap_probe/17-1702-T11303__pca__drop0.csv \\
        --llr-csv artifacts/gap_kball_llr_refine/T11303_n1000__llr_refined.csv \\
        --top-n 10 --num-samples 6
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from artifacts_scripts.hdbscan_gap_probe import load_scope


def load_kball_csv(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            gid = int(r["gap_idx"])
            rows[gid] = {
                "gap_idx": gid,
                "gap_size": int(r["gap_size"]),
                "article_indices": json.loads(r["article_indices_json"]),
            }
    return rows


def load_llr_csv(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            gid = int(r["kball_gap_id"])
            rows[gid] = {
                "kball_gap_id": gid,
                "kball_size": int(r["kball_size"]),
                "refined_k_star": int(r["refined_k_star"]),
                "refined_n_articles": int(r["refined_n_articles"]),
                "median_logit": float(r["median_logit"]),
                "mean_logit": float(r["mean_logit"]),
                "min_logit": float(r["min_logit"]),
                "effective_size": float(r["effective_size"]),
                "kball_size_delta": int(r["kball_size_delta"]),
                "kball_jaccard": float(r["kball_jaccard"]),
                "survivor": int(r["survivor"]),
                "merged_into_gap_id": r["merged_into_gap_id"],
                "dedup_reason": r["dedup_reason"],
                "article_indices": json.loads(r["article_indices_json"]),
            }
    return rows


def fmt_titles(titles: list[str], indices: list[int], n: int, rng: random.Random) -> list[str]:
    if not indices:
        return []
    if len(indices) <= n:
        picks = indices
    else:
        picks = rng.sample(indices, n)
    out = []
    for idx in picks:
        if 0 <= idx < len(titles):
            t = titles[idx] or "<no title>"
            out.append(t.strip()[:160])
        else:
            out.append(f"<idx {idx} out of range>")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scope", required=True, help="e.g. 17/1702/T11303")
    p.add_argument("--kball-csv", required=True, type=Path)
    p.add_argument("--llr-csv", required=True, type=Path)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)

    print(f"[load] scope {args.scope}...", flush=True)
    _emb, _ids, _types, titles, _topic_ids, _refs = load_scope(args.scope, with_refs=False)
    print(f"  {len(titles)} works total", flush=True)

    kball = load_kball_csv(args.kball_csv)
    llr = load_llr_csv(args.llr_csv)
    print(f"  KBALL gaps: {len(kball)}, LLR rows: {len(llr)}", flush=True)

    survivors = [g for g in llr.values() if g["survivor"] == 1]
    print(f"  LLR survivors: {len(survivors)} ({100*len(survivors)/max(1,len(llr)):.1f}%)")

    # ── Block 1: KBALL top-N, with LLR fate ────────────────────────────
    kball_top = sorted(kball.values(), key=lambda r: r["gap_idx"])[: args.top_n]
    print("\n" + "=" * 88)
    print(f"KBALL TOP {args.top_n}   (ranked by KBALL gap_idx, i.e. native gap_size order)")
    print("=" * 88)
    # Build LLR-rank index by median_logit on survivors
    llr_rank: dict[int, int] = {}
    for rank, g in enumerate(
        sorted(survivors, key=lambda g: -g["median_logit"]), start=1
    ):
        llr_rank[g["kball_gap_id"]] = rank

    for kg in kball_top:
        gid = kg["gap_idx"]
        lr = llr.get(gid)
        print(f"\n— KBALL rank {gid}   (gap_size={kg['gap_size']})")
        if lr is None:
            print("    [no LLR row]")
        else:
            fate = "SURVIVOR" if lr["survivor"] == 1 else f"DROPPED ({lr['dedup_reason']})"
            if lr["survivor"] != 1 and lr["merged_into_gap_id"]:
                fate += f" → into kball_gap_id={lr['merged_into_gap_id']}"
            llr_r = llr_rank.get(gid)
            print(
                f"    LLR: k*={lr['refined_k_star']} (Δ={lr['kball_size_delta']:+d}, "
                f"jaccard={lr['kball_jaccard']:.2f})  "
                f"med_logit={lr['median_logit']:.2f}  "
                f"min_logit={lr['min_logit']:.2f}  "
                f"eff_size={lr['effective_size']:.1f}  "
                f"{fate}"
                f"{f'  [LLR rank #{llr_r}]' if llr_r else ''}"
            )
            print(f"    KBALL members sample ({args.num_samples} of {len(kg['article_indices'])}):")
            for t in fmt_titles(titles, kg["article_indices"], args.num_samples, rng):
                print(f"       · {t}")
            if lr["survivor"] == 1 and lr["article_indices"] != kg["article_indices"]:
                print(f"    LLR members sample ({args.num_samples} of {len(lr['article_indices'])}):")
                for t in fmt_titles(titles, lr["article_indices"], args.num_samples, rng):
                    print(f"       · {t}")

    # ── Block 2: LLR top-N by median_logit (survivors), with KBALL origin
    llr_top = sorted(survivors, key=lambda g: -g["median_logit"])[: args.top_n]
    print("\n" + "=" * 88)
    print(f"LLR TOP {args.top_n}   (survivors only, ranked by median_logit DESC)")
    print("=" * 88)
    for rank, lr in enumerate(llr_top, start=1):
        gid = lr["kball_gap_id"]
        kg = kball.get(gid)
        print(
            f"\n— LLR rank #{rank}   kball_gap_id={gid}  "
            f"(KBALL native rank #{gid}, size {kg['gap_size'] if kg else '?'})"
        )
        print(
            f"    LLR: k*={lr['refined_k_star']} (Δ={lr['kball_size_delta']:+d}, "
            f"jaccard={lr['kball_jaccard']:.2f})  "
            f"med_logit={lr['median_logit']:.2f}  "
            f"min_logit={lr['min_logit']:.2f}  "
            f"eff_size={lr['effective_size']:.1f}"
        )
        print(f"    LLR members sample ({args.num_samples} of {len(lr['article_indices'])}):")
        for t in fmt_titles(titles, lr["article_indices"], args.num_samples, rng):
            print(f"       · {t}")

    # ── Summary stats ─────────────────────────────────────────────────
    kball_top_ids = {kg["gap_idx"] for kg in kball_top}
    llr_top_ids = {g["kball_gap_id"] for g in llr_top}
    overlap = kball_top_ids & llr_top_ids
    kball_top_dropped = [
        llr[g] for g in kball_top_ids if g in llr and llr[g]["survivor"] != 1
    ]
    kball_top_surviving = [
        llr[g] for g in kball_top_ids if g in llr and llr[g]["survivor"] == 1
    ]
    print("\n" + "=" * 88)
    print(f"SUMMARY (top-{args.top_n})")
    print("=" * 88)
    print(f"  KBALL top-{args.top_n} ∩ LLR top-{args.top_n}: {len(overlap)}/{args.top_n}")
    print(
        f"  KBALL top-{args.top_n} survivors: {len(kball_top_surviving)} / {args.top_n}; "
        f"dropped: {len(kball_top_dropped)}"
    )
    if kball_top_surviving:
        sizes = sorted([g["refined_k_star"] for g in kball_top_surviving])
        orig = sorted([g["kball_size"] for g in kball_top_surviving])
        print(
            f"  KBALL-top survivors: kball_size range {orig[0]}–{orig[-1]}, "
            f"refined k* range {sizes[0]}–{sizes[-1]}"
        )
    if llr_top:
        kb_ranks = sorted([g["kball_gap_id"] for g in llr_top])
        print(f"  LLR top-{args.top_n} originate from KBALL ranks: {kb_ranks}")


if __name__ == "__main__":
    main()
