"""Compare alternative rank metrics on existing LLR-refined CSVs.

Re-ranks survivors under several candidate metrics and prints top-N with
sample titles, so we can eyeball which metric pushes the templated-leak
and off-topic clusters down the leaderboard.

Variants:
  M1  median_logit
  M2  median_logit, restricted to eff_size >= 30
  M3  median_logit, restricted to eff_size >= 75
  M4  min(mean_logit, median_logit)            -- penalises bimodal clusters
  M5  min(mean_logit, median_logit), eff_size >= 30
  M6  median_logit * log(eff_size)             -- bonus, multiplicative size weight

Usage::

    uv run python -u -m artifacts_scripts.gap_llr_rank_variants \\
        --scope 17/1702/T11303 \\
        --llr-csv artifacts/gap_kball_llr_refine/T11303_n1000__llr_refined.csv \\
        --top-n 10 --num-samples 4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

from artifacts_scripts.hdbscan_gap_probe import load_scope


def load_llr_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "kball_gap_id": int(r["kball_gap_id"]),
                "kball_size": int(r["kball_size"]),
                "refined_k_star": int(r["refined_k_star"]),
                "median_logit": float(r["median_logit"]),
                "mean_logit": float(r["mean_logit"]),
                "min_logit": float(r["min_logit"]),
                "effective_size": float(r["effective_size"]),
                "kball_jaccard": float(r["kball_jaccard"]),
                "survivor": int(r["survivor"]),
                "article_indices": json.loads(r["article_indices_json"]),
            })
    return rows


def fmt_titles(titles: list[str], indices: list[int], n: int, rng: random.Random) -> list[str]:
    if not indices:
        return []
    picks = indices if len(indices) <= n else rng.sample(indices, n)
    out = []
    for idx in picks:
        if 0 <= idx < len(titles):
            t = (titles[idx] or "<no title>").strip()[:150]
            out.append(t)
        else:
            out.append(f"<idx {idx} OOR>")
    return out


def variant_M1(g: dict) -> float | None:
    return g["median_logit"]


def variant_M2(g: dict) -> float | None:
    return g["median_logit"] if g["effective_size"] >= 30 else None


def variant_M3(g: dict) -> float | None:
    return g["median_logit"] if g["effective_size"] >= 75 else None


def variant_M4(g: dict) -> float | None:
    return min(g["mean_logit"], g["median_logit"])


def variant_M5(g: dict) -> float | None:
    return min(g["mean_logit"], g["median_logit"]) if g["effective_size"] >= 30 else None


def variant_M6(g: dict) -> float | None:
    if g["effective_size"] <= 1:
        return None
    return g["median_logit"] * math.log(g["effective_size"])


def variant_M7(g: dict) -> float | None:
    return g["median_logit"] * g["effective_size"]


def variant_M8(g: dict) -> float | None:
    if g["effective_size"] <= 0:
        return None
    return g["median_logit"] * math.sqrt(g["effective_size"])


def variant_M9(g: dict) -> float | None:
    return g["effective_size"]


VARIANTS = [
    ("M1", "median_logit",                          variant_M1),
    ("M2", "median_logit @ eff_size>=30",           variant_M2),
    ("M3", "median_logit @ eff_size>=75",           variant_M3),
    ("M4", "min(mean_logit, median_logit)",         variant_M4),
    ("M5", "min(mean_logit, median_logit) @ eff>=30", variant_M5),
    ("M6", "median_logit * log(eff_size)",          variant_M6),
    ("M7", "median_logit * eff_size",               variant_M7),
    ("M8", "median_logit * sqrt(eff_size)",         variant_M8),
    ("M9", "effective_size alone",                  variant_M9),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scope", required=True)
    p.add_argument("--llr-csv", required=True, type=Path)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)

    print(f"[load] scope {args.scope}...", flush=True)
    _emb, _ids, _types, titles, _topic_ids, _refs = load_scope(args.scope, with_refs=False)
    print(f"  {len(titles)} works total", flush=True)

    rows = load_llr_csv(args.llr_csv)
    survivors = [g for g in rows if g["survivor"] == 1]
    print(f"  survivors: {len(survivors)}", flush=True)

    # Bookkeeping: which kball_gap_ids appear in M1 top-N? (baseline reference)
    baseline_ids: set[int] = set()
    for code, label, fn in VARIANTS:
        scored = [(fn(g), g) for g in survivors]
        scored = [(s, g) for s, g in scored if s is not None]
        scored.sort(key=lambda x: -x[0])
        topk = scored[: args.top_n]
        if code == "M1":
            baseline_ids = {g["kball_gap_id"] for _, g in topk}
        n_eligible = sum(1 for s, _ in scored)
        print("\n" + "=" * 88)
        print(f"[{code}] {label}   (eligible: {n_eligible}/{len(survivors)} survivors)")
        print("=" * 88)
        for rank, (score, g) in enumerate(topk, start=1):
            gid = g["kball_gap_id"]
            flag = "" if code == "M1" or gid in baseline_ids else "  NEW"
            print(
                f"\n#{rank}  gap_id={gid}  k*={g['refined_k_star']}  "
                f"eff={g['effective_size']:.1f}  "
                f"med={g['median_logit']:.2f}  mean={g['mean_logit']:.2f}  "
                f"min={g['min_logit']:.2f}  "
                f"score={score:.2f}{flag}"
            )
            for t in fmt_titles(titles, g["article_indices"], args.num_samples, rng):
                print(f"     · {t}")

    # ── Cross-variant overlap matrix ─────────────────────────────────
    tops: dict[str, set[int]] = {}
    for code, label, fn in VARIANTS:
        scored = [(fn(g), g) for g in survivors]
        scored = [(s, g) for s, g in scored if s is not None]
        scored.sort(key=lambda x: -x[0])
        tops[code] = {g["kball_gap_id"] for _, g in scored[: args.top_n]}

    print("\n" + "=" * 88)
    print(f"OVERLAP MATRIX (top-{args.top_n} intersection counts)")
    print("=" * 88)
    codes = [c for c, _, _ in VARIANTS]
    header = "      " + " ".join(f"{c:>4}" for c in codes)
    print(header)
    for c1 in codes:
        row = [f"{len(tops[c1] & tops[c2]):>4}" for c2 in codes]
        print(f"  {c1:<4}" + " ".join(row))


if __name__ == "__main__":
    main()
