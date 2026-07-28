"""Compare LLR-refined gap rankings under book-only vs chapter-only landmarks.

Given the same topic refined under two landmark configurations, compute
how much the top-10 rankings under M9 (effective_size alone) actually
differ. Useful for asking: does swapping books for chapters identify
genuinely different gaps, or mostly the same ones with relabeled IDs?

Note: gap_id semantics differ across the two CSVs because they come
from independent K-ball runs. So we can't compare by gap_id directly.
Instead we use article-index Jaccard between gap member sets.

Usage::

    uv run python -u -m artifacts_scripts.gap_chapter_vs_book_overlap \\
        --scope 17/1702/T11303 \\
        --book-llr-csv  artifacts/gap_kball_llr_refine/T11303_n1000__llr_refined.csv \\
        --chap-llr-csv  artifacts/gap_kball_llr_refine/chapter_only/T11303__llr_refined.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from artifacts_scripts.hdbscan_gap_probe import load_scope


def load_llr_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "kball_gap_id": int(r["kball_gap_id"]),
                "refined_k_star": int(r["refined_k_star"]),
                "median_logit": float(r["median_logit"]),
                "mean_logit": float(r["mean_logit"]),
                "effective_size": float(r["effective_size"]),
                "survivor": int(r["survivor"]),
                "members": frozenset(json.loads(r["article_indices_json"])),
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scope", required=True)
    p.add_argument("--book-llr-csv", required=True, type=Path)
    p.add_argument("--chap-llr-csv", required=True, type=Path)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)

    print(f"[load] scope {args.scope}...", flush=True)
    _emb, _ids, _types, titles, _topic_ids, _refs = load_scope(args.scope, with_refs=False)

    book_rows = load_llr_csv(args.book_llr_csv)
    chap_rows = load_llr_csv(args.chap_llr_csv)
    book_surv = [g for g in book_rows if g["survivor"] == 1]
    chap_surv = [g for g in chap_rows if g["survivor"] == 1]
    print(f"  book-only survivors: {len(book_surv)};  chapter-only: {len(chap_surv)}")

    book_top = sorted(book_surv, key=lambda g: -g["effective_size"])[: args.top_n]
    chap_top = sorted(chap_surv, key=lambda g: -g["effective_size"])[: args.top_n]

    def sample_titles(members, n):
        members = list(members)
        if len(members) <= n:
            picks = members
        else:
            picks = rng.sample(members, n)
        return [(titles[i] or "<no title>").strip()[:120] for i in picks if 0 <= i < len(titles)]

    print("\n" + "=" * 88)
    print(f"BOOK-ONLY M9 top-{args.top_n}  vs  CHAPTER-ONLY M9 top-{args.top_n}")
    print("(matched by max-Jaccard member overlap)")
    print("=" * 88)

    # For each book-top gap, find the best-matching chapter-top gap by member Jaccard
    for i, bg in enumerate(book_top, start=1):
        best_j, best_jacc = -1, 0.0
        for j, cg in enumerate(chap_top):
            inter = len(bg["members"] & cg["members"])
            union = len(bg["members"] | cg["members"])
            jacc = inter / union if union else 0.0
            if jacc > best_jacc:
                best_jacc, best_j = jacc, j
        print(
            f"\nBook #{i}  gid={bg['kball_gap_id']}  k*={bg['refined_k_star']}  "
            f"eff={bg['effective_size']:.1f}  med={bg['median_logit']:.2f}"
        )
        for t in sample_titles(bg["members"], args.num_samples):
            print(f"     · {t}")
        if best_jacc >= 0.1:
            cg = chap_top[best_j]
            print(
                f"   ⟷ Chapter #{best_j+1}  gid={cg['kball_gap_id']}  "
                f"k*={cg['refined_k_star']}  eff={cg['effective_size']:.1f}  "
                f"jaccard={best_jacc:.2f}"
            )
        else:
            print(f"   ⟶ NO MATCH in chapter top-{args.top_n} (best Jaccard {best_jacc:.2f})")

    # Now the chapter-top picks: are they covered by book-top?
    print("\n" + "-" * 88)
    print(f"Chapter-only picks NOT in book-only top-{args.top_n}:")
    print("-" * 88)
    book_top_members_unioned = set().union(*[bg["members"] for bg in book_top])
    for i, cg in enumerate(chap_top, start=1):
        best_jacc = 0.0
        for bg in book_top:
            inter = len(cg["members"] & bg["members"])
            union = len(cg["members"] | bg["members"])
            jacc = inter / union if union else 0.0
            best_jacc = max(best_jacc, jacc)
        if best_jacc < 0.3:
            print(
                f"\nChapter #{i}  gid={cg['kball_gap_id']}  k*={cg['refined_k_star']}  "
                f"eff={cg['effective_size']:.1f}  med={cg['median_logit']:.2f}  "
                f"max_jaccard_to_book_top={best_jacc:.2f}"
            )
            for t in sample_titles(cg["members"], args.num_samples):
                print(f"     · {t}")


if __name__ == "__main__":
    main()
