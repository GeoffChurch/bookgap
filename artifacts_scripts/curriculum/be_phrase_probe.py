"""Compare three bigram-mining recipes on a curriculum's sampled corpus:

    A) NPMI · count       (production, what we just shipped)
    B) Branching entropy alone     (boundary entropy of "_ a b _")
    C) NPMI · count · BE   (multiplicative combination)

Specifically diagnostic for whether branching entropy correctly demotes
the boilerplate aggregates that survive NPMI · count alone — the 25-gram
of citation-export UI text that surfaced in __24's Topic 2 was the headline
example.

Usage:
    python -m artifacts_scripts.curriculum.be_phrase_probe \\
        artifacts/curriculum/<run_dir>

Reads the same per-curriculum filtered_work_ids.json layout as
``bigram_keywords_probe.py``; reuses the per-text loading and tokenisation
code from that module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from artifacts_scripts.curriculum.bigram_keywords_probe import (
    _load_texts_for_filtered_ids,
    _mine_bigrams,
    _mine_bigrams_be_only,
    _mine_bigrams_npmi_be,
    _mine_marginal_be,
    _mine_normalized_be,
    _mine_quantile_be,
)
from artifacts_scripts.curriculum.keyword_extraction import tokenize


def _format_score(s: float, w: int = 8) -> str:
    return f"{s:>{w}.3f}"


def run(
    run_dir: Path,
    *,
    top_n: int = 30,
    npmi_threshold: float = 0.4,
    npmi_count_floor: int = 5,
    be_min: float = 1.5,
    be_npmi_min: float = 0.5,
    be_count_floor: int = 3,
    quantile: float = 0.5,
    marginal_alpha: float = 0.4,
    norm_be_min: float = 0.5,
    max_bigrams: int = 5000,
) -> None:
    print(f"=== Branching-entropy probe on {run_dir} ===")
    with open(run_dir / "filtered_work_ids.json") as f:
        filtered_ids: List[str] = json.load(f)
    print(f"  top-k: {len(filtered_ids)} works")

    parquet_root = Path("cache/works_parquet")
    print("  loading text for top-k work IDs...")
    id_to_text = _load_texts_for_filtered_ids(
        filtered_ids, parquet_root,
    )
    print(f"  found texts: {len(id_to_text)} / {len(filtered_ids)}")
    texts = [id_to_text.get(wid, "") for wid in filtered_ids]
    print("  tokenising...")
    raw = [tokenize(t) for t in texts]
    print(f"  total tokens: {sum(len(d) for d in raw):,}")

    print()
    print("=== A) NPMI · count (production recipe) ===")
    a = _mine_bigrams(
        raw, count_floor=npmi_count_floor, npmi_threshold=npmi_threshold,
        max_bigrams=max_bigrams,
    )
    print(f"  candidates: {len(a)}")
    print(f"  top {top_n} (sorted by NPMI · count):")
    for npmi, w1, w2, c in a[:top_n]:
        print(f"    {npmi:>7.3f}  count={c:>5}  {w1}_{w2}")

    print()
    print("=== B) Boundary entropy alone (option 1) ===")
    b = _mine_bigrams_be_only(
        raw, min_boundary_be=be_min, min_count=be_count_floor,
        max_bigrams=max_bigrams,
    )
    print(f"  candidates: {len(b)}")
    print(f"  top {top_n} (sorted by min(left_be, right_be)):")
    for s, w1, w2, c in b[:top_n]:
        print(f"    {s:>7.3f} bits  count={c:>5}  {w1}_{w2}")

    print()
    print(f"=== C) NPMI · count · BE  (BE > {be_npmi_min} bits, hand-set) ===")
    c_recipe = _mine_bigrams_npmi_be(
        raw, npmi_threshold=npmi_threshold, min_boundary_be=be_npmi_min,
        min_count=be_count_floor, max_bigrams=max_bigrams,
    )
    print(f"  candidates: {len(c_recipe)}")
    print(f"  top {top_n} (sorted by NPMI · count · BE):")
    for s, w1, w2, c in c_recipe[:top_n]:
        print(f"    {s:>10.3f}  count={c:>5}  {w1}_{w2}")

    print()
    print(f"=== D) NPMI · count · BE  (BE quantile threshold @ {quantile}) ===")
    d_recipe = _mine_quantile_be(
        raw, npmi_threshold=npmi_threshold, be_quantile=quantile,
        min_count=be_count_floor, max_bigrams=max_bigrams,
    )
    print(f"  candidates: {len(d_recipe)}")
    print(f"  top {top_n}:")
    for s, w1, w2, c in d_recipe[:top_n]:
        print(f"    {s:>10.3f}  count={c:>5}  {w1}_{w2}")

    print()
    print(f"=== E) NPMI · count · BE  (BE > {marginal_alpha} · H_W) ===")
    e_recipe = _mine_marginal_be(
        raw, npmi_threshold=npmi_threshold, alpha=marginal_alpha,
        min_count=be_count_floor, max_bigrams=max_bigrams,
    )
    print(f"  candidates: {len(e_recipe)}")
    print(f"  top {top_n}:")
    for s, w1, w2, c in e_recipe[:top_n]:
        print(f"    {s:>10.3f}  count={c:>5}  {w1}_{w2}")

    print()
    print(f"=== F) NPMI · count · normalized_BE  (norm_BE > {norm_be_min}) ===")
    f_recipe = _mine_normalized_be(
        raw, npmi_threshold=npmi_threshold, min_norm_be=norm_be_min,
        min_count=be_count_floor, max_bigrams=max_bigrams,
    )
    print(f"  candidates: {len(f_recipe)}")
    print(f"  top {top_n}:")
    for s, w1, w2, c in f_recipe[:top_n]:
        print(f"    {s:>10.3f}  count={c:>5}  {w1}_{w2}")

    # Diagnostic: which recipe recovers known wins, which rejects boilerplate?
    KNOWN_WINS = [
        ("bell", "theorem"),
        ("single", "photon"),
        ("photon", "counting"),
        ("photon", "pairs"),
        ("mach", "zehnder"),
        ("two", "photon"),
        ("quantum", "cryptography"),
        ("quantum", "mechanics"),
        ("key", "distribution"),
        ("einstein", "podolsky"),
        ("podolsky", "rosen"),
    ]
    KNOWN_BOILERPLATE = [
        ("articles", "citing"),
        ("reflect", "usage"),
        ("get", "alerts"),
        ("advertisement", "return"),
        ("article", "views"),
        ("alert", "save"),
        ("plain", "text"),
        ("citation", "alert"),
        ("save", "article"),
        ("post", "reddit"),
        ("reddit", "share"),
        ("manager", "easybib"),
        ("easybib", "bookends"),
        ("dropdown", "menu"),
    ]

    def in_top(scored, target, n=200):
        for i, (s, w1, w2, c) in enumerate(scored[:n]):
            if (w1, w2) == target:
                return i + 1
        return None

    recipes = [("A:NPMI·c", a), ("B:BE", b), ("C:NPMI·c·BE", c_recipe),
               ("D:quantile", d_recipe), ("E:α·H_W", e_recipe),
               ("F:norm_BE", f_recipe)]

    def show(label: str, target: Tuple[str, str]) -> str:
        ranks = []
        for _, scored in recipes:
            r = in_top(scored, target)
            ranks.append(f"#{r}" if r else "—")
        return f"  {label} {target[0]}_{target[1]:<28}  " + "  ".join(
            f"{rk:>10}" for rk in ranks
        )

    print("\n=== Diagnostic: known-win recovery and boilerplate rejection ===")
    print(f"  Looking at top-200 of each recipe (out of {max_bigrams} max).")
    header = "  " + " " * 35 + "  " + "  ".join(
        f"{name:>10}" for name, _ in recipes
    )
    print(header)
    for w1, w2 in KNOWN_WINS:
        print(show("WIN ", (w1, w2)))
    print()
    for w1, w2 in KNOWN_BOILERPLATE:
        print(show("BOIL", (w1, w2)))

    print("\n=== Summary tallies ===")
    print(f"  {'recipe':<14}  {'wins (of 11)':>14}  {'boilerplate (of 14)':>22}")
    for name, scored in recipes:
        n_wins = sum(1 for w in KNOWN_WINS if in_top(scored, w))
        n_boil = sum(1 for b_pair in KNOWN_BOILERPLATE if in_top(scored, b_pair))
        print(f"  {name:<14}  {n_wins:>14}  {n_boil:>22}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--be-min", type=float, default=1.5,
                        help="minimum min(left_be, right_be) bits for BE-only")
    parser.add_argument("--be-npmi-min", type=float, default=0.5,
                        help="minimum BE bits as a guard inside the NPMI·count·BE recipe")
    parser.add_argument("--quantile", type=float, default=0.5,
                        help="quantile threshold for recipe D")
    parser.add_argument("--marginal-alpha", type=float, default=0.4,
                        help="alpha · H_W threshold for recipe E")
    parser.add_argument("--norm-be-min", type=float, default=0.5,
                        help="normalized-BE threshold (0..1) for recipe F")
    parser.add_argument("--max-bigrams", type=int, default=5000)
    args = parser.parse_args()
    run(
        args.run_dir,
        top_n=args.top_n,
        be_min=args.be_min,
        be_npmi_min=args.be_npmi_min,
        quantile=args.quantile,
        marginal_alpha=args.marginal_alpha,
        norm_be_min=args.norm_be_min,
        max_bigrams=args.max_bigrams,
    )


if __name__ == "__main__":
    main()
