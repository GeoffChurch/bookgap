"""Effective gap size via centroid distance-ratio (likelihood-free).

Parametric alternative to the per-article KDE in gap_effective_size.py.
Fits a single isotropic Gaussian to each gap (mean = centroid, variance
= empirical RMS-to-centroid squared), then uses a likelihood-free
distance-ratio posterior to give each gap article a soft membership.

For each gap G:
    μ_G = mean(x_i for i ∈ G)
    σ_G² = (1/|G|) Σ_{i ∈ G} ||x_i − μ_G||²
    μ_C = mean over full corpus; σ_C² analogous
    For each i ∈ G:
        d²_G(i) = ||x_i − μ_G||²
        d²_C(i) = ||x_i − μ_C||²
        score(i) = d²_C(i) / (d²_G(i) + d²_C(i))    ∈ [0, 1]
    effective_size(G) = Σ_{i ∈ G} score(i)

Why distance-ratio, not Bayesian likelihood: in 384-dim, the isotropic
Gaussian log-density is dominated by `−(d/2) log σ²`, so any moderately-
tight gap (σ_G < σ_C) gets ~e^135 advantage from the normalizer alone
and the posterior trivially saturates at 1.0. Distance-ratio sidesteps
the curse of dimensionality.

Tight isolated gap (d²_G ≪ d²_C) → score → 1 → effective ≈ |G|.
Gap centroid coincident with corpus centroid (d²_G ≈ d²_C) → score ≈ 0.5
→ effective ≈ |G|/2. Pure noise (gap centroid is farther from i than
corpus centroid) → score < 0.5.

Usage::

    python -m artifacts_scripts.gap_effective_size_centroid \\
        --scope 17/1702/T10181 \\
        --kball-csv artifacts/kball_gap_probe/17-1702-T10181__pca__drop0.csv \\
        --maximal-bookless-csv artifacts/maximal_bookless/T10181_no_denoise.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from artifacts_scripts.hdbscan_gap_probe import load_scope


def _load_kball(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "gap_id": int(r["gap_idx"]),
                "size": int(r["gap_size"]),
                "topic_id": r["topic_id"],
                "article_indices": json.loads(r["article_indices_json"]),
            })
    return rows


def _load_mb(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "gap_id": int(r["gap_id"]),
                "size": int(r["n_articles"]),
                "n_boundary_books": int(r["n_boundary_books"]),
                "article_indices": json.loads(r["article_indices_json"]),
            })
    return rows


def _topic_id_from_path(path: Path) -> str:
    stem = path.stem
    parts = stem.split("__")[0].split("-")
    if len(parts) >= 3 and parts[-1].startswith("T"):
        return parts[-1]
    if stem.startswith("T") and "_" in stem:
        return stem.split("_")[0]
    return stem


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True)
    p.add_argument("--kball-csv", required=True)
    p.add_argument("--maximal-bookless-csv", required=True)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--output-dir", default="artifacts/gap_effective_size_centroid")
    args = p.parse_args()

    print(f"Loading scope {args.scope}...")
    emb, ids, types, titles, topic_ids, _refs = load_scope(args.scope, with_refs=False)
    n, d = emb.shape
    print(f"  {n:,} works, {d}-d")

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)

    # Corpus centroid + squared distance to it (precomputed once).
    mu_corpus = emb.mean(axis=0)
    sq_dist_corpus = ((emb - mu_corpus) ** 2).sum(axis=1)
    sigma_sq_corpus = float(sq_dist_corpus.mean())
    print(f"  corpus σ²: {sigma_sq_corpus:.4f}, σ: {np.sqrt(sigma_sq_corpus):.4f}")

    kball = _load_kball(Path(args.kball_csv))
    mb = _load_mb(Path(args.maximal_bookless_csv))
    topic_id = _topic_id_from_path(Path(args.maximal_bookless_csv))
    kball = [r for r in kball if r["topic_id"] == topic_id]
    print(f"  K-ball: {len(kball)} gaps;  max-bookless: {len(mb)} gaps;  topic: {topic_id}")

    def compute(gap: dict) -> tuple[float, float, float]:
        G = gap["article_indices"]
        size = len(G)
        if size <= 1:
            return float(size), 1.0, 0.0
        G_arr = np.asarray(G, dtype=np.int64)
        gap_emb = emb[G_arr]
        mu_g = gap_emb.mean(axis=0)
        d_sq_gap = ((gap_emb - mu_g) ** 2).sum(axis=1)
        d_sq_corp = sq_dist_corpus[G_arr]
        sigma_sq_g = float(d_sq_gap.mean())
        # score(i) = ‖x_i − μ_C‖² / (‖x_i − μ_G‖² + ‖x_i − μ_C‖²); range [0, 1].
        score = d_sq_corp / (d_sq_gap + d_sq_corp + 1e-12)
        eff = float(score.sum())
        return eff, float(score.mean()), float(np.sqrt(sigma_sq_g))

    print(f"\nScoring {len(mb)} mb gaps + {len(kball)} K-ball gaps...")
    for g in mb:
        eff, mp, sg = compute(g)
        g["effective_size"] = eff
        g["mean_score"] = mp
        g["sigma_g"] = sg
    for g in kball:
        eff, mp, sg = compute(g)
        g["effective_size"] = eff
        g["mean_score"] = mp
        g["sigma_g"] = sg

    mb_by_eff = sorted(mb, key=lambda r: -r["effective_size"])
    kball_by_eff = sorted(kball, key=lambda r: -r["effective_size"])

    def _rank(lst): return {g["gap_id"]: i for i, g in enumerate(lst)}
    mb_rank_eff = _rank(mb_by_eff)
    mb_rank_raw = _rank(sorted(mb, key=lambda r: -r["size"]))
    kball_rank_eff = _rank(kball_by_eff)
    kball_rank_raw = _rank(sorted(kball, key=lambda r: -r["size"]))

    def _print_top(name, sorted_list, rank_raw):
        print()
        print(f"## Top-{args.top_n} {name} by centroid distance-ratio effective size")
        print(f"  {'eff_rk':>6}  {'gap_id':>6}  {'raw_sz':>6}  {'eff_sz':>7}  "
              f"{'score':>6}  {'σ_G':>5}  {'raw_rk':>6}  Δrank")
        for i, g in enumerate(sorted_list[:args.top_n]):
            raw_rk_1 = rank_raw[g["gap_id"]] + 1
            delta = rank_raw[g["gap_id"]] - i
            delta_str = f"{'+' if delta > 0 else ''}{delta}"
            print(f"  {i+1:>6}  {g['gap_id']:>6}  {g['size']:>6}  "
                  f"{g['effective_size']:>7.1f}  {g['mean_score']:>6.3f}  "
                  f"{g['sigma_g']:>5.3f}  {raw_rk_1:>6}  {delta_str}")

    _print_top("K-ball", kball_by_eff, kball_rank_raw)
    _print_top("max-bookless", mb_by_eff, mb_rank_raw)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for items, name in [(mb, "mb"), (kball, "kball")]:
        rank_raw = mb_rank_raw if name == "mb" else kball_rank_raw
        rank_eff = mb_rank_eff if name == "mb" else kball_rank_eff
        path = out_dir / f"{topic_id}__{name}_centroid_eff.csv"
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["gap_id", "raw_size", "effective_size", "mean_score",
                         "sigma_g", "raw_rank", "eff_rank", "delta_rank"])
            for g in sorted(items, key=lambda r: -r["effective_size"]):
                wr.writerow([
                    g["gap_id"], g["size"],
                    f"{g['effective_size']:.3f}",
                    f"{g['mean_score']:.6f}",
                    f"{g['sigma_g']:.4f}",
                    rank_raw[g["gap_id"]] + 1,
                    rank_eff[g["gap_id"]] + 1,
                    rank_raw[g["gap_id"]] - rank_eff[g["gap_id"]],
                ])
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
