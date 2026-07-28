"""Effective gap size via self-tuning Gaussian-weighted K-NN purity.

For each gap, computes per-article soft membership using a
self-tuning Gaussian kernel:

    σ_i² = median squared distance from i to its K_sigma nearest neighbors
    w(i, j) = exp(-d²_{ij} / (σ_i σ_j))                    (i ≠ j)
    purity(i) = Σ_{j ∈ G ∩ N_K(i)} w(i,j) / Σ_{j ∈ N_K(i)} w(i,j)
    effective_size(G) = Σ_{i ∈ G} purity(i)

Where N_K(i) is i's K_query nearest neighbors (excluding self). K_query
is sized to capture ≥99% of Gaussian-kernel mass.

Distances and σ_i are derived from L2-normalized embeddings (the same
indexing as load_scope). Per-point σ_i adapts to local density;
the symmetric form σ_i × σ_j makes the kernel scale-invariant and
eliminates the inverse-square singularity at d → 0.

Usage::

    python -m artifacts_scripts.gap_effective_size \\
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
from bookgap.index import build_index, FlatL2


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
    p.add_argument("--k-sigma", type=int, default=7,
                   help="number of NN for self-tuning bandwidth (default 7, "
                        "the Zelnik-Manor & Perona convention)")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--output-dir", default="artifacts/gap_effective_size")
    args = p.parse_args()

    print(f"Loading scope {args.scope}...")
    emb, ids, types, titles, topic_ids, _refs = load_scope(args.scope, with_refs=False)
    n = len(ids)
    print(f"  {n:,} works")

    # L2-normalise so distances live on the unit sphere.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)
    emb = np.ascontiguousarray(emb)

    print(f"Building FlatL2 index over {n:,} points...")
    index = build_index(FlatL2(), emb)

    print(f"Computing self-tuning σ for all points (K_sigma={args.k_sigma})...")
    D_sigma, _ = index.search(emb, k=args.k_sigma + 1)
    # FAISS FlatL2 returns squared L2 distances. Drop self at column 0.
    sigma_sq = np.median(D_sigma[:, 1:args.k_sigma + 1], axis=1)
    sigma_sq = np.maximum(sigma_sq, 1e-8).astype(np.float32)
    sigma = np.sqrt(sigma_sq)
    print(f"  σ stats: min={sigma.min():.4f}, median={np.median(sigma):.4f}, "
          f"max={sigma.max():.4f}, mean={sigma.mean():.4f}")

    kball = _load_kball(Path(args.kball_csv))
    mb = _load_mb(Path(args.maximal_bookless_csv))
    topic_id = _topic_id_from_path(Path(args.maximal_bookless_csv))
    kball = [r for r in kball if r["topic_id"] == topic_id]
    print(f"  K-ball: {len(kball)} gaps;  max-bookless: {len(mb)} gaps;  topic: {topic_id}")

    # In-gap lookup is reused per gap; allocate once outside the function.
    in_gap_lookup = np.zeros(n, dtype=bool)

    def compute(gap: dict) -> tuple[float, float]:
        G = gap["article_indices"]
        size = len(G)
        if size <= 1:
            return float(size), 1.0
        G_arr = np.asarray(G, dtype=np.int64)
        # Refresh the in-gap mask
        in_gap_lookup[:] = False
        in_gap_lookup[G_arr] = True
        # K_query: enough to capture the Gaussian tail. Size×5 with floor 500.
        K_query = min(max(size * 5, 500), n - 1)
        gap_emb = np.ascontiguousarray(emb[G_arr])
        D, I = index.search(gap_emb, k=K_query + 1)
        # Drop self.
        D = D[:, 1:]
        I = I[:, 1:]
        sigma_i = sigma[G_arr][:, None]
        sigma_j = sigma[I]
        # Self-tuning Gaussian kernel.
        w = np.exp(-D / (sigma_i * sigma_j))
        w_total = w.sum(axis=1) + 1e-30
        w_in_gap = (w * in_gap_lookup[I]).sum(axis=1)
        purities = w_in_gap / w_total
        return float(purities.sum()), float(purities.mean())

    print(f"\nScoring {len(mb)} mb gaps + {len(kball)} K-ball gaps...")
    for g in mb:
        eff, mp = compute(g)
        g["effective_size"] = eff
        g["mean_purity"] = mp
    for g in kball:
        eff, mp = compute(g)
        g["effective_size"] = eff
        g["mean_purity"] = mp

    mb_by_eff = sorted(mb, key=lambda r: -r["effective_size"])
    mb_by_raw = sorted(mb, key=lambda r: -r["size"])
    kball_by_eff = sorted(kball, key=lambda r: -r["effective_size"])
    kball_by_raw = sorted(kball, key=lambda r: -r["size"])

    def _rank(lst: list[dict]) -> dict[int, int]:
        return {g["gap_id"]: i for i, g in enumerate(lst)}

    mb_rank_eff = _rank(mb_by_eff)
    mb_rank_raw = _rank(mb_by_raw)
    kball_rank_eff = _rank(kball_by_eff)
    kball_rank_raw = _rank(kball_by_raw)

    def _print_top(name, sorted_list, rank_raw, rank_eff):
        print()
        print(f"## Top-{args.top_n} {name} by effective size")
        print(f"  {'eff_rk':>6}  {'gap_id':>6}  {'raw_sz':>6}  {'eff_sz':>7}  "
              f"{'purity':>7}  {'raw_rk':>6}  Δrank")
        for i, g in enumerate(sorted_list[:args.top_n]):
            raw_rk_1 = rank_raw[g["gap_id"]] + 1
            delta = rank_raw[g["gap_id"]] - i  # positive = promoted
            delta_str = f"{'+' if delta > 0 else ''}{delta}"
            print(f"  {i+1:>6}  {g['gap_id']:>6}  {g['size']:>6}  "
                  f"{g['effective_size']:>7.1f}  {g['mean_purity']:>7.3f}  "
                  f"{raw_rk_1:>6}  {delta_str}")

    _print_top("K-ball", kball_by_eff, kball_rank_raw, kball_rank_eff)
    _print_top("max-bookless", mb_by_eff, mb_rank_raw, mb_rank_eff)

    def _print_movers(name, items, rank_raw, rank_eff):
        # Sort by (raw rank − eff rank): negative = demoted, positive = promoted
        scored = sorted(items, key=lambda g: rank_raw[g["gap_id"]] - rank_eff[g["gap_id"]])
        print()
        print(f"## {name} biggest demotions (raw-high → eff-low)")
        print(f"  {'gap_id':>6}  {'raw_sz':>6}  {'eff_sz':>7}  {'purity':>7}  "
              f"{'raw_rk':>6}  {'eff_rk':>6}")
        for g in scored[:5]:
            print(f"  {g['gap_id']:>6}  {g['size']:>6}  {g['effective_size']:>7.1f}  "
                  f"{g['mean_purity']:>7.3f}  {rank_raw[g['gap_id']]+1:>6}  "
                  f"{rank_eff[g['gap_id']]+1:>6}")
        print(f"## {name} biggest promotions (raw-low → eff-high)")
        for g in scored[-5:][::-1]:
            print(f"  {g['gap_id']:>6}  {g['size']:>6}  {g['effective_size']:>7.1f}  "
                  f"{g['mean_purity']:>7.3f}  {rank_raw[g['gap_id']]+1:>6}  "
                  f"{rank_eff[g['gap_id']]+1:>6}")

    _print_movers("K-ball", kball, kball_rank_raw, kball_rank_eff)
    _print_movers("max-bookless", mb, mb_rank_raw, mb_rank_eff)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(items: list[dict], name: str, rank_raw, rank_eff):
        path = out_dir / f"{topic_id}__{name}_eff.csv"
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["gap_id", "raw_size", "effective_size", "mean_purity",
                         "raw_rank", "eff_rank", "delta_rank"])
            for g in sorted(items, key=lambda r: -r["effective_size"]):
                wr.writerow([
                    g["gap_id"], g["size"],
                    f"{g['effective_size']:.3f}",
                    f"{g['mean_purity']:.6f}",
                    rank_raw[g["gap_id"]] + 1,
                    rank_eff[g["gap_id"]] + 1,
                    rank_raw[g["gap_id"]] - rank_eff[g["gap_id"]],
                ])
        print(f"Wrote {path}")

    _save(mb, "mb", mb_rank_raw, mb_rank_eff)
    _save(kball, "kball", kball_rank_raw, kball_rank_eff)


if __name__ == "__main__":
    main()
