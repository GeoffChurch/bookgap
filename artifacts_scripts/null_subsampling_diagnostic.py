"""How reliable is a sub-sampled corpus null for LLR-vs-k?

The Phase 1-3 LLR machinery uses a corpus null at each k in K_GRID,
computed as ``mean(log(M_k))`` over **all** article seeds in the topic.
On T10181 with N_articles ≈ 88k the full null requires top-K_GRID kNN
for every article seed, which is the slow part of setup() when we want
to extend K_GRID past 1280. A sub-sampled null (m article seeds chosen
at random) would let us extend the grid cheaply if it's accurate
enough.

This diagnostic answers two questions:
  - How does the sample-mean SE shrink with m?
  - Is the SE roughly k-independent (so the same m works at all k in the
    extended grid)?

The answer is: at fixed k, the SE shrinks as σ_k / √m by the CLT. We
report σ_k and the empirical relative-SE-vs-m curve at each k in K_GRID
so callers can pick m for a target precision.

Usage::

    uv run python -u -m artifacts_scripts.null_subsampling_diagnostic \\
        --scope 17/1702/T11303 \\
        --out cache/null_subsampling_diagnostic/T11303.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from artifacts_scripts.gap_static_seed_diagnostic import (
    K_GRID,
    kernel_mass_matrix,
    setup,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True, help="e.g. 17/1702/T11303")
    p.add_argument("--out", required=True, type=Path,
                   help="Output CSV path")
    p.add_argument("--sample-sizes", default="50,100,250,500,1000,2500,5000",
                   help="Comma-separated m values to test")
    p.add_argument("--n-trials", type=int, default=50,
                   help="Number of random subsamples per m")
    p.add_argument("--rng-seed", type=int, default=0)
    args = p.parse_args()

    sample_sizes = [int(s) for s in args.sample_sizes.split(",")]
    rng = np.random.default_rng(args.rng_seed)

    print(f"[setup] loading scope {args.scope}...", flush=True)
    state = setup(args.scope)
    art_idx = state["article_global_idx"]
    n_articles = len(art_idx)
    print(f"  N_articles={n_articles}", flush=True)

    print(f"[kernel] computing kernel mass at each k in K_GRID for all article seeds...",
          flush=True)
    M_all = kernel_mass_matrix(
        D_art_NN=state["D_art_NN_all"][art_idx],
        I_art_NN_global=state["I_art_NN_all_global"][art_idx],
        sigma=state["sigma"],
        k_grid=K_GRID,
        seed_global_idx=art_idx,
    )
    log_M = np.log(np.maximum(M_all, 1e-30))  # (N_articles, len(K_GRID))

    # Full-corpus null at each k
    full_null = log_M.mean(axis=0)
    full_sd = log_M.std(axis=0)
    print(f"  full_null per k: {[f'{v:.3f}' for v in full_null]}", flush=True)
    print(f"  full_sd   per k: {[f'{v:.3f}' for v in full_sd]}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "k", "m", "full_null", "full_sd",
            "subsample_mean_of_means", "bias",
            "subsample_se", "predicted_se",
            "relative_se",
        ])
        for k_idx, k in enumerate(K_GRID):
            for m in sample_sizes:
                if m >= n_articles:
                    continue
                means = np.empty(args.n_trials)
                for t in range(args.n_trials):
                    sample = rng.choice(n_articles, size=m, replace=False)
                    means[t] = log_M[sample, k_idx].mean()
                mean_of_means = float(means.mean())
                bias = mean_of_means - float(full_null[k_idx])
                se = float(means.std(ddof=1))
                predicted_se = float(full_sd[k_idx] / np.sqrt(m))
                rel_se = se / max(abs(full_null[k_idx]), 1e-12)
                w.writerow([
                    k, m,
                    f"{full_null[k_idx]:.6f}",
                    f"{full_sd[k_idx]:.6f}",
                    f"{mean_of_means:.6f}",
                    f"{bias:.6f}",
                    f"{se:.6f}",
                    f"{predicted_se:.6f}",
                    f"{rel_se:.6f}",
                ])
            print(f"  k={k}: full_null={full_null[k_idx]:.3f}, "
                  f"σ_k={full_sd[k_idx]:.3f}, "
                  f"SE@m=1000={float(full_sd[k_idx]/np.sqrt(1000)):.4f}",
                  flush=True)

    print(f"[done] csv written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
