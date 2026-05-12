"""Static-seed cohesion diagnostic.

Resolves five algorithmic decisions for a proposed static-seed gap-
detection variant before committing to a full implementation. See
``docs/superpowers/specs/2026-05-12-static-seed-cohesion-diagnostic-design.md``
for context.

Pipeline diagnosed (per spec):

  per-seed (k*, weight vector) -> Jaccard confluence merge ->
  joint normalize -> rank by effective_size + coverage readout.

This script answers:
  1. LLR-vs-k or z-score-vs-k for the surprise curve?
  2. Self-tuning Gaussian / linear-rank / linear-distance decay?
  3. Static or one-step recentering?
  4. Is joint normalization sufficient as a redundancy handler?
  5. Jaccard or containment for the pre-norm confluence merge?

Usage::

    conda run -p .conda python -m artifacts_scripts.gap_static_seed_diagnostic \
        --scope 17/1702/T11303 \
        --out cache/static_seed_diagnostic/T11303
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix

# Module-level constants (no magic numbers in section code).
K_SIGMA = 7
K_QUERY = 1500
K_GRID = (5, 10, 20, 40, 80, 160, 320, 640, 1280)


# ============================================================
# Math primitives
# ============================================================

def generalized_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Generalized Jaccard on non-negative weight vectors.

    J(a, b) = sum(min(a, b)) / sum(max(a, b)).
    Returns 0.0 when both vectors are all-zero.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.minimum(a, b).sum()
    denom = np.maximum(a, b).sum()
    if denom <= 0:
        return 0.0
    return float(num / denom)


def generalized_containment(a: np.ndarray, b: np.ndarray) -> float:
    """Generalized Szymkiewicz-Simpson on non-negative weight vectors.

    C(a, b) = sum(min(a, b)) / min(sum(a), sum(b)).
    Returns 0.0 when either vector sums to zero.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.minimum(a, b).sum()
    denom = min(a.sum(), b.sum())
    if denom <= 0:
        return 0.0
    return float(num / denom)


def joint_normalize(M: csr_matrix) -> tuple[csr_matrix, np.ndarray]:
    """Normalize each row of M to sum to 1; return (M_normalized, column_sums).

    Rows that sum to zero are left unchanged (no division). Column sums of
    the normalized matrix are returned as effective_size per cluster.
    """
    from scipy.sparse import diags
    M = M.tocsr()
    row_sums = np.asarray(M.sum(axis=1)).flatten()
    inv = np.where(row_sums > 0, 1.0 / np.where(row_sums > 0, row_sums, 1.0), 0.0)
    M_norm = (diags(inv) @ M).tocsr()
    eff = np.asarray(M_norm.sum(axis=0)).flatten()
    return M_norm, eff


def kernel_mass_matrix(
    D_art_NN: np.ndarray,
    I_art_NN_global: np.ndarray,
    sigma: np.ndarray,
    k_grid: Sequence[int],
    seed_global_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Compute cumulative kernel mass M_k(seed) for each seed at each k in k_grid.

    Arguments:
      D_art_NN: (N_rows, K_query) squared L2 distances; row i corresponds
        to the i-th row of seed_global_idx (or i-th seed if seed_global_idx
        is None and rows correspond directly to sigma's indexing).
      I_art_NN_global: (N_rows, K_query) global indices of neighbors per seed.
      sigma: (N_total,) self-tuning bandwidths over all works (article+book).
      k_grid: sequence of k values; all values <= D_art_NN.shape[1].
      seed_global_idx: (N_rows,) global index of each seed row. If None,
        sigma is assumed to be already aligned to D_art_NN's rows.

    Returns (N_rows, len(k_grid)) cumulative kernel mass.
    """
    if seed_global_idx is None:
        sigma_seed = sigma[: D_art_NN.shape[0], None]
    else:
        sigma_seed = sigma[seed_global_idx][:, None]
    sigma_n = sigma[I_art_NN_global]
    W = np.exp(-D_art_NN / (sigma_seed * sigma_n))
    cum = np.cumsum(W, axis=1)
    cols = np.array([k - 1 for k in k_grid], dtype=np.int64)
    return cum[:, cols]


# ============================================================
# Setup
# ============================================================

def setup(scope: str, k_sigma: int = K_SIGMA, k_query: int = K_QUERY) -> dict:
    """Build the shared kernel/index state used by all five sections.

    Returns a dict with keys:
      emb, ids, types, is_article, is_book, article_global_idx, book_global_idx,
      sigma, D_art_NN_all, I_art_NN_all_global,
      full_index, article_index.

    All seeds (every article + every book) get a (k_query,)-row of nearest-
    article neighbors — these are the candidate cluster members under the
    static-seed variant's "articles-only" rule.
    """
    # Local imports keep the script importable without these (for unit tests).
    from artifacts_scripts.hdbscan_gap_probe import load_scope
    from bookgap.index import build_index, FlatL2

    emb, ids, types, _t, topic_ids, _refs = load_scope(scope, with_refs=False)
    types_arr = np.array(types)
    is_book = types_arr == "book"
    is_article = types_arr == "article"
    book_global_idx = np.where(is_book)[0]
    article_global_idx = np.where(is_article)[0]

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)
    emb = np.ascontiguousarray(emb)

    full_index = build_index(FlatL2(), emb)
    article_emb = np.ascontiguousarray(emb[article_global_idx])
    article_index = build_index(FlatL2(), article_emb)

    D_sigma, _ = full_index.search(emb, k=k_sigma + 1)
    sigma_sq = np.median(D_sigma[:, 1: k_sigma + 1], axis=1)
    sigma = np.sqrt(np.maximum(sigma_sq, 1e-8)).astype(np.float32)

    # For every work (book + article), find its top-k_query nearest articles.
    # article_index returns local indices into article_global_idx.
    D_art_NN_all, I_art_NN_local = article_index.search(emb, k=k_query)
    I_art_NN_all_global = article_global_idx[I_art_NN_local]

    return {
        "emb": emb,
        "ids": np.array(ids),
        "types": types_arr,
        "is_article": is_article,
        "is_book": is_book,
        "article_global_idx": article_global_idx,
        "book_global_idx": book_global_idx,
        "sigma": sigma,
        "D_art_NN_all": D_art_NN_all,
        "I_art_NN_all_global": I_art_NN_all_global,
        "full_index": full_index,
        "article_index": article_index,
    }


# ============================================================
# Seed sampling
# ============================================================

def stratified_cohesive_seeds(
    article_seed_csv: Path,
    n_buckets: int = 8,
    per_bucket: int = 3,
) -> list[int]:
    """Sample seeds from existing article-as-seed basins, stratified by size.

    For each of n_buckets log-spaced final_size buckets, take the per_bucket
    basins with the highest logit_sum, and return their seed_article indices.
    Returns up to n_buckets * per_bucket global article indices.
    """
    rows = []
    with open(article_seed_csv) as f:
        for r in csv.DictReader(f):
            rows.append({
                "seed_article": int(r["seed_article"]),
                "final_size": int(r["final_size"]),
                "logit_sum": float(r["logit_sum"]),
            })
    if not rows:
        return []
    sizes = np.array([r["final_size"] for r in rows], dtype=np.float64)
    log_sizes = np.log(np.maximum(sizes, 1.0))
    edges = np.linspace(log_sizes.min(), log_sizes.max() + 1e-6, n_buckets + 1)
    seeds: list[int] = []
    for b in range(n_buckets):
        in_bucket = [r for r, ls in zip(rows, log_sizes) if edges[b] <= ls < edges[b + 1]]
        in_bucket.sort(key=lambda r: -r["logit_sum"])
        seeds.extend([r["seed_article"] for r in in_bucket[:per_bucket]])
    return seeds


def sample_diagnostic_seeds(
    state: dict,
    article_seed_csv: Path | None,
    n_random_articles: int = 50,
    rng_seed: int = 0,
) -> dict:
    """Build the three sample populations: cohesive, random, book."""
    cohesive = stratified_cohesive_seeds(article_seed_csv) if article_seed_csv else []
    rng = np.random.default_rng(rng_seed)
    art = state["article_global_idx"]
    random_articles = rng.choice(art, size=min(n_random_articles, len(art)), replace=False).tolist()
    books = state["book_global_idx"].tolist()
    return {
        "cohesive": [int(x) for x in cohesive],
        "random": [int(x) for x in random_articles],
        "book": [int(x) for x in books],
    }


# ============================================================
# Section 1: surprise curve shape (LLR vs z-score)
# ============================================================

def section1_surprise_curves(state: dict, sample_seeds: dict, out_dir: Path) -> dict:
    """Sweep k for each sampled seed; compare LLR vs z-score curve shape.

    Computes M_k(seed) for ALL article seeds at every k in K_GRID (the
    empirical null at each k), then evaluates each sampled seed's
    z-score and LLR.

    Writes:
      out_dir/curves_<seed_kind>.png — small-multiples of curves per population.
      out_dir/curves_summary.csv — per-seed argmax-k for each metric and
        peak/floor ratio.

    Returns a decision-rule dict with the recommended metric.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma = state["sigma"]
    art_idx = state["article_global_idx"]
    # Null distribution: kernel mass for EVERY article seed at every k in K_GRID.
    M_all_articles = kernel_mass_matrix(
        D_art_NN=state["D_art_NN_all"][art_idx],
        I_art_NN_global=state["I_art_NN_all_global"][art_idx],
        sigma=sigma,
        k_grid=K_GRID,
        seed_global_idx=art_idx,
    )  # (N_articles, K_count)
    null_mean = M_all_articles.mean(axis=0)
    null_std = M_all_articles.std(axis=0)
    null_log_mean = np.log(np.maximum(M_all_articles, 1e-30)).mean(axis=0)

    rows = []
    all_seeds = []
    for kind, seeds in sample_seeds.items():
        for s in seeds:
            all_seeds.append((kind, int(s)))

    seed_arr = np.array([s for _, s in all_seeds], dtype=np.int64)
    M_seeds = kernel_mass_matrix(
        D_art_NN=state["D_art_NN_all"][seed_arr],
        I_art_NN_global=state["I_art_NN_all_global"][seed_arr],
        sigma=sigma,
        k_grid=K_GRID,
        seed_global_idx=seed_arr,
    )  # (N_seeds, K_count)

    z_curves = (M_seeds - null_mean[None, :]) / np.maximum(null_std[None, :], 1e-30)
    llr_curves = np.log(np.maximum(M_seeds, 1e-30)) - null_log_mean[None, :]

    for i, (kind, s) in enumerate(all_seeds):
        z = z_curves[i]
        l = llr_curves[i]
        k_z = int(K_GRID[int(np.argmax(z))])
        k_l = int(K_GRID[int(np.argmax(l))])
        z_peak_floor = float((z.max() - z.min()) / max(abs(z.min()), 1e-30))
        l_peak_floor = float((l.max() - l.min()) / max(abs(l.min()), 1e-30))
        rows.append({
            "kind": kind, "seed": s,
            "k_argmax_z": k_z, "k_argmax_llr": k_l,
            "z_peak": float(z.max()), "llr_peak": float(l.max()),
            "z_peak_floor_ratio": z_peak_floor,
            "llr_peak_floor_ratio": l_peak_floor,
        })

    summary_path = out_dir / "curves_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plot small-multiples per kind.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for kind in sample_seeds:
        ix = [i for i, (k, _) in enumerate(all_seeds) if k == kind]
        if not ix:
            continue
        n = len(ix)
        cols = 5
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(3 * cols, 2.2 * rows_n), squeeze=False)
        for j, i in enumerate(ix):
            ax = axes[j // cols][j % cols]
            ax.plot(K_GRID, z_curves[i], label="z", color="tab:blue")
            ax2 = ax.twinx()
            ax2.plot(K_GRID, llr_curves[i], label="LLR", color="tab:orange")
            ax.set_xscale("log")
            ax.set_title(f"{kind} {all_seeds[i][1]}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax2.tick_params(labelsize=6)
        # Hide unused axes
        for j in range(n, rows_n * cols):
            axes[j // cols][j % cols].axis("off")
        fig.suptitle(f"z (blue) & LLR (orange) curves — {kind} seeds")
        fig.tight_layout()
        fig.savefig(out_dir / f"curves_{kind}.png", dpi=110)
        plt.close(fig)

    # Decision rule: prefer LLR if its peak/floor ratio is higher on >=60% of seeds.
    llr_wins = sum(1 for r in rows if r["llr_peak_floor_ratio"] > r["z_peak_floor_ratio"])
    pct_llr_wins = llr_wins / len(rows) if rows else 0.0
    recommended = "LLR" if pct_llr_wins >= 0.6 else "z-score"

    decision = {
        "section": 1,
        "recommended_metric": recommended,
        "pct_llr_wins": pct_llr_wins,
        "n_seeds": len(rows),
        "summary_csv": str(summary_path),
    }
    with open(out_dir / "decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    return decision


def main() -> None:
    raise NotImplementedError("main() wired in Task 7")


if __name__ == "__main__":
    main()
