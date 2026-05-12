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


def _sparse_weighted_jaccard(Mc, ci: int, cj: int) -> float:
    """Generalized Jaccard on two sparse CSC columns.

    Same value as ``generalized_jaccard`` on the densified columns, but
    O(nnz_i + nnz_j) — avoids materializing N_rows-long dense vectors.
    Assumes CSC storage (row indices sorted within each column).
    """
    a_start, a_end = int(Mc.indptr[ci]), int(Mc.indptr[ci + 1])
    b_start, b_end = int(Mc.indptr[cj]), int(Mc.indptr[cj + 1])
    a_rows = Mc.indices[a_start:a_end]
    a_vals = Mc.data[a_start:a_end]
    b_rows = Mc.indices[b_start:b_end]
    b_vals = Mc.data[b_start:b_end]
    i = j = 0
    min_sum = 0.0
    max_sum = 0.0
    na, nb = len(a_rows), len(b_rows)
    while i < na and j < nb:
        ra, rb = a_rows[i], b_rows[j]
        if ra < rb:
            max_sum += a_vals[i]
            i += 1
        elif ra > rb:
            max_sum += b_vals[j]
            j += 1
        else:
            av, bv = a_vals[i], b_vals[j]
            if av < bv:
                min_sum += av
                max_sum += bv
            else:
                min_sum += bv
                max_sum += av
            i += 1
            j += 1
    while i < na:
        max_sum += a_vals[i]
        i += 1
    while j < nb:
        max_sum += b_vals[j]
        j += 1
    if max_sum <= 0:
        return 0.0
    return float(min_sum / max_sum)


def _sparse_weighted_containment(Mc, ci: int, cj: int) -> float:
    """Generalized containment on two sparse CSC columns.

    Same value as ``generalized_containment`` on the densified columns,
    O(nnz_i + nnz_j).
    """
    a_start, a_end = int(Mc.indptr[ci]), int(Mc.indptr[ci + 1])
    b_start, b_end = int(Mc.indptr[cj]), int(Mc.indptr[cj + 1])
    a_rows = Mc.indices[a_start:a_end]
    a_vals = Mc.data[a_start:a_end]
    b_rows = Mc.indices[b_start:b_end]
    b_vals = Mc.data[b_start:b_end]
    a_sum = float(a_vals.sum())
    b_sum = float(b_vals.sum())
    denom = a_sum if a_sum < b_sum else b_sum
    if denom <= 0:
        return 0.0
    i = j = 0
    min_sum = 0.0
    na, nb = len(a_rows), len(b_rows)
    while i < na and j < nb:
        ra, rb = a_rows[i], b_rows[j]
        if ra < rb:
            i += 1
        elif ra > rb:
            j += 1
        else:
            av, bv = a_vals[i], b_vals[j]
            min_sum += av if av < bv else bv
            i += 1
            j += 1
    return float(min_sum / denom)


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


# ============================================================
# Section 2: decay function comparison
# ============================================================

def _gaussian_weights(D_row: np.ndarray, sigma_seed: float, sigma_n: np.ndarray) -> np.ndarray:
    return np.exp(-D_row / (sigma_seed * sigma_n))


def _triangular_weights(k: int) -> np.ndarray:
    ranks = np.arange(k, dtype=np.float64)
    return np.maximum(0.0, 1.0 - ranks / k)


def _epanechnikov_weights(D_row: np.ndarray) -> np.ndarray:
    d_max = float(D_row[-1])
    if d_max <= 0:
        return np.ones_like(D_row, dtype=np.float64)
    return np.maximum(0.0, 1.0 - D_row / d_max)


def section2_decay_comparison(
    state: dict, sample_seeds: dict, k_star_by_seed: dict[int, int], out_dir: Path,
) -> dict:
    """Compare Gaussian / triangular / Epanechnikov weight vectors at each seed's k*.

    All three share the same top-k* member set; only the weights differ.
    We compare pairwise weighted Jaccard on the weight vectors.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma = state["sigma"]
    rows = []
    for kind, seeds in sample_seeds.items():
        for s in seeds:
            k_star = k_star_by_seed.get(s)
            if not k_star or k_star < 2:
                continue
            D_row = state["D_art_NN_all"][s, :k_star]
            I_row = state["I_art_NN_all_global"][s, :k_star]
            w_g = _gaussian_weights(D_row, float(sigma[s]), sigma[I_row])
            w_t = _triangular_weights(k_star)
            w_e = _epanechnikov_weights(D_row)
            rows.append({
                "kind": kind, "seed": s, "k_star": k_star,
                "wj_g_t": generalized_jaccard(w_g, w_t),
                "wj_g_e": generalized_jaccard(w_g, w_e),
                "wj_t_e": generalized_jaccard(w_t, w_e),
                "gaussian_sum": float(w_g.sum()),
                "triangular_sum": float(w_t.sum()),
                "epanechnikov_sum": float(w_e.sum()),
            })
    summary_path = out_dir / "decay_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if rows:
        wj_pairs = np.array([
            [r["wj_g_t"], r["wj_g_e"], r["wj_t_e"]] for r in rows
        ])
        all_high = (wj_pairs >= 0.85).all(axis=1).mean()
    else:
        all_high = 0.0

    recommended = "self-tuning-gaussian" if all_high >= 0.5 else "DEFER"
    decision = {
        "section": 2,
        "recommended_decay": recommended,
        "pct_seeds_all_pairs_wj_ge_0.85": float(all_high),
        "n_seeds": len(rows),
        "summary_csv": str(summary_path),
    }
    with open(out_dir / "decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    return decision


# ============================================================
# Section 3: static vs one-step recentering
# ============================================================

def section3_recentering(
    state: dict, sample_seeds: dict, k_star_by_seed: dict[int, int], out_dir: Path,
) -> dict:
    """For each seed, compute the new top-k* around the kernel-weighted centroid
    of the seed's original top-k* members, then compare via Jaccard.

    Writes:
      out_dir/recenter_summary.csv — per-seed Jaccard and member counts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma = state["sigma"]
    article_index = state["article_index"]
    article_global_idx = state["article_global_idx"]
    emb = state["emb"]

    rows = []
    for kind, seeds in sample_seeds.items():
        for s in seeds:
            k_star = k_star_by_seed.get(s)
            if not k_star or k_star < 2:
                continue
            D_row = state["D_art_NN_all"][s, :k_star]
            I_row = state["I_art_NN_all_global"][s, :k_star]
            w = _gaussian_weights(D_row, float(sigma[s]), sigma[I_row])
            members_emb = emb[I_row]
            centroid = (w[:, None] * members_emb).sum(axis=0) / max(w.sum(), 1e-30)
            n = float(np.linalg.norm(centroid))
            if n < 1e-12:
                continue
            centroid = (centroid / n).astype(np.float32)
            _, I_new_local = article_index.search(centroid[None, :], k=k_star)
            I_new_global = article_global_idx[I_new_local[0]]
            inter = len(set(I_row.tolist()) & set(I_new_global.tolist()))
            jaccard = inter / (2 * k_star - inter)
            rows.append({
                "kind": kind, "seed": s, "k_star": k_star,
                "intersection": inter, "jaccard": jaccard,
            })

    summary_path = out_dir / "recenter_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if rows:
        jaccards = np.array([r["jaccard"] for r in rows])
        median_j = float(np.median(jaccards))
        p25_j = float(np.percentile(jaccards, 25))
    else:
        median_j = p25_j = 1.0

    recommend_recenter = (median_j < 0.90) or (p25_j < 0.80)
    decision = {
        "section": 3,
        "recommended_step": "one-recentering" if recommend_recenter else "static",
        "median_jaccard": median_j,
        "p25_jaccard": p25_j,
        "n_seeds": len(rows),
        "summary_csv": str(summary_path),
    }
    with open(out_dir / "decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    return decision


# ============================================================
# Section 4: joint normalization + confluence merge
# ============================================================

def _build_membership_matrix(
    state: dict,
    seed_k_pairs: list[tuple[int, int]],
) -> tuple[csr_matrix, np.ndarray]:
    """Construct (N_articles, N_seeds) sparse membership matrix.

    seed_k_pairs: list of (seed_global_idx, k_star). Each seed's top-k*
    article neighbors get the Gaussian kernel weight in the seed's column.

    Returns:
      M: sparse matrix.
      llr_per_seed: (N_seeds,) array of seed's LLR at k_star (computed
        on the fly here so we can break confluence-merge ties later).
    """
    sigma = state["sigma"]
    art_idx = state["article_global_idx"]
    n_art = len(art_idx)
    n_total = sigma.shape[0]
    global_to_local = np.full(n_total, -1, dtype=np.int64)
    global_to_local[art_idx] = np.arange(n_art)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    llrs = np.zeros(len(seed_k_pairs), dtype=np.float64)
    for col, (s, k_star) in enumerate(seed_k_pairs):
        D_row = state["D_art_NN_all"][s, :k_star]
        I_row = state["I_art_NN_all_global"][s, :k_star]
        w = _gaussian_weights(D_row, float(sigma[s]), sigma[I_row])
        local = global_to_local[I_row]
        valid = local >= 0
        rows.extend(local[valid].tolist())
        cols.extend([col] * int(valid.sum()))
        vals.extend(w[valid].tolist())
        llrs[col] = float(np.log(max(w.sum(), 1e-30)))
    M = csr_matrix((vals, (rows, cols)), shape=(n_art, len(seed_k_pairs)))
    return M, llrs


def _confluence_components(
    seed_k_pairs: list[tuple[int, int]],
    M: csr_matrix,
    threshold: float,
) -> list[list[int]]:
    """Union-find over weighted-Jaccard-high seed pairs.

    Identifies all globally-overlapping seed pairs via the nonzero
    pattern of ``M.T @ M`` — entry (i, j) is nonzero iff seeds i and j
    share at least one article (both columns have a nonzero in the
    same row). Iterates the upper triangle once; no per-article walk,
    no seen-set, no pair cap. Memory bounded by the sparse output of
    M.T @ M (≈16 bytes per overlapping pair).
    """
    import scipy.sparse as sp
    n = M.shape[1]
    Mc = M.tocsc()

    overlap = (M.T @ M).tocsr()
    overlap = sp.triu(overlap, k=1).tocoo()

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    n_pairs = overlap.nnz
    n_unioned = 0
    print(f"    confluence: {n_pairs} overlapping pairs from M.T @ M", flush=True)
    for idx in range(n_pairs):
        ci = int(overlap.row[idx])
        cj = int(overlap.col[idx])
        if _sparse_weighted_jaccard(Mc, ci, cj) >= threshold:
            union(ci, cj)
            n_unioned += 1
        if (idx + 1) % 100_000 == 0:
            print(f"    confluence: {idx + 1}/{n_pairs} pairs evaluated, "
                  f"{n_unioned} unioned", flush=True)

    print(f"    confluence: done, {n_unioned} pairs unioned", flush=True)
    components: dict[int, list[int]] = {}
    for x in range(n):
        components.setdefault(find(x), []).append(x)
    return list(components.values())


def section4_joint_norm(
    state: dict, k_star_by_seed: dict[int, int], out_dir: Path,
    jaccard_threshold: float = 0.9,
) -> tuple[dict, list[int]]:
    """Run joint normalization on all seeds with a chosen k*; report
    effective-size distribution + confluence-merge survivor count.

    Returns (decision_dict, survivors). ``survivors`` is a list of
    global seed indices, one per confluence-merge component
    (highest-LLR seed in each component).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_k_pairs = [
        (s, k) for s, k in k_star_by_seed.items()
        if k is not None and k >= 2
    ]
    if not seed_k_pairs:
        decision = {"section": 4, "n_seeds": 0, "verdict": "no-seeds"}
        with open(out_dir / "decision.json", "w") as f:
            json.dump(decision, f, indent=2)
        return decision, []

    print(f"    building membership matrix: {len(seed_k_pairs)} seeds", flush=True)
    M, llrs = _build_membership_matrix(state, seed_k_pairs)
    print(f"    M shape={M.shape}, nnz={M.nnz}; joint-normalizing", flush=True)
    M_norm, eff = joint_normalize(M)

    components = _confluence_components(
        seed_k_pairs, M_norm, threshold=jaccard_threshold)
    survivor_cols: list[int] = []
    for comp in components:
        best = max(comp, key=lambda c: llrs[c])
        survivor_cols.append(best)
    survivors = [int(seed_k_pairs[c][0]) for c in survivor_cols]
    print(f"    components={len(components)}, survivors={len(survivors)}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(eff, bins=80, color="tab:blue", alpha=0.7)
    ax.set_xlabel("effective_size")
    ax.set_ylabel("count")
    ax.set_title(f"Effective size distribution ({len(eff)} clusters)")
    fig.tight_layout()
    fig.savefig(out_dir / "eff_size_hist.png", dpi=110)
    plt.close(fig)

    top20_idx = np.argsort(-eff)[:20]
    types_arr = state["types"]
    is_book = state["is_book"]
    rows = []
    for rank, i in enumerate(top20_idx):
        s, k = seed_k_pairs[i]
        rows.append({
            "rank": rank + 1,
            "seed_global": int(s),
            "seed_type": "book" if is_book[s] else "article",
            "k_star": int(k),
            "effective_size": float(eff[i]),
            "llr": float(llrs[i]),
        })
    top20_path = out_dir / "top20_by_eff.csv"
    with open(top20_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    median_eff = float(np.median(eff)) if len(eff) > 0 else 0.0
    top50 = np.sort(eff)[-50:] if len(eff) >= 50 else eff
    pct_top50_above_2x_median = float((top50 >= 2 * median_eff).mean()) if median_eff > 0 else 0.0

    sufficient = (len(survivors) <= 500) and (pct_top50_above_2x_median >= 0.5)
    decision = {
        "section": 4,
        "joint_norm_sufficient": sufficient,
        "n_seeds_before_merge": len(seed_k_pairs),
        "n_survivors_after_merge": len(survivors),
        "median_effective_size": median_eff,
        "pct_top50_above_2x_median": pct_top50_above_2x_median,
        "top20_csv": str(top20_path),
    }
    with open(out_dir / "decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    return decision, survivors


# ============================================================
# Section 5: Jaccard vs containment
# ============================================================

def section5_jaccard_vs_containment(
    state: dict, k_star_by_seed: dict[int, int], out_dir: Path,
    pair_cap: int = 200_000,
) -> dict:
    """Compute generalized Jaccard and containment for every overlapping seed pair.

    Identifies pairs where J < 0.5 but C >= 0.8 (candidate nested-distinct
    pairs that containment would incorrectly merge).

    pair_cap caps the number of candidate pairs evaluated to avoid blowups
    on dense regions; pairs are sampled uniformly from the overlap graph.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_k_pairs = [
        (s, k) for s, k in k_star_by_seed.items() if k is not None and k >= 2
    ]
    M, _ = _build_membership_matrix(state, seed_k_pairs)
    Mc = M.tocsc()
    Mr = M.tocsr()

    candidate_pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    rng = np.random.default_rng(0)
    for a in range(Mr.shape[0]):
        col_idx = Mr.indices[Mr.indptr[a]: Mr.indptr[a + 1]]
        if len(col_idx) < 2:
            continue
        for i in range(len(col_idx)):
            for j in range(i + 1, len(col_idx)):
                ci, cj = int(col_idx[i]), int(col_idx[j])
                key = (ci, cj) if ci < cj else (cj, ci)
                if key in seen:
                    continue
                seen.add(key)
                candidate_pairs.append(key)
                if len(candidate_pairs) >= pair_cap:
                    break
            if len(candidate_pairs) >= pair_cap:
                break
        if len(candidate_pairs) >= pair_cap:
            break

    rows = []
    for (ci, cj) in candidate_pairs:
        J = _sparse_weighted_jaccard(Mc, ci, cj)
        C = _sparse_weighted_containment(Mc, ci, cj)
        rows.append({"seed_a": seed_k_pairs[ci][0], "seed_b": seed_k_pairs[cj][0],
                     "k_a": seed_k_pairs[ci][1], "k_b": seed_k_pairs[cj][1],
                     "jaccard": J, "containment": C})

    summary_path = out_dir / "pair_metrics.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Scatter plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Js = np.array([r["jaccard"] for r in rows])
    Cs = np.array([r["containment"] for r in rows])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Js, Cs, s=4, alpha=0.3)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=0.7)
    ax.axhline(0.8, color="red", linestyle="--", linewidth=0.7)
    ax.set_xlabel("generalized Jaccard")
    ax.set_ylabel("generalized containment")
    ax.set_title("Pair overlap: Jaccard vs containment")
    fig.tight_layout()
    fig.savefig(out_dir / "jaccard_vs_containment.png", dpi=110)
    plt.close(fig)

    nested_pairs = [(r["seed_a"], r["seed_b"]) for r in rows
                    if r["jaccard"] < 0.5 and r["containment"] >= 0.8]
    decision = {
        "section": 5,
        "n_pairs_evaluated": len(rows),
        "n_nested_candidates": len(nested_pairs),
        "recommended_metric": "jaccard" if len(nested_pairs) >= 3 else (
            "jaccard (default; no nested cases observed)"
        ),
        "summary_csv": str(summary_path),
    }
    with open(out_dir / "decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    return decision


# ============================================================
# Section 6: survivors recentering (Phase 2 addition)
# ============================================================

def section6_survivors_recentering(
    state: dict,
    k_star_by_seed: dict[int, int],
    survivors: list[int],
    out_dir: Path,
) -> dict:
    """Section 3 restricted to confluence-merge survivors.

    Answers: do dedup survivors (highest-LLR-per-component) need
    recentering, or are they already near their local-density centers?

    Writes:
      out_dir/recenter_survivors_summary.csv — per-survivor seed, k_star,
        intersection, jaccard.
      out_dir/decision.json — section=6, recommended_step
        ("skip-recentering-post-dedup" if median_jaccard >= 0.90 AND
        p25 >= 0.80 else "one-recentering"), median_jaccard,
        p25_jaccard, n_seeds, summary_csv.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma = state["sigma"]
    article_index = state["article_index"]
    article_global_idx = state["article_global_idx"]
    emb = state["emb"]
    is_book = state.get("is_book")

    rows = []
    for s in survivors:
        k_star = k_star_by_seed.get(s)
        if not k_star or k_star < 2:
            continue
        D_row = state["D_art_NN_all"][s, :k_star]
        I_row = state["I_art_NN_all_global"][s, :k_star]
        w = _gaussian_weights(D_row, float(sigma[s]), sigma[I_row])
        members_emb = emb[I_row]
        centroid = (w[:, None] * members_emb).sum(axis=0) / max(w.sum(), 1e-30)
        n = float(np.linalg.norm(centroid))
        if n < 1e-12:
            continue
        centroid = (centroid / n).astype(np.float32)
        _, I_new_local = article_index.search(centroid[None, :], k=k_star)
        I_new_global = article_global_idx[I_new_local[0]]
        inter = len(set(I_row.tolist()) & set(I_new_global.tolist()))
        jaccard = inter / (2 * k_star - inter)
        kind = "book" if (is_book is not None and is_book[s]) else "article"
        rows.append({
            "kind": kind, "seed": int(s), "k_star": int(k_star),
            "intersection": int(inter), "jaccard": float(jaccard),
        })

    summary_path = out_dir / "recenter_survivors_summary.csv"
    with open(summary_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    if rows:
        jaccards = np.array([r["jaccard"] for r in rows])
        median_j = float(np.median(jaccards))
        p25_j = float(np.percentile(jaccards, 25))
    else:
        median_j = p25_j = 1.0

    skip_recenter = (median_j >= 0.90) and (p25_j >= 0.80)
    decision = {
        "section": 6,
        "recommended_step": "skip-recentering-post-dedup" if skip_recenter else "one-recentering",
        "median_jaccard": median_j,
        "p25_jaccard": p25_j,
        "n_seeds": len(rows),
        "summary_csv": str(summary_path),
    }
    with open(out_dir / "decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    return decision


# ============================================================
# Main driver
# ============================================================

def _compute_k_star(state: dict, metric: str = "LLR") -> dict[int, int]:
    """For every seed (article + book), compute k* by argmax of the chosen
    surprise metric over K_GRID. Used as input to sections 2–5.
    """
    sigma = state["sigma"]
    art_idx = state["article_global_idx"]
    M_null = kernel_mass_matrix(
        D_art_NN=state["D_art_NN_all"][art_idx],
        I_art_NN_global=state["I_art_NN_all_global"][art_idx],
        sigma=sigma,
        k_grid=K_GRID,
        seed_global_idx=art_idx,
    )
    null_mean = M_null.mean(axis=0)
    null_std = M_null.std(axis=0)
    null_log_mean = np.log(np.maximum(M_null, 1e-30)).mean(axis=0)

    n_total = state["sigma"].shape[0]
    all_seeds = np.arange(n_total)
    M_all = kernel_mass_matrix(
        D_art_NN=state["D_art_NN_all"],
        I_art_NN_global=state["I_art_NN_all_global"],
        sigma=sigma,
        k_grid=K_GRID,
        seed_global_idx=all_seeds,
    )
    if metric == "LLR":
        curves = np.log(np.maximum(M_all, 1e-30)) - null_log_mean[None, :]
    else:
        curves = (M_all - null_mean[None, :]) / np.maximum(null_std[None, :], 1e-30)
    argmax_idx = np.argmax(curves, axis=1)
    return {int(s): int(K_GRID[argmax_idx[s]]) for s in range(n_total)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True,
                   help="e.g. 17/1702/T11303")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory, e.g. cache/static_seed_diagnostic/T11303")
    p.add_argument("--article-seed-csv", type=Path,
                   default=Path("artifacts/gap_article_as_seed/T11303__article_seed.csv"),
                   help="Source of stratified cohesive seeds for section 1")
    p.add_argument("--n-random-articles", type=int, default=50)
    p.add_argument("--jaccard-threshold", type=float, default=0.9)
    p.add_argument("--skip", default="",
                   help="Comma-separated section numbers to skip, e.g. '5'")
    args = p.parse_args()

    skip = {int(x) for x in args.skip.split(",") if x.strip()}
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[setup] loading scope {args.scope}...")
    state = setup(args.scope)
    print(f"  N_total={state['sigma'].shape[0]}, "
          f"N_articles={len(state['article_global_idx'])}, "
          f"N_books={len(state['book_global_idx'])}")

    print("[k_star] computing k* per seed via LLR argmax (will be overridden by section 1's decision)")
    k_star_by_seed = _compute_k_star(state, metric="LLR")

    sample_seeds = sample_diagnostic_seeds(
        state,
        args.article_seed_csv if args.article_seed_csv.exists() else None,
        n_random_articles=args.n_random_articles,
    )
    print(f"  cohesive={len(sample_seeds['cohesive'])}, "
          f"random={len(sample_seeds['random'])}, "
          f"book={len(sample_seeds['book'])}")

    decisions: dict[str, dict] = {}
    if 1 not in skip:
        print("[section 1] surprise curves...")
        decisions["section1"] = section1_surprise_curves(
            state, sample_seeds, args.out / "section1")
        # If LLR not preferred, recompute k_star with z-score.
        if decisions["section1"].get("recommended_metric") == "z-score":
            print("  section 1 prefers z-score; recomputing k* with z-score")
            k_star_by_seed = _compute_k_star(state, metric="z-score")
    if 2 not in skip:
        print("[section 2] decay comparison...")
        decisions["section2"] = section2_decay_comparison(
            state, sample_seeds, k_star_by_seed, args.out / "section2")
    if 3 not in skip:
        print("[section 3] recentering...")
        decisions["section3"] = section3_recentering(
            state, sample_seeds, k_star_by_seed, args.out / "section3")
    if 4 not in skip:
        print("[section 4] joint normalization...")
        decisions["section4"], survivors_s4 = section4_joint_norm(
            state, k_star_by_seed, args.out / "section4",
            jaccard_threshold=args.jaccard_threshold)
    else:
        survivors_s4 = None
    if 5 not in skip:
        print("[section 5] Jaccard vs containment...")
        decisions["section5"] = section5_jaccard_vs_containment(
            state, k_star_by_seed, args.out / "section5")
    if 6 not in skip:
        if survivors_s4 is None:
            print("[section 6] SKIPPED — section 4 was skipped; "
                  "survivors not available")
        else:
            print("[section 6] survivors recentering...")
            decisions["section6"] = section6_survivors_recentering(
                state, k_star_by_seed, survivors_s4, args.out / "section6")

    summary_path = args.out / "decisions.json"
    with open(summary_path, "w") as f:
        json.dump(decisions, f, indent=2)
    print(f"[done] decisions written to {summary_path}")


if __name__ == "__main__":
    main()
