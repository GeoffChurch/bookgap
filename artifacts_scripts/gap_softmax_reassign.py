"""Multi-cluster softmax post-hoc reassignment.

Phase 1 of the multicluster-softmax-membership backlog item. Reads an
enriched soft-EM CSV (the per-seed pairwise-posterior basin set) and
recomputes membership using a global softmax over kNN basin centroids,
with optional ∅ background (variant B) and optional iteration to
convergence. Emits per-config artifacts to --output-dir.

Spec: docs/superpowers/specs/2026-05-08-multicluster-softmax-phase1-design.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bookgap.index import build_index, FlatL2


def compute_self_tuning_sigma(
    emb: np.ndarray,
    k_sigma: int = 7,
) -> np.ndarray:
    """σ_i = sqrt(median squared distance to k_sigma nearest neighbors
    of i, excluding i itself).

    Matches gap_soft_em.py's formula. Returns shape (N,) float32.
    """
    index = build_index(FlatL2(), emb)
    D, _ = index.search(emb, k=k_sigma + 1)   # +1 for self
    # D[:, 0] is self (distance 0); take 1 .. k_sigma+1
    sigma_sq = np.median(D[:, 1:k_sigma + 1], axis=1)
    sigma_sq = np.maximum(sigma_sq, 1e-8).astype(np.float32)
    return np.sqrt(sigma_sq).astype(np.float32)


def load_enriched_csv(
    csv_path: Path,
    is_article: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Read an enriched soft-EM CSV. Build (N_articles, N_basins) M0
    dense matrix of original article-side soft-EM weights.

    Returns (M0, basin_meta) where basin_meta[k] holds metadata
    needed downstream (rank, n_books, book_anchored, etc.) for the
    k-th basin.
    """
    article_global_idx = np.where(is_article)[0]
    article_local_of_global = np.full(len(is_article), -1, dtype=np.int64)
    article_local_of_global[article_global_idx] = np.arange(len(article_global_idx))

    rows: list[dict] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)

    N_articles = len(article_global_idx)
    N_basins = len(rows)
    M0 = np.zeros((N_articles, N_basins), dtype=np.float32)
    basin_meta: list[dict] = []

    for k, row in enumerate(rows):
        weights = json.loads(row["member_weights_json"])
        for w in weights:
            if w["type"] != "article":
                continue
            local = article_local_of_global[int(w["idx"])]
            if local >= 0:
                M0[local, k] = float(w["weight"])
        basin_meta.append({
            "rank": int(row["rank"]),
            "first_seed": int(row["seed_article"]),   # CSV header is seed_article, not first_seed
            "n_books": int(row.get("n_books", 0)),
            "n_articles": int(row.get("n_articles", 0)),
            "book_anchored": int(row.get("book_anchored", 0)),
        })

    return M0, basin_meta


def softmax_with_optional_empty(
    log_K: np.ndarray,
    log_K_empty: np.ndarray | None,
    tau: float,
) -> np.ndarray:
    """Numerically-stable softmax over log_K / tau (and optional log_K_empty / tau).

    Args:
        log_K: (N, K) array of log K̃_k(a) for kNN basins.
        log_K_empty: (N,) or None. If provided, append as an extra
            column.
        tau: temperature.

    Returns:
        (N, K) or (N, K+1) array of probabilities. Rows sum to 1.
    """
    if log_K_empty is not None:
        logits = np.concatenate(
            [log_K / tau, (log_K_empty / tau)[:, None]], axis=1
        )
    else:
        logits = log_K / tau
    logits = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(logits)
    return exps / exps.sum(axis=1, keepdims=True)


def mean_entropy(probs: np.ndarray) -> float:
    """Mean per-row entropy (nats)."""
    p = np.clip(probs, 1e-30, 1.0)
    return float((-p * np.log(p)).sum(axis=1).mean())


def tune_tau_for_perplexity(
    log_K: np.ndarray,
    log_K_empty: np.ndarray | None,
    P_target: float,
    log_tau_low: float = -3.0,
    log_tau_high: float = 1.0,
    tol: float = 0.01,
    max_iters: int = 30,
) -> float:
    """Bisect on log τ until mean per-row entropy is within tol of
    log(P_target). Returns the tuned τ.

    Higher τ → flatter softmax → higher entropy → higher perplexity.
    """
    target_H = float(np.log(P_target))
    lo, hi = float(log_tau_low), float(log_tau_high)
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        probs = softmax_with_optional_empty(log_K, log_K_empty, np.exp(mid))
        H = mean_entropy(probs)
        if abs(H - target_H) < tol:
            return float(np.exp(mid))
        if H < target_H:
            lo = mid    # need higher τ → higher entropy
        else:
            hi = mid    # need lower τ → lower entropy
    return float(np.exp(0.5 * (lo + hi)))


def compute_log_K_empty(
    article_emb: np.ndarray,
    article_sigma: np.ndarray,
    k_marginal: int,
) -> np.ndarray:
    """Marginal-density background kernel K̃_∅(a) = (1/N) · Σ_y κ(a, y),
    truncated to a's k_marginal nearest articles.

    The kernel is essentially zero beyond ~5σ; this matches the
    truncation strategy gap_soft_em uses internally for K_G/K_B.

    Returns log(K̃_∅) of shape (N_articles,) in float64.
    """
    N = article_emb.shape[0]
    k = min(k_marginal, N)
    index = build_index(FlatL2(), article_emb.astype(np.float32))
    D, I = index.search(article_emb.astype(np.float32), k=k)  # squared L2

    sigma_a = article_sigma.astype(np.float64)
    sigma_y = sigma_a[I]                       # (N, k)
    d2 = D.astype(np.float64)                  # (N, k)
    kern = np.exp(-d2 / (sigma_a[:, None] * sigma_y + 1e-30))  # (N, k)

    K_emp = kern.sum(axis=1) / float(N)        # (N,) — divide by N (not k)
    # Sanity-check the kernel tail: log the value at the K-th neighbor
    tail = kern[:, -1]
    median_tail = float(np.median(tail))
    if median_tail > 1e-4:
        print(f"WARNING: K_empty kNN tail not negligible "
              f"(median tail κ = {median_tail:.2e}); consider larger k_marginal")
    return np.log(K_emp + 1e-30)


def compute_basin_centroids(
    article_emb: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """Weighted mean of article embeddings, unit-normalized per basin.

    Args:
        article_emb: (N_articles, d) float32, unit-normalized.
        M: (N_articles, N_basins) float32, soft membership weights.

    Returns:
        (N_basins, d) float32, unit-normalized.
    """
    weight_sum = M.sum(axis=0)                  # (N_basins,)
    weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    centroids = (M.T @ article_emb) / weight_sum[:, None]  # (N_basins, d)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    return (centroids / np.where(norms > 0, norms, 1.0)).astype(np.float32)


def knn_basin_lookup(
    article_emb: np.ndarray,
    centroids: np.ndarray,
    k: int,
) -> np.ndarray:
    """For each article, return indices of its top-k nearest basin
    centroids by cosine (= dot product on the unit sphere).

    Brute matmul + argpartition; basin set is small (~hundreds).
    """
    sims = article_emb @ centroids.T          # (N_articles, N_basins)
    k = min(k, centroids.shape[0])
    if k >= sims.shape[1]:
        return np.argsort(-sims, axis=1)
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    sorted_part = part[rows, np.argsort(-sims[rows, part], axis=1)]
    return sorted_part.astype(np.int64)


def compute_log_K_basins(
    article_emb: np.ndarray,
    article_sigma: np.ndarray,
    M: np.ndarray,
    knn_basin_idx: np.ndarray,
) -> np.ndarray:
    """For each (article a, basin k in a's kNN), compute
        K̃_k(a) = (1 / Σ_x M[x,k]) · Σ_x M[x,k] · κ(a, x)
    and return log K̃ of shape (N_articles, K_kNN) in float64.

    Args:
        article_emb: (N, d) float32, unit-normalized.
        article_sigma: (N,) float32.
        M: (N, N_basins) float32, soft membership weights.
        knn_basin_idx: (N, K_kNN) int64, indices into M's columns.

    Vectorized over articles; loops over the basin-side index columns
    of knn_basin_idx (K_kNN ≤ 10 in practice).
    """
    N, d = article_emb.shape
    K = knn_basin_idx.shape[1]
    sigma_a = article_sigma.astype(np.float64)
    emb_f64 = article_emb.astype(np.float64)
    sigma_y = sigma_a   # member-side sigma is same array indexed by member

    # Pairwise squared distances between every article and every other
    # article, computed on demand per basin's member set.
    K_tilde = np.zeros((N, K), dtype=np.float64)
    for k_col in range(K):
        basin_ids = knn_basin_idx[:, k_col]   # (N,)
        # For each article a, look up its assigned basin (basin_ids[a]),
        # then sum κ(a, x) · M[x, basin] over members x.
        # Group articles by basin to vectorize.
        unique_basins = np.unique(basin_ids)
        for b in unique_basins:
            arts_with_b = np.where(basin_ids == b)[0]            # (m,)
            members = np.where(M[:, b] > 0)[0]                   # (s,)
            if len(members) == 0:
                K_tilde[arts_with_b, k_col] = 1e-30
                continue
            denom = M[members, b].sum()
            d2 = ((emb_f64[arts_with_b, None, :] - emb_f64[None, members, :]) ** 2).sum(axis=2)
            sig_o = sigma_a[arts_with_b, None] * sigma_y[None, members]
            kern = np.exp(-d2 / (sig_o + 1e-30))                 # (m, s)
            num = (kern * M[members, b][None, :]).sum(axis=1)    # (m,)
            K_tilde[arts_with_b, k_col] = num / max(denom, 1e-30)

    return np.log(K_tilde + 1e-30)


def softmax_e_step(
    article_emb: np.ndarray,
    article_sigma: np.ndarray,
    M: np.ndarray,
    log_K_empty: np.ndarray | None,
    k_knn: int,
    P_target: float,
    log_tau_low: float = -3.0,
    log_tau_high: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """One E-step. Returns new M (full N_articles × N_basins or
    N_basins+1 if ∅ is included) and a meta dict.

    Steps:
        1. centroids ← weighted mean of M's article members
        2. knn_idx[a] ← top-k_knn basins by cosine to centroids
        3. log_K[a, k] ← log K̃_kNN(a) for k ∈ kNN(a)
        4. tau ← bisection on target perplexity
        5. probs ← softmax over kNN basins (+ ∅ if log_K_empty given)
        6. M_new ← scatter probs back into full N_basins (+1 if ∅)
    """
    N, _ = article_emb.shape
    N_basins = M.shape[1]

    centroids = compute_basin_centroids(article_emb, M)
    knn_idx = knn_basin_lookup(article_emb, centroids, k=k_knn)
    log_K = compute_log_K_basins(article_emb, article_sigma, M, knn_idx)
    tau = tune_tau_for_perplexity(
        log_K, log_K_empty, P_target,
        log_tau_low=log_tau_low, log_tau_high=log_tau_high,
    )
    probs = softmax_with_optional_empty(log_K, log_K_empty, tau)

    out_cols = N_basins + (1 if log_K_empty is not None else 0)
    M_new = np.zeros((N, out_cols), dtype=np.float32)
    K_kNN = knn_idx.shape[1]
    rows = np.arange(N)[:, None]
    M_new[rows, knn_idx] = probs[:, :K_kNN].astype(np.float32)
    if log_K_empty is not None:
        M_new[:, -1] = probs[:, -1].astype(np.float32)

    achieved_H = mean_entropy(probs)
    return M_new, {
        "tau": float(tau),
        "achieved_H": float(achieved_H),
        "achieved_P": float(np.exp(achieved_H)),
        "k_knn": int(k_knn),
        "with_empty": log_K_empty is not None,
    }


def run_softmax_reassign(
    article_emb: np.ndarray,
    article_sigma: np.ndarray,
    M0: np.ndarray,
    log_K_empty: np.ndarray | None,
    k_knn: int,
    P_target: float,
    iter_mode: str,
    max_iters: int = 20,
    conv_tol: float = 1e-3,
) -> tuple[np.ndarray, dict]:
    """Drive one or more E-steps. Returns final M and meta dict
    including τ history, iter count, final L_inf delta, converged flag.

    iter_mode='oneshot': one E-step starting from M0.
    iter_mode='iterated': E-steps until ‖M_new − M‖_∞ < conv_tol or
        iter count reaches max_iters.
    """
    deltas: list[float] = []
    taus: list[float] = []
    achieved_Hs: list[float] = []

    if iter_mode == "oneshot":
        M_new, step_meta = softmax_e_step(
            article_emb, article_sigma, M0, log_K_empty, k_knn, P_target,
        )
        return M_new, {
            "iters": 1,
            "final_delta": float("nan"),
            "converged": False,
            "taus": [step_meta["tau"]],
            "achieved_Hs": [step_meta["achieved_H"]],
            "achieved_Ps": [step_meta["achieved_P"]],
        }

    if iter_mode != "iterated":
        raise ValueError(f"unknown iter_mode {iter_mode}")

    M_iter = M0.copy()
    M_new = M_iter
    for it in range(1, max_iters + 1):
        M_new, step_meta = softmax_e_step(
            article_emb, article_sigma, M_iter, log_K_empty, k_knn, P_target,
        )
        taus.append(step_meta["tau"])
        achieved_Hs.append(step_meta["achieved_H"])
        M_basins_new = M_new[:, :M0.shape[1]]
        delta = float(np.abs(M_basins_new - M_iter).max())
        deltas.append(delta)
        M_iter = M_basins_new
        if delta < conv_tol:
            return M_new, {
                "iters": it,
                "final_delta": delta,
                "converged": True,
                "taus": taus,
                "achieved_Hs": achieved_Hs,
                "achieved_Ps": [float(np.exp(h)) for h in achieved_Hs],
            }

    return M_new, {
        "iters": max_iters,
        "final_delta": deltas[-1],
        "converged": False,
        "taus": taus,
        "achieved_Hs": achieved_Hs,
        "achieved_Ps": [float(np.exp(h)) for h in achieved_Hs],
    }


def crisp_argmax(M: np.ndarray, with_empty: bool) -> np.ndarray:
    """Argmax over basin columns. With ∅, last column is ∅; if it wins,
    return -1 (no basin claim)."""
    am = M.argmax(axis=1)
    if with_empty:
        empty_col = M.shape[1] - 1
        am = np.where(am == empty_col, -1, am)
    return am.astype(np.int64)


def crisp_threshold(M: np.ndarray, with_empty: bool, threshold: float) -> np.ndarray:
    """For each row, return the unique basin with p > threshold; -1 if
    none. ∅ column is ignored (cannot be a 'support')."""
    cols = M.shape[1] - (1 if with_empty else 0)
    M_basins = M[:, :cols]
    above = M_basins > threshold
    counts = above.sum(axis=1)
    am = M_basins.argmax(axis=1)
    out = np.where(counts >= 1, am, -1)
    return out.astype(np.int64)


def crisp_confidence_margin(
    M: np.ndarray, with_empty: bool, k_knn: int,
) -> np.ndarray:
    """Article supports basin k iff p_k − p_(2nd over basins) > 1/(2*k_knn).
    ∅ is excluded from the comparison."""
    cols = M.shape[1] - (1 if with_empty else 0)
    M_basins = M[:, :cols]
    sorted_desc = -np.sort(-M_basins, axis=1)
    top1 = sorted_desc[:, 0]
    top2 = sorted_desc[:, 1] if M_basins.shape[1] > 1 else np.zeros_like(top1)
    margin = top1 - top2
    threshold = 1.0 / (2.0 * k_knn)
    am = M_basins.argmax(axis=1)
    out = np.where(margin > threshold, am, -1)
    return out.astype(np.int64)


def write_basin_csv(
    out_path: Path,
    M: np.ndarray,
    crisp: np.ndarray,
    basin_meta: list[dict],
    article_global_idx: np.ndarray,
    with_empty: bool,
) -> None:
    """Write a basin CSV in the same schema as gap_soft_em's enriched
    output. One row per basin; members are articles whose crisp ==
    basin index, sorted by softmax weight desc.

    crisp: (N_articles,) int64; values in [0, N_basins) or -1 (no support).

    Note: the CSV column header for the seed index is `seed_article`
    (matches gap_soft_em.py:1327); the internal basin_meta dict uses
    key `first_seed` for the same value.
    """
    n_basins = len(basin_meta)
    fieldnames = [
        "rank", "seed_article", "n_articles", "n_books", "book_anchored",
        "member_indices_json", "member_weights_json",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for k in range(n_basins):
            mask = (crisp == k)
            members_local = np.where(mask)[0]
            if len(members_local) == 0:
                w.writerow({
                    "rank": basin_meta[k]["rank"],
                    "seed_article": basin_meta[k]["first_seed"],
                    "n_articles": 0,
                    "n_books": basin_meta[k]["n_books"],
                    "book_anchored": basin_meta[k]["book_anchored"],
                    "member_indices_json": json.dumps([]),
                    "member_weights_json": json.dumps([]),
                })
                continue
            weights = M[members_local, k]
            order = np.argsort(-weights)
            members_local = members_local[order]
            members_global = article_global_idx[members_local]
            member_weights = [
                {"idx": int(g), "type": "article", "weight": float(M[l, k])}
                for g, l in zip(members_global, members_local)
            ]
            w.writerow({
                "rank": basin_meta[k]["rank"],
                "seed_article": basin_meta[k]["first_seed"],
                "n_articles": len(members_local),
                "n_books": basin_meta[k]["n_books"],
                "book_anchored": basin_meta[k]["book_anchored"],
                "member_indices_json": json.dumps([int(g) for g in members_global]),
                "member_weights_json": json.dumps(member_weights),
            })


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True)
    p.add_argument("--enriched-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--variant", choices=["A", "B"], required=True)
    p.add_argument("--perplexity", type=float, required=True)
    p.add_argument("--iter-mode", choices=["oneshot", "iterated"], required=True)
    p.add_argument("--k-knn", type=int, default=10)
    p.add_argument("--k-sigma", type=int, default=7)
    p.add_argument("--k-marginal", type=int, default=200)
    p.add_argument("--max-iters", type=int, default=20)
    p.add_argument("--conv-tol", type=float, default=1e-3)
    args = p.parse_args()

    # Imports kept local to avoid heavy startup cost in unit tests
    from artifacts_scripts.hdbscan_gap_probe import load_scope
    from scipy.sparse import csr_matrix, save_npz

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_tag = f"variant{args.variant}__P{args.perplexity}__{args.iter_mode}"

    print(f"Loading scope {args.scope}...", flush=True)
    emb, ids, types, _titles, _topic_ids, _refs = load_scope(args.scope, with_refs=False)
    n = len(ids)
    types_arr = np.array(types)
    is_article = types_arr == "article"
    article_global_idx = np.where(is_article)[0]
    article_emb = emb[article_global_idx]
    norms = np.linalg.norm(article_emb, axis=1, keepdims=True)
    article_emb = (article_emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)

    print(f"  {n:,} works ({(~is_article).sum():,} books, {is_article.sum():,} articles)",
          flush=True)

    print("Loading enriched CSV...", flush=True)
    M0, basin_meta = load_enriched_csv(Path(args.enriched_csv), is_article)
    print(f"  M0 shape: {M0.shape}", flush=True)

    print(f"Computing self-tuning σ (k={args.k_sigma})...", flush=True)
    article_sigma = compute_self_tuning_sigma(article_emb, k_sigma=args.k_sigma)

    log_K_empty = None
    if args.variant == "B":
        print(f"Computing K̃_∅ via article-kNN truncation (k_marginal={args.k_marginal})...",
              flush=True)
        log_K_empty = compute_log_K_empty(article_emb, article_sigma, args.k_marginal)

    print(f"Running softmax reassign ({config_tag})...", flush=True)
    M, meta = run_softmax_reassign(
        article_emb, article_sigma, M0, log_K_empty,
        k_knn=args.k_knn, P_target=args.perplexity,
        iter_mode=args.iter_mode, max_iters=args.max_iters, conv_tol=args.conv_tol,
    )
    print(f"  iters={meta['iters']}, converged={meta.get('converged', None)}, "
          f"final_delta={meta.get('final_delta', float('nan')):.6f}, "
          f"final τ={meta['taus'][-1]:.4f}, achieved P={meta['achieved_Ps'][-1]:.3f}",
          flush=True)

    M_basins_only = M[:, :M0.shape[1]]
    sparse = csr_matrix(M_basins_only)
    save_npz(out_dir / f"memberships__{config_tag}.npz", sparse)

    with_empty = log_K_empty is not None
    rules = {
        "argmax": crisp_argmax(M, with_empty),
        "p_gt_0.5": crisp_threshold(M, with_empty, 0.5),
        "confidence_margin": crisp_confidence_margin(M, with_empty, args.k_knn),
    }
    for rule_name, crisp in rules.items():
        write_basin_csv(
            out_dir / f"basins__{config_tag}__{rule_name}.csv",
            M_basins_only, crisp, basin_meta, article_global_idx, with_empty=False,
        )

    meta_full = dict(meta)
    meta_full["config"] = config_tag
    meta_full["variant"] = args.variant
    meta_full["perplexity_target"] = args.perplexity
    meta_full["iter_mode"] = args.iter_mode
    meta_full["k_knn"] = args.k_knn
    meta_full["k_sigma"] = args.k_sigma
    meta_full["k_marginal"] = args.k_marginal if args.variant == "B" else None
    meta_full["with_empty"] = with_empty
    meta_full["n_articles"] = int(len(article_global_idx))
    meta_full["n_basins"] = int(M0.shape[1])
    (out_dir / f"meta__{config_tag}.json").write_text(json.dumps(meta_full, indent=2))
    print(f"Done. Wrote artifacts to {out_dir}/*__{config_tag}.*", flush=True)


if __name__ == "__main__":
    main()
