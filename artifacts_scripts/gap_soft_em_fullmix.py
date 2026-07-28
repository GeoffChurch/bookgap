"""Soft-EM with full-data mixture posterior — no separate boundary set.

Theoretical alternative to ``gap_soft_em.py``. Replaces the local
CSLS-balanced boundary construction with the strict 2-component
mixture posterior:

    P(z=G | y) = K_G(y) / K_total(y)
              = Σ_x w(x) · K(y, x) / Σ_x K(y, x)

over y's K-NN neighbourhood (sparsified for tractability — far points
contribute negligibly to the kernel sum). Each point contributes
``w(x) · K(y,x)`` to ``K_G`` and ``(1 − w(x)) · K(y,x)`` to ``K_¬G``;
their sum is exactly ``K(y,x)``. No separate B set, no Task-7 in-G
skip rule, no size-matched local-boundary calibration.

Design source:
  ``docs/backlog/cohesion-metric-design/full-data-soft-mixture-posterior.md``

Key differences from ``gap_soft_em.py``:

  Posterior:
    gap_soft_em        — K_G / (K_G + K_B[size-matched local sample of ¬G])
    gap_soft_em_fullmix — K_G / K_total[K-NN neighbourhood of y]

  Boundary construction:
    gap_soft_em        — 1-NN-book per article in support, CSLS-balanced
                          B_articles around boundary books (Task-7 in-G skip)
    gap_soft_em_fullmix — none. Each candidate's K-NN gives both K_G and
                          K_total directly.

  In-G books:
    gap_soft_em        — explicitly skipped from boundary lookup (Task 7)
    gap_soft_em_fullmix — handled implicitly by membership weighting
                          (a book in G with w=0.9 contributes 0.9·K to K_G
                          and 0.1·K to K_¬G; no special-case rule)

  Threshold τ_supp:
    gap_soft_em        — empirically calibrated (the 0.5 on the
                          local-boundary scale)
    gap_soft_em_fullmix — theoretically motivated ("more likely than not
                          to be in G" under the 2-component mixture)

CSV schema: identical to gap_soft_em (same columns; drop-in compatible
with gap_multi_scale_coverage.py and gap_method_comparison_table.py).

Usage::

    python -m artifacts_scripts.gap_soft_em_fullmix \\
        --scope 17/1702/T11303 \\
        --output-dir /tmp/t11303_soft_em_fullmix
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from artifacts_scripts.hdbscan_gap_probe import load_scope
from bookgap.index import build_index, FlatL2, HNSW


def expand_soft_fullmix(
    seed_idx: int,
    emb: np.ndarray,
    types_arr: np.ndarray,
    sigma: np.ndarray,
    full_index,
    n: int,
    *,
    tau_supp: float = 0.5,
    neighborhood_K: int = 200,
    max_iters: int = 50,
    conv_eps: float = 1e-3,
    claimed: set[int] | None = None,
    claimed_overlap_threshold: float = 0.5,
    claimed_check_every: int = 5,
) -> np.ndarray | None:
    """Soft-EM expansion with full-data mixture posterior.

    Returns weight vector w (length n), or None on claimed-overlap abort.

    Inner loop:
      1. Initialize w[seed] = 1.0.
      2. Each iteration:
         a. Determine support (w > tau_supp). Bootstrap if singleton.
         b. Compute weighted centroid; pull top-K candidates from
            centroid neighbourhood (FAISS K-NN).
         c. For each candidate y: get y's K-NN neighbourhood. Compute
            K_G(y) = Σ_x w(x)·K(y,x) and K_total(y) = Σ_x K(y,x) over
            y's neighbourhood. Set new_w(y) = K_G / K_total.
         d. Convergence: ‖new_w − w‖₂ < conv_eps.

    Self-contribution handling: y appears in its own K-NN. We include
    self in both K_G and K_total — the standard mixture-model treatment.
    The self term contributes w(y) to K_G and 1.0 to K_total. This is
    NOT self-fulfilling: the surrounding neighbours dominate at typical
    K (200), and the iteration converges to a real fixed point.
    """
    w = np.zeros(n, dtype=np.float64)
    w[seed_idx] = 1.0

    K_eff = min(neighborhood_K, n)

    for it in range(max_iters):
        supp = np.where(w > tau_supp)[0]

        # ---- Bootstrap: singleton support → force-add nearest neighbour ----
        if len(supp) < 2:
            _, I = full_index.search(emb[seed_idx:seed_idx + 1], k=2)
            nn = int(I[0, 1]) if int(I[0, 0]) == seed_idx else int(I[0, 0])
            w[nn] = 1.0
            continue

        # ---- Weighted centroid of support ----
        w_supp = w[supp]
        centroid = (emb[supp] * w_supp[:, None]).sum(axis=0) / w_supp.sum()
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-12:
            break
        centroid = (centroid / norm).astype(np.float32)

        # ---- Candidate set: centroid K-NN ∪ current support ----
        _, I_centroid = full_index.search(centroid[None, :], k=K_eff)
        candidates = set(int(c) for c in I_centroid[0])
        candidates.update(int(s) for s in supp)
        candidates_arr = np.array(sorted(candidates), dtype=np.int64)
        C = len(candidates_arr)

        # ---- Per-candidate K-NN neighbourhood (batched) ----
        # knn_per_cand[c, k] = global index of c's k-th nearest neighbour
        cand_emb_for_knn = emb[candidates_arr]
        _, knn_per_cand = full_index.search(cand_emb_for_knn, k=K_eff)
        # shape: (C, K_eff)

        # ---- Posterior update (vectorised) ----
        # For each candidate c with neighbours {x_1, ..., x_K}:
        #   K_G(c)     = Σ_k w[x_k] · kern(c, x_k)
        #   K_total(c) = Σ_k             kern(c, x_k)
        #   new_w(c)   = K_G / K_total

        # Gather neighbour embeddings: (C, K, d)
        neighbor_emb = emb[knn_per_cand]                 # (C, K, d)
        cand_emb = emb[candidates_arr][:, None, :]       # (C, 1, d) for broadcast

        # d2[c, k] = ‖cand_c − neighbor[c,k]‖²
        diff = neighbor_emb - cand_emb                    # (C, K, d)
        d2 = (diff * diff).sum(axis=-1)                   # (C, K)

        # σ outer product (per-pair denominator for self-tuning Gaussian)
        sigma_cand = sigma[candidates_arr][:, None].astype(np.float64)   # (C, 1)
        sigma_neighbor = sigma[knn_per_cand].astype(np.float64)           # (C, K)
        sigma_outer = sigma_cand * sigma_neighbor                          # (C, K)

        kern = np.exp(-d2.astype(np.float64) / (sigma_outer + 1e-30))    # (C, K)

        # Weights of neighbours
        w_neighbor = w[knn_per_cand]                                      # (C, K)

        K_G = (w_neighbor * kern).sum(axis=1)                             # (C,)
        K_total = kern.sum(axis=1)                                         # (C,)

        new_w_cands = K_G / (K_total + 1e-30)                             # (C,)

        new_w = w.copy()
        new_w[candidates_arr] = new_w_cands

        delta = float(np.linalg.norm(new_w - w))
        w = new_w

        # ---- Claimed-overlap early-abort ----
        if (
            claimed is not None
            and claimed
            and (it + 1) % claimed_check_every == 0
        ):
            hard_supp = set(int(i) for i in np.where(w > tau_supp)[0])
            overlap = len(hard_supp & claimed)
            if overlap / max(len(hard_supp), 1) >= claimed_overlap_threshold:
                return None

        if delta < conv_eps:
            break

    return w


def _compute_csv_stats(
    w: np.ndarray,
    tau_supp: float,
    is_article: np.ndarray,
    is_book: np.ndarray,
) -> dict | None:
    """Compute CSV columns from converged weight vector. Same as gap_soft_em."""
    supp_mask = w > tau_supp
    supp_idxs = np.where(supp_mask)[0]
    if len(supp_idxs) == 0:
        return None

    w_supp = w[supp_idxs]
    posts = w_supp.copy()
    posts_clipped = np.clip(posts, 1e-3, 1 - 1e-3)
    logits = np.log(posts_clipped / (1 - posts_clipped))

    g_articles = [int(i) for i in supp_idxs if is_article[i]]
    g_books = [int(i) for i in supp_idxs if is_book[i]]
    g_all = supp_idxs.tolist()

    member_weights = sorted(
        [
            {
                "idx": int(i),
                "type": "book" if is_book[i] else "article",
                "weight": float(w[i]),
            }
            for i in supp_idxs
        ],
        key=lambda x: -x["weight"],
    )

    return {
        "n_articles": len(g_articles),
        "n_books": len(g_books),
        "total_size": len(g_all),
        "articles": g_articles,
        "all_members": g_all,
        "members_typed": [
            {"idx": int(i), "type": "book" if is_book[i] else "article"}
            for i in g_all
        ],
        "logit_sum": float(logits.sum()),
        "mean_post": float(posts.mean()),
        "min_post": float(posts.min()),
        "member_weights": member_weights,
        "members_set": frozenset(int(i) for i in supp_idxs),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--k-sigma", type=int, default=7)
    p.add_argument("--frontier-width", type=int, default=200,
                   help="Top-K candidates pulled from centroid neighbourhood AND "
                        "K-NN size for per-candidate K_total. Sweep on T11303 "
                        "showed 200 is the right choice — smaller fragments basins.")
    p.add_argument("--tau-supp", type=float, default=0.5,
                   help="Support / hard-threshold cutoff for dedup and output. "
                        "Theoretically motivated under mixture posterior: 'more "
                        "likely than not to be in G'.")
    p.add_argument("--max-iters", type=int, default=50,
                   help="Maximum soft-EM iterations per seed.")
    p.add_argument("--conv-eps", type=float, default=1e-3,
                   help="Weight-vector L2 norm convergence threshold.")
    p.add_argument("--min-final-size", type=int, default=5,
                   help="Minimum hard-threshold support size to accept a basin.")
    p.add_argument("--top-n", type=int, default=25,
                   help="Top-N basins to print on stdout.")
    p.add_argument("--seed-stride", type=int, default=1,
                   help="Iterate every Nth article (1 = every article).")
    p.add_argument("--claimed-overlap-threshold", type=float, default=0.5,
                   help="Abort expansion when |hard_supp ∩ claimed| / |hard_supp| "
                        "exceeds this fraction.")
    p.add_argument("--claimed-check-every", type=int, default=5)
    p.add_argument("--disable-fast-filters", action="store_true")
    args = p.parse_args()

    print(f"Loading scope {args.scope}...")
    emb, ids, types, _titles, _topic_ids, _refs = load_scope(args.scope, with_refs=False)
    n = len(ids)
    types_arr = np.array(types)
    is_book = types_arr == "book"
    is_article = types_arr == "article"
    print(f"  {n:,} works ({is_book.sum():,} books, {is_article.sum():,} articles)")

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)
    emb = np.ascontiguousarray(emb)

    scope_parts = args.scope.split("/")
    topic_id = scope_parts[-1] if scope_parts[-1].startswith("T") else "scope"

    print("Building indices...")
    # HNSW for fast K-NN at scale. efSearch ≥ frontier_width*4 ensures
    # recall matches FlatL2.
    efs = max(args.frontier_width * 4 + 50, args.k_sigma + 20)
    full_index = build_index(HNSW(hnsw_m=32, hnsw_efc=400, hnsw_efs=efs), emb)

    print(f"Computing self-tuning σ (K_sigma={args.k_sigma})...")
    D_sigma, _ = full_index.search(emb, k=args.k_sigma + 1)
    sigma_sq = np.median(D_sigma[:, 1:args.k_sigma + 1], axis=1)
    sigma_sq = np.maximum(sigma_sq, 1e-8).astype(np.float32)
    sigma = np.sqrt(sigma_sq)
    print(f"  σ stats: min={sigma.min():.4f}, median={np.median(sigma):.4f}, "
          f"max={sigma.max():.4f}")

    article_global_idx = np.where(is_article)[0]
    seeds = article_global_idx[::args.seed_stride]
    print(f"\nIterating {len(seeds):,} article seeds (stride {args.seed_stride})...")

    t_start = time.time()
    results: list[dict] = []
    claimed: set[int] = set()
    gap_hashes: dict[frozenset, dict] = {}
    iter_counts: list[int] = []
    n_skipped_claimed = 0
    n_aborted_overlap = 0
    n_below_min_size = 0
    n_dup = 0

    for seed_pos, seed in enumerate(seeds):
        if (seed_pos + 1) % 1000 == 0 or seed_pos == len(seeds) - 1:
            elapsed = time.time() - t_start
            rate = (seed_pos + 1) / max(elapsed, 0.01)
            eta = (len(seeds) - seed_pos - 1) / max(rate, 0.01)
            print(
                f"  processed {seed_pos + 1}/{len(seeds)} seeds in {elapsed:.0f}s "
                f"({rate:.0f}/s, eta {eta:.0f}s); "
                f"discovered {len(results)} gaps; "
                f"skipped-claimed {n_skipped_claimed}, "
                f"early-abort {n_aborted_overlap}, "
                f"too-small {n_below_min_size}, "
                f"dup {n_dup}",
                flush=True,
            )

        s = int(seed)
        if not args.disable_fast_filters and s in claimed:
            n_skipped_claimed += 1
            continue

        claimed_arg = None if args.disable_fast_filters else claimed
        w = expand_soft_fullmix(
            seed_idx=s,
            emb=emb,
            types_arr=types_arr,
            sigma=sigma,
            full_index=full_index,
            n=n,
            tau_supp=args.tau_supp,
            neighborhood_K=args.frontier_width,
            max_iters=args.max_iters,
            conv_eps=args.conv_eps,
            claimed=claimed_arg,
            claimed_overlap_threshold=args.claimed_overlap_threshold,
            claimed_check_every=args.claimed_check_every,
        )

        if w is None:
            n_aborted_overlap += 1
            continue

        stats = _compute_csv_stats(w, args.tau_supp, is_article, is_book)
        if stats is None or stats["total_size"] < args.min_final_size:
            n_below_min_size += 1
            continue

        iter_counts.append(1)

        members_set = stats["members_set"]
        if members_set in gap_hashes:
            n_dup += 1
            gap_hashes[members_set]["seed_types"].add("article")
            continue

        gap_hashes[members_set] = {
            "seed_types": {"article"},
            "first_seed": s,
        }
        results.append({
            "first_seed": s,
            "first_seed_type": "article",
            "seed_types": gap_hashes[members_set]["seed_types"],
            **stats,
        })
        claimed |= members_set

    elapsed = time.time() - t_start
    n_distinct = len(gap_hashes)
    print(f"\nDiscovery complete: {n_distinct} distinct frozensets in {elapsed:.0f}s wall")
    print(
        f"  breakdown: skipped-claimed {n_skipped_claimed} "
        f"({n_skipped_claimed / len(seeds) * 100:.0f}%), "
        f"early-abort {n_aborted_overlap}, too-small {n_below_min_size}, "
        f"dup {n_dup}"
    )

    if args.disable_fast_filters and iter_counts:
        n_completed = len(iter_counts)
        confluence_rate = 1.0 - n_distinct / max(n_completed, 1)
        print(f"  confluence rate: {confluence_rate * 100:.1f}%")

    results.sort(key=lambda r: -r["logit_sum"])

    print()
    print(f"## Top-{args.top_n} discovered gaps (by logit_sum)")
    print(
        f"  {'rk':>3}  {'seed':>6}  {'type':>7}  {'arts':>5}  "
        f"{'books':>5}  {'logit':>7}  {'mean_p':>6}  {'min_p':>5}"
    )
    for i, r in enumerate(results[: args.top_n]):
        print(
            f"  {i + 1:>3}  {r['first_seed']:>6}  "
            f"{r['first_seed_type']:>7}  "
            f"{r['n_articles']:>5}  {r['n_books']:>5}  "
            f"{r['logit_sum']:>7.1f}  "
            f"{r['mean_post']:>6.3f}  {r['min_post']:>5.3f}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{topic_id}__article_seed.csv"

    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "rank", "seed_article", "first_seed_type",
            "n_articles", "n_books", "total_size",
            "book_anchored", "seed_types",
            "final_size",
            "logit_sum", "mean_post", "min_post",
            "article_indices_json",
            "member_indices_json",
            "member_weights_json",
        ])
        for i, r in enumerate(results):
            wr.writerow([
                i + 1,
                r["first_seed"],
                r["first_seed_type"],
                r["n_articles"], r["n_books"], r["total_size"],
                int(r["n_books"] > 0),
                "|".join(sorted(r["seed_types"])),
                r["total_size"],
                f"{r['logit_sum']:.3f}",
                f"{r['mean_post']:.6f}",
                f"{r['min_post']:.4f}",
                json.dumps(r["articles"]),
                json.dumps(r["members_typed"]),
                json.dumps(r["member_weights"]),
            ])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
