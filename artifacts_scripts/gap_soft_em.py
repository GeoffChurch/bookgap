"""Soft-EM-based gap discovery with local CSLS-balanced boundary.

EM-style alternative to ``gap_article_as_seed.py``. Replaces the
Viterbi-style hard-threshold inner loop (ordered cumulative ADD + REMOVE)
with a soft posterior update: each candidate's weight is directly set to
its posterior ``K_G / (K_G + K_B)`` each iteration, converging when the
weight-vector L2 norm stabilises.

Design source:
  ``docs/backlog/cohesion-metric-design/fuzzy-membership-preserving-boundary.md``

Key differences from ``gap_article_as_seed.py --batch-addition``:

  Inner-loop policy:
    Phase 3 — ordered cumulative ADD + symmetric REMOVE (Viterbi hard EM)
    Soft EM  — weight vector updated by posterior each iteration

  Membership:
    Phase 3 — set[int] (binary)
    Soft EM  — dict[int, float] (sparse weights)

  Frontier filter:
    Phase 3 — article-only (unless --with-book-seeds)
    Soft EM  — always unconditional (books may enter G geometrically)

  Book seeds:
    Phase 3 — optional via --with-book-seeds
    Soft EM  — not implemented; articles do discovery, books join as members

  Convergence:
    Phase 3 — frozenset stable across iters
    Soft EM  — weight L2 norm < eps

CSV schema (superset of Phase 3, drop-in compatible with
``gap_multi_scale_coverage.py`` and ``gap_method_comparison_table.py``):
  rank, seed_article, first_seed_type,
  n_articles, n_books, total_size, book_anchored, seed_types,
  final_size, logit_sum, mean_post, min_post,
  article_indices_json, member_indices_json,
  member_weights_json   <-- NEW: [{idx, type, weight}, ...] sorted desc

Usage::

    python -m artifacts_scripts.gap_soft_em \\
        --scope 17/1702/T11303 \\
        --output-dir /tmp/t11303_soft_em
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from artifacts_scripts.hdbscan_gap_probe import load_scope
from bookgap.index import build_index, FlatL2, HNSW


def expand_soft(
    seed_idx: int,
    emb: np.ndarray,
    types_arr: np.ndarray,
    sigma: np.ndarray,
    full_index,
    book_index,
    book_global_idx: np.ndarray,
    article_global_idx: np.ndarray,
    article_to_nearest_books_global: np.ndarray,
    article_index,
    n: int,
    *,
    tau_supp: float = 0.5,
    neighborhood_K: int = 200,
    max_iters: int = 50,
    conv_eps: float = 1e-3,
    claimed: set[int] | None = None,
    claimed_overlap_threshold: float = 0.5,
    claimed_check_every: int = 5,
    in_G_lookup: np.ndarray | None = None,
) -> np.ndarray | None:
    """Soft-EM expansion from *seed_idx*. Returns weight vector w (length n).

    Returns None if the expansion fails (empty boundary, no article
    members after bootstrap, or claimed-overlap abort).

    Inner loop (adapted from ``/tmp/fuzzy_em_prototype.py:expand_fuzzy``):

    1. Initialize w[seed] = 1.0, all others 0.
    2. Each iteration:
       a. Determine support: indices where w > tau_supp. Bootstrap
          (force-add nearest neighbor) if singleton.
       b. Compute weighted centroid of support; pull top-K candidates
          from centroid neighborhood.
       c. For each article in support, find first 1-NN-book OUTSIDE support
          → boundary books with weighted f(b) = Σ_{i: 1NN(i)=b} w(i).
       d. CSLS-balanced B_articles: ceil(f(b)) nearest non-support
          articles per boundary book.
       e. For each candidate y:
            K_G(y) = Σ_x w(x) · K(y,x)
            K_B(y) = Σ_b f(b) · K(y,b)  +  Σ_a 1 · K(y,a)
            new_w(y) = K_G / (K_G + K_B)
       f. Convergence: ‖new_w − w‖₂ < conv_eps.
    """
    is_article = types_arr == "article"
    is_book = types_arr == "book"
    if in_G_lookup is None:
        in_G_lookup = np.zeros(n, dtype=bool)

    w = np.zeros(n, dtype=np.float64)
    w[seed_idx] = 1.0

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

        # ---- Neighbourhood candidates (centroid K-NN ∪ current support) ----
        K_eff = min(neighborhood_K, n)
        _, I_centroid = full_index.search(centroid[None, :], k=K_eff)
        candidates = set(int(c) for c in I_centroid[0])
        candidates.update(int(s) for s in supp)
        candidates_arr = np.array(sorted(candidates), dtype=np.int64)

        # ---- Boundary books: first not-in-support book per article member ----
        in_G_lookup[:] = False
        in_G_lookup[supp] = True

        article_supp = supp[is_article[supp]]
        if len(article_supp) == 0:
            break  # degenerate: no article members at all

        # Use precomputed nearest-book table; iterate columns to find first
        # book not in support. Shape (m, K_books).
        cand_books = article_to_nearest_books_global[article_supp]  # (m, K_books)
        mask_valid = (cand_books != -1) & ~in_G_lookup[np.where(cand_books != -1, cand_books, 0)]
        has_valid = mask_valid.any(axis=1)
        if not has_valid.any():
            break  # boundary saturated

        first_valid_col = mask_valid.argmax(axis=1)
        # f(b) weighted by w[article]
        f_b: dict[int, float] = {}
        for row_i, (ai, has_v) in enumerate(zip(article_supp, has_valid)):
            if not has_v:
                continue
            gb = int(cand_books[row_i, first_valid_col[row_i]])
            f_b[gb] = f_b.get(gb, 0.0) + float(w[ai])

        if not f_b:
            break  # no valid boundary books

        # ---- CSLS-balanced B_articles: ceil(f(b)) nearest non-support articles ----
        B_articles: set[int] = set()
        for b_global, fb in f_b.items():
            k_want = max(1, int(np.ceil(fb)))
            K_search = min(k_want + len(supp) + 10, len(article_global_idx))
            _, I_arts = article_index.search(emb[b_global:b_global + 1], k=K_search)
            taken = 0
            for al_idx in I_arts[0]:
                if taken >= k_want:
                    break
                a_global = int(article_global_idx[al_idx])
                if not in_G_lookup[a_global] and a_global not in B_articles:
                    B_articles.add(a_global)
                    taken += 1

        B_books_arr = np.array(list(f_b.keys()), dtype=np.int64)
        B_books_mass = np.array(list(f_b.values()), dtype=np.float64)
        B_arts_arr = np.array(list(B_articles), dtype=np.int64)

        # ---- Weight update (fully vectorised over all candidates) ----
        # cand_emb: (C, d),  supp_emb: (S, d),  sigma_cand: (C,),  sigma_supp: (S,)
        # Kernel matrix K[c, s] = exp(-‖cand_c - supp_s‖² / (σ_c · σ_s))
        # For unit-normalised embeddings: ‖a-b‖² = 2(1 - a·b), computed via
        #   ‖a-b‖² = ‖a‖²+‖b‖²-2a·b = 2 - 2(a·b)  [both unit vectors]
        # which is equivalent to the direct difference for float32 emb.
        cand_emb = emb[candidates_arr].astype(np.float64)  # (C, d)
        sigma_cand = sigma[candidates_arr].astype(np.float64)  # (C,)
        supp_emb = emb[supp].astype(np.float64)              # (S, d)
        supp_sigma = sigma[supp].astype(np.float64)          # (S,)

        # d2[c, s] = ‖cand_c − supp_s‖²  via broadcasting (C, S)
        # = (cand_emb² row-sum) + (supp_emb² row-sum) - 2 * cand @ supp.T
        c2 = (cand_emb * cand_emb).sum(axis=1, keepdims=True)   # (C, 1)
        s2 = (supp_emb * supp_emb).sum(axis=1, keepdims=True).T  # (1, S)
        dot_cs = cand_emb @ supp_emb.T                           # (C, S)
        d2_cs = np.maximum(c2 + s2 - 2 * dot_cs, 0.0)           # (C, S)

        # σ_cand × σ_supp outer product → (C, S) denominator
        sigma_outer = sigma_cand[:, None] * supp_sigma[None, :]  # (C, S)
        kern_cs = np.exp(-d2_cs / (sigma_outer + 1e-30))         # (C, S)

        # K_G[c] = Σ_s w[s] · kern[c,s]  — (C,)
        K_G_vec = kern_cs @ w_supp  # (C,)

        # Subtract self-contribution for candidates that are also in support
        supp_set = set(int(s) for s in supp)
        supp_pos_in_cand = {int(s): i for i, s in enumerate(candidates_arr)
                            if int(s) in supp_set}
        for s_global, c_idx in supp_pos_in_cand.items():
            K_G_vec[c_idx] -= float(w[s_global])  # kern(y,y)=1; remove w[y]·1

        # K_B_books[c] = Σ_b f(b) · kern(cand_c, book_b)  — (C,)
        if len(B_books_arr) > 0:
            book_emb_b = emb[B_books_arr].astype(np.float64)    # (B, d)
            sigma_b = sigma[B_books_arr].astype(np.float64)      # (B,)
            c2_for_b = c2  # reuse (C, 1)
            b2 = (book_emb_b * book_emb_b).sum(axis=1, keepdims=True).T  # (1, B)
            dot_cb = cand_emb @ book_emb_b.T                    # (C, B)
            d2_cb = np.maximum(c2_for_b + b2 - 2 * dot_cb, 0.0)  # (C, B)
            sigma_outer_b = sigma_cand[:, None] * sigma_b[None, :]  # (C, B)
            kern_cb = np.exp(-d2_cb / (sigma_outer_b + 1e-30))  # (C, B)
            K_B_books_vec = kern_cb @ B_books_mass               # (C,)
        else:
            K_B_books_vec = np.zeros(len(candidates_arr))

        # K_B_arts[c] = Σ_a kern(cand_c, art_a)  — mass=1 each  — (C,)
        if len(B_arts_arr) > 0:
            art_emb_a = emb[B_arts_arr].astype(np.float64)      # (A, d)
            sigma_a = sigma[B_arts_arr].astype(np.float64)       # (A,)
            a2 = (art_emb_a * art_emb_a).sum(axis=1, keepdims=True).T  # (1, A)
            dot_ca = cand_emb @ art_emb_a.T                     # (C, A)
            d2_ca = np.maximum(c2 + a2 - 2 * dot_ca, 0.0)      # (C, A)
            sigma_outer_a = sigma_cand[:, None] * sigma_a[None, :]  # (C, A)
            kern_ca = np.exp(-d2_ca / (sigma_outer_a + 1e-30))  # (C, A)
            K_B_arts_vec = kern_ca.sum(axis=1)                   # (C,)
        else:
            K_B_arts_vec = np.zeros(len(candidates_arr))

        K_B_vec = K_B_books_vec + K_B_arts_vec                  # (C,)
        new_w_cands = K_G_vec / (K_G_vec + K_B_vec + 1e-30)    # (C,)

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
    """Compute CSV columns from the converged weight vector.

    Returns a dict of stats, or None if support is below min size
    (caller checks min_final_size).
    """
    supp_mask = w > tau_supp
    supp_idxs = np.where(supp_mask)[0]
    if len(supp_idxs) == 0:
        return None

    w_supp = w[supp_idxs]
    # Posteriors for supported members: their weights ARE the posteriors
    posts = w_supp.copy()
    posts_clipped = np.clip(posts, 1e-3, 1 - 1e-3)
    logits = np.log(posts_clipped / (1 - posts_clipped))

    g_articles = [int(i) for i in supp_idxs if is_article[i]]
    g_books = [int(i) for i in supp_idxs if is_book[i]]
    g_all = supp_idxs.tolist()

    # member_weights_json: [{idx, type, weight}, ...] sorted by weight desc
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
                   help="Top-K candidates pulled from centroid neighbourhood "
                        "each iteration. Higher than Phase 3's W=50 because "
                        "no threshold-cliff — soft updates handle all candidates "
                        "gracefully.")
    p.add_argument("--tau-supp", type=float, default=0.5,
                   help="Support / hard-threshold cutoff for dedup and output.")
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
    p.add_argument("--claimed-check-every", type=int, default=5,
                   help="Check claimed-overlap every N iterations.")
    p.add_argument("--disable-fast-filters", action="store_true",
                   help="Disable seed-skip and early-abort; only exact-match "
                        "dedup catches duplicates. Slow; for testing confluence.")
    args = p.parse_args()

    print(f"Loading scope {args.scope}...")
    emb, ids, types, _titles, topic_ids, _refs = load_scope(args.scope, with_refs=False)
    n = len(ids)
    types_arr = np.array(types)
    is_book = types_arr == "book"
    is_article = types_arr == "article"
    book_global_idx = np.where(is_book)[0]
    article_global_idx = np.where(is_article)[0]
    print(f"  {n:,} works ({is_book.sum():,} books, {is_article.sum():,} articles)")

    # Unit-normalise embeddings (inner-product distance on unit sphere)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)
    emb = np.ascontiguousarray(emb)

    # Topic id from scope path
    scope_parts = args.scope.split("/")
    topic_id = scope_parts[-1] if scope_parts[-1].startswith("T") else "scope"

    print("Building indices...")
    # Full index: HNSW for fast centroid K-NN queries at scale.
    # efSearch ≥ frontier_width*4 ensures HNSW recall matches FlatL2.
    efs = max(args.frontier_width * 4 + 50, args.k_sigma + 20)
    full_index = build_index(HNSW(hnsw_m=32, hnsw_efc=400, hnsw_efs=efs), emb)
    book_emb = np.ascontiguousarray(emb[book_global_idx])
    book_index = build_index(FlatL2(), book_emb)  # small; stays Flat
    article_emb_arr = np.ascontiguousarray(emb[article_global_idx])
    article_index = build_index(FlatL2(), article_emb_arr)  # small; stays Flat

    print(f"Computing self-tuning σ (K_sigma={args.k_sigma})...")
    D_sigma, _ = full_index.search(emb, k=args.k_sigma + 1)
    sigma_sq = np.median(D_sigma[:, 1:args.k_sigma + 1], axis=1)
    sigma_sq = np.maximum(sigma_sq, 1e-8).astype(np.float32)
    sigma = np.sqrt(sigma_sq)
    print(f"  σ stats: min={sigma.min():.4f}, median={np.median(sigma):.4f}, "
          f"max={sigma.max():.4f}")

    # Precompute top-5 nearest books for every article (one batched query).
    NEAREST_BOOKS_K = 5
    K_books_eff = min(NEAREST_BOOKS_K, len(book_global_idx))
    print(f"Precomputing top-{K_books_eff} nearest books for all "
          f"{len(article_global_idx):,} articles...")
    article_emb_for_book_q = np.ascontiguousarray(emb[article_global_idx])
    _, I_b_all_local = book_index.search(article_emb_for_book_q, k=K_books_eff)
    # Map to global indices; non-article rows stay -1.
    article_to_nearest_books_global = np.full((n, K_books_eff), -1, dtype=np.int64)
    article_to_nearest_books_global[article_global_idx] = book_global_idx[I_b_all_local]

    # Shared scratch boolean lookup array (mutated inside expand_soft)
    in_G_lookup = np.zeros(n, dtype=bool)

    seeds = article_global_idx[::args.seed_stride]
    print(f"\nIterating {len(seeds):,} article seeds (stride {args.seed_stride})...")

    t_start = time.time()
    results: list[dict] = []
    claimed: set[int] = set()
    gap_hashes: dict[frozenset, dict] = {}  # frozenset → metadata
    iter_counts: list[int] = []
    n_skipped_claimed = 0
    n_aborted_overlap = 0
    n_below_min_size = 0
    n_dup = 0

    for seed_pos, seed in enumerate(seeds):
        # Progress every 1000 seeds (before any short-circuits so counters are accurate)
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
        w = expand_soft(
            seed_idx=s,
            emb=emb,
            types_arr=types_arr,
            sigma=sigma,
            full_index=full_index,
            book_index=book_index,
            book_global_idx=book_global_idx,
            article_global_idx=article_global_idx,
            article_to_nearest_books_global=article_to_nearest_books_global,
            article_index=article_index,
            n=n,
            tau_supp=args.tau_supp,
            neighborhood_K=args.frontier_width,
            max_iters=args.max_iters,
            conv_eps=args.conv_eps,
            claimed=claimed_arg,
            claimed_overlap_threshold=args.claimed_overlap_threshold,
            claimed_check_every=args.claimed_check_every,
            in_G_lookup=in_G_lookup,
        )

        if w is None:
            # Claimed-overlap abort
            n_aborted_overlap += 1
            continue

        stats = _compute_csv_stats(w, args.tau_supp, is_article, is_book)
        if stats is None or stats["total_size"] < args.min_final_size:
            n_below_min_size += 1
            continue

        iter_counts.append(1)  # expand_soft doesn't return iter count; placeholder

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
            "seed_types": gap_hashes[members_set]["seed_types"],  # live ref
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

    # Sort by logit_sum descending for display and output
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
            "final_size",            # backwards-compat alias for total_size
            "logit_sum", "mean_post", "min_post",
            "article_indices_json",  # article subset (backwards-compat)
            "member_indices_json",   # full typed member list
            "member_weights_json",   # NEW: sparse weights for support members
        ])
        for i, r in enumerate(results):
            wr.writerow([
                i + 1,
                r["first_seed"],
                r["first_seed_type"],
                r["n_articles"],
                r["n_books"],
                r["total_size"],
                int(r["n_books"] > 0),
                "|".join(sorted(r["seed_types"])),
                r["total_size"],     # final_size mirror
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
