"""Multi-scale post-hoc coverage labeling for cohesion-aware gap basins.

SUPERSEDED for new runs by ``gap_softmax_enrich.py`` (multi-cluster
softmax Phase 2), which produces inherited_coverage_median_post and
effective_uncovered_articles as direct softmax readouts (formula
differs; see the Phase 2 design doc). Retained for fallback /
regression.

Post-processes a Phase-3-enriched gap CSV (with ``member_indices_json``,
``book_anchored``, ``n_books``, ``logit_sum`` columns) and adds
*inherited coverage* labels to non-book-anchored basins.

Design rationale: ``docs/coverage_design_tradeoffs.md``, Axis II
("Membership type") and Axis IV ("Coverage labeling"). The core finding
is that a narrow non-book-anchored basin G can be *structurally inside*
a broader book-anchored basin H even when their frozensets are disjoint:
every article in G may have high posterior under H's ``compute_state``.
The Dependency Parsing example (89-article cross-lingual UD basin, rank-4
H with 64 books) shows median posterior 0.61–0.68.

Algorithm::

    1. For each book-anchored basin H, precompute its boundary structure
       (B_books + B_articles) once — NOT per article.
    2. For each non-book-anchored basin G:
         for each article a in G:
           for each candidate book-anchored basin H (top-K by centroid dist):
             post(a|H) = K_G(a,H) / (K_G(a,H) + K_B(a,H))
           track best H per article
       aggregate: median over articles of max_H post(a|H)
       label as 'inherited-coverage' if median >= threshold
    3. Add columns:
       inherited_coverage_max_basin_rank
       inherited_coverage_max_basin_books
       inherited_coverage_median_post   (continuous depth-of-containment)
       effective_uncovered_articles      (volume-aware: Σ_a w(a)·(1 - best_post),
                                          membership-marginalized expectation
                                          of "uncovered article-equivalents";
                                          sort descending for the strongest
                                          textbook-proposal candidates)
       coverage_label_v2                 (tiered)

Output label hierarchy (descending priority):
  templated-leak     (if templated_leak_flag column present and set)
  book-anchored      (n_books > 0)
  book-exists        (if book_exists_flag column present and set)
  inherited-strong   (median_post ≥ 0.75 — tightly inside a book-anchored basin)
  inherited-moderate (0.65 ≤ median_post < 0.75 — partially contained)
  inherited-weak     (0.55 ≤ median_post < 0.65 — peripheral overlap)
  real-gap           (median_post < 0.55 — minimal containment, the genuine gaps)

Empirically (T10181, 4174 non-book-anchored basins): median_post is
unimodal around 0.70 (std 0.07, min 0.53, max 0.97), so a single cutoff
is arbitrary — the tiered split is honest about the continuous nature
and lets downstream consumers pick their tolerance. The inheritance
thresholds (0.55 / 0.65 / 0.75) split T10181 roughly into 11% / 25% /
52% / 100% cumulative — well-spread tiers.

Usage::

    python -m artifacts_scripts.gap_multi_scale_coverage \\
        --scope 17/1702/T11303 \\
        --input /tmp/t11303_phase3/T11303__article_seed.csv \\
        --output /tmp/t11303_phase3/T11303__article_seed_multiscale.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Large basins produce member_indices_json fields that exceed the csv default.
csv.field_size_limit(sys.maxsize)

import numpy as np

from artifacts_scripts.hdbscan_gap_probe import load_scope
from bookgap.index import FlatL2, build_index


# ---------------------------------------------------------------------------
# Boundary precomputation helpers
# ---------------------------------------------------------------------------

def _precompute_boundary(
    H_members: np.ndarray,       # global indices (articles + books)
    is_article: np.ndarray,      # bool[n]
    article_to_nearest_books_global: np.ndarray,  # (n, K_books)
    book_global_idx: np.ndarray,
    article_index,               # FlatL2 over article embeddings
    article_global_idx: np.ndarray,
    emb: np.ndarray,
    in_G_lookup: np.ndarray,     # scratch bool[n]; will be clobbered
) -> tuple[list[int], list[int]]:
    """Return (B_books, B_articles) for a book-anchored basin H.

    B_books: boundary books (1-NN book outside H per article member).
    B_articles: CSLS-balanced f(b) articles around each boundary book.

    Mirrors compute_state's boundary logic from gap_article_as_seed.py.
    """
    n = len(emb)
    K_books = article_to_nearest_books_global.shape[1]

    in_G_lookup[:] = False
    in_G_lookup[H_members] = True

    # -- B_books: first not-in-H book per article member ----
    article_mask = is_article[H_members]
    article_members_arr = H_members[article_mask]
    if len(article_members_arr) == 0:
        return [], []

    cand = article_to_nearest_books_global[article_members_arr]  # (m, K)
    mask_valid = (cand != -1) & ~in_G_lookup[np.where(cand != -1, cand, 0)]
    first_valid_col = mask_valid.argmax(axis=1)
    has_any = mask_valid.any(axis=1)
    if not has_any.any():
        return [], []

    b_per_member = cand[np.arange(len(cand)), first_valid_col][has_any]
    book_counts = Counter(b_per_member.tolist())
    B_books = list(book_counts.keys())

    # -- B_articles: CSLS-balanced articles around each boundary book -----
    in_B_lookup = np.zeros(n, dtype=bool)
    B_articles: list[int] = []
    for b in B_books:
        in_B_lookup[b] = True
        f_b = book_counts[b]
        K_want = max(f_b + 50, 100)
        K_want = min(K_want, len(article_global_idx))
        _, I_a_local = article_index.search(emb[b:b + 1], k=K_want)
        arts_global = article_global_idx[I_a_local[0]]
        taken = 0
        for a_g in arts_global:
            if taken >= f_b:
                break
            a_int = int(a_g)
            if not in_G_lookup[a_int] and not in_B_lookup[a_int]:
                in_B_lookup[a_int] = True
                B_articles.append(a_int)
                taken += 1

    return B_books, B_articles


# ---------------------------------------------------------------------------
# Kernel computation
# ---------------------------------------------------------------------------

def _kernel_sums(
    a: int,
    H_members: np.ndarray,       # global indices of H (articles + books)
    B_all: np.ndarray,            # global indices of boundary (B_books + B_articles)
    sigma: np.ndarray,            # self-tuning sigma per point
    emb: np.ndarray,
) -> tuple[float, float]:
    """Return (K_G_val, K_B_val) for article a against basin H.

    Uses self-tuning Gaussian kernel: w(a,x) = exp(-d(a,x)^2 / (sigma_a * sigma_x)).
    K_G = sum_{x in H} w(a,x)     (includes self if a in H; no subtraction needed
                                    for the multi-scale evaluation context — a is
                                    in a *different* basin G, so self-overlap is
                                    semantic, not numerical artefact).
    K_B = sum_{b in B} w(a,b)

    Both computed directly from embeddings; no index needed since |H| + |B|
    is typically ~200 per basin, making a direct matmul faster than FAISS.
    """
    sigma_a = float(sigma[a])
    a_emb = emb[a]  # (d,)

    # K_G
    if len(H_members) > 0:
        H_emb = emb[H_members]                     # (|H|, d)
        diff_G = H_emb - a_emb                     # (|H|, d)
        d_G = (diff_G ** 2).sum(axis=1)            # (|H|,)
        sigma_H = sigma[H_members]                  # (|H|,)
        w_G = np.exp(-d_G / (sigma_a * sigma_H + 1e-30))
        # exclude self-contribution (a == H member)
        self_mask = H_members == a
        w_G[self_mask] = 0.0
        K_G_val = float(w_G.sum())
    else:
        K_G_val = 0.0

    # K_B
    if len(B_all) > 0:
        B_emb = emb[B_all]                         # (|B|, d)
        diff_B = B_emb - a_emb                     # (|B|, d)
        d_B = (diff_B ** 2).sum(axis=1)            # (|B|,)
        sigma_B = sigma[B_all]                      # (|B|,)
        w_B = np.exp(-d_B / (sigma_a * sigma_B + 1e-30))
        K_B_val = float(w_B.sum())
    else:
        K_B_val = 0.0

    return K_G_val, K_B_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True,
                   help="e.g. 17/1702/T11303")
    p.add_argument("--input", required=True,
                   help="Phase-3 gap CSV (with member_indices_json, "
                        "n_books, book_anchored, logit_sum)")
    p.add_argument("--output", required=True,
                   help="Enriched CSV output path")
    p.add_argument("--real-gap-threshold", type=float, default=0.55,
                   help="Median posterior below this is labeled 'real-gap'. "
                        "Above this is split into inherited-weak/moderate/strong "
                        "tiers (cutoffs 0.65 / 0.75). Default 0.55 — matches the "
                        "T10181 distribution where < 0.55 covers ~2%% of basins.")
    p.add_argument("--k-sigma", type=int, default=7,
                   help="Self-tuning sigma: K nearest neighbors (default 7)")
    p.add_argument("--top-k-basins", type=int, default=10,
                   help="Per article, only evaluate against the K nearest "
                        "book-anchored basins by centroid distance (default 10). "
                        "Set 0 to evaluate all.")
    args = p.parse_args()

    t0 = time.time()
    print(f"Loading scope {args.scope}...", flush=True)
    emb, ids, types, _titles, _topic_ids, _refs = load_scope(args.scope, with_refs=False)
    n = len(ids)
    types_arr = np.array(types)
    is_article = types_arr == "article"
    is_book = types_arr == "book"
    book_global_idx = np.where(is_book)[0]
    article_global_idx = np.where(is_article)[0]
    print(f"  {n:,} works ({is_book.sum():,} books, {is_article.sum():,} articles)",
          flush=True)

    # L2-normalise embeddings (matches gap_article_as_seed.py convention)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)
    emb = np.ascontiguousarray(emb)

    print("Building indexes...", flush=True)
    full_index = build_index(FlatL2(), emb)
    book_emb = np.ascontiguousarray(emb[book_global_idx])
    book_index = build_index(FlatL2(), book_emb)
    article_emb = np.ascontiguousarray(emb[article_global_idx])
    article_index = build_index(FlatL2(), article_emb)

    print(f"Computing self-tuning sigma (K_sigma={args.k_sigma})...", flush=True)
    K_sigma_eff = min(args.k_sigma, n - 1)
    D_sigma, _ = full_index.search(emb, k=K_sigma_eff + 1)
    sigma_sq = np.median(D_sigma[:, 1:K_sigma_eff + 1], axis=1)
    sigma = np.sqrt(np.maximum(sigma_sq, 1e-8)).astype(np.float32)
    print(f"  sigma: min={sigma.min():.4f} median={np.median(sigma):.4f} "
          f"max={sigma.max():.4f}", flush=True)

    # Precompute top-K nearest books for all articles (vectorized)
    NEAREST_BOOKS_K = 5
    K_books_eff = min(NEAREST_BOOKS_K, len(book_global_idx))
    if K_books_eff > 0:
        print(f"Precomputing top-{K_books_eff} nearest books for all articles...",
              flush=True)
        _, I_b_all_local = book_index.search(article_emb, k=K_books_eff)
        article_to_nearest_books_global = np.full((n, K_books_eff), -1, dtype=np.int64)
        article_to_nearest_books_global[article_global_idx] = book_global_idx[I_b_all_local]
    else:
        article_to_nearest_books_global = np.full((n, 1), -1, dtype=np.int64)

    # Load gap CSV
    print(f"Reading {args.input}...", flush=True)
    with open(args.input) as f:
        reader = csv.DictReader(f)
        in_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    print(f"  {len(rows):,} gaps", flush=True)

    # Parse member lists
    def parse_members(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (all_members_arr, article_members_arr, article_weights_arr).

        ``article_weights_arr`` is parallel to ``article_members_arr`` and
        contains soft-EM weights when ``member_weights_json`` is present
        (gap_soft_em.py output) or 1.0 for every article in the
        hard-membership case (gap_article_as_seed.py output). With
        ``w(a) = 1`` for all hard members, the weighted-sum metric collapses
        to the unweighted sum — Option B reduces to Option A as a special
        case.
        """
        raw = json.loads(row["member_indices_json"])
        weights_raw = row.get("member_weights_json")
        weights_lookup: dict[int, float] = {}
        if weights_raw:
            for m in json.loads(weights_raw):
                weights_lookup[int(m["idx"])] = float(m.get("weight", 1.0))

        if raw and isinstance(raw[0], dict):
            all_m = np.array([m["idx"] for m in raw], dtype=np.int64)
            art_idx_list = [m["idx"] for m in raw if m.get("type") == "article"]
            art_m = np.array(art_idx_list, dtype=np.int64)
            if weights_lookup:
                art_w = np.array([weights_lookup.get(int(i), 1.0) for i in art_m],
                                 dtype=np.float64)
            else:
                art_w = np.ones(len(art_m), dtype=np.float64)
        else:
            all_m = np.array(raw, dtype=np.int64)
            art_m = all_m[is_article[all_m]]
            art_w = np.ones(len(art_m), dtype=np.float64)
        return all_m, art_m, art_w

    # Partition into book-anchored and non-book-anchored
    book_anchored_rows: list[tuple[int, dict]] = []   # (original_row_idx, row)
    non_book_anchored_rows: list[tuple[int, dict]] = []

    for i, row in enumerate(rows):
        if int(row.get("n_books", 0)) > 0:
            book_anchored_rows.append((i, row))
        else:
            non_book_anchored_rows.append((i, row))

    print(f"  Book-anchored basins: {len(book_anchored_rows):,}", flush=True)
    print(f"  Non-book-anchored basins: {len(non_book_anchored_rows):,}", flush=True)

    # ------------------------------------------------------------------
    # Phase 1: precompute boundary structures for all book-anchored basins
    # ------------------------------------------------------------------
    print("\nPrecomputing boundary structures for book-anchored basins...", flush=True)
    in_G_scratch = np.zeros(n, dtype=bool)

    # Store per book-anchored basin:
    #   members_arr, article_members_arr, B_books, B_articles,
    #   B_all_arr (concatenated), centroid
    anchored_basins: list[dict] = []

    for ba_idx, (row_i, row) in enumerate(book_anchored_rows):
        all_m, art_m, _art_w = parse_members(row)
        B_books, B_articles = _precompute_boundary(
            H_members=all_m,
            is_article=is_article,
            article_to_nearest_books_global=article_to_nearest_books_global,
            book_global_idx=book_global_idx,
            article_index=article_index,
            article_global_idx=article_global_idx,
            emb=emb,
            in_G_lookup=in_G_scratch,
        )
        B_all = np.array(B_books + B_articles, dtype=np.int64)
        # Centroid of member embeddings (for top-K pruning)
        if len(all_m) > 0:
            centroid = emb[all_m].mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroid = centroid / max(norm, 1e-12)
        else:
            centroid = np.zeros(emb.shape[1], dtype=np.float32)
        anchored_basins.append({
            "rank": int(row.get("rank", ba_idx + 1)),
            "n_books": int(row.get("n_books", 0)),
            "members_arr": all_m,
            "article_members_arr": art_m,
            "B_all": B_all,
            "centroid": centroid.astype(np.float32),
        })

    print(f"  {len(anchored_basins):,} boundary structures precomputed in "
          f"{time.time() - t0:.1f}s", flush=True)

    # Stack centroids for fast batched dot-product distance
    if anchored_basins:
        centroids_mat = np.stack([b["centroid"] for b in anchored_basins], axis=0)
        # (N_anchored, d) — unit-normalised
    else:
        centroids_mat = np.zeros((0, emb.shape[1]), dtype=np.float32)

    # ------------------------------------------------------------------
    # Phase 2: for each non-book-anchored basin G, compute inherited coverage
    # ------------------------------------------------------------------
    print(f"\nComputing inherited coverage for {len(non_book_anchored_rows):,} "
          f"non-anchored basins...", flush=True)

    N_nb = len(non_book_anchored_rows)
    N_ba = len(anchored_basins)
    top_k = args.top_k_basins if args.top_k_basins > 0 else N_ba

    # Results: indexed by original row position
    inherited_results: dict[int, dict] = {}

    t_phase2 = time.time()
    n_inherited = 0
    n_real_gap = 0

    for proc_idx, (row_i, row) in enumerate(non_book_anchored_rows):
        all_m, art_m, art_w = parse_members(row)

        if len(art_m) == 0 or N_ba == 0:
            inherited_results[row_i] = {
                "max_basin_rank": "",
                "max_basin_books": "",
                "median_post": "",
                "effective_uncovered": "",
            }
            n_real_gap += 1
            if (proc_idx + 1) % 50 == 0 or proc_idx == N_nb - 1:
                elapsed = time.time() - t_phase2
                rate = (proc_idx + 1) / max(elapsed, 0.01)
                eta = (N_nb - proc_idx - 1) / max(rate, 0.01)
                print(f"  processed {proc_idx+1}/{N_nb} non-anchored basins in "
                      f"{elapsed:.0f}s ({rate:.1f}/s, eta {eta/60:.1f}min); "
                      f"inherited {n_inherited}, real-gap {n_real_gap}",
                      flush=True)
            continue

        # Compute G centroid to rank book-anchored basins by proximity
        G_centroid = emb[art_m].mean(axis=0)
        G_norm = np.linalg.norm(G_centroid)
        G_centroid = G_centroid / max(G_norm, 1e-12)

        if N_ba > 0:
            # Dot product with unit centroids → larger = closer (cosine similarity)
            sims = centroids_mat @ G_centroid    # (N_ba,)
            # Take top-K nearest basins
            if top_k >= N_ba:
                basin_order = np.argsort(-sims)
            else:
                basin_order = np.argpartition(-sims, top_k)[:top_k]
                basin_order = basin_order[np.argsort(-sims[basin_order])]
        else:
            basin_order = np.array([], dtype=np.int64)

        # Per-article, per-basin posterior
        # best_post_per_article[i] = max over candidate basins of post(art_m[i] | H)
        best_post_per_article = np.zeros(len(art_m), dtype=np.float64)

        # Track per-basin posterior arrays to compute per-basin medians.
        # basin_posts_by_idx[ba_idx] = np.ndarray of posteriors (one per G article)
        basin_posts_by_idx: dict[int, np.ndarray] = {}

        art_emb_batch = emb[art_m]               # (|G_arts|, d) — precompute once
        sigma_a_batch = sigma[art_m].astype(np.float64)  # (|G_arts|,)

        for ba_idx in basin_order:
            H = anchored_basins[int(ba_idx)]
            H_members = H["members_arr"]
            B_all = H["B_all"]
            if len(H_members) == 0:
                continue

            # Batch kernel computation for all articles in G vs this basin H.
            # Uses matmul-based squared distances (all embeddings are unit-normalised):
            #   d^2(a, x) = 2*(1 - a·x)
            H_emb = emb[H_members]                  # (|H|, d)
            sigma_H = sigma[H_members].astype(np.float64)  # (|H|,)

            dot_GH = art_emb_batch @ H_emb.T        # (|G_arts|, |H|)
            d2_GH = np.maximum(2.0 * (1.0 - dot_GH), 0.0)
            ker_GH = np.exp(
                -d2_GH / (sigma_a_batch[:, None] * sigma_H[None, :] + 1e-30)
            )  # (|G_arts|, |H|)
            # Zero out self-contributions: a == H_member (rare cross-basin overlap)
            for local_a_idx, a_g in enumerate(art_m):
                same = H_members == int(a_g)
                if same.any():
                    ker_GH[local_a_idx, same] = 0.0
            K_G_batch = ker_GH.sum(axis=1)          # (|G_arts|,)

            # K_B
            if len(B_all) > 0:
                B_emb = emb[B_all]
                sigma_B = sigma[B_all].astype(np.float64)
                dot_GB = art_emb_batch @ B_emb.T     # (|G_arts|, |B|)
                d2_GB = np.maximum(2.0 * (1.0 - dot_GB), 0.0)
                ker_GB = np.exp(
                    -d2_GB / (sigma_a_batch[:, None] * sigma_B[None, :] + 1e-30)
                )
                K_B_batch = ker_GB.sum(axis=1)       # (|G_arts|,)
            else:
                K_B_batch = np.zeros(len(art_m), dtype=np.float64)

            denom = K_G_batch + K_B_batch + 1e-30
            post_batch = K_G_batch / denom           # (|G_arts|,)

            basin_posts_by_idx[int(ba_idx)] = post_batch

            # Update best-per-article tracker (for final aggregated median)
            improved = post_batch > best_post_per_article
            best_post_per_article[improved] = post_batch[improved]

        # Aggregate: median over articles of max-H posterior
        # (where max is taken per article across all candidate basins)
        if len(best_post_per_article) > 0:
            median_post = float(np.median(best_post_per_article))
        else:
            median_post = 0.0

        # Effective uncovered articles: Σ_a w(a, G) · (1 - best_post(a, H*)).
        # Membership-marginalized expectation of "uncovered article-equivalents."
        # With hard membership (gap_article_as_seed.py output) every w(a)=1 and
        # this collapses to Σ_a (1 - best_post). With soft-EM weights it
        # correctly handles fractional membership; it's strictly more general.
        if len(best_post_per_article) > 0:
            effective_uncovered = float(np.sum(art_w * (1.0 - best_post_per_article)))
        else:
            effective_uncovered = 0.0

        # argmax_basin: the H that achieves the highest median_post(G, H)
        # i.e. median over all articles a in G of post(a | H), for each H separately.
        if basin_posts_by_idx:
            best_ba_idx_by_median = max(
                basin_posts_by_idx,
                key=lambda idx: float(np.median(basin_posts_by_idx[idx])),
            )
            best_basin = anchored_basins[best_ba_idx_by_median]
            max_basin_rank = best_basin["rank"]
            max_basin_books = best_basin["n_books"]
        else:
            max_basin_rank = ""
            max_basin_books = ""

        # Always record the per-basin inheritance signal — tiering happens
        # at label-assignment time, not here. Only the explicit threshold
        # below decides whether the rank+books fields are populated (so a
        # basin labeled real-gap doesn't suggest a "best containing basin"
        # in the output).
        if max_basin_rank != "" and median_post >= args.real_gap_threshold:
            n_inherited += 1
            inherited_results[row_i] = {
                "max_basin_rank": max_basin_rank,
                "max_basin_books": max_basin_books,
                "median_post": f"{median_post:.4f}",
                "effective_uncovered": f"{effective_uncovered:.3f}",
            }
        else:
            n_real_gap += 1
            inherited_results[row_i] = {
                "max_basin_rank": "",
                "max_basin_books": "",
                "median_post": f"{median_post:.4f}" if median_post > 0 else "",
                "effective_uncovered": f"{effective_uncovered:.3f}" if median_post > 0 else "",
            }

        if (proc_idx + 1) % 50 == 0 or proc_idx == N_nb - 1:
            elapsed = time.time() - t_phase2
            rate = (proc_idx + 1) / max(elapsed, 0.01)
            eta = (N_nb - proc_idx - 1) / max(rate, 0.01)
            print(f"  processed {proc_idx+1}/{N_nb} non-anchored basins in "
                  f"{elapsed:.0f}s ({rate:.1f}/s, eta {eta/60:.1f}min); "
                  f"inherited {n_inherited}, real-gap {n_real_gap}",
                  flush=True)

    # ------------------------------------------------------------------
    # Phase 3: write enriched CSV
    # ------------------------------------------------------------------
    def coverage_label_v2(row: dict, inh: dict | None) -> str:
        """7-label tiered hierarchy (descending priority).

        Multi-scale inheritance is *continuous*, not bimodal — empirically
        every basin in T10181 has SOME containment in some book-anchored
        basin (median_post min observed: 0.53). A single threshold doesn't
        cut cleanly, so we tier the inheritance labels and surface the
        continuous median_post as a sortable column.
        """
        if int(row.get("templated_leak_flag", 0)):
            return "templated-leak"
        if int(row.get("n_books", 0)) > 0:
            return "book-anchored"
        if int(row.get("book_exists_flag", 0)):
            return "book-exists"
        if inh and inh.get("max_basin_rank", ""):
            mp_str = inh.get("median_post", "")
            if mp_str:
                try:
                    mp = float(mp_str)
                    if mp >= 0.75:
                        return "inherited-strong"
                    if mp >= 0.65:
                        return "inherited-moderate"
                    if mp >= 0.55:
                        return "inherited-weak"
                    # else falls through to weakest-tier (rare in practice)
                except ValueError:
                    pass
        return "real-gap"

    new_fieldnames = [
        "inherited_coverage_max_basin_rank",
        "inherited_coverage_max_basin_books",
        "inherited_coverage_median_post",
        "effective_uncovered_articles",
        "coverage_label_v2",
    ]
    out_fieldnames = in_fieldnames + [c for c in new_fieldnames if c not in in_fieldnames]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for row_i, row in enumerate(rows):
            inh = inherited_results.get(row_i)
            if inh is None:
                # Book-anchored basin: no inheritance computed
                row.update({
                    "inherited_coverage_max_basin_rank": "",
                    "inherited_coverage_max_basin_books": "",
                    "inherited_coverage_median_post": "",
                    "effective_uncovered_articles": "",
                    "coverage_label_v2": coverage_label_v2(row, None),
                })
            else:
                row.update({
                    "inherited_coverage_max_basin_rank": inh.get("max_basin_rank", ""),
                    "inherited_coverage_max_basin_books": inh.get("max_basin_books", ""),
                    "inherited_coverage_median_post": inh.get("median_post", ""),
                    "effective_uncovered_articles": inh.get("effective_uncovered", ""),
                    "coverage_label_v2": coverage_label_v2(row, inh),
                })
            writer.writerow(row)

    elapsed_total = time.time() - t0
    print(f"\nDone in {elapsed_total:.1f}s", flush=True)
    print(f"Wrote {args.output}", flush=True)

    # Summary
    with open(args.output) as f:
        out_rows = list(csv.DictReader(f))
    label_counts: Counter = Counter(r["coverage_label_v2"] for r in out_rows)
    print("\nSummary by coverage_label_v2:")
    for label in ["templated-leak", "book-anchored", "book-exists",
                  "inherited-strong", "inherited-moderate", "inherited-weak",
                  "real-gap"]:
        print(f"  {label}: {label_counts.get(label, 0):,}")
    print(f"  total: {len(out_rows):,}", flush=True)


if __name__ == "__main__":
    main()
