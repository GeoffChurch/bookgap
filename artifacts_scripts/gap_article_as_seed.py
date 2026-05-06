"""Article-as-seed gap discovery via moving-centroid expansion.

REFERENCE IMPLEMENTATION for seeded-expansion gap detection. New gap
methods using this pattern (single-seed-at-a-time expansion with cohesion
boundary criterion + exact-match dedup) should copy from this file. See
``docs/gap-method-development.md`` for conventions on CSV schemas,
naming, progress reporting, HNSW backend defaults, and validation order.

Iterates *every article* in scope as a candidate seed (instead of using
the K-ball CSV's 200 seeds). Maintains a "claimed" set of articles
already in some discovered gap; seeds inside claimed regions are
skipped, and seeds whose expansion drifts into a claimed region are
aborted early. This exploits the confluence property of moving-centroid
expansion: many seeds converge to the same final gap, so most articles
land in already-discovered basins and are cheap to reject.

Inner loop is strict moving-centroid (single candidate per iter,
accept iff min_post ≥ τ; matches gap_kball_moving_centroid --strict).

Outputs the discovered set of distinct cohesive gaps. Gaps that K-ball's
greedy missed (low-density regions, tiny clusters) become reachable.

Usage::

    python -m artifacts_scripts.gap_article_as_seed \\
        --scope 17/1702/T11303
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from artifacts_scripts.hdbscan_gap_probe import load_scope
from bookgap.index import build_index, FlatL2, HNSW


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
    p.add_argument("--k-sigma", type=int, default=7)
    p.add_argument("--k-query", type=int, default=1000)
    p.add_argument("--min-post-threshold", type=float, default=0.5)
    p.add_argument("--max-iters", type=int, default=500)
    p.add_argument("--min-final-size", type=int, default=5)
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--dedup-jaccard", type=float, default=0.5)
    p.add_argument("--claimed-overlap-threshold", type=float, default=0.5,
                   help="abort expansion when |G ∩ claimed| / |G| exceeds "
                        "this fraction — seed has converged to a known basin")
    p.add_argument("--claimed-check-every", type=int, default=5,
                   help="check claimed-overlap every N expansion steps")
    p.add_argument("--seed-stride", type=int, default=1,
                   help="iterate every Nth article (1 = every article); "
                        "useful for sub-sampling on large scopes")
    p.add_argument("--disable-fast-filters", action="store_true",
                   help="disable seed-skip and early-abort; only exact-match "
                        "dedup catches duplicates. Slow; for testing confluence.")
    p.add_argument("--output-dir", default="artifacts/gap_article_as_seed")
    args = p.parse_args()

    print(f"Loading scope {args.scope}...")
    emb, ids, types, _t, topic_ids, _refs = load_scope(args.scope, with_refs=False)
    n = len(ids)
    types_arr = np.array(types)
    is_book = types_arr == "book"
    is_article = types_arr == "article"
    book_global_idx = np.where(is_book)[0]
    article_global_idx = np.where(is_article)[0]
    print(f"  {n:,} works ({is_book.sum():,} books, {is_article.sum():,} articles)")

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)
    emb = np.ascontiguousarray(emb)

    print(f"Building indexes...")
    # full_index uses HNSW for fast K-NN at scale; book and article
    # indices stay Flat (smaller, fine for batched calls). efSearch must
    # be ≥ K_query so HNSW returns enough neighbors to match Flat.
    efs = max(args.k_query + 50, args.k_sigma + 20)
    full_index = build_index(HNSW(hnsw_m=32, hnsw_efc=400, hnsw_efs=efs), emb)
    book_emb = np.ascontiguousarray(emb[book_global_idx])
    book_index = build_index(FlatL2(), book_emb)
    article_emb_arr = np.ascontiguousarray(emb[article_global_idx])
    article_index = build_index(FlatL2(), article_emb_arr)

    print(f"Computing self-tuning σ for all points (K_sigma={args.k_sigma})...")
    D_sigma, _ = full_index.search(emb, k=args.k_sigma + 1)
    sigma_sq = np.median(D_sigma[:, 1:args.k_sigma + 1], axis=1)
    sigma_sq = np.maximum(sigma_sq, 1e-8).astype(np.float32)
    sigma = np.sqrt(sigma_sq)
    print(f"  σ stats: min={sigma.min():.4f}, median={np.median(sigma):.4f}, "
          f"max={sigma.max():.4f}")

    # Topic id derived from scope, since we're not using the K-ball CSV.
    scope_parts = args.scope.split("/")
    topic_id = scope_parts[-1] if scope_parts[-1].startswith("T") else "scope"

    K_query = min(args.k_query, n - 1)
    # Per-book article neighborhood prefetch budget (size-of-G + buffer)
    K_per_book_default = min(500, len(article_global_idx))

    # Precompute 1-NN book for ALL articles in scope (one batched query).
    # We index by global article position; the lookup is cheap thereafter.
    print(f"Precomputing 1-NN book for all {len(article_global_idx):,} articles...")
    article_emb_for_book_q = np.ascontiguousarray(emb[article_global_idx])
    _, I_b_all_local = book_index.search(article_emb_for_book_q, k=1)
    article_to_nearest_book_global = np.full(n, -1, dtype=np.int64)
    article_to_nearest_book_global[article_global_idx] = book_global_idx[I_b_all_local[:, 0]]

    # Cache K-NN results per article-index so we don't requery FAISS repeatedly.
    knn_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get_knn(i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (D, I) of K_query nearest neighbors (excluding self)."""
        if i in knn_cache:
            return knn_cache[i]
        D, I = full_index.search(emb[i:i+1], k=K_query + 1)
        D = D[0, 1:]
        I = I[0, 1:]
        knn_cache[i] = (D, I)
        return D, I

    # Cache per-book sorted nearest article list, expanded on demand.
    book_arts_cache: dict[int, np.ndarray] = {}

    def get_book_arts(b: int, want: int) -> np.ndarray:
        """Return sorted nearest articles (global indices) for book b."""
        if b in book_arts_cache and len(book_arts_cache[b]) >= want:
            return book_arts_cache[b]
        K = max(want, K_per_book_default)
        K = min(K, len(article_global_idx))
        _, I_a_local = article_index.search(emb[b:b+1], k=K)
        arts_global = article_global_idx[I_a_local[0]]
        book_arts_cache[b] = arts_global
        return arts_global

    in_G_lookup = np.zeros(n, dtype=bool)

    def compute_state(G_arr: np.ndarray) -> dict:
        """Score G: K_G(i), K_B(i), post(i), min_post, logit_sum, weakest."""
        if len(G_arr) < 2:
            return None
        in_G_lookup[:] = False
        in_G_lookup[G_arr] = True
        # Boundary
        b_per_member = article_to_nearest_book_global[G_arr]
        book_counts = Counter(b_per_member.tolist())
        B_books = list(book_counts.keys())
        B_articles_list: list[int] = []
        in_B_lookup = np.zeros(n, dtype=bool)
        for b in B_books:
            in_B_lookup[b] = True
            f_b = book_counts[b]
            arts = get_book_arts(b, want=f_b + 50)
            taken = 0
            for a in arts:
                if taken >= f_b:
                    break
                a_int = int(a)
                if not in_G_lookup[a_int]:
                    in_B_lookup[a_int] = True
                    B_articles_list.append(a_int)
                    taken += 1
        # K_G, K_B for each i
        K_G = np.zeros(len(G_arr), dtype=np.float64)
        K_B = np.zeros(len(G_arr), dtype=np.float64)
        for ix, i in enumerate(G_arr):
            D, I = get_knn(int(i))
            sigma_i = sigma[int(i)]
            sigma_j = sigma[I]
            w = np.exp(-D / (sigma_i * sigma_j))
            in_G_mask = in_G_lookup[I] & (I != int(i))
            in_B_mask = in_B_lookup[I]
            K_G[ix] = (w * in_G_mask).sum()
            K_B[ix] = (w * in_B_mask).sum()
        denom = K_G + K_B + 1e-30
        post = K_G / denom
        post_clip = np.clip(post, 1e-3, 1 - 1e-3)
        logits = np.log(post_clip / (1 - post_clip))
        return {
            "G": G_arr,
            "post": post,
            "min_post": float(post.min()),
            "weakest_pos": int(np.argmin(post)),
            "logit_sum": float(logits.sum()),
        }

    def find_frontier(G_arr: np.ndarray, top: int) -> list[int]:
        """Closest-to-current-centroid non-gap articles. Up to `top`."""
        # Recompute centroid as the unit-normalized mean of current G.
        centroid = emb[G_arr].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm < 1e-12:
            return []
        centroid = (centroid / norm).astype(np.float32)
        # Pull a generous K-NN around the centroid; we'll filter to
        # articles only, excluding G members.
        over_k = min(top * 8 + len(G_arr), n - 1)
        _, I_centroid = full_index.search(centroid[None, :], k=over_k)
        I_centroid = I_centroid[0]
        is_art_mask = is_article[I_centroid]
        in_g_mask = in_G_lookup[I_centroid]
        mask = is_art_mask & (~in_g_mask)
        return I_centroid[mask][:top].tolist()

    def expand_seed(seed: int, claimed: set[int]) -> dict | None:
        """Strict moving-centroid expansion from a single article seed,
        with early-abort when G drifts heavily into already-claimed areas.

        Returns the final state on convergence, or None on early-abort.
        """
        G = [seed]
        state = None
        for it in range(args.max_iters):
            # Find the single nearest candidate (centroid of current G,
            # excluding G members and books).
            in_G_lookup[:] = False
            in_G_lookup[np.asarray(G, dtype=np.int64)] = True
            cands = find_frontier(np.asarray(G, dtype=np.int64), top=1)
            if not cands:
                break
            j = cands[0]
            G_test = G + [j]
            test_state = compute_state(np.asarray(G_test, dtype=np.int64))
            if test_state is None or test_state["min_post"] < args.min_post_threshold:
                break
            G.append(j)
            state = test_state
            # Periodic claimed-overlap early-abort
            if (
                not args.disable_fast_filters
                and (it + 1) % args.claimed_check_every == 0
                and claimed
            ):
                overlap = len(set(G) & claimed)
                if overlap / len(G) >= args.claimed_overlap_threshold:
                    return None  # converged to a known basin
        if state is None or len(G) < args.min_final_size:
            return None
        return state

    seeds = article_global_idx[::args.seed_stride]
    print(f"\nIterating {len(seeds):,} article seeds (stride {args.seed_stride})...")
    t_start = time.time()
    results: list[dict] = []
    claimed: set[int] = set()
    # Exact-set dedup: hash the gap's frozen member set; O(|G|) per check.
    # Relies on confluence — moving-centroid is deterministic enough that
    # seeds in the same basin converge to identical final sets.
    gap_hashes: set[frozenset] = set()
    n_skipped_claimed = 0
    n_aborted_overlap = 0
    n_below_min_size = 0
    n_dup = 0
    for seed_idx, seed in enumerate(seeds):
        # Progress print at every 1000-seed boundary, BEFORE any short-circuits
        if (seed_idx + 1) % 1000 == 0 or seed_idx == len(seeds) - 1:
            elapsed = time.time() - t_start
            rate = (seed_idx + 1) / max(elapsed, 0.01)
            eta = (len(seeds) - seed_idx - 1) / max(rate, 0.01)
            print(f"  processed {seed_idx + 1}/{len(seeds)} seeds in {elapsed:.0f}s "
                  f"({rate:.0f}/s, eta {eta:.0f}s); "
                  f"discovered {len(results)} gaps; "
                  f"skipped-claimed {n_skipped_claimed}, "
                  f"early-abort {n_aborted_overlap}, "
                  f"too-small {n_below_min_size}, "
                  f"dup {n_dup}",
                  flush=True)
        s = int(seed)
        if not args.disable_fast_filters and s in claimed:
            n_skipped_claimed += 1
            continue
        final = expand_seed(s, claimed)
        if final is None:
            n_aborted_overlap += 1
            continue
        G = final["G"]
        if len(G) < args.min_final_size:
            n_below_min_size += 1
            continue
        members_set = frozenset(int(x) for x in G)
        if members_set in gap_hashes:
            n_dup += 1
            continue
        gap_hashes.add(members_set)
        results.append({
            "seed_article": s,
            "final_size": int(len(G)),
            "logit_sum": final["logit_sum"],
            "min_post": final["min_post"],
            "mean_post": float(final["post"].mean()),
            "members": [int(x) for x in G],
        })
        claimed |= members_set
    elapsed = time.time() - t_start
    print(f"\nDiscovery complete: {len(results)} gaps in pool ({elapsed:.0f}s wall)")
    print(f"  breakdown: skipped-claimed {n_skipped_claimed} ({n_skipped_claimed/len(seeds)*100:.0f}%), "
          f"early-abort {n_aborted_overlap}, too-small {n_below_min_size}, dup {n_dup}")

    # Sort by logit_sum for display (we already deduped during discovery
    # via the claimed-set + per-acceptance Jaccard check).
    results.sort(key=lambda r: -r["logit_sum"])

    print()
    print(f"## Top-{args.top_n} discovered gaps (by logit_sum)")
    print(f"  {'rk':>3}  {'seed':>6}  {'final':>5}  "
          f"{'logit':>7}  {'mean_p':>6}  {'min_p':>5}")
    for i, r in enumerate(results[:args.top_n]):
        print(f"  {i+1:>3}  {r['seed_article']:>6}  "
              f"{r['final_size']:>5}  {r['logit_sum']:>7.1f}  "
              f"{r['mean_post']:>6.3f}  {r['min_post']:>5.3f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{topic_id}__article_seed.csv"
    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "seed_article", "final_size",
                     "logit_sum", "mean_post", "min_post",
                     "article_indices_json"])
        for i, r in enumerate(results):
            wr.writerow([
                i + 1, r["seed_article"],
                r["final_size"],
                f"{r['logit_sum']:.3f}",
                f"{r['mean_post']:.6f}",
                f"{r['min_post']:.4f}",
                json.dumps(r["members"]),
            ])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
