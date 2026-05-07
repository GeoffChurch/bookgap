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
from typing import Callable

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


def _expand_symmetric_loop(
    seed: int,
    find_frontier: Callable[[np.ndarray, int], list[int]],
    compute_state: Callable[[np.ndarray], dict | None],
    tau: float,
    frontier_width: int,
    max_iters: int,
    claimed: set[int] | None = None,
    claimed_overlap_threshold: float = 0.5,
    claimed_check_every: int = 5,
) -> tuple[dict | None, int]:
    """Symmetric add+remove fixed-point loop.

    Pure: takes pre-built ``find_frontier`` and ``compute_state``
    callables. The caller is responsible for building the indices and
    closures that compute_state/find_frontier rely on.

    On each iteration:
      ADD: any frontier candidate whose individual addition keeps
        ``min_post(G ∪ {c}) ≥ tau``.
      REMOVE: any current member whose ``post(i | G_new) < tau``.
      Stop when G is unchanged (fixed point) or shrinks to empty.

    Returns ``(final_state, iter_count)``. ``final_state`` is the
    ``compute_state`` output for the final G, or None on failure
    (didn't stabilize, shrunk to empty, no candidates, etc.).
    """
    G: set[int] = {int(seed)}
    iter_count = 0
    for it in range(max_iters):
        iter_count = it + 1
        G_arr = np.asarray(sorted(G), dtype=np.int64)
        # Singleton bootstrap: force-add nearest neighbor on iter 1
        # because compute_state requires |G| >= 2.
        if len(G_arr) < 2:
            cands = find_frontier(G_arr, 1)
            if not cands:
                return None, iter_count
            G.add(int(cands[0]))
            continue
        # ADD phase: ordered cumulative — each candidate evaluated against
        # the running G (which grows as we add). Stop on first failure;
        # later candidates are skipped (we've found this iteration's
        # acceptance frontier). The plan's "ordered, stop-when-collective-
        # would-drop" variant, replacing the batch-independent ADD that
        # over-admitted on T11303.
        cands = find_frontier(G_arr, frontier_width)
        G_candidate = set(G)
        for c in cands:
            cand_arr = np.asarray(sorted(G_candidate | {int(c)}), dtype=np.int64)
            st = compute_state(cand_arr)
            if st is None or st["min_post"] < tau:
                break
            G_candidate.add(int(c))
        G_arr_new = np.asarray(sorted(G_candidate), dtype=np.int64)
        st_new = compute_state(G_arr_new)
        if st_new is None:
            return None, iter_count
        # REMOVE phase
        post_by_idx = dict(zip(G_arr_new.tolist(), st_new["post"].tolist()))
        to_remove = {i for i, p in post_by_idx.items() if p < tau}
        G_new = G_candidate - to_remove
        if G_new == G:
            # Fixed point; recompute state on the final G to return.
            G_arr_final = np.asarray(sorted(G), dtype=np.int64)
            return compute_state(G_arr_final), iter_count
        G = G_new
        if not G:
            return None, iter_count
        # Optional claimed-overlap early-abort (matches single-add path).
        if (
            claimed is not None
            and (it + 1) % claimed_check_every == 0
            and claimed
        ):
            overlap = len(G & claimed)
            if overlap / max(len(G), 1) >= claimed_overlap_threshold:
                return None, iter_count
    return None, iter_count  # didn't stabilize within max_iters


def _expand_symmetric(
    seed: int,
    emb: np.ndarray,
    types: np.ndarray,
    tau: float = 0.5,
    frontier_width: int = 50,
    max_iters: int = 500,
    k_sigma: int = 7,
) -> dict | None:
    """Standalone symmetric expansion for unit testing. Builds its own
    indices/sigma from (emb, types) and calls ``_expand_symmetric_loop``.

    Returns ``{"members_frozen": frozenset[int], "min_post": float,
    "logit_sum": float, "iter_count": int}`` on convergence; None on
    failure.
    """
    from bookgap.index import FlatL2, build_index
    from collections import Counter

    n = len(emb)
    types_arr = np.asarray(types)
    is_book_arr = types_arr == "book"
    is_article_arr = types_arr == "article"
    book_global = np.where(is_book_arr)[0]

    full_index = build_index(FlatL2(), emb)
    if len(book_global) == 0:
        # No books → boundary undefined; can't run the cohesion metric.
        return None
    book_emb = np.ascontiguousarray(emb[book_global])
    book_index = build_index(FlatL2(), book_emb)

    K_sigma_eff = min(k_sigma, n - 1)
    D_sigma, _ = full_index.search(emb, k=K_sigma_eff + 1)
    sigma = np.sqrt(np.maximum(
        np.median(D_sigma[:, 1:K_sigma_eff + 1], axis=1), 1e-8
    )).astype(np.float32)

    K_query = min(frontier_width * 4 + 50, n - 1)

    def get_knn(i: int) -> tuple[np.ndarray, np.ndarray]:
        D, I = full_index.search(emb[i:i + 1], k=K_query + 1)
        return D[0, 1:], I[0, 1:]

    def compute_state(G_arr: np.ndarray) -> dict | None:
        if len(G_arr) < 2:
            return None
        in_G = np.zeros(n, dtype=bool)
        in_G[G_arr] = True
        article_members = G_arr[is_article_arr[G_arr]]
        if len(article_members) == 0:
            return None
        b_per_member: list[int] = []
        K_books = min(5, len(book_global))
        for a in article_members:
            _, I_b = book_index.search(emb[a:a + 1], k=K_books)
            for j_local in I_b[0]:
                gb = int(book_global[j_local])
                if not in_G[gb]:
                    b_per_member.append(gb)
                    break
        if not b_per_member:
            return None
        book_counts = Counter(b_per_member)
        in_B = np.zeros(n, dtype=bool)
        for b, f_b in book_counts.items():
            in_B[b] = True
            _, I_a = full_index.search(emb[b:b + 1], k=f_b + len(G_arr) + 5)
            taken = 0
            for ia in I_a[0]:
                ia_int = int(ia)
                if taken >= f_b:
                    break
                if in_G[ia_int] or in_B[ia_int] or not is_article_arr[ia_int]:
                    continue
                in_B[ia_int] = True
                taken += 1
        K_G = np.zeros(len(G_arr), dtype=np.float64)
        K_B = np.zeros(len(G_arr), dtype=np.float64)
        for ix, i in enumerate(G_arr):
            D, I = get_knn(int(i))
            sigma_i = sigma[int(i)]
            sigma_j = sigma[I]
            w = np.exp(-D / (sigma_i * sigma_j))
            in_G_mask = in_G[I] & (I != int(i))
            in_B_mask = in_B[I]
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
            "logit_sum": float(logits.sum()),
        }

    def find_frontier(G_arr: np.ndarray, top: int) -> list[int]:
        centroid = emb[G_arr].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm < 1e-12:
            return []
        centroid = (centroid / norm).astype(np.float32)
        over_k = min(top * 4 + len(G_arr) + 5, n - 1)
        _, I = full_index.search(centroid[None, :], k=over_k)
        in_G = np.zeros(n, dtype=bool)
        in_G[G_arr] = True
        return [int(i) for i in I[0] if not in_G[i]][:top]

    final_state, iter_count = _expand_symmetric_loop(
        seed, find_frontier, compute_state,
        tau=tau, frontier_width=frontier_width, max_iters=max_iters,
    )
    if final_state is None:
        return None
    G_arr = final_state["G"]
    return {
        "members_frozen": frozenset(int(x) for x in G_arr),
        "min_post": final_state["min_post"],
        "logit_sum": final_state["logit_sum"],
        "iter_count": iter_count,
    }


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
    p.add_argument("--batch-addition", action="store_true",
                   help="use symmetric add+remove fixed-point inner loop instead of "
                        "the default strict-single-add. Subsumes batch-addition + "
                        "canonicalization; see docs/cohesion_aware_gap_quality.md Phase 2.")
    p.add_argument("--batch-frontier-width", type=int, default=50,
                   help="number of candidates per iteration when --batch-addition is set "
                        "(ignored otherwise). Default 50; tune based on observed iter counts.")
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

    # Precompute top-K nearest books for ALL articles in scope (one batched query).
    # Phase 3's compute_state filters to the first not-in-G book; K=5 is comfortably
    # above the number of books any article ever has in G in practice.
    NEAREST_BOOKS_K = 5
    K_books_eff = min(NEAREST_BOOKS_K, len(book_global_idx))
    print(f"Precomputing top-{K_books_eff} nearest books for all "
          f"{len(article_global_idx):,} articles...")
    article_emb_for_book_q = np.ascontiguousarray(emb[article_global_idx])
    _, I_b_all_local = book_index.search(article_emb_for_book_q, k=K_books_eff)
    article_to_nearest_books_global = np.full((n, K_books_eff), -1, dtype=np.int64)
    article_to_nearest_books_global[article_global_idx] = book_global_idx[I_b_all_local]

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
        # Boundary: for each article member, pick the first not-in-G book
        # from its precomputed top-K list. Vectorized: build a (m, K) mask
        # of valid book candidates (not -1 sentinel, not in G), argmax to
        # find the first valid column per row, gather. Books inside G are
        # interior and don't count as boundary anchors.
        article_mask = is_article[G_arr]
        article_members_arr = G_arr[article_mask]
        if len(article_members_arr) == 0:
            return None  # G has no article members — boundary undefined
        cand = article_to_nearest_books_global[article_members_arr]  # (m, K)
        # Valid := not the -1 sentinel AND not in G. The -1 sentinel filter
        # is load-bearing — without it, in_G_lookup[-1] would be read (the
        # last work's slot) and could spuriously mark padding as in-G.
        mask_valid = (cand != -1) & ~in_G_lookup[np.where(cand != -1, cand, 0)]
        first_valid_col = mask_valid.argmax(axis=1)
        has_any = mask_valid.any(axis=1)
        if not has_any.any():
            return None  # all members saturated their K-NN books — degenerate
        b_per_member = cand[np.arange(len(cand)), first_valid_col][has_any]
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
        # Always sync in_G_lookup from G_arr so the symmetric path works
        # correctly even when find_frontier is called between compute_state
        # calls that may have updated in_G_lookup to a different G.
        in_G_lookup[:] = False
        in_G_lookup[G_arr] = True
        in_g_mask = in_G_lookup[I_centroid]
        mask = is_art_mask & (~in_g_mask)
        return I_centroid[mask][:top].tolist()

    def expand_seed(seed: int, claimed: set[int]) -> tuple[dict | None, int]:
        if args.batch_addition:
            # Symmetric path: reuse the shared closures find_frontier and
            # compute_state already in scope. claimed-overlap early-abort
            # is honored unless --disable-fast-filters is set.
            claimed_arg = None if args.disable_fast_filters else claimed
            return _expand_symmetric_loop(
                seed, find_frontier, compute_state,
                tau=args.min_post_threshold,
                frontier_width=args.batch_frontier_width,
                max_iters=args.max_iters,
                claimed=claimed_arg,
                claimed_overlap_threshold=args.claimed_overlap_threshold,
                claimed_check_every=args.claimed_check_every,
            )
        # Existing strict-single-add path below — keep unchanged.
        G = [seed]
        state = None
        iters_done = 0
        for it in range(args.max_iters):
            iters_done = it + 1
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
            if (
                not args.disable_fast_filters
                and (it + 1) % args.claimed_check_every == 0
                and claimed
            ):
                overlap = len(set(G) & claimed)
                if overlap / len(G) >= args.claimed_overlap_threshold:
                    return None, iters_done
        if state is None or len(G) < args.min_final_size:
            return None, iters_done
        return state, iters_done

    seeds = article_global_idx[::args.seed_stride]
    print(f"\nIterating {len(seeds):,} article seeds (stride {args.seed_stride})...")
    t_start = time.time()
    results: list[dict] = []
    claimed: set[int] = set()
    # Exact-set dedup: hash the gap's frozen member set; O(|G|) per check.
    # Relies on confluence — moving-centroid is deterministic enough that
    # seeds in the same basin converge to identical final sets.
    gap_hashes: set[frozenset] = set()
    iter_counts: list[int] = []  # iterations per converged seed
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
        final, iters_done = expand_seed(s, claimed)
        if final is None:
            n_aborted_overlap += 1
            continue
        iter_counts.append(iters_done)
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
    n_distinct_frozensets = len(gap_hashes)  # before Jaccard dedup
    print(f"\nDiscovery complete: {n_distinct_frozensets} distinct frozensets "
          f"in {elapsed:.0f}s wall")
    print(f"  breakdown: skipped-claimed {n_skipped_claimed} "
          f"({n_skipped_claimed/len(seeds)*100:.0f}%), "
          f"early-abort {n_aborted_overlap}, too-small {n_below_min_size}, "
          f"dup {n_dup}")

    if iter_counts:
        arr = np.asarray(iter_counts)
        print(f"  iterations per converged seed: mean={arr.mean():.1f} "
              f"p50={int(np.median(arr))} p99={int(np.percentile(arr, 99))} "
              f"max={int(arr.max())}")

    if args.disable_fast_filters:
        n_seeds_completed = len(iter_counts)
        confluence_rate = (1.0 - n_distinct_frozensets / max(n_seeds_completed, 1))
        print(f"  confluence rate (fraction of completed seeds collapsing "
              f"into existing frozensets): {confluence_rate*100:.1f}%")

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
