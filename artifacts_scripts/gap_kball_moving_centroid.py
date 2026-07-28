"""K-ball-init + moving-centroid expansion cohesion gap finder.

Variant of gap_kball_hillclimb.py: same scoring (self-tuning Gaussian
KDE + CSLS-balanced boundary, logit_sum + min_post>=0.5 boundary), but
the frontier candidates are ranked by distance from the *current* gap
centroid (recomputed after each acceptance), not by min-distance to any
G member. This is the intermediate point on the spectrum between
gap_kball_cohesion.py (fixed-centroid radial) and gap_kball_hillclimb.py
(aspherical union-of-K-NN frontier): the centroid moves with G but the
candidate set is still ranked spherically around that moving centroid.

Optimization target:  logit_sum = Σ log(post(i) / (1 - post(i)))
Boundary constraint:  min_post(G) ≥ 0.5  (forced contraction otherwise)
Stop:                 no frontier candidate improves logit_sum without
                      violating the boundary, or we hit max_iters

Scoring uses the same self-tuning Gaussian KDE + CSLS-balanced
boundary as gap_effective_size_local / gap_kball_cohesion.

Usage::

    python -m artifacts_scripts.gap_kball_moving_centroid \\
        --scope 17/1702/T10181 \\
        --kball-csv artifacts/kball_gap_probe/17-1702-T10181__pca__drop0.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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
    p.add_argument("--k-sigma", type=int, default=7)
    p.add_argument("--k-query", type=int, default=1000,
                   help="K-NN size per article (used both for kernel mass "
                        "computation and for frontier construction)")
    p.add_argument("--frontier-top", type=int, default=30,
                   help="how many closest-to-current-centroid candidates "
                        "to evaluate per iteration")
    p.add_argument("--min-post-threshold", type=float, default=0.5)
    p.add_argument("--max-iters", type=int, default=500)
    p.add_argument("--min-final-size", type=int, default=5,
                   help="if forced shrinkage drops below this, abandon seed")
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--dedup-jaccard", type=float, default=0.5)
    p.add_argument("--output-dir", default="artifacts/gap_kball_moving_centroid")
    p.add_argument("--strict", action="store_true",
                   help="strict policy: include nearest candidate to current "
                        "centroid; stop at first min_post violation. No logit "
                        "improvement check. Matches gap_kball_cohesion's "
                        "fixed-centroid policy except the centroid drifts.")
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

    print(f"Building FlatL2 indexes...")
    full_index = build_index(FlatL2(), emb)
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

    kball = _load_kball(Path(args.kball_csv))
    topic_id = _topic_id_from_path(Path(args.kball_csv))
    print(f"  K-ball seeds: {len(kball)}; topic: {topic_id}")

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

    def hill_climb(G_init: np.ndarray) -> dict | None:
        G = list(G_init)
        state = compute_state(np.asarray(G, dtype=np.int64))
        if state is None:
            return None
        for _it in range(args.max_iters):
            # Forced shrinkage
            while state["min_post"] < args.min_post_threshold and len(G) > args.min_final_size:
                weakest = state["weakest_pos"]
                del G[weakest]
                state = compute_state(np.asarray(G, dtype=np.int64))
                if state is None:
                    return None
            if state["min_post"] < args.min_post_threshold:
                return None  # degenerate seed
            # Try expansion
            in_G_lookup[:] = False
            in_G_lookup[np.asarray(G, dtype=np.int64)] = True
            top = 1 if args.strict else args.frontier_top
            cands = find_frontier(np.asarray(G, dtype=np.int64), top=top)
            if not cands:
                break
            if args.strict:
                # Single-candidate (nearest to current centroid) acceptance:
                # accept iff min_post stays ≥ threshold. No logit-improvement
                # gate, matching gap_kball_cohesion.py's policy.
                j = cands[0]
                G_test = G + [j]
                test_state = compute_state(np.asarray(G_test, dtype=np.int64))
                if test_state is None or test_state["min_post"] < args.min_post_threshold:
                    break
                G.append(j)
                state = test_state
            else:
                # Lenient: among top-K candidates, accept the one that most
                # improves logit_sum without violating min_post.
                best_state = None
                best_logit = state["logit_sum"]
                best_j = None
                for j in cands:
                    G_test = G + [j]
                    test_state = compute_state(np.asarray(G_test, dtype=np.int64))
                    if test_state is None:
                        continue
                    if (test_state["min_post"] >= args.min_post_threshold
                            and test_state["logit_sum"] > best_logit):
                        best_logit = test_state["logit_sum"]
                        best_state = test_state
                        best_j = j
                if best_state is None:
                    break
                G.append(best_j)
                state = best_state
        return state

    print(f"\nHill-climbing from {len(kball)} K-ball seeds...")
    results: list[dict] = []
    for seed_idx, kg in enumerate(kball):
        members = np.asarray(kg["article_indices"], dtype=np.int64)
        if len(members) < 2:
            continue
        final = hill_climb(members)
        if final is None:
            continue
        G = final["G"]
        results.append({
            "kball_gap_id": kg["gap_id"],
            "kball_size": kg["size"],
            "final_size": int(len(G)),
            "logit_sum": final["logit_sum"],
            "min_post": final["min_post"],
            "mean_post": float(final["post"].mean()),
            "members": [int(x) for x in G],
        })
        if (seed_idx + 1) % 25 == 0:
            top_logit = max(r["logit_sum"] for r in results) if results else 0
            print(f"  processed {seed_idx + 1}/{len(kball)} seeds; "
                  f"current top logit={top_logit:.1f}; pool size={len(results)}",
                  flush=True)

    # Greedy Jaccard dedup, ranked by logit_sum
    results.sort(key=lambda r: -r["logit_sum"])
    deduped: list[dict] = []
    for r in results:
        rs = set(r["members"])
        keep = True
        for d in deduped:
            ds = set(d["members"])
            inter = len(rs & ds)
            union = len(rs) + len(ds) - inter
            if union == 0:
                continue
            if inter / union >= args.dedup_jaccard:
                keep = False
                break
        if keep:
            deduped.append(r)
    print(f"\nDedup: {len(results)} → {len(deduped)} unique gaps "
          f"(Jaccard threshold {args.dedup_jaccard})")

    print()
    print(f"## Top-{args.top_n} K-ball-init moving-centroid gaps (by logit_sum)")
    print(f"  {'rk':>3}  {'kb_id':>5}  {'kb_sz':>5}  {'final':>5}  "
          f"{'logit':>7}  {'mean_p':>6}  {'min_p':>5}  Δsize")
    for i, r in enumerate(deduped[:args.top_n]):
        delta = r["final_size"] - r["kball_size"]
        delta_str = f"{'+' if delta > 0 else ''}{delta}"
        print(f"  {i+1:>3}  {r['kball_gap_id']:>5}  {r['kball_size']:>5}  "
              f"{r['final_size']:>5}  {r['logit_sum']:>7.1f}  "
              f"{r['mean_post']:>6.3f}  {r['min_post']:>5.3f}  {delta_str}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{topic_id}__kball_moving_centroid.csv"
    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "kball_gap_id", "kball_size", "final_size",
                     "logit_sum", "mean_post", "min_post",
                     "article_indices_json"])
        for i, r in enumerate(deduped):
            wr.writerow([
                i + 1, r["kball_gap_id"], r["kball_size"],
                r["final_size"],
                f"{r['logit_sum']:.3f}",
                f"{r['mean_post']:.6f}",
                f"{r['min_post']:.4f}",
                json.dumps(r["members"]),
            ])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
