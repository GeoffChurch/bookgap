"""K-ball-seeded cohesion-expansion gap finder.

Two-stage gap selection:
1. K-ball outputs candidate gap centers (already done; consumed via the
   K-ball CSV as a list of (centroid, members) tuples).
2. For each K-ball centroid, sweep over K-NN radii and pick the radius
   that maximizes a logit-sum cohesion score. Yields the cohesion-
   optimized gap centered at that K-ball seed.

Replaces K-ball's rigid `radius_fraction × dist-to-nearest-book` rule
with a cohesion-aware radius. The seed (centroid) is set by K-ball;
only the radius is reoptimized.

Why logit-sum, not effective_size_local: effective_size_local =
Σ post(i) is monotone non-decreasing in |G|, so it has no natural peak
and just ratchets up to K_max. The logit-sum score Σ log(post/(1-post))
is positive for post>0.5 and negative for post<0.5, so adding a leaky
article (post<0.5) actively decreases the score. The peak identifies
the "tight core" boundary — the largest k for which the marginal
article is still confidently in-gap.

Scoring uses local-bg effective_size (same as gap_effective_size_local):
    σ_i = sqrt(median squared dist to K_sigma-NN)
    K(x_i, x_j) = exp(-d²_{ij} / (σ_i σ_j))
    For G with each article's 1-NN book b_i:
        f(b) = |{i ∈ G : b_i = b}|
        A(b) = b's f(b) nearest articles (excluding G)
        B = {b_i} ∪ ⋃ A(b)
        post(i) = K_{G\\{i}}(i) / (K_{G\\{i}}(i) + K_B(i))
    effective_size(G) = Σ post(i)

Implementation: per seed, precompute the K_query nearest neighbors and
kernel weights for each of the top K_max candidate articles ranked by
distance from centroid. Then sweep k = 1..K_max; at each step the score
is a vectorized mask-and-sum over precomputed data.

Usage::

    python -m artifacts_scripts.gap_kball_cohesion \\
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
    p.add_argument("--k-max", type=int, default=200,
                   help="max gap size to consider (sweep depth)")
    p.add_argument("--k-query", type=int, default=1000,
                   help="K-NN size for kernel-mass capture per gap article")
    p.add_argument("--min-post-threshold", type=float, default=0.5,
                   help="include articles in distance order until min(post) over G "
                        "drops below this threshold; that's the gap's natural boundary")
    p.add_argument("--warmup-k", type=int, default=5,
                   help="don't apply the min-post stop criterion until k >= this "
                        "(small k is degenerate because K_G is tiny)")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--dedup-jaccard", type=float, default=0.5,
                   help="dedup gaps with Jaccard overlap above this")
    p.add_argument("--output-dir", default="artifacts/gap_kball_cohesion")
    args = p.parse_args()

    print(f"Loading scope {args.scope}...")
    emb, ids, types, titles, topic_ids, _refs = load_scope(args.scope, with_refs=False)
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
    article_emb = np.ascontiguousarray(emb[article_global_idx])
    article_index = build_index(FlatL2(), article_emb)

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

    K_max = args.k_max
    K_query = min(args.k_query, n - 1)
    K_per_book = min(K_max + 200, len(article_global_idx))

    in_G_lookup = np.zeros(n, dtype=bool)
    in_B_lookup = np.zeros(n, dtype=bool)

    def process_seed(kg: dict) -> dict | None:
        members = np.asarray(kg["article_indices"], dtype=np.int64)
        # Centroid on the unit sphere
        centroid = emb[members].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm < 1e-12:
            return None
        centroid = (centroid / norm).astype(np.float32)

        # Step 1: K_max nearest articles to centroid (in distance order).
        # Over-query because we'll filter out books.
        over_k = min(K_max * 4, n - 1)
        _, I_centroid = full_index.search(centroid[None, :], k=over_k)
        I_centroid = I_centroid[0]
        cand_mask = is_article[I_centroid]
        cand = I_centroid[cand_mask][:K_max]
        if len(cand) == 0:
            return None
        K = len(cand)

        # Step 2: precompute K-NN and kernel weights for each candidate
        cand_emb = np.ascontiguousarray(emb[cand])
        D_NN, I_NN = full_index.search(cand_emb, k=K_query + 1)
        # Drop self
        D_NN = D_NN[:, 1:]
        I_NN = I_NN[:, 1:]
        sigma_i = sigma[cand][:, None]
        sigma_j = sigma[I_NN]
        W = np.exp(-D_NN / (sigma_i * sigma_j))  # (K, K_query)

        # Step 3: 1-NN book per candidate
        _, I_b = book_index.search(cand_emb, k=1)
        nearest_book = book_global_idx[I_b[:, 0]]

        # Step 4: precompute sorted nearest-article list per unique book
        unique_books_in_cand = np.unique(nearest_book)
        book_emb_q = np.ascontiguousarray(emb[unique_books_in_cand])
        _, I_book_arts_local = article_index.search(book_emb_q, k=K_per_book)
        book_to_arts: dict[int, np.ndarray] = {}
        for r, b in enumerate(unique_books_in_cand):
            book_to_arts[int(b)] = article_global_idx[I_book_arts_local[r]]

        # Step 5: sweep k = 1..K, stopping when min(post) drops below threshold.
        # Track the last "good" k where min_post still exceeded the threshold;
        # that k is the gap's natural boundary.
        best_logit = 0.0
        best_eff_size = 0.0
        best_min_post = 0.0
        best_k = 1
        best_G = cand[:1].copy()

        # In-G lookup, kept up to date as we sweep.
        in_G_lookup[:] = False

        # Track current B_books and f(b) and a cursor for each book's article list.
        book_counts: Counter = Counter()
        book_cursor: dict[int, int] = {}
        # We'll reset in_B_lookup per step from scratch (cheap: small B).

        for k in range(1, K + 1):
            new_a = int(cand[k - 1])
            in_G_lookup[new_a] = True
            b_new = int(nearest_book[k - 1])
            book_counts[b_new] += 1
            book_cursor.setdefault(b_new, 0)

            # Rebuild B_articles from scratch (fast: small set)
            in_B_lookup[:] = False
            for b, f_b in book_counts.items():
                in_B_lookup[b] = True  # the book itself
                arts = book_to_arts[b]
                taken = 0
                for a in arts:
                    if taken >= f_b:
                        break
                    if not in_G_lookup[int(a)]:
                        in_B_lookup[int(a)] = True
                        taken += 1

            # Score: vectorized over G's first k rows
            W_sub = W[:k]
            I_sub = I_NN[:k]
            self_mask = I_sub == cand[:k][:, None]
            G_mask = in_G_lookup[I_sub] & ~self_mask
            B_mask = in_B_lookup[I_sub]
            K_G = (W_sub * G_mask).sum(axis=1)
            K_B = (W_sub * B_mask).sum(axis=1)
            denom = K_G + K_B + 1e-30
            post = K_G / denom
            min_post = float(post.min()) if len(post) > 0 else 0.0
            post_clip = np.clip(post, 1e-3, 1 - 1e-3)
            logits = np.log(post_clip / (1 - post_clip))
            score = float(logits.sum())
            eff_size = float(post.sum())

            # Stopping rule: once we're past warmup, if min_post ever drops
            # below the threshold, we've passed the gap's natural boundary —
            # roll back to the last good k and stop.
            if k >= args.warmup_k and min_post < args.min_post_threshold:
                break
            best_logit = score
            best_eff_size = eff_size
            best_min_post = min_post
            best_k = k
            best_G = cand[:k].copy()

        # Reset in_G_lookup so the next seed starts clean
        for j in cand[:k]:
            in_G_lookup[int(j)] = False

        return {
            "kball_gap_id": kg["gap_id"],
            "kball_size": kg["size"],
            "best_size": int(best_k),
            "best_logit_score": best_logit,
            "best_effective_size": best_eff_size,
            "best_min_post": best_min_post,
            "members": best_G.tolist(),
        }

    print(f"\nProcessing {len(kball)} K-ball seeds (K_max={K_max}, K_query={K_query})...")
    results: list[dict] = []
    for seed_idx, kg in enumerate(kball):
        r = process_seed(kg)
        if r is None:
            continue
        results.append(r)
        if (seed_idx + 1) % 25 == 0:
            top = max(r["best_logit_score"] for r in results)
            print(f"  scored {seed_idx + 1}/{len(kball)} seeds; current top logit={top:.1f}")

    # Greedy Jaccard dedup, sorted by logit score
    results.sort(key=lambda r: -r["best_logit_score"])
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
            jac = inter / union
            if jac >= args.dedup_jaccard:
                keep = False
                break
        if keep:
            deduped.append(r)

    print(f"\nDedup: {len(results)} → {len(deduped)} unique gaps "
          f"(Jaccard threshold {args.dedup_jaccard})")

    print()
    print(f"## Top-{args.top_n} cohesion-optimized gaps (K-ball seeded, "
          f"min-post stop @ {args.min_post_threshold})")
    print(f"  {'rk':>3}  {'kb_id':>5}  {'kb_sz':>5}  {'opt_sz':>6}  "
          f"{'logit':>7}  {'eff_sz':>7}  {'purity':>7}  {'min_p':>5}  Δsize")
    for i, r in enumerate(deduped[:args.top_n]):
        purity = r["best_effective_size"] / max(r["best_size"], 1)
        delta = r["best_size"] - r["kball_size"]
        delta_str = f"{'+' if delta > 0 else ''}{delta}"
        print(f"  {i+1:>3}  {r['kball_gap_id']:>5}  {r['kball_size']:>5}  "
              f"{r['best_size']:>6}  {r['best_logit_score']:>7.1f}  "
              f"{r['best_effective_size']:>7.1f}  {purity:>7.3f}  "
              f"{r['best_min_post']:>5.3f}  {delta_str}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{topic_id}__kball_cohesion.csv"
    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "kball_gap_id", "kball_size", "optimized_size",
                     "logit_score", "effective_size", "mean_purity", "min_post",
                     "article_indices_json"])
        for i, r in enumerate(deduped):
            wr.writerow([
                i + 1, r["kball_gap_id"], r["kball_size"],
                r["best_size"],
                f"{r['best_logit_score']:.3f}",
                f"{r['best_effective_size']:.3f}",
                f"{r['best_effective_size'] / max(r['best_size'], 1):.6f}",
                f"{r['best_min_post']:.4f}",
                json.dumps(r["members"]),
            ])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
