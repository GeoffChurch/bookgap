"""Probe: dump per-k score curves for selected K-ball seeds.

Same scoring as gap_kball_cohesion.py, but instead of finding the peak
and returning a single best gap, this prints the full
(k, post_min, post_mean, eff_size, logit_sum) curve for each requested
K-ball gap_id. Useful for diagnosing whether the cohesion objective has
a natural peak at the cluster's true boundary or just monotonically
ratchets up.

Usage::

    python -m artifacts_scripts.gap_kball_cohesion_curve \\
        --scope 17/1702/T10181 \\
        --kball-csv artifacts/kball_gap_probe/17-1702-T10181__pca__drop0.csv \\
        --kball-gap-ids 13,144,2,30 \\
        --k-max 500
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True)
    p.add_argument("--kball-csv", required=True)
    p.add_argument("--kball-gap-ids", required=True,
                   help="comma-separated K-ball gap IDs to curve-trace")
    p.add_argument("--k-sigma", type=int, default=7)
    p.add_argument("--k-max", type=int, default=500)
    p.add_argument("--k-query", type=int, default=2000)
    p.add_argument("--print-every", type=int, default=10,
                   help="print every Nth k value (default 10)")
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

    D_sigma, _ = full_index.search(emb, k=args.k_sigma + 1)
    sigma_sq = np.median(D_sigma[:, 1:args.k_sigma + 1], axis=1)
    sigma_sq = np.maximum(sigma_sq, 1e-8).astype(np.float32)
    sigma = np.sqrt(sigma_sq)

    kball_all = _load_kball(Path(args.kball_csv))
    wanted = {int(x) for x in args.kball_gap_ids.split(",")}
    kball = [g for g in kball_all if g["gap_id"] in wanted]
    print(f"  curve-tracing {len(kball)} of {len(kball_all)} K-ball gaps")

    K_max = args.k_max
    K_query = min(args.k_query, n - 1)
    K_per_book = min(K_max + 200, len(article_global_idx))

    in_G_lookup = np.zeros(n, dtype=bool)
    in_B_lookup = np.zeros(n, dtype=bool)

    for kg in kball:
        members = np.asarray(kg["article_indices"], dtype=np.int64)
        centroid = emb[members].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm < 1e-12:
            continue
        centroid = (centroid / norm).astype(np.float32)

        over_k = min(K_max * 4, n - 1)
        _, I_centroid = full_index.search(centroid[None, :], k=over_k)
        I_centroid = I_centroid[0]
        cand_mask = is_article[I_centroid]
        cand = I_centroid[cand_mask][:K_max]
        if len(cand) == 0:
            continue
        K = len(cand)

        cand_emb = np.ascontiguousarray(emb[cand])
        D_NN, I_NN = full_index.search(cand_emb, k=K_query + 1)
        D_NN = D_NN[:, 1:]
        I_NN = I_NN[:, 1:]
        sigma_i = sigma[cand][:, None]
        sigma_j = sigma[I_NN]
        W = np.exp(-D_NN / (sigma_i * sigma_j))

        _, I_b = book_index.search(cand_emb, k=1)
        nearest_book = book_global_idx[I_b[:, 0]]
        unique_books_in_cand = np.unique(nearest_book)
        book_emb_q = np.ascontiguousarray(emb[unique_books_in_cand])
        _, I_book_arts_local = article_index.search(book_emb_q, k=K_per_book)
        book_to_arts: dict[int, np.ndarray] = {}
        for r, b in enumerate(unique_books_in_cand):
            book_to_arts[int(b)] = article_global_idx[I_book_arts_local[r]]

        in_G_lookup[:] = False
        book_counts: Counter = Counter()

        print()
        print(f"## kball_gap_id={kg['gap_id']}, raw_size={kg['size']}")
        print(f"  {'k':>4}  {'min_post':>8}  {'mean_post':>9}  "
              f"{'eff_sz':>7}  {'logit':>7}  {'marg_post':>9}")

        prev_eff = 0.0
        for k in range(1, K + 1):
            new_a = int(cand[k - 1])
            in_G_lookup[new_a] = True
            b_new = int(nearest_book[k - 1])
            book_counts[b_new] += 1

            in_B_lookup[:] = False
            for b, f_b in book_counts.items():
                in_B_lookup[b] = True
                arts = book_to_arts[b]
                taken = 0
                for a in arts:
                    if taken >= f_b:
                        break
                    if not in_G_lookup[int(a)]:
                        in_B_lookup[int(a)] = True
                        taken += 1

            W_sub = W[:k]
            I_sub = I_NN[:k]
            self_mask = I_sub == cand[:k][:, None]
            G_mask = in_G_lookup[I_sub] & ~self_mask
            B_mask = in_B_lookup[I_sub]
            K_G = (W_sub * G_mask).sum(axis=1)
            K_B = (W_sub * B_mask).sum(axis=1)
            denom = K_G + K_B + 1e-30
            post = K_G / denom

            eff_sz = float(post.sum())
            min_p = float(post.min()) if len(post) > 0 else 0.0
            mean_p = float(post.mean()) if len(post) > 0 else 0.0
            post_clip = np.clip(post, 1e-3, 1 - 1e-3)
            logits = np.log(post_clip / (1 - post_clip))
            logit_sum = float(logits.sum())
            marg_post = float(post[-1])

            if k % args.print_every == 0 or k == 1 or k == K:
                print(f"  {k:>4}  {min_p:>8.3f}  {mean_p:>9.3f}  "
                      f"{eff_sz:>7.1f}  {logit_sum:>7.1f}  {marg_post:>9.3f}")
            prev_eff = eff_sz

        # Reset for next seed
        for j in cand[:K]:
            in_G_lookup[int(j)] = False


if __name__ == "__main__":
    main()
