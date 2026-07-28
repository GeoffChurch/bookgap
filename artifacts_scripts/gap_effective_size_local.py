"""Effective gap size via locally-bounded background (mutual-NN flavored).

Per-article KDE soft membership where the "background" is constructed
locally from the gap's own boundary books and the articles those books
claim, rather than the global corpus. Inspired by the CSLS / mutual-NN
correction for hubness: a boundary book b that is nearest to f(b) gap
articles contributes f(b) competing non-gap articles to the background,
so its bounding strength on the foreground is discounted by an equally-
sized outside neighborhood it equally well bounds.

For each gap G:
    b_i = nearest book to x_i, for i ∈ G
    B_books = {b_i : i ∈ G}                       # unique 1-NN books
    For each b ∈ B_books:
        f(b) = |{i ∈ G : b_i = b}|
        A(b) = b's f(b) nearest articles (excluding G)
    B = B_books ∪ ⋃_b A(b)                        # local background

Soft membership uses self-tuning Gaussian KDE on G ∪ B, asking:
how much kernel mass at x_i comes from G \\ {i} vs B?

    σ_i² = median squared dist to K_sigma-NN over the full scope
    K(x_i, x_j) = exp(-d²_{ij} / (σ_i σ_j))
    K_G(i) = Σ_{j ∈ G \\ {i}} K(x_i, x_j)
    K_B(i) = Σ_{j ∈ B}        K(x_i, x_j)
    post_local(i) = K_G(i) / (K_G(i) + K_B(i))
    effective_size(G) = Σ_{i ∈ G} post_local(i)

The class priors and 1/|S| normalizers cancel because B is sized
commensurately with G (total |B_articles| = |G| by construction).

Usage::

    python -m artifacts_scripts.gap_effective_size_local \\
        --scope 17/1702/T10181 \\
        --kball-csv artifacts/kball_gap_probe/17-1702-T10181__pca__drop0.csv \\
        --maximal-bookless-csv artifacts/maximal_bookless/T10181_no_denoise.csv
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


def _load_mb(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "gap_id": int(r["gap_id"]),
                "size": int(r["n_articles"]),
                "n_boundary_books": int(r["n_boundary_books"]),
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
    p.add_argument("--maximal-bookless-csv", required=True)
    p.add_argument("--k-sigma", type=int, default=7)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--output-dir", default="artifacts/gap_effective_size_local")
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

    print(f"Building FlatL2 index over {n:,} points (full)...")
    full_index = build_index(FlatL2(), emb)

    print(f"Building FlatL2 index over {len(book_global_idx):,} books...")
    book_emb = np.ascontiguousarray(emb[book_global_idx])
    book_index = build_index(FlatL2(), book_emb)

    print(f"Building FlatL2 index over {len(article_global_idx):,} articles...")
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
    mb = _load_mb(Path(args.maximal_bookless_csv))
    topic_id = _topic_id_from_path(Path(args.maximal_bookless_csv))
    kball = [r for r in kball if r["topic_id"] == topic_id]
    print(f"  K-ball: {len(kball)} gaps;  max-bookless: {len(mb)} gaps;  topic: {topic_id}")

    in_gap_lookup = np.zeros(n, dtype=bool)

    def compute(gap: dict) -> tuple[float, float, int, int]:
        G = gap["article_indices"]
        size = len(G)
        if size <= 1:
            return float(size), 1.0, 0, 0
        G_arr = np.asarray(G, dtype=np.int64)
        in_gap_lookup[:] = False
        in_gap_lookup[G_arr] = True

        # Step 1: 1-NN book per gap article. f(b) = count.
        gap_emb_q = np.ascontiguousarray(emb[G_arr])
        _, I_book_local = book_index.search(gap_emb_q, k=1)
        nearest_books_global = book_global_idx[I_book_local[:, 0]]
        book_counts = Counter(nearest_books_global.tolist())
        unique_books = np.array(sorted(book_counts.keys()), dtype=np.int64)

        # Step 2: for each unique book b, find its f(b) nearest non-gap articles.
        #         Query article_index with k = f(b) + buffer for excluding gap members.
        # We over-query then filter — the buffer accounts for any gap articles
        # that happen to be closest to b.
        max_f = max(book_counts.values())
        buffer = min(max(size, 50), len(article_global_idx))
        max_k = min(max_f + buffer, len(article_global_idx))
        b_emb_q = np.ascontiguousarray(emb[unique_books])
        _, I_art_local = article_index.search(b_emb_q, k=max_k)
        # Map article-local → global indices
        I_art_global = article_global_idx[I_art_local]  # (n_unique_books, max_k)
        # For each book b, take the first f(b) non-gap articles
        background_articles_set: set[int] = set()
        for row, b in enumerate(unique_books):
            f_b = book_counts[int(b)]
            taken = 0
            for j in I_art_global[row]:
                if taken >= f_b:
                    break
                if not in_gap_lookup[int(j)]:
                    background_articles_set.add(int(j))
                    taken += 1
        background_articles = np.array(sorted(background_articles_set), dtype=np.int64)
        background_books = unique_books
        background_set = set(background_articles.tolist()) | set(background_books.tolist())
        # Also build a boolean lookup for fast membership
        in_bg_lookup = np.zeros(n, dtype=bool)
        in_bg_lookup[background_articles] = True
        in_bg_lookup[background_books] = True

        # Step 3: per-article KDE soft membership using self-tuning Gaussian.
        # For each gap article i, compute K_G(i) over G \ {i} and K_B(i) over B.
        # Use a single full-corpus K-NN query per gap article (large enough K) and
        # filter to G/B. Keep K_query proportional to the larger of |G|, |B|.
        target = len(G_arr) + len(background_set)
        K_query = min(max(target * 3, 500), n - 1)
        D, I = full_index.search(gap_emb_q, k=K_query + 1)
        D = D[:, 1:]
        I = I[:, 1:]
        sigma_i_arr = sigma[G_arr][:, None]
        sigma_j_arr = sigma[I]
        w = np.exp(-D / (sigma_i_arr * sigma_j_arr))
        in_G = in_gap_lookup[I]
        in_B = in_bg_lookup[I]
        K_G = (w * in_G).sum(axis=1)
        K_B = (w * in_B).sum(axis=1)
        denom = K_G + K_B + 1e-30
        post = K_G / denom
        eff = float(post.sum())
        return eff, float(post.mean()), len(unique_books), len(background_set)

    print(f"\nScoring {len(mb)} mb gaps + {len(kball)} K-ball gaps with local background...")
    for g in mb:
        eff, mp, n_b_books, n_bg = compute(g)
        g["effective_size"] = eff
        g["mean_post"] = mp
        g["n_boundary_books"] = n_b_books
        g["n_background"] = n_bg
    for g in kball:
        eff, mp, n_b_books, n_bg = compute(g)
        g["effective_size"] = eff
        g["mean_post"] = mp
        g["n_boundary_books"] = n_b_books
        g["n_background"] = n_bg

    mb_by_eff = sorted(mb, key=lambda r: -r["effective_size"])
    kball_by_eff = sorted(kball, key=lambda r: -r["effective_size"])

    def _rank(lst): return {g["gap_id"]: i for i, g in enumerate(lst)}
    mb_rank_eff = _rank(mb_by_eff)
    mb_rank_raw = _rank(sorted(mb, key=lambda r: -r["size"]))
    kball_rank_eff = _rank(kball_by_eff)
    kball_rank_raw = _rank(sorted(kball, key=lambda r: -r["size"]))

    def _print_top(name, sorted_list, rank_raw):
        print()
        print(f"## Top-{args.top_n} {name} by local-bg effective size")
        print(f"  {'eff_rk':>6}  {'gap_id':>6}  {'raw_sz':>6}  {'eff_sz':>7}  "
              f"{'post':>6}  {'#bks':>4}  {'#bg':>4}  {'raw_rk':>6}  Δrank")
        for i, g in enumerate(sorted_list[:args.top_n]):
            raw_rk_1 = rank_raw[g["gap_id"]] + 1
            delta = rank_raw[g["gap_id"]] - i
            delta_str = f"{'+' if delta > 0 else ''}{delta}"
            print(f"  {i+1:>6}  {g['gap_id']:>6}  {g['size']:>6}  "
                  f"{g['effective_size']:>7.1f}  {g['mean_post']:>6.3f}  "
                  f"{g['n_boundary_books']:>4}  {g['n_background']:>4}  "
                  f"{raw_rk_1:>6}  {delta_str}")

    _print_top("K-ball", kball_by_eff, kball_rank_raw)
    _print_top("max-bookless", mb_by_eff, mb_rank_raw)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for items, name in [(mb, "mb"), (kball, "kball")]:
        rank_raw = mb_rank_raw if name == "mb" else kball_rank_raw
        rank_eff = mb_rank_eff if name == "mb" else kball_rank_eff
        path = out_dir / f"{topic_id}__{name}_local_eff.csv"
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["gap_id", "raw_size", "effective_size", "mean_post",
                         "n_boundary_books", "n_background",
                         "raw_rank", "eff_rank", "delta_rank"])
            for g in sorted(items, key=lambda r: -r["effective_size"]):
                wr.writerow([
                    g["gap_id"], g["size"],
                    f"{g['effective_size']:.3f}",
                    f"{g['mean_post']:.6f}",
                    g["n_boundary_books"], g["n_background"],
                    rank_raw[g["gap_id"]] + 1,
                    rank_eff[g["gap_id"]] + 1,
                    rank_raw[g["gap_id"]] - rank_eff[g["gap_id"]],
                ])
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
