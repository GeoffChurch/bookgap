"""Inspect books associated with each algorithm's top-N gaps.

REFERENCE IMPLEMENTATION for qualitative gap inspection. New
inspection scripts (per-gap article titles + nearest-book lookup)
should copy from this file. See ``docs/gap-method-development.md``
for the inspector pattern and how to use vote concentration as a
"book exists" diagnostic.

For each top gap on each side, print article titles (sample) and
associated book titles.

- Max-bookless: "associated books" = the boundary books in the lowest
  book-containing dendrogram ancestor (already in the CSV). All printed.
- K-ball: "associated books" = the nearest book to each article in the
  gap (since K-ball's per-article radius is defined by distance to the
  k-th nearest book, k=1). Printed sorted by frequency.

Both outputs share the same scope-global indexing as ``load_scope``.

Usage::

    python -m artifacts_scripts.gap_books_inspector \\
        --scope 17/1702/T10181 \\
        --kball-csv artifacts/kball_gap_probe/17-1702-T10181__pca__drop0.csv \\
        --maximal-bookless-csv artifacts/maximal_bookless/T10181_no_denoise.csv \\
        --top-n 5
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


def _load_kball_csv(path: Path) -> list[dict]:
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


def _load_maximal_bookless_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "gap_id": int(r["gap_id"]),
                "size": int(r["n_articles"]),
                "n_boundary_books": int(r["n_boundary_books"]),
                "article_indices": json.loads(r["article_indices_json"]),
                "boundary_book_indices": json.loads(r["boundary_book_indices_json"]),
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True)
    p.add_argument("--kball-csv", required=True)
    p.add_argument("--maximal-bookless-csv", required=True)
    p.add_argument("--top-n", type=int, default=5,
                   help="number of largest gaps from each side")
    p.add_argument("--top-titles", type=int, default=8,
                   help="sample article titles to print per gap")
    p.add_argument("--top-books", type=int, default=10,
                   help="max books to print per gap (mb prints all up to this)")
    p.add_argument("--kball-gap-ids", default=None,
                   help="comma-separated K-ball gap IDs to inspect "
                        "(overrides --top-n for K-ball side)")
    p.add_argument("--mb-gap-ids", default=None,
                   help="comma-separated max-bookless gap IDs to inspect "
                        "(overrides --top-n for mb side)")
    args = p.parse_args()

    print(f"Loading scope {args.scope}...")
    emb, ids, types, titles, topic_ids, _refs = load_scope(args.scope, with_refs=False)
    n = len(ids)
    n_books = sum(t == "book" for t in types)
    n_articles = sum(t == "article" for t in types)
    print(f"  {n:,} works ({n_books:,} books, {n_articles:,} articles)")

    # L2-normalise so distances match the K-ball bypass path.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)

    types_arr = np.array(types)
    book_mask = types_arr == "book"
    book_global_idx = np.where(book_mask)[0]
    book_emb = np.ascontiguousarray(emb[book_mask])
    print(f"Building FlatL2 index over {len(book_emb):,} books...")
    book_index = build_index(FlatL2(), book_emb)

    titles_arr = np.array(titles, dtype=object)

    kball_all = _load_kball_csv(Path(args.kball_csv))
    mb_all = _load_maximal_bookless_csv(Path(args.maximal_bookless_csv))
    if args.kball_gap_ids is not None:
        wanted = {int(x) for x in args.kball_gap_ids.split(",")}
        kball = [g for g in kball_all if g["gap_id"] in wanted]
    else:
        kball = sorted(kball_all, key=lambda r: -r["size"])[:args.top_n]
    if args.mb_gap_ids is not None:
        wanted = {int(x) for x in args.mb_gap_ids.split(",")}
        mb = [g for g in mb_all if g["gap_id"] in wanted]
    else:
        mb = sorted(mb_all, key=lambda r: -r["size"])[:args.top_n]

    # K-ball: nearest-book per article in each gap, aggregated.
    print(f"Computing K-ball associated books for top {args.top_n} gaps...")
    for g in kball:
        article_idxs = np.array(g["article_indices"], dtype=np.int64)
        q = np.ascontiguousarray(emb[article_idxs])
        D, I = book_index.search(q, k=1)
        local_book_idxs = I[:, 0]
        global_book_idxs = book_global_idx[local_book_idxs].tolist()
        # Distances to those nearest books (same order as articles)
        nearest_dists = np.sqrt(np.maximum(D[:, 0], 0.0))
        counts = Counter(global_book_idxs)
        # For each unique nearest-book, also record min distance among articles
        # so we can sort by frequency (primary) and proximity (secondary).
        per_book_min_d: dict[int, float] = {}
        for bi, d in zip(global_book_idxs, nearest_dists.tolist()):
            if bi not in per_book_min_d or d < per_book_min_d[bi]:
                per_book_min_d[bi] = d
        ranked = sorted(
            counts.items(),
            key=lambda kv: (-kv[1], per_book_min_d[kv[0]]),
        )
        g["associated_books"] = ranked
        g["per_book_min_d"] = per_book_min_d

    # ---------------- Print markdown report ----------------
    print()
    print(f"# Gap-book inspection: {args.scope}")
    print()

    print(f"## Max-bookless top-{args.top_n}")
    for i, g in enumerate(mb, 1):
        print()
        print(f"### mb-rank {i}: gap_id={g['gap_id']}, size={g['size']}, "
              f"n_boundary_books={g['n_boundary_books']}")
        sample = g["article_indices"][:args.top_titles]
        print(f"  Sample articles ({len(sample)} of {g['size']}):")
        for j in sample:
            t = titles_arr[j]
            print(f"    - {(t or '(no title)')[:110]}")
        print(f"  Boundary books ({len(g['boundary_book_indices'])} total, "
              f"showing up to {args.top_books}):")
        if not g["boundary_book_indices"]:
            print(f"    (none)")
        for bi in g["boundary_book_indices"][:args.top_books]:
            t = titles_arr[bi]
            print(f"    - {(t or '(no title)')[:110]}")

    print()
    print(f"## K-ball top-{args.top_n}")
    for i, g in enumerate(kball, 1):
        print()
        print(f"### kb-rank {i}: gap_id={g['gap_id']}, size={g['size']}")
        sample = g["article_indices"][:args.top_titles]
        print(f"  Sample articles ({len(sample)} of {g['size']}):")
        for j in sample:
            t = titles_arr[j]
            print(f"    - {(t or '(no title)')[:110]}")
        ab = g["associated_books"][:args.top_books]
        n_unique = len(g["associated_books"])
        print(f"  Nearest-book lookup: {n_unique} unique books across "
              f"{g['size']} articles, showing top {len(ab)} by frequency:")
        for bi, count in ab:
            t = titles_arr[bi]
            d = g["per_book_min_d"][bi]
            print(f"    - [×{count}, min_d={d:.3f}] {(t or '(no title)')[:100]}")


if __name__ == "__main__":
    main()
