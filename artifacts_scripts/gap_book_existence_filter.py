"""Phase 1: enrich a gap CSV with book-existence signal columns.

SUPERSEDED for new runs by ``gap_softmax_enrich.py`` (multi-cluster
softmax Phase 2), which produces these columns plus the multi-scale
coverage columns and the direct softmax effective_uncovered in one
pass. Retained for fallback / regression and for use against gap CSVs
from non-soft-EM pipelines (K-ball, HDBSCAN max-bookless).

Reads any gap CSV produced by the gap-detection pipelines (K-ball,
article-as-seed, max-bookless), computes per-gap top keywords + top
nearest book + Jaccard overlap + flags, and writes an enriched CSV.

The flags are diagnostic, not destructive — callers (e.g. the
comparison-table generator) decide how to use them.

Usage::

    python -m artifacts_scripts.gap_book_existence_filter \\
        --scope 17/1702/T10181 \\
        --input artifacts/gap_article_as_seed/T10181__article_seed.csv \\
        --output artifacts/gap_article_as_seed/T10181__article_seed_enriched.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from artifacts_scripts.keyword_helpers import (
    compute_book_existence_signal,
    compute_gap_keywords,
    compute_top_books,
)
from artifacts_scripts.hdbscan_gap_probe import load_scope
from bookgap.index import FlatL2, build_index


# Maps from possible input-CSV column names to the canonical name used here.
_MEMBER_COLUMNS = ("article_indices_json", "member_indices_json")
_SIZE_COLUMNS = ("final_size", "gap_size", "total_size")


def _load_articles_parquet(scope: str) -> tuple[list[str], list[str]]:
    parts = scope.split("/")
    parquet_dir = Path("cache/works_parquet").joinpath(*parts)
    article_pq = parquet_dir / "article.parquet"
    if not article_pq.exists():
        raise FileNotFoundError(f"No article.parquet at {article_pq}")
    tbl = pq.read_table(article_pq, columns=["display_name", "abstract"])
    return tbl.column("display_name").to_pylist(), tbl.column("abstract").to_pylist()


def _row_members(row: dict) -> list[int]:
    """Read article-only members from whichever column the CSV provides."""
    for col in _MEMBER_COLUMNS:
        if col in row and row[col]:
            data = json.loads(row[col])
            # member_indices_json carries dicts with type tags; filter to articles.
            if data and isinstance(data[0], dict):
                return [m["idx"] for m in data if m.get("type") == "article"]
            return list(data)
    raise KeyError(f"row has no member column (looked for {_MEMBER_COLUMNS})")


def _row_size(row: dict, members: list[int]) -> int:
    for col in _SIZE_COLUMNS:
        if col in row and row[col]:
            return int(row[col])
    return len(members)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True)
    p.add_argument("--input", required=True, help="gap CSV (any pipeline)")
    p.add_argument("--output", required=True)
    p.add_argument("--keywords-per-gap", type=int, default=5)
    p.add_argument("--bigram-iterations", type=int, default=2)
    p.add_argument("--vote-concentration-threshold", type=float, default=0.25)
    p.add_argument("--title-overlap-threshold", type=float, default=0.2)
    p.add_argument("--templated-concentration-threshold", type=float, default=0.5)
    args = p.parse_args()

    print(f"Loading scope {args.scope}...")
    titles, abstracts = _load_articles_parquet(args.scope)
    emb, ids, types, all_titles, _topic_ids, _refs = load_scope(args.scope, with_refs=False)
    types_arr = np.array(types)
    book_global_idx = np.where(types_arr == "book")[0]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)
    emb = np.ascontiguousarray(emb)
    book_emb = np.ascontiguousarray(emb[book_global_idx])
    book_index = build_index(FlatL2(), book_emb)
    all_titles_arr = np.array(all_titles, dtype=object)

    print(f"Reading {args.input}...")
    with open(args.input) as f:
        reader = csv.DictReader(f)
        in_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    print(f"  {len(rows)} gaps")

    members_per_gap = [_row_members(r) for r in rows]
    sizes = [_row_size(r, m) for r, m in zip(rows, members_per_gap)]

    print("Computing per-gap keywords...")
    keywords = compute_gap_keywords(
        members_per_gap, titles, abstracts,
        top_k=args.keywords_per_gap,
        bigram_iterations=args.bigram_iterations,
    )

    print("Computing top books per gap...")
    new_cols = []
    for members, size, kw_list in zip(members_per_gap, sizes, keywords):
        top_books = compute_top_books(members, emb, book_index, book_global_idx,
                                      all_titles_arr, k=1)
        if top_books:
            top_book_id, top_book_title, top_book_votes = top_books[0]
        else:
            top_book_id, top_book_title, top_book_votes = -1, "", 0
        top_keywords = [kw for kw, _ in kw_list]
        signal = compute_book_existence_signal(
            gap_size=size,
            top_book_votes=top_book_votes,
            top_book_title=top_book_title,
            top_keywords=top_keywords,
            vote_concentration_threshold=args.vote_concentration_threshold,
            title_overlap_threshold=args.title_overlap_threshold,
            templated_concentration_threshold=args.templated_concentration_threshold,
        )
        new_cols.append({
            "top_book_id": top_book_id,
            "top_book_title": top_book_title,
            "top_book_votes": top_book_votes,
            "vote_concentration": f"{signal['vote_concentration']:.4f}",
            "top_keywords": "|".join(top_keywords),
            "title_keyword_overlap": f"{signal['title_keyword_overlap']:.4f}",
            "book_exists_flag": int(signal["book_exists_flag"]),
            "templated_leak_flag": int(signal["templated_leak_flag"]),
        })

    new_fieldnames = [
        "top_book_id", "top_book_title", "top_book_votes",
        "vote_concentration", "top_keywords", "title_keyword_overlap",
        "book_exists_flag", "templated_leak_flag",
    ]
    out_fieldnames = in_fieldnames + [c for c in new_fieldnames if c not in in_fieldnames]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for row, extra in zip(rows, new_cols):
            row.update(extra)
            writer.writerow(row)

    n_book_exists = sum(c["book_exists_flag"] for c in new_cols)
    n_templated = sum(c["templated_leak_flag"] for c in new_cols)
    n_clean = len(new_cols) - n_book_exists - n_templated
    print()
    print(f"Total gaps: {len(new_cols)}")
    print(f"  book_exists_flag: {n_book_exists}")
    print(f"  templated_leak_flag: {n_templated}")
    print(f"  clean (neither): {n_clean}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
