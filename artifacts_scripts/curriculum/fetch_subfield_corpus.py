"""Fetch a field corpus for a book's ToC-matching generalization test.

OpenAlex book records carry noisy `primary_topic` tags, but their SUBFIELD is
reliable. This pulls the top-K most-cited works of a book's subfield and writes
them as gz pages in the same `{results: [...]}` format `toc_dendrogram_cut`'s
load_corpus reads — so a niche book gets a clean field corpus, matching how
Jurafsky used the whole NLP topic.

    PYTHONPATH=. uv run python -m artifacts_scripts.curriculum.fetch_subfield_corpus \
        --book https://openalex.org/W4211017872 --out cache/oa/colorimage_cv --top-k 4000
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pyalex

pyalex.config.email = "geoffrey.a.churchill@gmail.com"
pyalex.config.max_retries = 3

KEEP = ("id", "abstract_inverted_index", "referenced_works", "cited_by_count",
        "publication_date", "primary_topic")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True, help="OpenAlex book work id/url")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=4000)
    args = ap.parse_args()

    book = pyalex.Works()[args.book]
    pt = book.get("primary_topic") or {}
    sub = (pt.get("subfield") or {})
    sub_id = sub.get("id", "").rsplit("/", 1)[-1]
    print(f"book: {book.get('title')!r} | subfield {sub_id} ({sub.get('display_name')})")
    if not sub_id:
        raise SystemExit("no subfield on the book's primary_topic")

    args.out.mkdir(parents=True, exist_ok=True)
    q = (pyalex.Works()
         .filter(primary_topic={"subfield": {"id": sub_id}})
         .filter(has_abstract=True)
         .sort(cited_by_count="desc"))
    page_idx = n = 0
    for page in q.paginate(per_page=200, n_max=args.top_k):
        slim = [{k: w.get(k) for k in KEEP} for w in page]
        with gzip.open(args.out / f"page_{page_idx:05d}.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"results": slim}, f)
        page_idx += 1
        n += len(slim)
        print(f"  page {page_idx}: +{len(slim)} (total {n})", flush=True)
    print(f"wrote {n} works -> {args.out}")


if __name__ == "__main__":
    main()
