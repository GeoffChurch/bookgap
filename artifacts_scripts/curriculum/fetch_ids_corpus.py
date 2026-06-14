"""Fetch a tight, book-specific corpus from an explicit OpenAlex ID list.

Used for the corpus-fit causal test: instead of a broad subfield slice, cluster
each book's own neighborhood (its gap-ball / referenced works). Writes gz pages
in the format `toc_dendrogram_cut.load_corpus` reads.

    PYTHONPATH=. .conda/bin/python -m artifacts_scripts.curriculum.fetch_ids_corpus \
        --gap-json artifacts/book_gaps/color_image_gap.json \
        --ids-path book.gap_article_ids --out cache/oa/tight_colorimage
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


def _dig(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-json", required=True)
    ap.add_argument("--ids-path", default="book.gap_article_ids",
                    help="dotted path to the ID list inside the json")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ids = _dig(json.load(open(args.gap_json)), args.ids_path)
    ids = [str(i).rsplit("/", 1)[-1] for i in ids]  # bare W-ids
    args.out.mkdir(parents=True, exist_ok=True)
    n = page_idx = 0
    for i in range(0, len(ids), 50):
        batch = ids[i:i + 50]
        works = pyalex.Works().filter(ids={"openalex": "|".join(batch)}).get(per_page=50)
        slim = [{k: w.get(k) for k in KEEP} for w in works]
        with gzip.open(args.out / f"page_{page_idx:05d}.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"results": slim}, f)
        page_idx += 1
        n += len(slim)
    print(f"wrote {n}/{len(ids)} works -> {args.out}")


if __name__ == "__main__":
    main()
