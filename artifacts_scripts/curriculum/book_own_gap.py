#!/usr/bin/env python3
"""Compute a book's OWN gap, then emit the in-ball article ids that would seed a
curriculum from it.

This is the "book-first" front-end for the real-book ToC reconstruction eval
(docs/backlog/curriculum-real-toc-baseline.md). Instead of starting from the gap
detector's chosen gaps, we treat a chosen book as a mere article and find its
nearest NON-SELF neighboring book — exactly how the main pipeline defines a
book's gap for the target/proposal gap percentiles (bookgap/gaps.py:97-101,
kball self-drop at kball_algorithm.py:203). The k-ball around the book =
articles within radius_fraction x dist-to-nearest-non-self-book; those become
the curriculum seed.

Querying with the WHOLE book set (is_intra_cloud=True) yields every book's gap
in one pass, so we also get the gap-size distribution (the percentile) for free
— the basis for the two-stage "pick a percentile spread of books" plan.

Usage:
  .conda/bin/python -m artifacts_scripts.curriculum.book_own_gap \
      --book-id https://openalex.org/W3160730739 --subfield 1702 \
      --out artifacts/book_gaps/W3160730739_ai.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from artifacts_scripts.curriculum.config import CurriculumConfig
from bookgap.cloud_index import get_kball_coverage
from bookgap.subcloud import SubCloud
from bookgap.util import Filter, LOGGER, setup_logging

RADIUS_FRACTION_DEFAULT = 0.75  # bookgap/configs/gap/gap_config.py default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book-id", default=None, help="OpenAlex id of the reference book")
    ap.add_argument("--rank-top", type=int, default=0,
                    help="if >0, also dump the top-N books by own-gap size (the niche, "
                         "real-gap textbooks that make good reconstruction targets)")
    ap.add_argument("--subfield", default=None,
                    help="scope to this OpenAlex subfield id (e.g. 1702 = AI)")
    ap.add_argument("--field", default=None,
                    help="scope to this OpenAlex field id (e.g. 17 = Computer Science); "
                         "broader than --subfield. Default (neither) = whole domain=3.")
    ap.add_argument("--radius-fraction", type=float, default=RADIUS_FRACTION_DEFAULT)
    ap.add_argument("--sample", type=int, default=25, help="n nearest gap-article titles to log")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    # Without a handler the global "bookgap" LOGGER drops INFO records (they fall
    # through to root's lastResort handler = stderr, WARNING+ only). setup_logging
    # attaches a per-record-flushing FileHandler (live + persisted) + console.
    setup_logging(log_name="book_own_gap", log_level="INFO", include_time=False,
                  log_path=args.out.with_suffix(".log"))

    cfg = OmegaConf.structured(CurriculumConfig())
    if args.subfield:
        cfg.experiment.cloud.filter_pair.pre.subfield = str(args.subfield)
    if args.field:
        cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cloud = instantiate(cfg.experiment.cloud)
    LOGGER.info("Cloud: %s", cloud.name())

    books = SubCloud(parent=cloud, post_filter=Filter(work_types=("book",)))
    articles = SubCloud(parent=cloud, post_filter=Filter(work_types=("article",)))
    book_ids = books.work_ids()
    article_ids = articles.work_ids()
    LOGGER.info("scope: %d books, %d articles", len(book_ids), len(article_ids))

    if not args.book_id and not args.rank_top:
        LOGGER.error("pass --book-id and/or --rank-top")
        sys.exit(2)

    # Every book's own gap in one pass; is_intra_cloud=True drops the self-book so
    # the radius is set by the nearest OTHER book.
    covered = get_kball_coverage(ambient=articles, landmark=books, query=books,
                                 radius_fraction=args.radius_fraction, k=1,
                                 is_intra_cloud=True)
    gap_sizes = np.array([len(ids) for _d, ids in covered])
    a_titles = articles.work_titles()
    b_titles = books.work_titles()
    LOGGER.info("own-gap-size distribution over %d books: max=%d p99=%.0f p90=%.0f median=%.0f zeros=%d",
                len(book_ids), int(gap_sizes.max()), float(np.percentile(gap_sizes, 99)),
                float(np.percentile(gap_sizes, 90)), float(np.median(gap_sizes)),
                int((gap_sizes == 0).sum()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "scope_subfield": args.subfield, "scope_field": args.field,
        "radius_fraction": args.radius_fraction,
        "n_books_in_scope": len(book_ids), "n_articles_in_scope": len(article_ids),
        "gap_size_max": int(gap_sizes.max()),
        "gap_size_distribution": {p: float(np.percentile(gap_sizes, p)) for p in (50, 90, 95, 99)},
    }

    def gap_for(bi: int) -> dict:
        sqd, local = covered[bi]
        local = np.asarray(local)[np.argsort(sqd)]            # nearest first
        return {"book": b_titles[bi], "book_id": book_ids[bi], "gap_size": int(len(local)),
                "gap_size_percentile": float(np.mean(gap_sizes < len(local))),
                "sample_titles": [a_titles[int(j)] for j in local[:args.sample]],
                "gap_article_ids": [article_ids[int(j)] for j in local]}

    if args.book_id:
        if args.book_id not in book_ids:
            LOGGER.error("book %s not found among %d books in scope", args.book_id, len(book_ids))
            sys.exit(2)
        g = gap_for(book_ids.index(args.book_id))
        LOGGER.info("Book %s own gap: %d articles, percentile=%.3f", args.book_id, g["gap_size"], g["gap_size_percentile"])
        for t in g["sample_titles"]:
            LOGGER.info("  - %s", t)
        payload["book"] = g

    if args.rank_top:
        top = np.argsort(gap_sizes)[::-1][:args.rank_top]
        LOGGER.info("top %d books by own-gap size:", args.rank_top)
        ranked = []
        for bi in top:
            g = gap_for(int(bi))
            LOGGER.info("  [%4d art] %s", g["gap_size"], g["book"][:70])
            ranked.append({k: g[k] for k in ("book", "book_id", "gap_size", "gap_size_percentile", "sample_titles")})
        payload["top_books"] = ranked

    json.dump(payload, open(args.out, "w"), indent=1)
    LOGGER.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
