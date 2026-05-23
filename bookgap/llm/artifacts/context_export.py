"""Per-(topic, gap) context export: writes one ContextRecord JSON per
gap into ``<output_dir>/contexts/``. Backend-agnostic — captures the
full gap composition (all articles ranked by centrality, all covering
books ranked by descending coverage) for later auditing and for any
consumer that wants the analytical view of a gap.

Filename: ``gap_{gap_idx}_topic_{topic_id}.json`` matching the
``custom_id`` already used in ``ollama_results.jsonl``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bookgap.llm.artifacts.common import group_gap_articles_by_covering_books
from bookgap.llm.data_structures import GapWithContext
from bookgap.llm.prompt_record import (
    ArticleEntry, BookCoverage, ContextRecord, FurthestArticle, GapStats,
    write_context_record,
)
from bookgap.util import LOGGER


def save_gap_context(gap_with_context: GapWithContext, output_dir: Path) -> Path:
    """Build a :class:`ContextRecord` and write it to
    ``<output_dir>/contexts/gap_{gap_idx}_topic_{topic_id}.json``.

    Returns the written path.
    """
    contexts_dir = output_dir / "contexts"
    record = _build_context_record(gap_with_context)
    return write_context_record(contexts_dir, record)


def generate_all_contexts(
    gaps: list[GapWithContext], output_dir: Path,
) -> list[Path]:
    """Write a ContextRecord JSON for every gap. Returns the list of
    written paths in iteration order."""
    paths: list[Path] = []
    for g in gaps:
        try:
            paths.append(save_gap_context(g, output_dir))
        except ValueError as exc:
            LOGGER.warning(
                "Skipping context for gap %d topic %s: %s",
                g.gap_idx, g.topic_id, exc,
            )
    return paths


def _build_context_record(g: GapWithContext) -> ContextRecord:
    gap = g.gap
    articles = g.articles
    books = g.books

    gap_dists_idxs = gap.get_dists_idxs()
    gap_size = g.gap_size
    if gap_size == 0:
        raise ValueError(f"Gap {g.gap_idx} has no articles")

    article_works = articles.works()
    book_works = books.works()

    articles_by_centrality = [
        _article_entry(article_works[int(idx)])
        for idx in gap_dists_idxs.idxs[:gap_size]
    ]

    furthest: FurthestArticle | None = None
    if gap_size > 1:
        furthest_pos = int(np.argmax(gap_dists_idxs.sqdists))
        furthest_idx = int(gap_dists_idxs.idxs[furthest_pos])
        w = article_works[furthest_idx]
        furthest = FurthestArticle(
            openalex_id=str(_get(w, "id", "")),
            display_name=str(_get(w, "display_name", "")),
            abstract=str(_get(w, "abstract", "")),
            sqdist=float(gap_dists_idxs.sqdists[furthest_pos]),
        )

    raw_coverage = group_gap_articles_by_covering_books(gap)
    book_coverage: dict[int, list[int]] = {
        int(b_idx): [int(a_idx) for a_idx in covered]
        for b_idx, covered in raw_coverage.items()
    }
    covering_books: list[BookCoverage] = []
    for b_idx, covered in sorted(
        book_coverage.items(), key=lambda item: (-len(item[1]), item[0]),
    ):
        w = book_works[b_idx]
        covering_books.append(BookCoverage(
            openalex_id=str(_get(w, "id", "")),
            display_name=str(_get(w, "display_name", "")),
            abstract=str(_get(w, "abstract", "")),
            articles_covered=len(covered),
            fraction_covered=(len(covered) / gap_size) if gap_size > 0 else 0.0,
        ))

    return ContextRecord(
        topic_id=g.topic_id,
        topic_name=g.topic_name,
        gap_idx=g.gap_idx,
        gap_stats=GapStats(
            gap_size=gap_size,
            gap_size_percentile=float(g.gap_size_percentile),
            gap_zscore=float(g.gap_zscore) if not np.isnan(g.gap_zscore) else float("nan"),
            shapiro_log10_pval=float(g.shapiro_log10_pval) if not np.isnan(g.shapiro_log10_pval) else float("nan"),
        ),
        topic_totals_articles=len(article_works),
        topic_totals_books=len(book_works),
        articles_by_centrality=articles_by_centrality,
        furthest_article=furthest,
        covering_books=covering_books,
    )


def _article_entry(w: Any) -> ArticleEntry:
    return ArticleEntry(
        openalex_id=str(_get(w, "id", "")),
        display_name=str(_get(w, "display_name", "")),
        abstract=str(_get(w, "abstract", "")),
    )


def _get(w: Any, key: str, default: Any) -> Any:
    return w.get(key, default) if isinstance(w, dict) else getattr(w, key, default)
