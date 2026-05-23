"""Tests for ContextRecord export."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from bookgap.llm.artifacts.context_export import save_gap_context, generate_all_contexts
from bookgap.llm.data_structures import GapWithContext
from bookgap.llm.prompt_record import (
    CONTEXT_SCHEMA_VERSION, read_context_record, record_filename,
)


def _make_gap_with_context() -> GapWithContext:
    mock_gap = Mock()
    mock_gap.get_dists_idxs.return_value = Mock(
        idxs=np.array([10, 15, 22, 8, 12]),
        sqdists=np.array([0.0, 0.3, 0.8, 0.5, 0.4]),
    )
    articles = Mock()
    books = Mock()
    articles.works.return_value = [
        {"id": f"https://openalex.org/A{i}",
         "display_name": f"Article {i}",
         "abstract": f"Abstract for article {i}"}
        for i in range(25)
    ]
    books.works.return_value = [
        {"id": f"https://openalex.org/B{i}",
         "display_name": f"Book {i}",
         "abstract": f"Abstract for book {i}"}
        for i in range(5)
    ]

    g = GapWithContext(
        gap=mock_gap, articles=articles, books=books,
        topic_name="Test Topic", topic_id="T9001", gap_idx=2,
        book_gap_sizes=[2, 5, 7, 3],
        gap_size=5, gap_size_percentile=0.75, gap_zscore=1.5,
        shapiro_log10_pval=-2.3,
    )
    return g


def test_save_gap_context_writes_json_with_correct_filename(tmp_path: Path) -> None:
    g = _make_gap_with_context()
    with patch(
        "bookgap.llm.artifacts.context_export.group_gap_articles_by_covering_books",
        return_value={0: [10, 15], 1: [22]},
    ):
        save_gap_context(g, tmp_path)

    out = tmp_path / "contexts" / record_filename(g.gap_idx, g.topic_id)
    assert out.exists(), f"expected {out}"


def test_saved_record_has_expected_shape(tmp_path: Path) -> None:
    g = _make_gap_with_context()
    with patch(
        "bookgap.llm.artifacts.context_export.group_gap_articles_by_covering_books",
        return_value={0: [10, 15], 1: [22]},
    ):
        save_gap_context(g, tmp_path)

    rec = read_context_record(
        tmp_path / "contexts" / record_filename(g.gap_idx, g.topic_id)
    )
    assert rec.schema_version == CONTEXT_SCHEMA_VERSION
    assert rec.topic_id == "T9001"
    assert rec.gap_idx == 2
    assert rec.gap_stats.gap_size == 5
    assert rec.topic_totals_articles == 25
    assert rec.topic_totals_books == 5
    # 5 gap articles ranked by centrality (0..4 of mock dists)
    assert len(rec.articles_by_centrality) == 5
    assert rec.articles_by_centrality[0].display_name == "Article 10"
    # Furthest article = max-sqdist entry = idx 22 (sqdist 0.8)
    assert rec.furthest_article is not None
    assert rec.furthest_article.display_name == "Article 22"
    # Books sorted by descending coverage: Book 0 (covered 2 articles) then Book 1 (1)
    assert len(rec.covering_books) == 2
    assert rec.covering_books[0].display_name == "Book 0"
    assert rec.covering_books[0].articles_covered == 2
    assert rec.covering_books[0].fraction_covered == pytest.approx(2 / 5)


def test_no_books_yields_empty_covering_books(tmp_path: Path) -> None:
    g = _make_gap_with_context()
    with patch(
        "bookgap.llm.artifacts.context_export.group_gap_articles_by_covering_books",
        return_value={},
    ):
        save_gap_context(g, tmp_path)
    rec = read_context_record(
        tmp_path / "contexts" / record_filename(g.gap_idx, g.topic_id)
    )
    assert rec.covering_books == []


def test_no_analysis_txt_file_written(tmp_path: Path) -> None:
    """The legacy human-readable .txt artifact is gone."""
    g = _make_gap_with_context()
    with patch(
        "bookgap.llm.artifacts.context_export.group_gap_articles_by_covering_books",
        return_value={0: [10]},
    ):
        save_gap_context(g, tmp_path)
    txt_files = list((tmp_path / "contexts").glob("*.txt"))
    assert txt_files == [], f"unexpected .txt files: {txt_files}"
