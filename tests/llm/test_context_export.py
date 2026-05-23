"""Tests for ContextRecord export with v2 fields."""
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


def _proposal_analysis_for(g: GapWithContext) -> dict:
    """Synthetic per-gap inputs for the coverage analysis.

    Book ordering: must match the order context_export's coverage-grouping
    yields after sorting by descending coverage with tie-broken by book_idx.
    With group_gap_articles_by_covering_books returning {0: [10, 15], 1: [22]},
    sorted_book_items is [(0, [10, 15]), (1, [22])]. We supply embeddings
    for book_0 and book_1 in that order (5 books in the SubCloud total but only
    2 are covering books — the proposal_analysis just needs embeddings for the
    covering books in that order).
    """
    gap_articles = np.array([
        [0.0, 0.1, 0.0],
        [0.0, -0.1, 0.0],
        [1.0, 0.1, 0.0],
        [1.0, -0.1, 0.0],
        [0.5, 0.5, 0.0],
    ], dtype=np.float32)
    # All topic articles: a superset of the gap articles. The PCA in
    # compute_local_projection_2d fits on this set; minimum size 2 to
    # produce a well-defined plane.
    all_topic_articles = np.vstack([
        gap_articles,
        np.array([
            [0.5, 0.0, 0.0],
            [0.2, 0.2, 0.1],
        ], dtype=np.float32),
    ])
    return {
        "gap_article_embeddings": gap_articles,
        "book_embeddings": np.array([
            [0.0, 0.0, 0.0],   # book_0 (covering book #1, near first 2 articles)
            [3.0, 0.0, 0.0],   # book_1 (covering book #2)
        ], dtype=np.float32),
        "proposal_embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "seed_embedding": np.array([0.5, 0.0, 0.0], dtype=np.float32),
        "all_topic_article_embeddings": all_topic_articles,
        "gap_radius": 1.2,
    }


def test_save_gap_context_writes_v2_record(tmp_path: Path) -> None:
    g = _make_gap_with_context()
    analysis = _proposal_analysis_for(g)
    with patch(
        "bookgap.llm.artifacts.context_export.group_gap_articles_by_covering_books",
        return_value={0: [10, 15], 1: [22]},
    ):
        save_gap_context(g, tmp_path, proposal_analysis=analysis)

    rec = read_context_record(
        tmp_path / "contexts" / record_filename(g.gap_idx, g.topic_id)
    )
    assert rec.schema_version == CONTEXT_SCHEMA_VERSION == 2
    # proposal_coverage populated
    assert rec.proposal_coverage.articles_covered >= 0
    assert 0.0 <= rec.proposal_coverage.fraction_covered <= 1.0
    # local_projection_2d populated and well-formed
    assert rec.local_projection_2d.gap_radius == 1.2
    # Seed projects to a finite (x, y) under PCA — no longer pinned to origin.
    assert np.isfinite(rec.local_projection_2d.seed.x)
    assert np.isfinite(rec.local_projection_2d.seed.y)
    assert len(rec.local_projection_2d.gap_articles) == 5
    assert len(rec.local_projection_2d.gap_articles_flipped_to_proposal) == 5
    # every book has articles_covered_after_proposal populated
    for b in rec.covering_books:
        assert isinstance(b.articles_covered_after_proposal, int)


def test_save_gap_context_raises_without_proposal_analysis(tmp_path: Path) -> None:
    """v2 is mandatory — saving without the analysis arg must raise."""
    g = _make_gap_with_context()
    with patch(
        "bookgap.llm.artifacts.context_export.group_gap_articles_by_covering_books",
        return_value={0: [10]},
    ):
        with pytest.raises(TypeError):  # missing required kw-only arg
            save_gap_context(g, tmp_path)  # type: ignore[call-arg]


def test_generate_all_contexts_raises_on_missing_entry(tmp_path: Path) -> None:
    """If proposal_analyses is missing an entry for some gap, raise loudly
    (don't silently skip — gaps without proposals shouldn't reach here)."""
    g = _make_gap_with_context()
    with patch(
        "bookgap.llm.artifacts.context_export.group_gap_articles_by_covering_books",
        return_value={0: [10]},
    ):
        with pytest.raises(KeyError):
            generate_all_contexts([g], tmp_path, proposal_analyses={})


def test_no_books_yields_empty_covering_books(tmp_path: Path) -> None:
    """v2 should still emit an empty covering_books list when no books cover the gap.
    Requires a proposal_analysis with zero covering-book embeddings."""
    g = _make_gap_with_context()
    analysis = _proposal_analysis_for(g)
    # Empty book_embeddings: a (0, D) array.
    analysis["book_embeddings"] = np.zeros((0, 3), dtype=np.float32)
    with patch(
        "bookgap.llm.artifacts.context_export.group_gap_articles_by_covering_books",
        return_value={},
    ):
        save_gap_context(g, tmp_path, proposal_analysis=analysis)
    rec = read_context_record(
        tmp_path / "contexts" / record_filename(g.gap_idx, g.topic_id)
    )
    assert rec.covering_books == []
    # All gap articles flip to proposal (only book in the set).
    assert all(rec.local_projection_2d.gap_articles_flipped_to_proposal)


def test_articles_by_centrality_content_and_order(tmp_path: Path) -> None:
    """articles_by_centrality preserves the gap's idxs order."""
    g = _make_gap_with_context()
    analysis = _proposal_analysis_for(g)
    with patch(
        "bookgap.llm.artifacts.context_export.group_gap_articles_by_covering_books",
        return_value={0: [10, 15], 1: [22]},
    ):
        save_gap_context(g, tmp_path, proposal_analysis=analysis)
    rec = read_context_record(
        tmp_path / "contexts" / record_filename(g.gap_idx, g.topic_id)
    )
    # gap.idxs = [10, 15, 22, 8, 12] → article display names in that order.
    assert [a.display_name for a in rec.articles_by_centrality] == [
        "Article 10", "Article 15", "Article 22", "Article 8", "Article 12",
    ]


def test_covering_books_sorted_by_descending_coverage(tmp_path: Path) -> None:
    """covering_books is sorted by descending coverage (book_idx tie-break)."""
    g = _make_gap_with_context()
    analysis = _proposal_analysis_for(g)
    with patch(
        "bookgap.llm.artifacts.context_export.group_gap_articles_by_covering_books",
        # book 1 covers 3 articles, book 0 covers 2 articles — book 1 should come first.
        return_value={0: [10, 15], 1: [22, 8, 12]},
    ):
        # Update analysis's book_embeddings to match the new ordering:
        # after sort, books = [(1, ...), (0, ...)], so embeddings must be in
        # that order too.
        analysis["book_embeddings"] = np.array([
            [3.0, 0.0, 0.0],   # book_1's embedding
            [0.0, 0.0, 0.0],   # book_0's embedding
        ], dtype=np.float32)
        save_gap_context(g, tmp_path, proposal_analysis=analysis)
    rec = read_context_record(
        tmp_path / "contexts" / record_filename(g.gap_idx, g.topic_id)
    )
    assert len(rec.covering_books) == 2
    assert rec.covering_books[0].display_name == "Book 1"  # higher coverage
    assert rec.covering_books[0].articles_covered == 3
    assert rec.covering_books[0].fraction_covered == pytest.approx(3 / 5)
    assert rec.covering_books[1].display_name == "Book 0"
    assert rec.covering_books[1].articles_covered == 2


def test_furthest_article_populated_when_gap_size_gt_1(tmp_path: Path) -> None:
    """furthest_article is the article at argmax(sqdists)."""
    g = _make_gap_with_context()  # sqdists = [0.0, 0.3, 0.8, 0.5, 0.4]
    analysis = _proposal_analysis_for(g)
    # Only one covering book in this test — trim book_embeddings to match.
    analysis["book_embeddings"] = np.array([
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    with patch(
        "bookgap.llm.artifacts.context_export.group_gap_articles_by_covering_books",
        return_value={0: [10]},
    ):
        save_gap_context(g, tmp_path, proposal_analysis=analysis)
    rec = read_context_record(
        tmp_path / "contexts" / record_filename(g.gap_idx, g.topic_id)
    )
    # argmax(sqdists) = index 2 → idxs[2] = 22 → "Article 22"
    assert rec.furthest_article is not None
    assert rec.furthest_article.display_name == "Article 22"
    assert rec.furthest_article.sqdist == pytest.approx(0.8)
