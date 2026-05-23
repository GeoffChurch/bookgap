"""Round-trip + filename-convention tests for ContextRecord/PromptRecord."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bookgap.llm.prompt_record import (
    ContextRecord, PromptRecord, ArticleEntry, BookEntry,
    GapStats, BookCoverage, FurthestArticle,
    ProposalCoverage, Point2D, LocalProjection2D,
    CONTEXT_SCHEMA_VERSION,
    record_filename,
    read_context_record, write_context_record,
    read_prompt_record, write_prompt_record,
)


def _sample_local_projection() -> LocalProjection2D:
    return LocalProjection2D(
        gap_radius=0.42,
        seed=Point2D(x=0.0, y=0.0),
        proposal=Point2D(x=0.1, y=0.2),
        gap_articles=[Point2D(x=0.05, y=0.1), Point2D(x=-0.05, y=0.08)],
        gap_articles_flipped_to_proposal=[True, False],
        books=[Point2D(x=0.5, y=0.3)],
    )


def _sample_proposal_coverage() -> ProposalCoverage:
    return ProposalCoverage(articles_covered=42, fraction_covered=0.21)


def _sample_prompt_record() -> PromptRecord:
    return PromptRecord(
        schema_version=1,
        custom_id="gap_0_topic_T10020",
        topic_id="T10020",
        topic_name="Quantum Information and Cryptography",
        gap_idx=0,
        requested_num_prompt_articles=10,
        requested_num_prompt_books=5,
        sampled_articles=[
            ArticleEntry(openalex_id="https://openalex.org/W1",
                         display_name="Sampled 1", abstract="Abs 1"),
        ],
        covering_books=[
            BookEntry(openalex_id="https://openalex.org/B1",
                      display_name="Book 1", abstract="BAbs 1",
                      articles_covered=10, fraction_covered=0.5),
        ],
        backend="ollama_local",
        model="gemma4:31b",
        generation_params={"temperature": 0.2, "num_ctx": 8192,
                           "num_predict": 8192},
    )


def _sample_context_record() -> ContextRecord:
    return ContextRecord(
        topic_id="T10020",
        topic_name="Quantum Information and Cryptography",
        gap_idx=0,
        gap_stats=GapStats(gap_size=898, gap_size_percentile=1.0,
                           gap_zscore=16.5, shapiro_log10_pval=-12.5),
        topic_totals_articles=46542,
        topic_totals_books=44,
        articles_by_centrality=[
            ArticleEntry(openalex_id="https://openalex.org/W1",
                         display_name="Closest", abstract="A1"),
        ],
        furthest_article=FurthestArticle(
            openalex_id="https://openalex.org/W999",
            display_name="Furthest", abstract="A999", sqdist=2.4,
        ),
        covering_books=[
            BookCoverage(openalex_id="https://openalex.org/B1",
                         display_name="Book 1", abstract="BAbs",
                         articles_covered=184, fraction_covered=0.3151,
                         articles_covered_after_proposal=120),
        ],
        proposal_coverage=_sample_proposal_coverage(),
        local_projection_2d=_sample_local_projection(),
    )


def test_prompt_record_round_trip(tmp_path: Path) -> None:
    rec = _sample_prompt_record()
    write_prompt_record(tmp_path, rec)
    loaded = read_prompt_record(
        tmp_path / record_filename(rec.gap_idx, rec.topic_id)
    )
    assert loaded == rec


def test_context_record_round_trip(tmp_path: Path) -> None:
    rec = _sample_context_record()
    write_context_record(tmp_path, rec)
    loaded = read_context_record(
        tmp_path / record_filename(rec.gap_idx, rec.topic_id)
    )
    assert loaded == rec


def test_record_filename_convention() -> None:
    assert record_filename(0, "T10020") == "gap_0_topic_T10020.json"
    assert record_filename(3, "T13567") == "gap_3_topic_T13567.json"


def test_context_record_with_no_furthest_article(tmp_path: Path) -> None:
    """A gap of size 1 has no 'furthest' entry. The None branch must
    round-trip cleanly."""
    rec = replace(_sample_context_record(), furthest_article=None)
    write_context_record(tmp_path, rec)
    loaded = read_context_record(
        tmp_path / record_filename(rec.gap_idx, rec.topic_id)
    )
    assert loaded == rec
    assert loaded.furthest_article is None


def test_prompt_record_with_empty_lists(tmp_path: Path) -> None:
    """Empty sampled_articles and covering_books round-trip cleanly
    (e.g., topics with no books)."""
    rec = replace(_sample_prompt_record(),
                  sampled_articles=[], covering_books=[])
    write_prompt_record(tmp_path, rec)
    loaded = read_prompt_record(
        tmp_path / record_filename(rec.gap_idx, rec.topic_id)
    )
    assert loaded == rec


def test_read_rejects_mismatched_schema_version(tmp_path: Path) -> None:
    """Schema-version mismatch must raise rather than silently coercing."""
    rec = _sample_prompt_record()
    write_prompt_record(tmp_path, rec)
    target = tmp_path / record_filename(rec.gap_idx, rec.topic_id)
    d = json.loads(target.read_text())
    d["schema_version"] = 999
    target.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="schema_version"):
        read_prompt_record(target)


def test_context_record_v2_round_trip(tmp_path: Path) -> None:
    """v2 ContextRecord with proposal_coverage + local_projection_2d
    + per-book articles_covered_after_proposal must round-trip cleanly."""
    rec = ContextRecord(
        topic_id="T10020",
        topic_name="Quantum Information and Cryptography",
        gap_idx=0,
        gap_stats=GapStats(gap_size=200, gap_size_percentile=1.0,
                           gap_zscore=15.0, shapiro_log10_pval=-10.0),
        topic_totals_articles=5000,
        topic_totals_books=10,
        articles_by_centrality=[
            ArticleEntry(openalex_id="https://openalex.org/W1",
                         display_name="A1", abstract="X1"),
            ArticleEntry(openalex_id="https://openalex.org/W2",
                         display_name="A2", abstract="X2"),
        ],
        furthest_article=None,
        covering_books=[
            BookCoverage(openalex_id="https://openalex.org/B1",
                         display_name="B1", abstract="BAbs1",
                         articles_covered=80, fraction_covered=0.4,
                         articles_covered_after_proposal=50),
            BookCoverage(openalex_id="https://openalex.org/B2",
                         display_name="B2", abstract="BAbs2",
                         articles_covered=30, fraction_covered=0.15,
                         articles_covered_after_proposal=30),
        ],
        proposal_coverage=_sample_proposal_coverage(),
        local_projection_2d=_sample_local_projection(),
    )
    write_context_record(tmp_path, rec)
    loaded = read_context_record(
        tmp_path / record_filename(rec.gap_idx, rec.topic_id)
    )
    assert loaded == rec
    assert loaded.schema_version == CONTEXT_SCHEMA_VERSION == 2


def test_book_coverage_v2_field_required() -> None:
    """BookCoverage requires articles_covered_after_proposal (not Optional)."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(BookCoverage)}
    assert "articles_covered_after_proposal" in fields
    # Field has no default, i.e. is required
    afp = next(f for f in dataclasses.fields(BookCoverage)
               if f.name == "articles_covered_after_proposal")
    assert afp.default is dataclasses.MISSING


def test_v1_file_is_rejected(tmp_path: Path) -> None:
    """Old v1 schema_version=1 files raise; backfill required."""
    target = tmp_path / record_filename(0, "T10020")
    # Hand-write a v1-shape file: missing proposal_coverage,
    # missing local_projection_2d, BookCoverage missing
    # articles_covered_after_proposal.
    v1 = {
        "schema_version": 1,
        "topic_id": "T10020", "topic_name": "X", "gap_idx": 0,
        "gap_stats": {"gap_size": 1, "gap_size_percentile": 1.0,
                      "gap_zscore": 0.0, "shapiro_log10_pval": 0.0},
        "topic_totals_articles": 1, "topic_totals_books": 0,
        "articles_by_centrality": [],
        "furthest_article": None,
        "covering_books": [],
    }
    target.write_text(json.dumps(v1))
    with pytest.raises(ValueError, match="schema_version"):
        read_context_record(target)
