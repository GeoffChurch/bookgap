from types import SimpleNamespace
from typing import cast

import faiss
import numpy as np
import pandas as pd
import pytest

from bookgap.subcloud import SubCloud
from bookgap.cloud_index import ProposalRankingResult
from bookgap.configs.config import ExperimentConfig
from bookgap.gap_types import DistsAndIdxs
from bookgap.llm.artifacts import results_export
from bookgap.llm.artifacts.results_export import (
    _sanitize_text_field,
    generate_gaps_analysis_csv,
)
from bookgap.llm.data_structures import GapWithContext
from bookgap.llm.identifiers import build_gap_proposal_id, parse_gap_proposal_id
from bookgap.llm.protocols import (
    Article,
    Book,
    # Chapter,
    ParsedProposal,
    Proposal,
    ProposalContext,
    ProposalCounts,
)


class _DummySubCloud:
    def __init__(self, embeddings: np.ndarray):
        self._embeddings = embeddings.astype(np.float32)
        self._index = faiss.IndexFlatL2(self._embeddings.shape[1])
        self._index.add(self._embeddings)  # type: ignore[call-arg]

    def embeddings(self) -> np.ndarray:
        return self._embeddings

    def index(self):
        return self._index


class _DummyGap:
    def __init__(self, center_idx: int = 0):
        self._dists_idxs = DistsAndIdxs(
            sqdists=np.array([0.0], dtype=np.float32),
            idxs=np.array([center_idx], dtype=np.intp),
        )

    def get_dists_idxs(self) -> DistsAndIdxs:
        return self._dists_idxs

    # The following methods satisfy the GapProtocol interface but aren't used in tests
    def get_ambient(self):  # pragma: no cover - not needed for ranking tests
        raise NotImplementedError

    def get_landmark(self):  # pragma: no cover - not needed for ranking tests
        raise NotImplementedError

    def get_covering_book_indices(self):  # pragma: no cover - not needed
        return []

    def get_book_to_articles_mapping(self):  # pragma: no cover - not needed
        return {}


def test_sanitize_text_replaces_newline_with_literal_escape():
    assert _sanitize_text_field("Line 1\nLine 2") == "Line 1\\nLine 2"


def test_sanitize_text_replaces_common_control_characters():
    text = "Before\rMiddle\tEnd\x0b"
    expected = "Before\\rMiddle\\tEnd\\x0b"
    assert _sanitize_text_field(text) == expected


def test_sanitize_text_preserves_unicode_characters():
    text = "Résumé – 数据"
    assert _sanitize_text_field(text) == text


def test_generate_gaps_analysis_csv_adds_rank_columns(tmp_path, monkeypatch):
    books_embeddings = np.ones((4, 3), dtype=np.float32)
    articles_embeddings = np.arange(12, dtype=np.float32).reshape(4, 3)
    gap = _DummyGap(center_idx=1)
    gap_with_context = GapWithContext(
        gap=gap,
        topic_name="Topic",
        topic_id="topic-1",
        gap_idx=0,
        books=cast(SubCloud, _DummySubCloud(books_embeddings)),
        articles=cast(SubCloud, _DummySubCloud(articles_embeddings)),
        book_gap_sizes=[2, 3, 4],
        gap_size=2,
        gap_size_percentile=0.5,
        gap_zscore=1.2,
        shapiro_log10_pval=-2.5,
    )

    proposal_model = Proposal(
        title="Example Title",
        abstract="Example Abstract",
        # chapters=[Chapter(chapter_title=f"Chapter {i}") for i in range(10)],
    )
    parsed_proposal = ParsedProposal(
        proposal=proposal_model,
        custom_id=build_gap_proposal_id(gap_idx=0, topic_id="topic-1"),
        embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )

    proposals_grouped = {("topic-1", 0): [parsed_proposal]}
    ordered_proposals = [
        (parsed_proposal, parse_gap_proposal_id(parsed_proposal.custom_id))
    ]

    context = ProposalContext(
        topic_name="Topic",
        topic_id="topic-1",
        gap_idx=0,
        requested_num_prompt_articles=10,
        requested_num_prompt_books=5,
        sampled_articles=[Article(display_name="Article", abstract="Abstract")],
        covering_books=[Book(display_name="Book", abstract="Abstract")],
        counts=ProposalCounts(num_articles_for_llm=3, num_books_for_llm=2),
    )

    config = cast(
        ExperimentConfig,
        SimpleNamespace(
            gap=SimpleNamespace(
                denoiser=SimpleNamespace(name="NONE"),
                drop_k=1,
                radius_fraction=0.5,
                cell_type=SimpleNamespace(name="KBALL"),
            ),
            llm=SimpleNamespace(
                num_articles=10,
                num_books=5,
            ),
            cloud=SimpleNamespace(
                denoiser="NONE",
                drop_k=1,
            ),
        ),
    )

    def fake_compute(*_args, **_kwargs):
        return ProposalRankingResult(
            book_ranks=np.array([2], dtype=np.int32),
            article_ranks=np.array([4], dtype=np.int32),
            proposal_distances=np.array([0.5], dtype=np.float32),
        )

    monkeypatch.setattr(results_export, "compute_proposal_rankings_for_gap", fake_compute)

    csv_path = generate_gaps_analysis_csv(
        gaps=[gap_with_context],
        proposals=[parsed_proposal],
        output_dir=tmp_path,
        config=config,
        contexts=[context],
        proposals_grouped=proposals_grouped,
        ordered_proposals=ordered_proposals,
    )

    df = pd.read_csv(csv_path)
    assert "proposal_book_rank" in df.columns
    assert "proposal_article_rank" in df.columns
    assert "proposal_book_rank_fraction" in df.columns
    assert "proposal_article_rank_fraction" in df.columns

    assert df.loc[0, "proposal_book_rank"] == 2
    assert df.loc[0, "proposal_article_rank"] == 4
    assert df.loc[0, "proposal_book_rank_fraction"] == pytest.approx(
        2 / len(books_embeddings)
    )
    assert df.loc[0, "proposal_article_rank_fraction"] == pytest.approx(
        4 / len(articles_embeddings)
    )


def test_generate_gaps_analysis_csv_real_rankings(tmp_path):
    books_embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )
    articles_embeddings = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )

    gap = _DummyGap(center_idx=0)
    gap_with_context = GapWithContext(
        gap=gap,
        topic_name="Topic",
        topic_id="topic-1",
        gap_idx=0,
        books=cast(SubCloud, _DummySubCloud(books_embeddings)),
        articles=cast(SubCloud, _DummySubCloud(articles_embeddings)),
        book_gap_sizes=[1, 2, 3],
        gap_size=2,
        gap_size_percentile=0.6,
        gap_zscore=1.0,
        shapiro_log10_pval=-1.0,
    )

    # proposal_chapters = [Chapter(chapter_title=f"Chapter {i}") for i in range(10)]
    proposal_a = ParsedProposal(
        proposal=Proposal(title="A", abstract="First"),#, chapters=proposal_chapters),
        custom_id=build_gap_proposal_id(gap_idx=0, topic_id="topic-1"),
        embedding=np.array([0.5, 0.0], dtype=np.float32),
    )
    proposal_b = ParsedProposal(
        proposal=Proposal(title="B", abstract="Second"),#, chapters=proposal_chapters),
        custom_id=build_gap_proposal_id(gap_idx=0, topic_id="topic-1"),
        embedding=np.array([2.5, 0.0], dtype=np.float32),
    )

    proposals_grouped = {("topic-1", 0): [proposal_a, proposal_b]}
    ordered_proposals = [
        (proposal_a, parse_gap_proposal_id(proposal_a.custom_id)),
        (proposal_b, parse_gap_proposal_id(proposal_b.custom_id)),
    ]

    context = ProposalContext(
        topic_name="Topic",
        topic_id="topic-1",
        gap_idx=0,
        requested_num_prompt_articles=10,
        requested_num_prompt_books=5,
        sampled_articles=[Article(display_name="Article", abstract="Abstract")],
        covering_books=[Book(display_name="Book", abstract="Abstract")],
        counts=ProposalCounts(num_articles_for_llm=3, num_books_for_llm=2),
    )

    config = cast(
        ExperimentConfig,
        SimpleNamespace(
            gap=SimpleNamespace(
                denoiser=SimpleNamespace(name="NONE"),
                drop_k=1,
                radius_fraction=0.5,
                cell_type=SimpleNamespace(name="KBALL"),
            ),
            llm=SimpleNamespace(
                num_articles=10,
                num_books=5,
            ),
            cloud=SimpleNamespace(
                denoiser="NONE",
                drop_k=1,
            ),
        ),
    )

    csv_path = generate_gaps_analysis_csv(
        gaps=[gap_with_context],
        proposals=[proposal_a, proposal_b],
        output_dir=tmp_path,
        config=config,
        contexts=[context],
        proposals_grouped=proposals_grouped,
        ordered_proposals=ordered_proposals,
    )

    df = pd.read_csv(csv_path).sort_values("proposal_title").reset_index(drop=True)

    assert list(df["proposal_title"]) == ["A", "B"]
    assert list(df["proposal_book_rank"]) == [1, 2]
    assert list(df["proposal_article_rank"]) == [1, 2]

    num_books = len(books_embeddings)
    num_articles = len(articles_embeddings)
    expected_book_frac = [1 / num_books, 2 / num_books]
    expected_article_frac = [1 / num_articles, 2 / num_articles]

    assert list(df["proposal_book_rank_fraction"]) == pytest.approx(expected_book_frac)
    assert list(df["proposal_article_rank_fraction"]) == pytest.approx(expected_article_frac)
