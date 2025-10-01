"""Tests for LLM context builder."""

import numpy as np
import pytest
from unittest.mock import patch
from types import SimpleNamespace
from typing import cast

from bookgap.cloud import SubCloud
from bookgap.config_manager import AnalysisConfiguration
from bookgap.gap_finder import GapProtocol
from bookgap.gap_types import DistsAndIdxs
from bookgap.llm.context_builder import build_proposal_context
from bookgap.llm.protocols import ProposalContext


class DummyGap:
    """Minimal gap object implementing just the interface under test."""

    def __init__(self, idxs):
        self._idxs = np.array(idxs, dtype=np.intp)
        self._sqdists = np.zeros(len(self._idxs), dtype=float)

    def get_dists_idxs(self):
        return DistsAndIdxs(sqdists=self._sqdists, idxs=self._idxs)

    def get_ambient(self):
        raise NotImplementedError

    def get_landmark(self):
        raise NotImplementedError

    def get_covering_book_indices(self):
        return []

    def get_book_to_articles_mapping(self):
        return {}


class DummySubCloud:
    """Minimal SubCloud stub returning configured works."""

    def __init__(self, works):
        self._works = works

    def works(self):
        return self._works


class TestBuildProposalContext:
    """Tests for build_proposal_context function."""
    
    def create_mock_gap(self, article_count=5):
        """Create a mock gap with specified article count."""
        return DummyGap(list(range(article_count)))
    
    def create_mock_config(self, articles=3, books=5):
        """Create a mock analysis config."""
        llm_config = SimpleNamespace(
            requested_num_prompt_articles=articles,
            requested_num_prompt_books=books,
        )
        return SimpleNamespace(llm_config=llm_config)
    
    def create_mock_subcloud(self, work_count=10, work_type="article"):
        """Create a mock SubCloud with test works."""
        works = []
        for i in range(work_count):
            if work_type == "article":
                works.append(
                    type("ArticleWork", (), {
                        "display_name": f"Test {work_type.title()} {i}",
                        "abstract": f"Abstract for {work_type} {i}",
                    })()
                )
            else:
                works.append({
                    "display_name": f"Test {work_type.title()} {i}",
                    "abstract": f"Abstract for {work_type} {i}",
                })
        return DummySubCloud(works)

    def build_context(self, gap, gap_idx, topic_name, topic_id, analysis_config, books, articles):
        """Helper to call build_proposal_context with proper casting."""
        return build_proposal_context(
            gap=cast(GapProtocol, gap),
            gap_idx=gap_idx,
            topic_name=topic_name,
            topic_id=topic_id,
            analysis_config=cast(AnalysisConfiguration, analysis_config),
            books=cast(SubCloud, books),
            articles=cast(SubCloud, articles),
        )
    
    @patch('bookgap.gap_finder.group_gap_articles_by_covering_books')
    def test_successful_context_building(self, mock_group_books):
        """Test successful context building."""
        # Set up mocks
        gap = self.create_mock_gap(article_count=10)
        config = self.create_mock_config(articles=3, books=2)
        articles = self.create_mock_subcloud(work_count=15, work_type="article")
        books = self.create_mock_subcloud(work_count=8, work_type="book")
        
        # Mock book coverage
        mock_group_books.return_value = {
            0: [0, 1, 2],  # Book 0 covers articles 0, 1, 2
            1: [2, 3],     # Book 1 covers articles 2, 3
        }
        
        # Build context
        context = self.build_context(
            gap=gap,
            gap_idx=5,
            topic_name="Test Topic",
            topic_id="test_topic",
            analysis_config=config,
            books=books,
            articles=articles,
        )
        
        # Verify result
        assert isinstance(context, ProposalContext)
        assert context.topic_name == "Test Topic"
        assert context.topic_id == "test_topic"
        assert context.gap_idx == 5
        assert context.requested_num_prompt_articles == 3
        assert context.requested_num_prompt_books == 2
        assert len(context.sampled_articles) == 3  # Should sample 3 articles
        assert len(context.covering_books) == 2    # Should get 2 covering books
        assert context.counts.num_articles_for_llm == 3
        assert context.counts.num_books_for_llm == 2
    
    def test_empty_gap_raises_error(self):
        """Test that empty gap raises ValueError."""
        gap = self.create_mock_gap(article_count=0)  # Empty gap
        config = self.create_mock_config()
        articles = self.create_mock_subcloud()
        books = self.create_mock_subcloud()
        
        with pytest.raises(ValueError) as exc_info:
            self.build_context(
                gap=gap,
                gap_idx=0,
                topic_name="Empty Topic",
                topic_id="empty_topic",
                analysis_config=config,
                books=books,
                articles=articles,
            )
        
        assert "contains no articles" in str(exc_info.value)
    
    @patch('bookgap.gap_finder.group_gap_articles_by_covering_books')
    def test_deterministic_sampling(self, mock_group_books):
        """Test that sampling is deterministic for same inputs."""
        mock_group_books.return_value = {}  # No book coverage
        
        gap = self.create_mock_gap(article_count=10)
        config = self.create_mock_config(articles=3, books=2)
        articles = self.create_mock_subcloud(work_count=15)
        books = self.create_mock_subcloud(work_count=8)
        
        # Build context twice with same parameters
        context1 = self.build_context(
            gap=gap,
            gap_idx=1,
            topic_name="Test Topic",
            topic_id="test_topic",
            analysis_config=config,
            books=books,
            articles=articles,
        )

        context2 = self.build_context(
            gap=gap,
            gap_idx=1,
            topic_name="Test Topic",
            topic_id="test_topic",
            analysis_config=config,
            books=books,
            articles=articles,
        )
        
        # Should get same sampled articles
        assert context1.sampled_articles == context2.sampled_articles
        assert context1.covering_books == context2.covering_books
    
    @patch('bookgap.gap_finder.group_gap_articles_by_covering_books')
    def test_different_seeds_give_different_results(self, mock_group_books):
        """Test that different topic_id or gap_idx gives different results."""
        mock_group_books.return_value = {}
        
        gap = self.create_mock_gap(article_count=10)
        config = self.create_mock_config(articles=5, books=2)
        articles = self.create_mock_subcloud(work_count=15)
        books = self.create_mock_subcloud(work_count=8)
        
        # Different topic_id
        context1 = self.build_context(
            gap=gap,
            gap_idx=1,
            topic_name="Test",
            topic_id="topic1",
            analysis_config=config,
            books=books,
            articles=articles,
        )
        
        context2 = self.build_context(
            gap=gap,
            gap_idx=1,
            topic_name="Test",
            topic_id="topic2",
            analysis_config=config,
            books=books,
            articles=articles,
        )
        
        # Should get different samples (with high probability)
        assert context1.sampled_articles != context2.sampled_articles
    
    def test_article_validation_missing_fields(self):
        """Test that articles missing required fields raise error."""
        gap = self.create_mock_gap(article_count=3)
        config = self.create_mock_config(articles=2, books=1)
        books = self.create_mock_subcloud()
        
        # Create articles with missing fields
        class ArticleWork:
            def __init__(self, name, abstract):
                self.display_name = name
                self.abstract = abstract

        articles_data = [
            ArticleWork('Valid Article', 'Valid abstract'),
            ArticleWork('', 'Missing display_name'),  # Invalid
            ArticleWork('Valid Title', ''),  # Invalid
        ]
        mock_articles = DummySubCloud(articles_data)
        
        with pytest.raises(ValueError) as exc_info:
            self.build_context(
                gap=gap,
                gap_idx=0,
                topic_name="Test",
                topic_id="test",
                analysis_config=config,
                books=books,
                articles=mock_articles,
            )
        
        assert "missing required fields" in str(exc_info.value)
    
    @patch('bookgap.gap_finder.group_gap_articles_by_covering_books')
    def test_book_validation_skips_invalid(self, mock_group_books):
        """Test that books with missing fields are skipped, not failed."""
        # Set up book coverage
        mock_group_books.return_value = {
            0: [0, 1],  # Book 0 (valid)
            1: [1, 2],  # Book 1 (invalid - missing fields)
            2: [2, 3],  # Book 2 (valid)
        }
        
        gap = self.create_mock_gap(article_count=5)
        config = self.create_mock_config(articles=2, books=3)
        articles = self.create_mock_subcloud(work_count=5)
        
        # Create books with some missing fields
        books_data = [
            {'display_name': 'Valid Book 1', 'abstract': 'Valid abstract 1'},
            {'display_name': '', 'abstract': 'Missing display_name'},  # Invalid
            {'display_name': 'Valid Book 2', 'abstract': 'Valid abstract 2'},
        ]
        mock_books = DummySubCloud(books_data)
        
        context = self.build_context(
            gap=gap,
            gap_idx=0,
            topic_name="Test",
            topic_id="test",
            analysis_config=config,
            books=mock_books,
            articles=articles,
        )
        
        # Should only have valid books (0 and 2), skipping book 1
        assert len(context.covering_books) == 2
        assert context.covering_books[0].display_name == 'Valid Book 1'
        assert context.covering_books[1].display_name == 'Valid Book 2'
        assert context.counts.num_books_for_llm == 2
    
    @patch('bookgap.gap_finder.group_gap_articles_by_covering_books')  
    def test_no_book_coverage(self, mock_group_books):
        """Test handling when no books cover the gap articles."""
        mock_group_books.return_value = {}  # No coverage
        
        gap = self.create_mock_gap(article_count=5)
        config = self.create_mock_config(articles=3, books=2)
        articles = self.create_mock_subcloud()
        books = self.create_mock_subcloud()
        
        context = self.build_context(
            gap=gap,
            gap_idx=0,
            topic_name="Test",
            topic_id="test",
            analysis_config=config,
            books=books,
            articles=articles,
        )
        
        assert len(context.covering_books) == 0
        assert context.counts.num_books_for_llm == 0
        assert len(context.sampled_articles) == 3  # Articles still sampled
    
    @patch('bookgap.gap_finder.group_gap_articles_by_covering_books')
    def test_sampling_limits(self, mock_group_books):
        """Test that sampling respects configured limits."""
        mock_group_books.return_value = {}
        
        # Small gap, large request
        gap = self.create_mock_gap(article_count=2)
        config = self.create_mock_config(articles=10, books=5)  # Request more than available
        articles = self.create_mock_subcloud()
        books = self.create_mock_subcloud()
        
        context = self.build_context(
            gap=gap,
            gap_idx=0,
            topic_name="Test",
            topic_id="test",
            analysis_config=config,
            books=books,
            articles=articles,
        )
        
        # Should only get what's available in the gap
        assert len(context.sampled_articles) == 2  # Limited by gap size
        assert context.counts.num_articles_for_llm == 2
