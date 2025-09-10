"""Tests for LLM context builder."""

import pytest
from unittest.mock import patch, MagicMock
import random

from bookgap.llm.context_builder import build_proposal_context
from bookgap.llm.protocols import ProposalContext


class TestBuildProposalContext:
    """Tests for build_proposal_context function."""
    
    def create_mock_gap(self, article_count=5):
        """Create a mock gap with specified article count."""
        mock_gap = MagicMock()
        mock_gap.dists_idxs.idxs = list(range(article_count))
        return mock_gap
    
    def create_mock_config(self, articles=3, books=5):
        """Create a mock analysis config."""
        mock_config = MagicMock()
        mock_config.llm_config.requested_num_prompt_articles = articles
        mock_config.llm_config.requested_num_prompt_books = books
        return mock_config
    
    def create_mock_subcloud(self, work_count=10, work_type="article"):
        """Create a mock SubCloud with test works."""
        mock_cloud = MagicMock()
        works = []
        for i in range(work_count):
            work = {
                'display_name': f'Test {work_type.title()} {i}',
                'abstract': f'Abstract for {work_type} {i}'
            }
            works.append(work)
        mock_cloud.works.return_value = works
        return mock_cloud
    
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
        context = build_proposal_context(
            gap=gap,
            gap_idx=5,
            topic_name="Test Topic",
            topic_id="test_topic",
            analysis_config=config,
            books=books,
            articles=articles
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
        assert context.counts["num_articles_for_llm"] == 3
        assert context.counts["num_books_for_llm"] == 2
    
    def test_empty_gap_raises_error(self):
        """Test that empty gap raises ValueError."""
        gap = self.create_mock_gap(article_count=0)  # Empty gap
        config = self.create_mock_config()
        articles = self.create_mock_subcloud()
        books = self.create_mock_subcloud()
        
        with pytest.raises(ValueError) as exc_info:
            build_proposal_context(
                gap=gap,
                gap_idx=0,
                topic_name="Empty Topic",
                topic_id="empty_topic",
                analysis_config=config,
                books=books,
                articles=articles
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
        context1 = build_proposal_context(
            gap=gap,
            gap_idx=1,
            topic_name="Test Topic",
            topic_id="test_topic",
            analysis_config=config,
            books=books,
            articles=articles
        )
        
        context2 = build_proposal_context(
            gap=gap,
            gap_idx=1,
            topic_name="Test Topic",
            topic_id="test_topic",
            analysis_config=config,
            books=books,
            articles=articles
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
        context1 = build_proposal_context(
            gap=gap, gap_idx=1, topic_name="Test", topic_id="topic1",
            analysis_config=config, books=books, articles=articles
        )
        
        context2 = build_proposal_context(
            gap=gap, gap_idx=1, topic_name="Test", topic_id="topic2",
            analysis_config=config, books=books, articles=articles
        )
        
        # Should get different samples (with high probability)
        assert context1.sampled_articles != context2.sampled_articles
    
    def test_article_validation_missing_fields(self):
        """Test that articles missing required fields raise error."""
        gap = self.create_mock_gap(article_count=3)
        config = self.create_mock_config(articles=2, books=1)
        books = self.create_mock_subcloud()
        
        # Create articles with missing fields
        mock_articles = MagicMock()
        articles_data = [
            {'display_name': 'Valid Article', 'abstract': 'Valid abstract'},
            {'display_name': '', 'abstract': 'Missing display_name'},  # Invalid
            {'display_name': 'Valid Title', 'abstract': ''},  # Invalid
        ]
        mock_articles.works.return_value = articles_data
        
        with pytest.raises(ValueError) as exc_info:
            build_proposal_context(
                gap=gap, gap_idx=0, topic_name="Test", topic_id="test",
                analysis_config=config, books=books, articles=mock_articles
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
        mock_books = MagicMock()
        books_data = [
            {'display_name': 'Valid Book 1', 'abstract': 'Valid abstract 1'},
            {'display_name': '', 'abstract': 'Missing display_name'},  # Invalid
            {'display_name': 'Valid Book 2', 'abstract': 'Valid abstract 2'},
        ]
        mock_books.works.return_value = books_data
        
        context = build_proposal_context(
            gap=gap, gap_idx=0, topic_name="Test", topic_id="test",
            analysis_config=config, books=mock_books, articles=articles
        )
        
        # Should only have valid books (0 and 2), skipping book 1
        assert len(context.covering_books) == 2
        assert context.covering_books[0]['display_name'] == 'Valid Book 1'
        assert context.covering_books[1]['display_name'] == 'Valid Book 2'
        assert context.counts["num_books_for_llm"] == 2
    
    @patch('bookgap.gap_finder.group_gap_articles_by_covering_books')  
    def test_no_book_coverage(self, mock_group_books):
        """Test handling when no books cover the gap articles."""
        mock_group_books.return_value = {}  # No coverage
        
        gap = self.create_mock_gap(article_count=5)
        config = self.create_mock_config(articles=3, books=2)
        articles = self.create_mock_subcloud()
        books = self.create_mock_subcloud()
        
        context = build_proposal_context(
            gap=gap, gap_idx=0, topic_name="Test", topic_id="test",
            analysis_config=config, books=books, articles=articles
        )
        
        assert len(context.covering_books) == 0
        assert context.counts["num_books_for_llm"] == 0
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
        
        context = build_proposal_context(
            gap=gap, gap_idx=0, topic_name="Test", topic_id="test",
            analysis_config=config, books=books, articles=articles
        )
        
        # Should only get what's available in the gap
        assert len(context.sampled_articles) == 2  # Limited by gap size
        assert context.counts["num_articles_for_llm"] == 2
