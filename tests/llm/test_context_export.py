"""Tests for context export functionality (Step 2.1)."""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from bookgap.llm.artifacts.context_export import save_gap_context, generate_all_contexts
from bookgap.llm.data_structures import GapWithContext


class TestContextExport:
    """Test context export functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock gap
        self.mock_gap = Mock()
        self.mock_gap.get_dists_idxs.return_value = Mock(
            idxs=np.array([10, 15, 22, 8, 12]),  # 5 articles in gap
            sqdists=np.array([0.1, 0.3, 0.8, 0.5, 0.4])  # Distances (first is center)
        )
        
        # Create mock SubClouds
        self.mock_articles = Mock()
        self.mock_books = Mock()
        
        # Mock work data - realistic article and book metadata
        self.article_works = [
            {'id': f'A{i}', 'display_name': f'Article {i}', 'abstract': f'Abstract for article {i}'}
            for i in range(25)  # 25 total articles
        ]
        
        self.book_works = [
            {'id': f'B{i}', 'display_name': f'Book {i}', 'abstract': f'Abstract for book {i}'}
            for i in range(5)  # 5 total books
        ]
        
        self.mock_articles.works.return_value = self.article_works
        self.mock_books.works.return_value = self.book_works

        # Precomputed gap statistics for testing
        self.book_gap_sizes = [2, 5, 7, 3]
        self.gap_size = len(self.mock_gap.get_dists_idxs.return_value.idxs)
        self.gap_size_percentile = 0.75
        self.gap_zscore = 1.5
        self.shapiro_log10_pval = -2.3

        # Create GapWithContext
        self.gap_with_context = GapWithContext(
            gap=self.mock_gap,
            topic_name="Machine Learning",
            topic_id="ml_123",
            gap_idx=0,
            books=self.mock_books,
            articles=self.mock_articles,
            book_gap_sizes=self.book_gap_sizes,
            gap_size=self.gap_size,
            gap_size_percentile=self.gap_size_percentile,
            gap_zscore=self.gap_zscore,
            shapiro_log10_pval=self.shapiro_log10_pval,
        )
    
    @patch('bookgap.llm.artifacts.context_export.group_gap_articles_by_covering_books')
    @patch('bookgap.llm.artifacts.context_export.get_book_to_articles_mapping')
    def test_save_gap_context_success(self, mock_book_mapping, mock_book_coverage):
        """Test successful gap context generation."""
        # Mock the gap analysis functions
        mock_book_coverage.return_value = {
            0: [10, 15],  # Book 0 covers articles 10, 15
            1: [22, 8, 12],  # Book 1 covers articles 22, 8, 12
        }
        mock_book_mapping.return_value = {
            0: Mock(idxs=np.array([10]), sqdists=np.array([0.2])),
            1: Mock(idxs=np.array([22]), sqdists=np.array([0.3])),
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Call the function
            result_paths = save_gap_context(self.gap_with_context, output_dir)
            
            # Verify files were created
            assert 'json' in result_paths
            assert 'txt' in result_paths
            assert result_paths['json'].exists()
            assert result_paths['txt'].exists()
            
            # Verify JSON structure
            with open(result_paths['json'], 'r') as f:
                context_data = json.load(f)
            
            # Check JSON structure
            assert 'metadata' in context_data
            assert 'gap_statistics' in context_data
            assert 'gap_articles' in context_data
            assert 'covering_books' in context_data
            
            # Check metadata
            metadata = context_data['metadata']
            assert metadata['topic_name'] == "Machine Learning"
            assert metadata['topic_id'] == "ml_123"
            assert metadata['gap_idx'] == 0
            
            # Check gap statistics
            stats = context_data['gap_statistics']
            assert stats['gap_size'] == 5
            assert stats['gap_size_percentile'] == 0.75
            assert stats['gap_zscore'] == 1.5
            assert stats['total_articles'] == 25
            assert stats['total_books'] == 5
            
            # Check gap articles
            gap_articles = context_data['gap_articles']
            assert len(gap_articles) == 5
            assert gap_articles[0]['is_center'] is True  # First article is center
            assert gap_articles[1]['is_center'] is False
            assert gap_articles[0]['id'] == 'A10'  # First article in gap has index 10
            
            # Check covering books
            covering_books = context_data['covering_books']
            assert len(covering_books) == 2
            # Books should be sorted by coverage (book 1 covers 3, book 0 covers 2)
            assert covering_books[0]['articles_covered'] == 3
            assert covering_books[1]['articles_covered'] == 2
            
            # Verify text file has content
            with open(result_paths['txt'], 'r') as f:
                text_content = f.read()
            
            assert "Gap Analysis for Topic: Machine Learning" in text_content
            assert "Topic ID: ml_123" in text_content
            assert "Gap Index: 0" in text_content
            assert "STATISTICAL ANALYSIS" in text_content
            assert "most central articles in the gap" in text_content
    
    def test_save_gap_context_empty_gap(self):
        """Test error handling for empty gap."""
        # Mock empty gap
        empty_gap = Mock()
        empty_gap.get_dists_idxs.return_value = Mock(
            idxs=np.array([]),
            sqdists=np.array([])
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            empty_gap_with_context = GapWithContext(
                gap=empty_gap,
                topic_name="Machine Learning",
                topic_id="ml_123",
                gap_idx=0,
                books=self.mock_books,
                articles=self.mock_articles,
                book_gap_sizes=[],
                gap_size=0,
                gap_size_percentile=float('nan'),
                gap_zscore=float('nan'),
                shapiro_log10_pval=float('nan'),
            )

            # Should raise ValueError for empty gap
            with pytest.raises(ValueError, match="Gap 0 has no articles"):
                save_gap_context(empty_gap_with_context, output_dir)
    
    @patch('bookgap.llm.artifacts.context_export.save_gap_context')
    def test_generate_all_contexts_success(self, mock_save_context):
        """Test batch context generation for multiple gaps."""
        # Mock successful context generation
        mock_save_context.return_value = {
            'json': Path('/fake/gap_0_context.json'),
            'txt': Path('/fake/gap_0_analysis.txt')
        }
        
        gaps = [self.gap_with_context]  # Single gap for testing
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            result_paths = generate_all_contexts(gaps, output_dir)
            
            # Should return both JSON and TXT paths
            assert len(result_paths) == 2
            mock_save_context.assert_called_once_with(self.gap_with_context, output_dir)
    
    @patch('bookgap.llm.artifacts.context_export.save_gap_context')
    def test_generate_all_contexts_partial_failure(self, mock_save_context):
        """Test batch context generation with some failures."""
        # First call succeeds, second fails
        mock_save_context.side_effect = [
            {'json': Path('/fake/gap_0_context.json'), 'txt': Path('/fake/gap_0_analysis.txt')},
            Exception("Mock failure")
        ]
        
        # Create two gaps
        gap2 = GapWithContext(
            gap=self.mock_gap,
            topic_name="Physics",
            topic_id="phys_456",
            gap_idx=1,
            books=self.mock_books,
            articles=self.mock_articles,
            book_gap_sizes=self.book_gap_sizes,
            gap_size=self.gap_size,
            gap_size_percentile=self.gap_size_percentile,
            gap_zscore=self.gap_zscore,
            shapiro_log10_pval=self.shapiro_log10_pval,
        )
        gaps = [self.gap_with_context, gap2]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Should not raise exception but continue processing
            result_paths = generate_all_contexts(gaps, output_dir)
            
            # Should only return paths from successful gap
            assert len(result_paths) == 2
            assert mock_save_context.call_count == 2
