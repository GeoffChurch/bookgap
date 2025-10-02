"""Unit tests for Phase 1 artifact generation functionality."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List, cast
import numpy as np

from bookgap.llm.data_structures import GapWithContext, BatchResults, ArtifactPaths

try:  # pragma: no cover - legacy module may not exist in modern codebase
    from bookgap.llm.generators.workflow_artifacts import generate_phase1_artifacts, generate_all_artifacts  # type: ignore[import,attr-defined]
except ImportError:  # pragma: no cover - gracefully skip obsolete suite when absent
    pytest.skip(
        "Legacy workflow_artifacts module missing; skipping obsolete Phase 1 artifact tests",
        allow_module_level=True,
    )

from bookgap.llm.artifacts.gap_plots import save_gap_plot_with_proposals, generate_gap_plots_for_results
from bookgap.llm.identifiers import build_gap_proposal_id
from bookgap.llm.protocols import ParsedProposal, Proposal, Chapter


class TestDataStructures:
    """Test the shared data structures."""
    
    def test_artifact_paths_creation(self):
        """Test ArtifactPaths can be created and is immutable."""
        plots = [Path("plot1.png"), Path("plot2.png")]
        artifact_paths = ArtifactPaths(plots=plots)
        
        assert artifact_paths.plots == plots
        assert artifact_paths.context_files is None
        assert artifact_paths.proposal_files is None
        assert artifact_paths.results_csv is None
        assert artifact_paths.latex_files is None
        
        # Test immutability
        with pytest.raises(AttributeError):
            artifact_paths.plots = [Path("new.png")]  # type: ignore[misc]
    
    def test_batch_results_structure(self):
        """Test BatchResults has correct fields."""
        # We can't easily create a full BatchResults without complex dependencies,
        # but we can test the structure exists
        fields = BatchResults.__dataclass_fields__
        expected_fields = {'proposals', 'metadata', 'artifact_paths'}
        assert set(fields.keys()) == expected_fields
        
    def test_gap_with_context_structure(self):
        """Test GapWithContext has correct fields."""
        fields = GapWithContext.__dataclass_fields__
        expected_fields = {'gap', 'topic_name', 'topic_id', 'gap_idx', 'books', 'articles'}
        assert set(fields.keys()) == expected_fields


class TestWorkflowArtifacts:
    """Test the workflow artifacts generation."""
    
    @patch('bookgap.llm.generators.workflow_artifacts.generate_gap_plots_for_results')
    def test_generate_phase1_artifacts(self, mock_generate_plots):
        """Test Phase 1 artifacts generation."""
        # Setup mocks
        mock_gap = Mock()
        mock_gap_with_context = Mock(spec=GapWithContext)
        mock_gap_with_context.gap = mock_gap
        gaps = [mock_gap_with_context]
        
        mock_results = Mock(spec=BatchResults)
        mock_results.proposals = []
        
        mock_output_dir = Path("/test/output")
        mock_plot_paths = [Path("/test/output/plot1.png")]
        mock_generate_plots.return_value = mock_plot_paths
        
        with patch('bookgap.llm.generators.workflow_artifacts.HydraConfig') as mock_hydra:
            mock_hydra.get.return_value.runtime.output_dir = "/test/output"
            
            # Call the function
            result = generate_phase1_artifacts(gaps, mock_results)
            
            # Verify results
            assert isinstance(result, ArtifactPaths)
            assert result.plots == mock_plot_paths
            mock_generate_plots.assert_called_once_with(gaps, mock_results.proposals, mock_output_dir)
    
    def test_generate_all_artifacts_calls_phase1(self):
        """Test that generate_all_artifacts calls Phase 1 generation."""
        with patch('bookgap.llm.generators.workflow_artifacts.generate_phase1_artifacts') as mock_phase1:
            mock_gap = Mock(spec=GapWithContext)
            mock_results = Mock(spec=BatchResults)
            
            mock_artifact_paths = Mock(spec=ArtifactPaths)
            mock_phase1.return_value = mock_artifact_paths
            
            result = generate_all_artifacts([mock_gap], mock_results)
            
            assert result == mock_artifact_paths
            mock_phase1.assert_called_once_with([mock_gap], mock_results)


class TestGapPlots:
    """Test the gap plotting functionality."""
    
    @patch('bookgap.llm.artifacts.gap_plots.plt')
    @patch('bookgap.llm.artifacts.gap_plots.get_embedding_model')
    @patch('bookgap.llm.artifacts.gap_plots._prepare_gap_visualization_data')
    def test_save_gap_plot_with_proposals(self, mock_prepare_data, mock_get_embedding, mock_plt):
        """Test saving gap plot with proposals."""
        # Setup mocks
        mock_gap_with_context = Mock(spec=GapWithContext)
        mock_gap_with_context.topic_name = "Test Topic"
        mock_gap_with_context.gap_idx = 0
        
        chapters = [Chapter(chapter_title=f"Chapter {i}") for i in range(10)]
        parsed_proposal = ParsedProposal(
            proposal=Proposal(title="Test Title", abstract="Test Abstract", chapters=chapters),
            custom_id=build_gap_proposal_id(gap_idx=0, topic_id="topic-1"),
            embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )
        proposals = [parsed_proposal]
        
        # Mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = [np.array([1.0, 2.0, 3.0])]
        mock_get_embedding.return_value = mock_embedding_model
        
        # Mock visualization data
        mock_prepare_data.return_value = (
            np.array([[0, 1], [2, 3]]),  # gap_coords
            np.array([[4, 5]]),          # proposal_coords  
            Mock(),                      # book_coords
            Mock(),                      # article_coords
            Mock()                       # book_to_articles
        )
        
        # Mock Hydra config
        with patch('bookgap.llm.artifacts.gap_plots.HydraConfig') as mock_hydra:
            mock_hydra.get.return_value.runtime.output_dir = "/test/output"
            
            # Call the function
            result_path = save_gap_plot_with_proposals(mock_gap_with_context, proposals)
            
            # Verify results
            assert isinstance(result_path, Path)
            assert str(result_path).endswith('.png')
            
            # Verify embedding was computed
            mock_get_embedding.assert_called_once()
            
            # Verify plot preparation was called
            mock_prepare_data.assert_called_once()
    
    def test_generate_gap_plots_for_results(self):
        """Test generating plots for multiple results."""
        # This is more of an integration test - we'll test the structure
        with patch('bookgap.llm.artifacts.gap_plots.save_gap_plot_with_proposals') as mock_save_plot:
            mock_gaps = [Mock(spec=GapWithContext), Mock(spec=GapWithContext)]
            
            chapters = [Chapter(chapter_title=f"Chapter {i}") for i in range(10)]
            proposal1 = ParsedProposal(
                proposal=Proposal(title="Title 1", abstract="Abstract 1", chapters=chapters),
                custom_id=build_gap_proposal_id(gap_idx=0, topic_id="topic-1"),
                embedding=np.zeros(4, dtype=np.float32),
            )
            proposal2 = ParsedProposal(
                proposal=Proposal(title="Title 2", abstract="Abstract 2", chapters=chapters),
                custom_id=build_gap_proposal_id(gap_idx=1, topic_id="topic-2"),
                embedding=np.zeros(4, dtype=np.float32),
            )
            parsed_results = [proposal1, proposal2]
            
            output_dir = Path("/test/output")
            
            mock_save_plot.return_value = Path("/test/output/plot.png")
            
            proposals_by_gap = {0: [proposal1], 1: [proposal2]}

            result_paths = generate_gap_plots_for_results(
                cast(List[GapWithContext], mock_gaps),
                parsed_results,
                output_dir,
                proposals_by_gap=proposals_by_gap,
            )
            
            # Should generate one plot per gap
            assert len(result_paths) == len(mock_gaps)
            assert all(isinstance(p, Path) for p in result_paths)
            assert mock_save_plot.call_count == len(mock_gaps)


class TestIntegration:
    """Integration tests for the complete Phase 1 workflow."""
    
    @patch('bookgap.llm.artifacts.gap_plots.plt')
    @patch('bookgap.llm.artifacts.gap_plots.get_embedding_model')
    def test_end_to_end_phase1_workflow(self, mock_get_embedding, mock_plt):
        """Test the complete Phase 1 workflow from gaps to artifact paths."""
        # Setup minimal mocks
        mock_gap_with_context = Mock(spec=GapWithContext)
        mock_gap_with_context.topic_name = "Test Topic"
        mock_gap_with_context.gap_idx = 0
        
        mock_results = Mock(spec=BatchResults)
        mock_results.proposals = []
        
        # Mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = []
        mock_get_embedding.return_value = mock_embedding_model
        
        with patch('bookgap.llm.artifacts.gap_plots._prepare_gap_visualization_data') as mock_prepare:
            mock_prepare.return_value = (
                np.array([[0, 1]]),  # gap_coords
                np.array([]),        # proposal_coords (empty)
                Mock(),              # book_coords
                Mock(),              # article_coords
                Mock()               # book_to_articles
            )
            
            with patch('bookgap.llm.generators.workflow_artifacts.HydraConfig') as mock_hydra:
                mock_hydra.get.return_value.runtime.output_dir = "/test/output"
                
                # Run the complete workflow
                artifact_paths = generate_phase1_artifacts([mock_gap_with_context], mock_results)
                
                # Verify results
                assert isinstance(artifact_paths, ArtifactPaths)
                assert len(artifact_paths.plots) == 1
                assert all(isinstance(p, Path) for p in artifact_paths.plots)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
