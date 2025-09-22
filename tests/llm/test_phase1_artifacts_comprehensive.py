"""Comprehensive tests for Phase 1 artifact generation functionality.

This module tests the complete Phase 1 implementation including:
- Shared data structures (data_structures.py)
- Workflow artifacts generation (workflow_artifacts.py) 
- Gap plotting functionality (gap_plots.py)
- End-to-end integration testing

All tests are consolidated here for the Phase 1 artifact generation feature.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List
import numpy as np

from bookgap.llm.data_structures import GapWithContext, BatchResults, ArtifactPaths
from bookgap.llm.generators.workflow_artifacts import generate_phase1_artifacts, generate_all_artifacts
from bookgap.llm.artifacts.gap_plots import save_gap_plot_with_proposals, generate_gap_plots_for_results
from bookgap.llm.protocols import ParsedProposal


class TestPhase1DataStructures:
    """Test the shared data structures for Phase 1."""
    
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
            artifact_paths.plots = [Path("new.png")]
    
    def test_artifact_paths_with_optional_fields(self):
        """Test ArtifactPaths with some optional fields set."""
        plots = [Path("plot.png")]
        context_files = [Path("context.txt")]
        
        artifact_paths = ArtifactPaths(
            plots=plots,
            context_files=context_files
        )
        
        assert artifact_paths.plots == plots
        assert artifact_paths.context_files == context_files
        assert artifact_paths.proposal_files is None
        assert artifact_paths.results_csv is None
        assert artifact_paths.latex_files is None
    
    def test_batch_results_structure(self):
        """Test BatchResults has correct fields and types."""
        fields = BatchResults.__dataclass_fields__
        expected_fields = {'proposals', 'metadata', 'artifact_paths'}
        assert set(fields.keys()) == expected_fields
        
        # Check field types
        assert fields['proposals'].type.__origin__ is list
        assert fields['artifact_paths'].default is None
        
    def test_gap_with_context_structure(self):
        """Test GapWithContext has correct fields."""
        fields = GapWithContext.__dataclass_fields__
        expected_fields = {'gap', 'topic_name', 'topic_id', 'gap_idx', 'books', 'articles'}
        assert set(fields.keys()) == expected_fields
        
        # All fields should be required (no defaults)
        import dataclasses
        for field in fields.values():
            assert field.default is dataclasses.MISSING
            assert field.default_factory is dataclasses.MISSING


class TestPhase1WorkflowArtifacts:
    """Test the workflow artifacts generation functions."""
    
    @patch('bookgap.llm.generators.workflow_artifacts.generate_gap_plots_for_results')
    @patch('bookgap.llm.generators.workflow_artifacts.HydraConfig')
    def test_generate_phase1_artifacts(self, mock_hydra, mock_generate_plots):
        """Test Phase 1 artifacts generation with mocked dependencies."""
        # Setup mocks
        mock_gap_with_context = Mock(spec=GapWithContext)
        gaps = [mock_gap_with_context]
        
        mock_results = Mock(spec=BatchResults)
        mock_results.proposals = []
        
        mock_output_dir = Path("/test/output")
        mock_plot_paths = [Path("/test/output/plot1.png")]
        
        mock_hydra.get.return_value.runtime.output_dir = "/test/output"
        mock_generate_plots.return_value = mock_plot_paths
        
        # Call the function
        result = generate_phase1_artifacts(gaps, mock_results)
        
        # Verify results
        assert isinstance(result, ArtifactPaths)
        assert result.plots == mock_plot_paths
        assert result.context_files is None
        assert result.proposal_files is None
        assert result.results_csv is None
        assert result.latex_files is None
        
        # Verify function calls
        mock_hydra.get.assert_called_once()
        mock_generate_plots.assert_called_once_with(gaps, mock_results.proposals, mock_output_dir)
    
    @patch('bookgap.llm.generators.workflow_artifacts.generate_phase1_artifacts')
    def test_generate_all_artifacts_calls_phase1(self, mock_phase1):
        """Test that generate_all_artifacts correctly delegates to Phase 1."""
        mock_gap = Mock(spec=GapWithContext)
        mock_results = Mock(spec=BatchResults)
        
        mock_artifact_paths = Mock(spec=ArtifactPaths)
        mock_phase1.return_value = mock_artifact_paths
        
        result = generate_all_artifacts([mock_gap], mock_results)
        
        assert result == mock_artifact_paths
        mock_phase1.assert_called_once_with([mock_gap], mock_results)
    
    def test_workflow_artifacts_function_signatures(self):
        """Test that workflow functions have correct signatures."""
        import inspect
        
        sig1 = inspect.signature(generate_phase1_artifacts)
        params1 = list(sig1.parameters.keys())
        assert 'gaps' in params1
        assert 'results' in params1
        assert len(params1) == 2
        
        sig2 = inspect.signature(generate_all_artifacts)
        params2 = list(sig2.parameters.keys())
        assert 'gaps' in params2
        assert 'results' in params2
        assert len(params2) == 2


class TestPhase1GapPlots:
    """Test the gap plotting functionality."""
    
    @patch('bookgap.llm.artifacts.gap_plots.plt')
    @patch('bookgap.llm.artifacts.gap_plots.get_embedding_model')
    @patch('bookgap.llm.artifacts.gap_plots._prepare_gap_visualization_data')
    @patch('bookgap.llm.artifacts.gap_plots.HydraConfig')
    def test_save_gap_plot_with_proposals_success(self, mock_hydra, mock_prepare_data, mock_get_embedding, mock_plt):
        """Test successful gap plot generation with proposals."""
        # Setup mocks with proper attributes
        mock_gap_with_context = Mock()
        mock_gap_with_context.topic_name = "Test Topic"
        mock_gap_with_context.gap_idx = 0
        
        # Mock the gap and its related objects
        mock_gap = Mock()
        mock_books = Mock()
        mock_books.embeddings.return_value = np.array([[1, 2, 3], [4, 5, 6]])  # Mock embeddings
        mock_articles = Mock()
        mock_articles.embeddings.return_value = np.array([[7, 8, 9]])
        
        mock_gap_with_context.gap = mock_gap
        mock_gap_with_context.books = mock_books
        mock_gap_with_context.articles = mock_articles
        
        # Create proper mock proposal with required attributes
        mock_proposal = Mock()
        mock_proposal.custom_id = "gap_0_proposal_1"
        mock_proposal.title = "Test Title"
        mock_proposal.abstract = "Test Abstract"
        mock_proposals = [mock_proposal]
        
        # Mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = [np.array([1.0, 2.0, 3.0])]
        mock_get_embedding.return_value = mock_embedding_model
        
        # Mock visualization data - need to return 7 values to match the unpacking
        mock_prepare_data.return_value = (
            np.array([[0, 1], [2, 3]]),  # base_embeddings
            np.array([0, 1]),            # item_types
            Mock(),                      # all_items
            Mock(),                      # gap_article_positions_map
            Mock(),                      # all_book_positions_map
            Mock(),                      # neighbor_data
            np.array([[4, 5]]),          # proposal_embeddings_arr
        )
        
        # Mock Hydra config
        mock_hydra.get.return_value.runtime.output_dir = "/test/output"
        
        # Call the function
        result_path = save_gap_plot_with_proposals(mock_gap_with_context, mock_proposals)
        
        # Verify results
        assert isinstance(result_path, Path)
        assert str(result_path).endswith('.png')
        
        # Verify embedding was computed
        mock_get_embedding.assert_called_once()
        
        # Verify plot preparation was called
        mock_prepare_data.assert_called_once()
    
    @patch('bookgap.llm.artifacts.gap_plots.save_gap_plot_with_proposals')
    def test_generate_gap_plots_for_results_success(self, mock_save_plot):
        """Test generating plots for multiple gaps with proper mocking."""
        # Create mock gaps with required attributes
        mock_gap1 = Mock()
        mock_gap1.gap_idx = 0
        mock_gap2 = Mock()
        mock_gap2.gap_idx = 1
        mock_gaps = [mock_gap1, mock_gap2]
        
        # Create proper mock proposals with custom_id
        mock_proposal1 = Mock()
        mock_proposal1.custom_id = "gap_0_proposal_1"
        mock_proposal2 = Mock()
        mock_proposal2.custom_id = "gap_1_proposal_1"
        mock_results = [mock_proposal1, mock_proposal2]
        
        output_dir = Path("/test/output")
        
        mock_save_plot.return_value = Path("/test/output/plot.png")
        
        result_paths = generate_gap_plots_for_results(mock_gaps, mock_results, output_dir)
        
        # Should generate one plot per gap
        assert len(result_paths) == len(mock_gaps)
        assert all(isinstance(p, Path) for p in result_paths)
        assert mock_save_plot.call_count == len(mock_gaps)
    
    def test_gap_plots_function_signatures(self):
        """Test that gap plotting functions have correct signatures."""
        import inspect
        
        sig1 = inspect.signature(save_gap_plot_with_proposals)
        params1 = list(sig1.parameters.keys())
        assert 'gap_with_context' in params1
        assert 'proposals' in params1
        assert 'method' in params1
        
        sig2 = inspect.signature(generate_gap_plots_for_results)
        params2 = list(sig2.parameters.keys())
        assert 'gaps' in params2
        assert 'results' in params2
        assert 'output_dir' in params2


class TestPhase1Integration:
    """Integration tests for the complete Phase 1 workflow."""
    
    def test_module_imports_successful(self):
        """Test that all Phase 1 modules can be imported without errors."""
        # This test verifies no circular imports or syntax errors
        try:
            from bookgap.llm.data_structures import GapWithContext, BatchResults, ArtifactPaths
            from bookgap.llm.generators.workflow_artifacts import generate_phase1_artifacts, generate_all_artifacts
            from bookgap.llm.artifacts.gap_plots import save_gap_plot_with_proposals, generate_gap_plots_for_results
            
            # Verify classes and functions exist
            assert GapWithContext is not None
            assert BatchResults is not None
            assert ArtifactPaths is not None
            assert generate_phase1_artifacts is not None
            assert generate_all_artifacts is not None
            assert save_gap_plot_with_proposals is not None
            assert generate_gap_plots_for_results is not None
            
        except ImportError as e:
            pytest.fail(f"Failed to import Phase 1 modules: {e}")
    
    def test_data_structure_creation(self):
        """Test that data structures can be created with valid arguments."""
        # Test ArtifactPaths creation
        plots = [Path("test1.png"), Path("test2.png")]
        artifact_paths = ArtifactPaths(plots=plots)
        
        assert artifact_paths.plots == plots
        assert len(artifact_paths.plots) == 2
        assert all(isinstance(p, Path) for p in artifact_paths.plots)
    
    @patch('bookgap.llm.artifacts.gap_plots.plt')
    @patch('bookgap.llm.artifacts.gap_plots.get_embedding_model')
    @patch('bookgap.llm.artifacts.gap_plots._prepare_gap_visualization_data')
    @patch('bookgap.llm.generators.workflow_artifacts.HydraConfig')
    def test_end_to_end_phase1_workflow_structure(self, mock_hydra, mock_prepare, mock_get_embedding, mock_plt):
        """Test the structural integrity of the complete Phase 1 workflow."""
        # Setup minimal mocks for structure testing
        mock_gap_with_context = Mock()
        mock_gap_with_context.topic_name = "Test Topic"
        mock_gap_with_context.gap_idx = 0
        
        mock_results = Mock()
        mock_results.proposals = []
        
        # Mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = []
        mock_get_embedding.return_value = mock_embedding_model
        
        # Mock visualization data for empty proposals
        mock_prepare.return_value = (
            np.array([[0, 1]]),  # gap_coords
            np.array([]),        # proposal_coords (empty)
            Mock(),              # book_coords
            Mock(),              # article_coords
            Mock()               # book_to_articles
        )
        
        mock_hydra.get.return_value.runtime.output_dir = "/test/output"
        
        # Run the complete workflow
        artifact_paths = generate_phase1_artifacts([mock_gap_with_context], mock_results)
        
        # Verify results structure
        assert isinstance(artifact_paths, ArtifactPaths)
        assert hasattr(artifact_paths, 'plots')
        assert hasattr(artifact_paths, 'context_files')
        assert hasattr(artifact_paths, 'proposal_files')
        assert hasattr(artifact_paths, 'results_csv')
        assert hasattr(artifact_paths, 'latex_files')
    
    def test_phase1_constants_and_configuration(self):
        """Test that Phase 1 modules have expected constants and configuration."""
        from bookgap.llm.artifacts import gap_plots
        
        # Verify plotting constants exist
        assert hasattr(gap_plots, 'FIGURE_SIZE')
        assert hasattr(gap_plots, 'DPI')
        assert hasattr(gap_plots, 'FORMAT')
        assert hasattr(gap_plots, 'ALPHA_GRID')
        assert hasattr(gap_plots, 'NEAREST_NEIGHBOR_K')
        
        # Verify values are reasonable
        assert gap_plots.FIGURE_SIZE == (12, 10)
        assert gap_plots.DPI == 300
        assert gap_plots.FORMAT == "png"
        assert 0 <= gap_plots.ALPHA_GRID <= 1
        assert gap_plots.NEAREST_NEIGHBOR_K > 0


class TestPhase1EndToEndDictConfig:
    """End-to-end tests with real DictConfig objects to test the exact code path that was breaking."""
    
    def test_serialize_config_with_real_dictconfig_from_asdict(self):
        """Test the exact code path that was failing: serialize_config called with DictConfig."""
        from omegaconf import OmegaConf
        from bookgap.llm.generators.workflow_functions import MetadataUtils
        
        # Create a DictConfig that mimics what Hydra provides
        # This is the exact scenario that was failing before the fix
        config_data = {
            "experiment": {
                "name": "test_batch_job",
                "description": "Testing DictConfig serialization"
            },
            "llm": {
                "model": {
                    "name": "gpt-4o-mini",
                    "uses_max_completion_tokens": True
                },
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "batch": {
                "size": 5,
                "timeout_seconds": 3600
            }
        }
        
        # Create DictConfig - this is what caused the original asdict() failure
        dict_config = OmegaConf.create(config_data)
        
        # Verify it's actually a DictConfig (not a regular dict)
        assert OmegaConf.is_config(dict_config)
        assert type(dict_config).__name__ == 'DictConfig'
        
        # This is the exact call that was failing before the fix
        # It should work now with OmegaConf.to_container() instead of asdict()
        try:
            metadata = MetadataUtils.serialize_config(dict_config)
            
            # Verify serialization succeeded
            assert isinstance(metadata, dict)
            assert len(metadata) > 0
            
            # Verify all keys are string indices and values are strings
            assert all(isinstance(k, str) and k.isdigit() for k in metadata.keys())
            assert all(isinstance(v, str) for v in metadata.values())
            
            # Verify we can reconstruct the original data
            combined_json = ''.join(metadata[str(i)] for i in range(len(metadata)))
            import json
            reconstructed_data = json.loads(combined_json)
            
            # Verify the data was preserved correctly
            assert reconstructed_data["experiment"]["name"] == "test_batch_job"
            assert reconstructed_data["llm"]["model"]["name"] == "gpt-4o-mini"
            assert reconstructed_data["batch"]["size"] == 5
            
            print("✅ DictConfig serialize_config works correctly")
            
        except Exception as e:
            pytest.fail(f"serialize_config failed with DictConfig (this was the original bug): {e}")
    
    def test_serialize_config_dictconfig_only(self):
        """Test that serialize_config now only handles DictConfig objects (after simplification)."""
        from omegaconf import OmegaConf
        from dataclasses import dataclass
        from bookgap.llm.generators.workflow_functions import MetadataUtils
        
        # Test with DictConfig - should work
        dict_config = OmegaConf.create({"model": "gpt-4", "temp": 0.5})
        metadata_dict = MetadataUtils.serialize_config(dict_config)
        assert isinstance(metadata_dict, dict)
        assert len(metadata_dict) > 0
        
        # Test with a regular dataclass - should now raise ValueError
        @dataclass
        class TestConfig:
            model: str
            temp: float
        
        dataclass_config = TestConfig(model="gpt-4", temp=0.5)
        
        # This should now raise an error since we simplified to DictConfig-only
        with pytest.raises(ValueError, match="Config must be an OmegaConf DictConfig"):
            MetadataUtils.serialize_config(dataclass_config)
        
        print("✅ serialize_config correctly enforces DictConfig-only behavior")
    
    @patch('bookgap.llm.generators.workflow_functions.OpenAI')
    def test_submit_batch_job_serialize_config_integration(self, mock_openai_class):
        """Test that submit_batch_job properly calls serialize_config with DictConfig."""
        from omegaconf import OmegaConf
        from bookgap.llm.generators.workflow_functions import MetadataUtils
        
        # Create a minimal DictConfig that matches what submit_batch_job expects
        config = OmegaConf.create({
            "llm": {
                "model": {
                    "uses_max_completion_tokens": True
                }
            }
        })
        
        # Test that serialize_config works with this config structure
        try:
            metadata = MetadataUtils.serialize_config(config)
            
            # Verify it worked and contains the expected structure
            combined_json = ''.join(metadata[str(i)] for i in range(len(metadata)))
            import json
            data = json.loads(combined_json)
            
            assert "llm" in data
            assert data["llm"]["model"]["uses_max_completion_tokens"] is True
            
            print("✅ serialize_config integration test passed")
            
        except Exception as e:
            pytest.fail(f"serialize_config integration failed: {e}")


class TestPhase1ErrorHandling:
    """Test error handling and edge cases for Phase 1."""
    
    def test_artifact_paths_requires_plots(self):
        """Test that ArtifactPaths requires plots argument."""
        with pytest.raises(TypeError):
            ArtifactPaths()  # Should fail - plots is required
    
    def test_empty_plots_list_allowed(self):
        """Test that empty plots list is allowed."""
        artifact_paths = ArtifactPaths(plots=[])
        assert artifact_paths.plots == []
        assert len(artifact_paths.plots) == 0
    
    @patch('bookgap.llm.artifacts.gap_plots.get_embedding_model')
    def test_gap_plots_with_empty_proposals(self, mock_get_embedding):
        """Test gap plotting handles empty proposals gracefully."""
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = []
        mock_get_embedding.return_value = mock_embedding_model
        
        from bookgap.llm.artifacts.gap_plots import _compute_proposal_embeddings
        
        # Should handle empty list without error
        result = _compute_proposal_embeddings([])
        assert result == []


if __name__ == "__main__":
    # Run tests with verbose output when executed directly
    pytest.main([__file__, "-v", "--tb=short"])
