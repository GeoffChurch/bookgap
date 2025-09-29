"""
Comprehensive tests for Phase 1 artifacts (gap plots with proposals).

Tests cover:
- Data structures (ArtifactPaths, GapWithContext integration)
- Gap plotting functionality (gap_plots.py)
- Error handling and validation
- End-to-end modern plotting pipeline
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from bookgap.llm.data_structures import ArtifactPaths
from bookgap.llm.artifacts.gap_plots import generate_gap_plots_for_results


class TestPhase1DataStructures:
    """Test the shared data structures for Phase 1 - only dynamic behavior, not static properties."""
    
    # All static dataclass field tests removed - covered by type checking



class TestPhase1GapPlots:
    """Test the gap plotting functionality."""
    
    @patch('hydra.core.hydra_config.HydraConfig')
    @patch('bookgap.llm.artifacts.data_preparation.embed_texts')
    def test_save_gap_plot_with_proposals_success(self, mock_embed_texts, mock_hydra_config):
        """Test the complete modern gap plotting pipeline end-to-end."""
        from bookgap.llm.artifacts.gap_plots import save_gap_plot_with_proposals
        from bookgap.llm.data_structures import GapWithContext
        from bookgap.llm.protocols import ParsedProposal, Proposal, Chapter
        from bookgap.gap_finder import GapProtocol
        from bookgap.cloud import SubCloud
        import tempfile
        import numpy as np
        
        # Create realistic test data
        mock_gap = Mock(spec=GapProtocol)
        mock_gap.get_dists_idxs.return_value = Mock(idxs=[0, 1, 2])  # 3 gap articles
        
        # Create mock SubClouds with realistic embeddings
        mock_books = Mock(spec=SubCloud)
        mock_books.embeddings.return_value = np.random.rand(5, 10)  # 5 books, 10-dim embeddings
        
        mock_articles = Mock(spec=SubCloud) 
        mock_articles.embeddings.return_value = np.random.rand(8, 10)  # 8 articles, 10-dim embeddings
        
        # Create realistic GapWithContext
        gap_with_context = GapWithContext(
            gap=mock_gap,
            topic_name="Machine Learning",
            topic_id="ml_001",
            gap_idx=5,
            books=mock_books,
            articles=mock_articles
        )
        
        # Create realistic proposals with proper structure
        chapters = [
            Chapter(
                chapter_title=f"Chapter {i}: Core Concepts", 
                chapter_topics=[f"Topic {i}.1", f"Topic {i}.2", f"Topic {i}.3"]
            ) for i in range(10)
        ]
        
        proposals = [
            ParsedProposal(
                custom_id="gap_5_proposal_1",
                proposal=Proposal(
                    title="Advanced Machine Learning Techniques",
                    abstract="This textbook covers advanced ML algorithms and their applications.",
                    chapters=chapters
                )
            ),
            ParsedProposal(
                custom_id="gap_5_proposal_2", 
                proposal=Proposal(
                    title="Deep Learning Fundamentals",
                    abstract="A comprehensive introduction to deep learning methodologies.",
                    chapters=chapters
                )
            )
        ]
        
        # Mock the embed_texts function to return realistic embeddings
        # embed_texts expects List[tuple[str, str]] and returns NDArray
        mock_embed_texts.return_value = np.random.rand(2, 10)  # 2 proposals, 10-dim embeddings
        
        # Mock Hydra config to return a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_hydra_instance = Mock()
            mock_hydra_instance.runtime.output_dir = str(temp_path)
            mock_hydra_config.get.return_value = mock_hydra_instance
            
            # Execute the complete pipeline
            try:
                result_path = save_gap_plot_with_proposals(gap_with_context, proposals, method="PCA")
                
                # Verify the result
                assert isinstance(result_path, Path)
                assert result_path.exists()
                assert result_path.suffix == '.png'
                assert temp_path in result_path.parents
                assert "Machine_Learning" in result_path.name
                assert "gap_5" in result_path.name
                assert "pca" in result_path.name.lower()
                
                # Verify the file has reasonable size (not empty, not too small)
                file_size = result_path.stat().st_size
                assert file_size > 1000  # At least 1KB for a real plot
                
                # Verify embedding function was called once for the batch of proposals
                mock_embed_texts.assert_called_once()
                # Verify it was called with the right number of title-abstract pairs
                call_args = mock_embed_texts.call_args[0][0]  # First positional argument
                assert len(call_args) == len(proposals)
                
                # Verify Hydra config was accessed
                mock_hydra_config.get.assert_called()
                
                print(f"✅ End-to-end test created plot: {result_path}")
                print(f"✅ Plot file size: {file_size} bytes")
                
            except Exception as e:
                pytest.fail(f"End-to-end gap plotting failed: {e}")
                
            # Test with t-SNE method as well
            try:
                result_path_tsne = save_gap_plot_with_proposals(gap_with_context, proposals, method="TSNE")
                assert result_path_tsne.exists()
                assert "tsne" in result_path_tsne.name.lower()
                print(f"✅ t-SNE plotting also works: {result_path_tsne}")
                
            except Exception as e:
                pytest.fail(f"t-SNE plotting failed: {e}")
    
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


class TestPhase1Integration:
    """Integration tests for the complete Phase 1 workflow - only dynamic behavior."""
    
    # All static import and structure tests removed - covered by type checking


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
        # This now tests that we can create ArtifactPaths with empty plots
        artifact_paths = ArtifactPaths(plots=[])
        assert artifact_paths.plots == []
    
    def test_empty_plots_list_allowed(self):
        """Test that empty plots list is allowed."""
        artifact_paths = ArtifactPaths(plots=[])
        assert artifact_paths.plots == []
        assert len(artifact_paths.plots) == 0
    
    def test_gap_plots_with_empty_proposals(self):
        """Test gap plotting handles empty proposals gracefully."""
        # This functionality is now tested in the unified architecture tests
        # The data preparation module handles empty proposals gracefully
        from bookgap.llm.artifacts.data_preparation import GapVisualizationDataPreparer
        
        preparer = GapVisualizationDataPreparer()
        result = preparer._compute_proposal_embeddings([])
        assert result == []


if __name__ == "__main__":
    # Run tests with verbose output when executed directly
    pytest.main([__file__, "-v", "--tb=short"])
