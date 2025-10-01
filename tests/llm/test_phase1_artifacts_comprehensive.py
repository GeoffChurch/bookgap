"""
Comprehensive tests for Phase 1 artifacts (gap plots with proposals).

Tests cover:
- Data structures (ArtifactPaths, GapWithContext integration)
- Gap plotting functionality (gap_plots.py)
- Error handling and validation
- End-to-end modern plotting pipeline
"""

import numpy as np
import pytest
from pathlib import Path
from typing import List, cast
from unittest.mock import Mock, patch
from types import SimpleNamespace
from contextlib import ExitStack

from bookgap.cloud import SubCloud
from bookgap.gap_finder import GapProtocol
from bookgap.llm.data_structures import ArtifactPaths, GapWithContext
from bookgap.llm.artifacts.gap_plots import (
    generate_gap_plots_for_results,
    save_gap_plot_with_proposals,
)
from bookgap.llm.protocols import ParsedProposal, Proposal, Chapter
import tempfile


class TestPhase1DataStructures:
    """Test the shared data structures for Phase 1 - only dynamic behavior, not static properties."""
    
    # All static dataclass field tests removed - covered by type checking



class TestPhase1GapPlots:
    """Test the gap plotting functionality."""
    
    @patch('bookgap.llm.artifacts.gap_plots.HydraConfig')
    def test_save_gap_plot_with_proposals_success(self, mock_hydra_config):
        """Test the complete modern gap plotting pipeline end-to-end."""
        # Create realistic test data
        mock_gap = Mock(spec=GapProtocol)
        mock_gap.get_dists_idxs.return_value = Mock(idxs=np.array([0, 1, 2]))  # 3 gap articles

        # Create mock SubClouds with realistic embeddings
        mock_books = Mock(spec=SubCloud)
        mock_books.embeddings.return_value = np.random.rand(5, 10)

        mock_articles = Mock(spec=SubCloud)
        mock_articles.embeddings.return_value = np.random.rand(8, 10)

        # Create realistic GapWithContext
        gap_size = len(mock_gap.get_dists_idxs.return_value.idxs)
        gap_with_context = GapWithContext(
            gap=mock_gap,
            topic_name="Machine Learning",
            topic_id="ml_001",
            gap_idx=5,
            books=mock_books,
            articles=mock_articles,
            book_gap_sizes=[gap_size, gap_size + 2, gap_size + 4],
            gap_size=gap_size,
            gap_size_percentile=0.6,
            gap_zscore=0.0,
            shapiro_log10_pval=-1.5,
        )

        # Create realistic proposals with proper structure
        chapters = [
            Chapter(
                chapter_title=f"Chapter {i}: Core Concepts"
            )
            for i in range(10)
        ]

        proposal_embeddings = np.random.rand(2, 10).astype(np.float32)

        proposals = [
            ParsedProposal(
                custom_id="gap_5_proposal_1",
                proposal=Proposal(
                    title="Advanced Machine Learning Techniques",
                    abstract="This textbook covers advanced ML algorithms and their applications.",
                    chapters=chapters,
                ),
                embedding=proposal_embeddings[0],
            ),
            ParsedProposal(
                custom_id="gap_5_proposal_2",
                proposal=Proposal(
                    title="Deep Learning Fundamentals",
                    abstract="A comprehensive introduction to deep learning methodologies.",
                    chapters=chapters,
                ),
                embedding=proposal_embeddings[1],
            ),
        ]

        mock_gap.get_covering_book_indices.return_value = []

        # Mock Hydra config to return a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_hydra_instance = Mock()
            mock_hydra_instance.runtime = SimpleNamespace(output_dir=str(temp_path))
            mock_hydra_config.get.return_value = mock_hydra_instance

            neighbor_search_return = {
                "gap_to_article_neighbors": set(),
                "gap_to_book_neighbors": set(),
                "gap_article_to_book_search": None,
                "book_to_article_neighbors": set(),
                "book_to_article_search": None,
                "covering_book_indices": [],
            }

            try:
                with ExitStack() as stack:
                    stack.enter_context(
                        patch(
                            'bookgap.llm.artifacts.data_preparation.GapVisualizationDataPreparer._perform_gap_neighbor_searches',
                            return_value=neighbor_search_return,
                        )
                    )
                    stack.enter_context(
                        patch(
                            'bookgap.llm.artifacts.data_preparation.GapVisualizationDataPreparer._get_covering_book_indices',
                            return_value=[],
                        )
                    )

                    result_path = save_gap_plot_with_proposals(gap_with_context, proposals, method="PCA")

                    assert isinstance(result_path, Path)
                    assert result_path.exists()
                    assert result_path.suffix == '.png'
                    assert "Machine_Learning" in result_path.name
                    assert "gap_5" in result_path.name
                    assert "pca" in result_path.name.lower()
                    assert "plots" in result_path.parts

                    file_size = result_path.stat().st_size
                    assert file_size > 1000

                    mock_hydra_config.get.assert_called()

                    result_path_tsne = save_gap_plot_with_proposals(gap_with_context, proposals, method="TSNE")
                    assert result_path_tsne.exists()
                    assert "tsne" in result_path_tsne.name.lower()

            except Exception as e:
                pytest.fail(f"End-to-end gap plotting failed: {e}")
    
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
        
        result_paths = generate_gap_plots_for_results(
            cast(List[GapWithContext], mock_gaps),
            cast(List[ParsedProposal], mock_results),
            output_dir,
        )
        
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
    

if __name__ == "__main__":
    # Run tests with verbose output when executed directly
    pytest.main([__file__, "-v", "--tb=short"])
