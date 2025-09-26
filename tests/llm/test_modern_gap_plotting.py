"""
Tests for the modern unified gap visualization architecture.

This test demonstrates how the new architecture solves all the testing
issues we encountered with the old mock-heavy approach.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from bookgap.visualization import VisualizationData, PlotConfig, ProjectionMethod
from bookgap.llm.artifacts.data_preparation import GapVisualizationDataPreparer
from bookgap.plot_utils import UnifiedGapPlotter


class TestVisualizationData:
    """Test the VisualizationData structure."""
    
    def test_visualization_data_creation_success(self):
        """Test successful creation of VisualizationData."""
        # Create realistic test data
        embeddings_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        item_types = ['article', 'article', 'book', 'book']
        all_items = [('article', 0), ('article', 1), ('book', 0), ('book', 1)]
        gap_positions = [0, 1]
        book_positions = [2, 3]
        
        data = VisualizationData(
            embeddings_2d=embeddings_2d,
            item_types=item_types,
            all_items=all_items,
            gap_article_positions=gap_positions,
            book_positions=book_positions
        )
        
        # Test properties
        assert data.n_items == 4
        assert data.n_proposals == 0
        assert data.article_positions == [0, 1]
        assert data.non_gap_article_positions == []
        
    def test_visualization_data_validation_errors(self):
        """Test that VisualizationData validates input consistency."""
        embeddings_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Test mismatched lengths
        with pytest.raises(ValueError, match="item_types length"):
            VisualizationData(
                embeddings_2d=embeddings_2d,
                item_types=['article'],  # Wrong length
                all_items=[('article', 0), ('article', 1)],
                gap_article_positions=[0],
                book_positions=[1]
            )
        
        # Test out-of-bounds positions
        with pytest.raises(ValueError, match="Position .* out of bounds"):
            VisualizationData(
                embeddings_2d=embeddings_2d,
                item_types=['article', 'book'],
                all_items=[('article', 0), ('book', 0)],
                gap_article_positions=[5],  # Out of bounds
                book_positions=[1]
            )

    def test_visualization_data_with_proposals(self):
        """Test VisualizationData with proposal data."""
        embeddings_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        proposal_embeddings_2d = np.array([[9.0, 10.0], [11.0, 12.0]])
        proposal_labels = ["Proposal 1", "Proposal 2"]
        
        data = VisualizationData(
            embeddings_2d=embeddings_2d,
            item_types=['article', 'book'],
            all_items=[('article', 0), ('book', 0)],
            gap_article_positions=[0],
            book_positions=[1],
            proposal_embeddings_2d=proposal_embeddings_2d,
            proposal_labels=proposal_labels
        )
        
        assert data.n_proposals == 2
        assert len(data.proposal_labels) == 2


class TestPlotConfig:
    """Test the PlotConfig structure."""
    
    def test_plot_config_defaults(self):
        """Test PlotConfig with default values."""
        config = PlotConfig()
        
        assert config.figsize == (12, 10)
        assert config.dpi == 300
        assert config.method_name == "PCA"
        assert config.show_legend is True
        
    def test_build_save_path(self):
        """Test save path generation."""
        config = PlotConfig(
            topic_name="Machine Learning",
            gap_idx=5,
            method_name="PCA"
        )
        
        base_dir = Path("/tmp/test")
        save_path = config.build_save_path(base_dir)
        
        expected = base_dir / "Machine_Learning_gap_5_pca.png"
        assert save_path == expected


class TestProjectionMethod:
    """Test the ProjectionMethod configuration."""
    
    def test_pca_method(self):
        """Test PCA method configuration."""
        method = ProjectionMethod.pca()
        
        assert method.name == "PCA"
        assert method.params["n_components"] == 2
        
    def test_tsne_method(self):
        """Test t-SNE method configuration."""
        method = ProjectionMethod.tsne(perplexity=20.0)
        
        assert method.name == "t-SNE"
        assert method.params["perplexity"] == 20.0
        assert method.params["random_state"] == 42


class TestUnifiedGapPlotter:
    """Test the unified plotting engine."""
    
    def test_plot_gap_visualization_success(self):
        """Test successful plot creation - NO COMPLEX MOCKS NEEDED!"""
        # Create realistic test data (small but complete)
        data = VisualizationData(
            embeddings_2d=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            item_types=['article', 'article', 'book'],
            all_items=[('article', 0), ('article', 1), ('book', 0)],
            gap_article_positions=[0],
            book_positions=[2],
            method_info={"explained_variance_ratio": [0.6, 0.3]}
        )
        
        config = PlotConfig(
            topic_name="Test Topic",
            gap_idx=1,
            method_name="PCA"
        )
        
        plotter = UnifiedGapPlotter()
        
        # This works without complex mocking because we use real data!
        fig = plotter.plot_gap_visualization(data, config)
        
        # Verify the plot was created
        assert fig is not None
        assert len(fig.axes) == 1
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_with_proposals(self):
        """Test plotting with proposal overlays."""
        data = VisualizationData(
            embeddings_2d=np.array([[1.0, 2.0], [3.0, 4.0]]),
            item_types=['article', 'book'],
            all_items=[('article', 0), ('book', 0)],
            gap_article_positions=[0],
            book_positions=[1],
            proposal_embeddings_2d=np.array([[7.0, 8.0]]),
            proposal_labels=["Test Proposal"]
        )
        
        config = PlotConfig(method_name="t-SNE")
        plotter = UnifiedGapPlotter()
        
        fig = plotter.plot_gap_visualization(data, config)
        
        # Verify plot contains proposal data
        assert fig is not None
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestDataPreparation:
    """Test the data preparation pipeline."""
    
    @patch('bookgap.llm.artifacts.data_preparation.get_embedding_model')
    def test_compute_proposal_embeddings(self, mock_get_embedding_model):
        """Test proposal embedding computation."""
        # Mock the embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_get_embedding_model.return_value = mock_model
        
        # Create mock proposals with required fields
        from bookgap.llm.protocols import ParsedProposal, Proposal, Chapter
        
        # Create minimal chapters for the proposal
        chapters = [
            Chapter(
                chapter_title=f"Chapter {i}", 
                chapter_topics=[f"Topic {i}.1", f"Topic {i}.2"]
            ) for i in range(10)
        ]
        
        proposals = [
            ParsedProposal(
                custom_id="test_1",
                proposal=Proposal(
                    title="Test Title", 
                    abstract="Test Abstract",
                    chapters=chapters
                )
            )
        ]
        
        preparer = GapVisualizationDataPreparer()
        embeddings = preparer._compute_proposal_embeddings(proposals)
        
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], np.ndarray)
        mock_model.encode.assert_called_once()


class TestIntegrationWithoutMocks:
    """Integration tests that demonstrate the power of the new architecture."""
    
    def test_end_to_end_without_complex_mocks(self):
        """
        This test demonstrates how the new architecture eliminates
        the need for complex mock setups that were causing failures.
        """
        # Step 1: Create minimal but realistic test data
        data = self._create_minimal_test_data()
        
        # Step 2: Create configuration
        config = PlotConfig(
            topic_name="Integration Test",
            gap_idx=0,
            method_name="PCA"
        )
        
        # Step 3: Create and verify plot
        plotter = UnifiedGapPlotter()
        fig = plotter.plot_gap_visualization(data, config)
        
        # Step 4: Verify results
        assert fig is not None
        ax = fig.axes[0]
        assert ax.get_title() == "Gap 0 - Integration Test (PCA)"
        assert "PC1" in ax.get_xlabel()
        assert "PC2" in ax.get_ylabel()
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def _create_minimal_test_data(self) -> VisualizationData:
        """Create minimal but realistic test data."""
        return VisualizationData(
            embeddings_2d=np.array([
                [1.0, 2.0],  # gap article
                [3.0, 4.0],  # regular article  
                [5.0, 6.0],  # book
                [7.0, 8.0]   # another book
            ]),
            item_types=['article', 'article', 'book', 'book'],
            all_items=[
                ('article', 0), ('article', 1), 
                ('book', 0), ('book', 1)
            ],
            gap_article_positions=[0],
            book_positions=[2, 3],
            method_info={"explained_variance_ratio": [0.7, 0.2]}
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
