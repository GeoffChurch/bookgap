"""
Test Step 9: ResultsManager Integration

Tests the integration of ResultsManager with centralized configuration management.
This step updates GapResults to use GapConfiguration objects instead of individual
parameters, ensuring consistency with the centralized system.

Test Coverage:
1. GapResults constructor with GapConfiguration
2. add_row method still works with config objects  
3. Configuration information properly stored and accessible
4. CSV import/export maintains compatibility
5. Factory creation with config objects
6. Backward compatibility for CSV loading
"""

import pytest
import tempfile
import pandas as pd
from pathlib import Path

from nextbook.config_manager import GapConfiguration, LLMConfiguration
from nextbook.results_manager import GapResults, ResultsFactory, ResultType


class TestGapResultsConfigIntegration:
    """Test GapResults integration with GapConfiguration objects."""
    
    def test_gap_results_constructor_with_config(self):
        """Test GapResults constructor accepts GapConfiguration object."""
        # Create configuration object
        gap_config = GapConfiguration(
            denoiser="median",
            drop_k=5, 
            radius_fraction=0.15
        )
        
        # Create GapResults with config object
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            results = GapResults(gap_config=gap_config, config_dir=config_dir)
            
            # Verify configuration is properly stored
            assert results.gap_config == gap_config
            assert results.gap_config.denoiser == "median"
            assert results.gap_config.drop_k == 5
            assert results.gap_config.radius_fraction == 0.15
    
    def test_gap_results_config_accessibility(self):
        """Test that config object properties are accessible through GapResults."""
        gap_config = GapConfiguration(
            denoiser="gauss",
            drop_k=10,
            radius_fraction=0.25
        )
        
        results = GapResults(gap_config=gap_config)
        
        # Verify config object is accessible
        assert results.gap_config.denoiser == "gauss"
        assert results.gap_config.drop_k == 10
        assert results.gap_config.radius_fraction == 0.25
        
        # Verify string representation works
        config_str = results.gap_config.to_string()
        assert "gauss" in config_str
        assert "k=10" in config_str
        assert "radiusfrac=0.25" in config_str
    
    def test_add_row_with_config_object(self):
        """Test add_row method works correctly with config objects."""
        gap_config = GapConfiguration(
            denoiser="bilateral",
            drop_k=8,
            radius_fraction=0.20
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            results = GapResults(gap_config=gap_config, config_dir=config_dir)
            
            # Add a row
            results.add_row(
                topic_name="Test Topic",
                topic_id="12345",
                gap_idx=1,
                num_books=50,
                num_articles=100,
                gap_size=25,
                gap_size_percentile=0.85,
                num_articles_for_llm=15,
                num_books_for_llm=10
            )
            
            # Verify row was added with correct config values
            df = results.get_df()
            assert len(df) == 1
            
            row = df.iloc[0]
            assert row['denoiser'] == "bilateral"
            assert row['drop_k'] == 8
            assert row['radius_fraction'] == 0.20
            assert row['topic_name'] == "Test Topic"
            assert row['gap_size'] == 25
    
    def test_csv_export_import_with_config(self):
        """Test CSV export/import maintains config object compatibility."""
        gap_config = GapConfiguration(
            denoiser="mean",
            drop_k=12,
            radius_fraction=0.18
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            csv_path = config_dir / "test_results.csv"
            
            # Create results and add data
            results = GapResults(gap_config=gap_config, config_dir=config_dir)
            results.add_row(
                topic_name="Export Test",
                topic_id="67890",
                gap_idx=2,
                num_books=75,
                num_articles=150,
                gap_size=30
            )
            
            # Export to CSV
            results.to_csv(csv_path)
            assert csv_path.exists()
            
            # Verify CSV content
            df = pd.read_csv(csv_path)
            assert len(df) == 1
            assert df.iloc[0]['denoiser'] == "mean"
            assert df.iloc[0]['drop_k'] == 12
            assert df.iloc[0]['radius_fraction'] == 0.18
    
    def test_factory_creation_with_config(self):
        """Test ResultsFactory can create GapResults with config objects."""
        gap_config = GapConfiguration(
            denoiser="adaptive",
            drop_k=7,
            radius_fraction=0.22
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # Create through factory
            results = ResultsFactory.create(
                ResultType.GAP_ANALYSIS,
                gap_config=gap_config,
                config_dir=config_dir
            )
            
            # Verify correct type and configuration
            assert isinstance(results, GapResults)
            assert results.gap_config == gap_config
            assert results.gap_config.denoiser == "adaptive"
    
    def test_config_object_immutability(self):
        """Test that config objects maintain immutability in results."""
        gap_config = GapConfiguration(
            denoiser="immutable_test",
            drop_k=3,
            radius_fraction=0.12
        )
        
        results = GapResults(gap_config=gap_config)
        
        # Verify config object is frozen (immutable)
        with pytest.raises(Exception):  # dataclass(frozen=True) raises FrozenInstanceError
            results.gap_config.denoiser = "modified"
        
        # Original config should remain unchanged
        assert results.gap_config.denoiser == "immutable_test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
