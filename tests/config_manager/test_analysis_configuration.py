#!/usr/bin/env python3
"""
Test-Driven Development for AnalysisConfiguration consolidation.

This module contains comprehensive test cases for the new consolidated
AnalysisConfiguration class that replaces GapConfiguration + LLMConfiguration.
"""

import pytest
import tempfile
from pathlib import Path
from nextbook.config_manager import AnalysisConfiguration  # Will fail initially
from nextbook.gap_directory_manager import GapDirectoryManager


class TestAnalysisConfiguration:
    """Test the consolidated AnalysisConfiguration class."""
    
    def test_valid_configuration_creation(self):
        """Test creating valid AnalysisConfiguration instances."""
        # Basic kball configuration
        config = AnalysisConfiguration(
            denoiser="NONE",
            drop_k=0,
            radius_fraction=0.75,
            cell_type="kball",
            requested_num_prompt_articles=50,
            requested_num_prompt_books=10
        )
        
        assert config.denoiser == "NONE"
        assert config.drop_k == 0
        assert config.radius_fraction == 0.75
        assert config.cell_type == "kball"
        assert config.requested_num_prompt_articles == 50
        assert config.requested_num_prompt_books == 10
        
        # Voronoi configuration
        voronoi_config = AnalysisConfiguration(
            denoiser="JAD_EQUAL",
            drop_k=5,
            radius_fraction=0.5,
            cell_type="voronoi",
            requested_num_prompt_articles=100,
            requested_num_prompt_books=25
        )
        
        assert voronoi_config.cell_type == "voronoi"
    
    def test_invalid_cell_type_validation(self):
        """Test that invalid cell types raise ValueError."""
        with pytest.raises(ValueError, match="cell_type must be either 'kball' or 'voronoi'"):
            AnalysisConfiguration(
                denoiser="NONE",
                drop_k=0,
                radius_fraction=0.75,
                cell_type="invalid",  # Invalid cell type
                requested_num_prompt_articles=50,
                requested_num_prompt_books=10
            )
    
    def test_edge_case_values(self):
        """Test edge case parameter values."""
        # Zero books (valid scenario)
        config_no_books = AnalysisConfiguration(
            denoiser="NONE",
            drop_k=0,
            radius_fraction=0.75,
            cell_type="kball",
            requested_num_prompt_articles=100,
            requested_num_prompt_books=0  # Valid: topics can have no books
        )
        assert config_no_books.requested_num_prompt_books == 0
        
        # Negative values should raise errors
        with pytest.raises(ValueError, match="num_articles must be non-negative"):
            AnalysisConfiguration(
                denoiser="NONE",
                drop_k=0,
                radius_fraction=0.75,
                cell_type="kball",
                requested_num_prompt_articles=-1,  # Invalid
                requested_num_prompt_books=10
            )
        
        with pytest.raises(ValueError, match="num_books must be non-negative"):
            AnalysisConfiguration(
                denoiser="NONE",
                drop_k=0,
                radius_fraction=0.75,
                cell_type="kball",
                requested_num_prompt_articles=50,
                requested_num_prompt_books=-1  # Invalid
            )
    
    def test_string_representation(self):
        """Test string conversion for directory naming."""
        config = AnalysisConfiguration(
            denoiser="JAD_EQUAL",
            drop_k=5,
            radius_fraction=0.25,
            cell_type="voronoi",
            requested_num_prompt_articles=75,
            requested_num_prompt_books=20
        )
        
        directory_name = config.to_directory_name()
        expected = "JAD_EQUAL_k=5_radiusfrac=0.25_voronoi"
        assert directory_name == expected
        
        full_string = config.to_string()
        expected_full = "JAD_EQUAL_k=5_radiusfrac=0.25_voronoi_articles=75_books=20"
        assert full_string == expected_full
    
    def test_string_parsing(self):
        """Test parsing configuration from strings."""
        config_str = "NONE_k=0_radiusfrac=0.75_kball_articles=50_books=10"
        config = AnalysisConfiguration.from_string(config_str)
        
        assert config.denoiser == "NONE"
        assert config.drop_k == 0
        assert config.radius_fraction == 0.75
        assert config.cell_type == "kball"
        assert config.requested_num_prompt_articles == 50
        assert config.requested_num_prompt_books == 10
    
    def test_malformed_string_parsing(self):
        """Test error handling for malformed config strings."""
        with pytest.raises(ValueError, match="Invalid configuration string format"):
            AnalysisConfiguration.from_string("invalid_format")
        
        with pytest.raises(ValueError, match="Invalid configuration string format"):
            AnalysisConfiguration.from_string("NONE_k=0_radiusfrac=0.75")  # Missing parts
    
    def test_round_trip_string_conversion(self):
        """Test that string conversion is consistent."""
        original = AnalysisConfiguration(
            denoiser="JAD_EQUAL",
            drop_k=10,
            radius_fraction=0.33,
            cell_type="voronoi",
            requested_num_prompt_articles=200,
            requested_num_prompt_books=50
        )
        
        config_str = original.to_string()
        parsed = AnalysisConfiguration.from_string(config_str)
        
        assert parsed.denoiser == original.denoiser
        assert parsed.drop_k == original.drop_k
        assert parsed.radius_fraction == original.radius_fraction
        assert parsed.cell_type == original.cell_type
        assert parsed.requested_num_prompt_articles == original.requested_num_prompt_articles
        assert parsed.requested_num_prompt_books == original.requested_num_prompt_books


class TestAlgorithmSelection:
    """Test algorithm selection based on cell_type."""
    
    def test_algorithm_dispatch_mechanism(self):
        """Test that cell_type correctly dispatches to different algorithms."""
        # This test will drive the implementation of algorithm selection
        from nextbook.gap_finder import _select_gap_algorithm  # Will need to create this
        
        kball_config = AnalysisConfiguration(
            denoiser="NONE", drop_k=0, radius_fraction=0.75,
            cell_type="kball", requested_num_prompt_articles=50, requested_num_prompt_books=10
        )
        
        voronoi_config = AnalysisConfiguration(
            denoiser="NONE", drop_k=0, radius_fraction=0.75,
            cell_type="voronoi", requested_num_prompt_articles=50, requested_num_prompt_books=10
        )
        
        # Should return different algorithm functions
        kball_algorithm = _select_gap_algorithm(kball_config)
        voronoi_algorithm = _select_gap_algorithm(voronoi_config)
        
        assert kball_algorithm.__name__ == "kball_gaps"
        assert voronoi_algorithm.__name__ == "voronoi_gaps"
    
    def test_invalid_cell_type_algorithm_selection(self):
        """Test error handling for invalid cell types in algorithm selection."""
        from nextbook.gap_finder import _select_gap_algorithm
        
        # Mock an invalid configuration (bypassing validation for testing)
        class MockConfig:
            cell_type = "invalid"
        
        with pytest.raises(ValueError, match="Unsupported cell_type"):
            _select_gap_algorithm(MockConfig())


class TestDirectoryStructure:
    """Test directory structure with consolidated configuration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Provide a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override the base directory
            original_base = GapDirectoryManager.get_base_artifacts_dir()
            GapDirectoryManager._base_artifacts_dir = Path(tmpdir)
            yield Path(tmpdir)
            GapDirectoryManager._base_artifacts_dir = original_base
    
    def test_concurrent_cell_types_no_interference(self, temp_dir):
        """Test that kball and voronoi runs create separate directories."""
        kball_config = AnalysisConfiguration(
            denoiser="NONE", drop_k=0, radius_fraction=0.75,
            cell_type="kball", requested_num_prompt_articles=50, requested_num_prompt_books=10
        )
        
        voronoi_config = AnalysisConfiguration(
            denoiser="NONE", drop_k=0, radius_fraction=0.75,
            cell_type="voronoi", requested_num_prompt_articles=50, requested_num_prompt_books=10
        )
        
        kball_dir = GapDirectoryManager.get_config_dir(kball_config)
        voronoi_dir = GapDirectoryManager.get_config_dir(voronoi_config)
        
        # Should be different directories
        assert kball_dir != voronoi_dir
        assert "kball" in str(kball_dir)
        assert "voronoi" in str(voronoi_dir)
    
    def test_llm_subdirectories_with_consolidated_config(self, temp_dir):
        """Test LLM subdirectory creation with consolidated config."""
        config = AnalysisConfiguration(
            denoiser="JAD_EQUAL", drop_k=5, radius_fraction=0.5,
            cell_type="voronoi", requested_num_prompt_articles=100, requested_num_prompt_books=25
        )
        
        prompts_dir = GapDirectoryManager.get_prompts_llm_dir(config, gap_idx=0)
        responses_dir = GapDirectoryManager.get_responses_llm_dir(config, gap_idx=0)
        
        # Should include LLM counts in path
        assert "articles_100_books_25" in str(prompts_dir)
        assert "articles_100_books_25" in str(responses_dir)
        
        # Should include cell type in base path
        assert "voronoi" in str(prompts_dir)
        assert "voronoi" in str(responses_dir)
    
    def test_special_characters_in_paths(self, temp_dir):
        """Test path building handles edge cases."""
        config = AnalysisConfiguration(
            denoiser="CCLIP_01",  # Using a denoiser with underscore/number combo
            drop_k=0,
            radius_fraction=0.75,
            cell_type="kball",
            requested_num_prompt_articles=50,
            requested_num_prompt_books=10
        )
        
        config_dir = GapDirectoryManager.get_config_dir(config)
        # Should not raise an error and should be a valid path
        assert isinstance(config_dir, Path)
        assert "CCLIP_01" in str(config_dir)
        assert "kball" in str(config_dir)


class TestCSVSchemaIntegration:
    """Test CSV schema with consolidated configuration."""
    
    def test_all_config_fields_in_csv(self):
        """Test that all configuration fields are included in CSV output."""
        from nextbook.results_manager import GapResults
        
        config = AnalysisConfiguration(
            denoiser="JAD_EQUAL", drop_k=10, radius_fraction=0.33,
            cell_type="voronoi", requested_num_prompt_articles=75, requested_num_prompt_books=15
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = GapResults(config)
            
            # Add a sample row (totals set equal to requested for this synthetic test)
            results.add_row(
                topic_name="Test Topic",
                topic_id="T12345",
                gap_idx=0,
                gap_size=25,
                gap_size_percentile=0.85,
                gap_zscore=2.1,
                shapiro_log10_pval=-3.2,
                num_articles=75,
                num_books=15,
                actual_num_prompt_articles=20,
                actual_num_prompt_books=5
            )
            
            # Read back and verify all config fields are present
            df = results.get_dataframe()
            
            assert "denoiser" in df.columns
            assert "drop_k" in df.columns
            assert "radius_fraction" in df.columns
            assert "cell_type" in df.columns  # NEW
            assert "num_articles" in df.columns
            assert "num_books" in df.columns
            
            # Verify values
            row = df.iloc[0]
            assert row["denoiser"] == "JAD_EQUAL"
            assert row["drop_k"] == 10
            assert row["radius_fraction"] == 0.33
            assert row["cell_type"] == "voronoi"  # NEW
            assert row["num_articles"] == 75
            assert row["num_books"] == 15
    
    def test_csv_filtering_by_cell_type(self):
        """Test filtering CSV results by cell type."""
        from nextbook.results_manager import GapResults
        
        kball_config = AnalysisConfiguration(
            denoiser="NONE", drop_k=0, radius_fraction=0.75,
            cell_type="kball", requested_num_prompt_articles=50, requested_num_prompt_books=10
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            results = GapResults(kball_config, config_dir)
            
            # Add multiple rows
            for i in range(3):
                results.add_row(
                    topic_name=f"Topic {i}",
                    topic_id=f"T{i}",
                    gap_idx=i,
                    gap_size=10 + i,
                    gap_size_percentile=0.5,
                    gap_zscore=1.0,
                    shapiro_log10_pval=-2.0,
                    num_articles=50,
                    num_books=10,
                    actual_num_prompt_articles=20,
                    actual_num_prompt_books=5
                )
            
            df = results.get_dataframe()
            
            # All rows should have kball cell type
            assert (df["cell_type"] == "kball").all()
            
            # Should be able to filter by cell type
            kball_rows = df[df["cell_type"] == "kball"]
            assert len(kball_rows) == 3
    
    def test_mixed_csv_handling_errors_cleanly(self):
        """Test that mixing old/new CSV formats errors cleanly."""
        # This test drives the "clean break" requirement
        from nextbook.results_manager import GapResults
        
        config = AnalysisConfiguration(
            denoiser="NONE", drop_k=0, radius_fraction=0.75,
            cell_type="kball", requested_num_prompt_articles=50, requested_num_prompt_books=10
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            
            # Create an "old format" CSV without cell_type column
            old_csv_path = config_dir / "gap_analysis.csv"
            old_csv_content = """topic_name,topic_id,denoiser,drop_k,radius_fraction,gap_idx,num_books,num_articles,gap_size,gap_size_percentile,gap_zscore,shapiro_log10_pval,num_articles_for_llm,num_books_for_llm
Test Topic,T123,NONE,0,0.75,0,10,50,25,0.85,2.1,-3.2,20,5"""
            
            old_csv_path.write_text(old_csv_content)
            
            # Attempting to load with new format should error clearly
            with pytest.raises(ValueError, match="CSV missing required 'cell_type' column"):
                GapResults.load_from_csv(old_csv_path, config)


if __name__ == "__main__":
    pytest.main([__file__])
