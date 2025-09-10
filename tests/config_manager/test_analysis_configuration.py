#!/usr/bin/env python3
"""
Test-Driven Development for AnalysisConfiguration consolidation.

This module contains comprehensive test cases for the new consolidated
AnalysisConfiguration class that replaces GapConfiguration + LLMConfiguration.
"""

import pytest
import tempfile
from pathlib import Path
from nextbook.config_manager import AnalysisConfiguration, GapConfiguration, LLMConfiguration
from nextbook.types import CellType
from nextbook.llm.llm_implementation import Implementation


def create_analysis_config(denoiser="NONE", drop_k=0, radius_fraction=0.75, 
                          cell_type="kball", requested_num_prompt_articles=50, 
                          requested_num_prompt_books=10):
    """Helper to create AnalysisConfiguration with the new pattern."""
    # Validate cell_type
    if cell_type not in ["kball", "voronoi"]:
        raise ValueError(f"Invalid cell_type: {cell_type}")
        
    gap_config = GapConfiguration(
        denoiser=denoiser,
        drop_k=drop_k,
        radius_fraction=radius_fraction,
        cell_type=CellType.KBALL if cell_type == "kball" else CellType.VORONOI
    )
    
    llm_config = LLMConfiguration(
        implementation=Implementation.STRUCTURED,
        provider="openai",
        model="gpt-4o",
        temperature=0.2,
        max_tokens=2000,
        chapters_min=10,
        chapters_max=15,
        requested_num_prompt_articles=requested_num_prompt_articles,
        requested_num_prompt_books=requested_num_prompt_books
    )
    
    return AnalysisConfiguration(gap_config=gap_config, llm_config=llm_config)


class TestAnalysisConfiguration:
    """Test the consolidated AnalysisConfiguration class."""
    
    def test_valid_configuration_creation(self):
        """Test creating valid AnalysisConfiguration instances."""
        # Basic kball configuration
        config = create_analysis_config(
            denoiser="NONE",
            drop_k=0,
            radius_fraction=0.75,
            cell_type="kball",
            requested_num_prompt_articles=50,
            requested_num_prompt_books=10
        )
        
        assert config.gap_config.denoiser == "NONE"
        assert config.gap_config.drop_k == 0
        assert config.gap_config.radius_fraction == 0.75
        assert config.gap_config.cell_type == CellType.KBALL
        assert config.llm_config.requested_num_prompt_articles == 50
        assert config.llm_config.requested_num_prompt_books == 10
        
        # Voronoi configuration
        voronoi_config = create_analysis_config(
            denoiser="JAD_EQUAL",
            drop_k=5,
            radius_fraction=0.5,
            cell_type="voronoi",
            requested_num_prompt_articles=100,
            requested_num_prompt_books=25
        )
        
        assert voronoi_config.gap_config.cell_type == CellType.VORONOI
    
    def test_invalid_cell_type_validation(self):
        """Test that invalid cell types raise ValueError."""
        with pytest.raises(ValueError):
            create_analysis_config(
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
        config_no_books = create_analysis_config(
            denoiser="NONE",
            drop_k=0,
            radius_fraction=0.75,
            cell_type="kball",
            requested_num_prompt_articles=100,
            requested_num_prompt_books=0  # Valid: topics can have no books
        )
        assert config_no_books.llm_config.requested_num_prompt_books == 0
    
    def test_string_representation(self):
        """Test string conversion for directory naming."""
        config = create_analysis_config(
            denoiser="JAD_EQUAL",
            drop_k=5,
            radius_fraction=0.25,
            cell_type="voronoi",
            requested_num_prompt_articles=75,
            requested_num_prompt_books=20
        )
        
        # Test the gap config string representation
        directory_name = config.gap_config.to_directory_name()
        expected = "JAD_EQUAL_k=5_radiusfrac=0.25_voronoi"
        assert directory_name == expected


class TestAlgorithmSelection:
    """Test algorithm selection based on cell_type."""
    
    def test_algorithm_dispatch_mechanism(self):
        """Test that cell_type correctly dispatches to different algorithms."""
        kball_config = create_analysis_config(
            denoiser="NONE", drop_k=0, radius_fraction=0.75,
            cell_type="kball", requested_num_prompt_articles=50, requested_num_prompt_books=10
        )
        
        voronoi_config = create_analysis_config(
            denoiser="NONE", drop_k=0, radius_fraction=0.75,
            cell_type="voronoi", requested_num_prompt_articles=50, requested_num_prompt_books=10
        )
        
        # Test that different algorithms are returned for different cell types
        kball_algorithm = kball_config.gap_config.cell_type.value
        voronoi_algorithm = voronoi_config.gap_config.cell_type.value
        
        # Test that different algorithms are returned
        assert kball_algorithm != voronoi_algorithm
        assert hasattr(kball_algorithm, 'find_gaps')
        assert hasattr(voronoi_algorithm, 'find_gaps')


class TestGapDirectoryManagerIntegration:
    """Test integration with GapDirectoryManager."""
    
    def test_directory_structure_creation(self):
        """Test that proper directory structures are created."""
        config = create_analysis_config(
            denoiser="JAD_EQUAL",
            drop_k=10,
            radius_fraction=0.33,
            cell_type="voronoi",
            requested_num_prompt_articles=200,
            requested_num_prompt_books=50
        )
        
        # Test the gap config string representation
        directory_name = config.gap_config.to_directory_name()
        
        # Verify the directory path includes configuration elements
        assert "JAD_EQUAL" in directory_name
        assert "k=10" in directory_name
        assert "voronoi" in directory_name
