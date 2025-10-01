"""
Tests for the LLM registry module and Implementation enum.

These tests verify the core enum structure and that the registry can
properly load and instantiate generators.
"""

import os
from enum import Enum
from unittest.mock import patch

from bookgap.config_manager import AnalysisConfiguration, GapConfiguration, LLMConfiguration
from bookgap.types import CellType
from bookgap.llm.llm_implementation import Implementation, get_generator


def _create_test_analysis_config() -> AnalysisConfiguration:
    """Build a minimal AnalysisConfiguration for registry smoke tests."""
    gap_config = GapConfiguration(
        denoiser="NONE",
        drop_k=0,
        radius_fraction=0.75,
        cell_type=CellType.KBALL,
    )
    llm_config = LLMConfiguration(
        implementation=Implementation.STRUCTURED,
        requested_num_prompt_articles=2,
        requested_num_prompt_books=1,
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=1024,
    )
    return AnalysisConfiguration.from_components(gap_config, llm_config)


class TestImplementation:
    """Tests for the Implementation enum."""
    
    def test_enum_values_exist(self):
        """Test that required enum values exist."""
        assert hasattr(Implementation, 'STRUCTURED')
        assert hasattr(Implementation, 'PROMPT')
        
        # Should have exactly two implementations
        assert len(list(Implementation)) == 2
    
    def test_enum_inheritance(self):
        """Test that Implementation inherits from Enum."""
        assert issubclass(Implementation, Enum)


class TestRegistryIntegration:
    """Integration tests for the current architecture."""

    def test_enum_structure_ready_for_generators(self):
        """Test that the enum structure is ready to receive generator classes."""
        # Verify enum exists and has expected values
        assert hasattr(Implementation, 'STRUCTURED')
        assert hasattr(Implementation, 'PROMPT')
        
        # Verify get_generator function exists
        assert callable(get_generator)
        
        # Verify the enum can be used as expected
        implementation_names = [impl.name for impl in Implementation]
        assert 'STRUCTURED' in implementation_names
        assert 'PROMPT' in implementation_names
        assert len(implementation_names) == 2
    
    def test_get_generator_with_real_config(self):
        """Test that get_generator works with properly configured config."""
        config = _create_test_analysis_config()

        # Mock the API key requirement and OpenAI client construction
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}), \
             patch("bookgap.llm.generators.openai_generator.OpenAI"):
            generator = get_generator(Implementation.STRUCTURED, config)
            assert generator is not None

            from bookgap.llm.generators.openai_generator import OpenAIProposalGenerator
            assert isinstance(generator, OpenAIProposalGenerator)
