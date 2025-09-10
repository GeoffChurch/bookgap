"""
Tests for the LLM registry module and Implementation enum.

These tests verify the core enum structure and that the registry can
properly load and instantiate generators.
"""

from enum import Enum

from bookgap.llm.llm_implementation import Implementation, get_generator


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
        from tests.llm.test_llm_generators import create_test_config
        
        config = create_test_config()
        
        # Should successfully create generator with valid config
        generator = get_generator(Implementation.STRUCTURED, config)
        assert generator is not None
        
        # Should have the expected type
        from bookgap.llm.generators.openai_generator import OpenAIProposalGenerator
        assert isinstance(generator, OpenAIProposalGenerator)
