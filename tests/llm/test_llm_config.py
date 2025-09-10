"""Tests for LLM configuration updates in config_manager."""

import pytest
from unittest.mock import patch, MagicMock

from bookgap.config_manager import LLMConfiguration
from bookgap.llm.llm_implementation import Implementation


@pytest.fixture(autouse=True)
def reset_enum_values():
    """Reset enum values before each test to ensure clean state."""
    # Store original values
    original_values = {}
    for impl in Implementation:
        original_values[impl] = impl._value_
    
    # Reset to placeholder values
    Implementation.STRUCTURED._value_ = "structured_placeholder"
    Implementation.PROMPT._value_ = "prompt_placeholder"
    
    yield
    
    # Restore original values after test
    for impl, value in original_values.items():
        impl._value_ = value


class TestLLMConfiguration:
    """Tests for the updated LLMConfiguration class."""
    
    def test_basic_creation_structured(self):
        """Test basic LLMConfiguration creation with structured implementation."""
        config = LLMConfiguration(
            implementation=Implementation.STRUCTURED,
            provider='openai',
            model='gpt-4',
            temperature=0.7,
            max_tokens=8000,
            chapters_min=3,
            chapters_max=8
        )
        
        assert config.implementation == Implementation.STRUCTURED
        assert config.provider == 'openai'
        assert config.model == 'gpt-4'
        assert config.temperature == 0.7
        assert config.max_tokens == 8000
        assert config.chapters_min == 3
        assert config.chapters_max == 8
    
    def test_basic_creation_prompt(self):
        """Test basic LLMConfiguration creation with prompt implementation."""
        config = LLMConfiguration(
            implementation=Implementation.PROMPT,
            provider='openai',
            model='gpt-3.5-turbo',
            temperature=0.9,
            max_tokens=4000,
            chapters_min=2,
            chapters_max=12
        )
        
        assert config.implementation == Implementation.PROMPT
        assert config.provider == 'openai'
        assert config.model == 'gpt-3.5-turbo'
        assert config.temperature == 0.9
        assert config.max_tokens == 4000
        assert config.chapters_min == 2
        assert config.chapters_max == 12
    
    def test_chapters_min_validation(self):
        """Test that chapters_min must be at least 1."""
        with pytest.raises(ValueError) as exc_info:
            LLMConfiguration(
                implementation=Implementation.STRUCTURED,
                provider='openai',
                model='gpt-4',
                temperature=0.7,
                max_tokens=8000,
                chapters_min=0,  # Invalid
                chapters_max=8
            )
        
        assert "chapters must satisfy 1 <= min <= max <= 50" in str(exc_info.value)
    
    def test_chapters_max_validation(self):
        """Test that chapters_max must be at least chapters_min."""
        with pytest.raises(ValueError) as exc_info:
            LLMConfiguration(
                implementation=Implementation.STRUCTURED,
                provider='openai',
                model='gpt-4',
                temperature=0.7,
                max_tokens=8000,
                chapters_min=5,
                chapters_max=3  # Invalid: less than chapters_min
            )
        
        assert "chapters must satisfy 1 <= min <= max <= 50" in str(exc_info.value)
    
    def test_temperature_validation_low(self):
        """Test temperature validation for values too low."""
        with pytest.raises(ValueError) as exc_info:
            LLMConfiguration(
                implementation=Implementation.STRUCTURED,
                provider='openai',
                model='gpt-4',
                temperature=-0.1,  # Invalid
                max_tokens=8000,
                chapters_min=3,
                chapters_max=8
            )
        
        assert "temperature must be in [0.0, 2.0]" in str(exc_info.value)
    
    def test_temperature_validation_high(self):
        """Test temperature validation for values too high."""
        with pytest.raises(ValueError) as exc_info:
            LLMConfiguration(
                implementation=Implementation.STRUCTURED,
                provider='openai',
                model='gpt-4',
                temperature=2.1,  # Invalid
                max_tokens=8000,
                chapters_min=3,
                chapters_max=8
            )
        
        assert "temperature must be in [0.0, 2.0]" in str(exc_info.value)
    
    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        with pytest.raises(ValueError) as exc_info:
            LLMConfiguration(
                implementation=Implementation.STRUCTURED,
                provider='openai',
                model='gpt-4',
                temperature=0.7,
                max_tokens=0,  # Invalid
                chapters_min=3,
                chapters_max=8
            )
        
        assert "max_tokens must be positive" in str(exc_info.value)
    
    def test_directory_name_structured(self):
        """Test directory name generation for structured implementation."""
        config = LLMConfiguration(
            implementation=Implementation.STRUCTURED,
            provider='openai',
            model='gpt-4',
            temperature=0.7,
            max_tokens=8000,
            chapters_min=3,
            chapters_max=8,
            requested_num_prompt_articles=5,
            requested_num_prompt_books=10
        )
        
        expected = "llm_structured_articles_5_books_10"
        assert config.to_subdir_name() == expected
    
    def test_directory_name_prompt(self):
        """Test directory name generation for prompt implementation."""
        config = LLMConfiguration(
            implementation=Implementation.PROMPT,
            provider='openai',
            model='gpt-3.5-turbo',
            temperature=0.9,
            max_tokens=4000,
            chapters_min=2,
            chapters_max=12,
            requested_num_prompt_articles=7,
            requested_num_prompt_books=3
        )

        expected = "llm_prompt_articles_7_books_3"
        assert config.to_subdir_name() == expected

    def test_from_subdir_name_structured(self):
        """Test parsing structured directory names."""
        # Test new format
        config = LLMConfiguration.from_subdir_name("llm_structured_articles_5_books_10")
        
        assert config.implementation == Implementation.STRUCTURED
        assert config.requested_num_prompt_articles == 5
        assert config.requested_num_prompt_books == 10
        # Other fields should have defaults
        assert config.provider == 'openai'
        assert config.model == 'gpt-5'  # Current default
        assert config.temperature == 0.2  # Current default
        assert config.max_tokens == 2000  # Current default
    
    def test_from_subdir_name_prompt(self):
        """Test parsing prompt directory names."""
        config = LLMConfiguration.from_subdir_name("llm_prompt_articles_3_books_15")
        
        assert config.implementation == Implementation.PROMPT
        assert config.requested_num_prompt_articles == 3
        assert config.requested_num_prompt_books == 15
        # Other fields should have defaults
        assert config.provider == 'openai'
        assert config.model == 'gpt-5'  # Current default
        assert config.temperature == 0.2  # Current default
        assert config.max_tokens == 2000  # Current default
    
    # Legacy format test removed - we eliminated backward compatibility
    
    def test_from_subdir_name_invalid_format(self):
        """Test that invalid directory names raise appropriate errors."""
        invalid_names = [
            "invalid_format",
            "llm_invalid_articles_3_books_5",
            "llm_structured_articles_books_5",
            "llm_prompt_articles_3_books",
            "llm_structured_articles_-1_books_5",
        ]
        
        for invalid_name in invalid_names:
            with pytest.raises(ValueError):
                LLMConfiguration.from_subdir_name(invalid_name)
    
    def test_boundary_values(self):
        """Test boundary values for validation."""
        # Minimum valid values
        config_min = LLMConfiguration(
            implementation=Implementation.STRUCTURED,
            provider='openai',
            model='gpt-4',
            temperature=0.0,
            max_tokens=1,
            chapters_min=1,
            chapters_max=1
        )
        assert config_min.temperature == 0.0
        assert config_min.max_tokens == 1
        assert config_min.chapters_min == 1
        assert config_min.chapters_max == 1
        
        # Maximum valid values
        config_max = LLMConfiguration(
            implementation=Implementation.PROMPT,
            provider='openai',
            model='gpt-4',
            temperature=2.0,
            max_tokens=100000,
            chapters_min=20,
            chapters_max=20
        )
        assert config_max.temperature == 2.0
        assert config_max.max_tokens == 100000
        assert config_max.chapters_min == 20
        assert config_max.chapters_max == 20
    
    def test_string_representation(self):
        """Test string representation of LLMConfiguration."""
        config = LLMConfiguration(
            implementation=Implementation.STRUCTURED,
            provider='openai',
            model='gpt-4',
            temperature=0.7,
            max_tokens=8000,
            chapters_min=3,
            chapters_max=8
        )
        
        str_repr = str(config)
        assert "Implementation.STRUCTURED" in str_repr
        assert "gpt-4" in str_repr
        assert "0.7" in str_repr
    
    def test_equality(self):
        """Test equality comparison of LLMConfiguration instances."""
        config1 = LLMConfiguration(
            implementation=Implementation.STRUCTURED,
            provider='openai',
            model='gpt-4',
            temperature=0.7,
            max_tokens=8000,
            chapters_min=3,
            chapters_max=8,
            requested_num_prompt_articles=10,
            requested_num_prompt_books=5
        )

        config2 = LLMConfiguration(
            implementation=Implementation.STRUCTURED,
            provider='openai',
            model='gpt-4',
            temperature=0.7,
            max_tokens=8000,
            chapters_min=3,
            chapters_max=8,
            requested_num_prompt_articles=10,
            requested_num_prompt_books=5
        )

        config3 = LLMConfiguration(
            implementation=Implementation.PROMPT,  # Different
            provider='openai',
            model='gpt-4',
            temperature=0.7,
            max_tokens=8000,
            chapters_min=3,
            chapters_max=8,
            requested_num_prompt_articles=10,
            requested_num_prompt_books=5
        )

        assert config1 == config2
        assert config1 != config3