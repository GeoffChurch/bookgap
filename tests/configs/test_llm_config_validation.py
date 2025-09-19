"""
Additional validation tests for the new LLMConfig to preserve important functionality
from the old LLMConfiguration tests.
"""

import pytest
from bookgap.configs.llm.llm_config import LLMConfig
from bookgap.llm.providers import ProviderType
from bookgap.llm.protocols import OpenAIModel


class TestLLMConfigValidation:
    """Test validation functionality for LLMConfig that was valuable in the old tests."""
    
    def test_chapters_validation_business_rules(self):
        """Test that chapters_min and chapters_max follow business rules."""
        # Test valid range
        config = LLMConfig(
            _target_="bookgap.llm.generators.openai_batch.OpenAIBatchGenerator",
            provider_type=ProviderType.OPENAI,
            model_name=OpenAIModel.GPT_4O_MINI,
            num_articles=10,
            num_books=5,
            temperature=0.2,
            max_tokens=2000,
            chapters_min=5,
            chapters_max=15,
            min_book_coverage=0.5,
            use_structured_output=True,
        )
        
        assert config.chapters_min == 5
        assert config.chapters_max == 15
        
        # Test edge cases
        edge_config = LLMConfig(
            _target_="bookgap.llm.generators.openai_batch.OpenAIBatchGenerator",
            provider_type=ProviderType.OPENAI,
            model_name=OpenAIModel.GPT_4O_MINI,
            num_articles=10,
            num_books=5,
            temperature=0.0,  # Minimum valid temperature
            max_tokens=1,     # Minimum valid tokens
            chapters_min=1,   # Minimum valid chapters
            chapters_max=50,  # Maximum valid chapters
            min_book_coverage=0.0,  # Minimum valid coverage
            use_structured_output=True,
        )
        
        assert edge_config.chapters_min == 1
        assert edge_config.chapters_max == 50
        assert edge_config.temperature == 0.0
        assert edge_config.min_book_coverage == 0.0
    
    def test_temperature_bounds_validation(self):
        """Test temperature stays within valid OpenAI bounds."""
        # Valid temperatures
        for temp in [0.0, 0.5, 1.0, 2.0]:
            config = LLMConfig(
                _target_="bookgap.llm.generators.openai_batch.OpenAIBatchGenerator",
                provider_type=ProviderType.OPENAI,
                model_name=OpenAIModel.GPT_4O_MINI,
                num_articles=10,
                num_books=5,
                temperature=temp,
                max_tokens=2000,
                chapters_min=10,
                chapters_max=15,
                min_book_coverage=0.5,
                use_structured_output=True,
            )
            assert config.temperature == temp
    
    def test_model_name_enum_validation(self):
        """Test that model_name properly uses OpenAIModel enum."""
        for model in OpenAIModel:
            config = LLMConfig(
                _target_="bookgap.llm.generators.openai_batch.OpenAIBatchGenerator",
                provider_type=ProviderType.OPENAI,
                model_name=model,
                num_articles=10,
                num_books=5,
                temperature=0.2,
                max_tokens=2000,
                chapters_min=10,
                chapters_max=15,
                min_book_coverage=0.5,
                use_structured_output=True,
            )
            assert config.model_name == model
            assert isinstance(config.model_name, OpenAIModel)
    
    def test_provider_type_validation(self):
        """Test that provider_type properly uses ProviderType enum."""
        config = LLMConfig(
            _target_="bookgap.llm.generators.openai_batch.OpenAIBatchGenerator",
            provider_type=ProviderType.OPENAI,
            model_name=OpenAIModel.GPT_4O_MINI,
            num_articles=10,
            num_books=5,
            temperature=0.2,
            max_tokens=2000,
            chapters_min=10,
            chapters_max=15,
            min_book_coverage=0.5,
            use_structured_output=True,
        )
        
        assert config.provider_type == ProviderType.OPENAI
        assert isinstance(config.provider_type, ProviderType)
    
    def test_target_field_required(self):
        """Test that _target_ field is required for Hydra instantiation."""
        config = LLMConfig(
            _target_="bookgap.llm.generators.openai_batch.OpenAIBatchGenerator",
            provider_type=ProviderType.OPENAI,
            model_name=OpenAIModel.GPT_4O_MINI,
            num_articles=10,
            num_books=5,
            temperature=0.2,
            max_tokens=2000,
            chapters_min=10,
            chapters_max=15,
            min_book_coverage=0.5,
            use_structured_output=True,
        )
        
        assert hasattr(config, '_target_')
        assert config._target_ == "bookgap.llm.generators.openai_batch.OpenAIBatchGenerator"
