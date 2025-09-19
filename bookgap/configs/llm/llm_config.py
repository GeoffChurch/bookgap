"""
LLM configuration for BookGap batch processing.

Defines model settings, provider configuration, and output constraints.
"""

from dataclasses import dataclass

from bookgap.llm.llm_implementation import Implementation
from bookgap.llm.providers import ProviderType
from bookgap.llm.protocols import OpenAIModel


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM parameters.
    
    All parameters are required - use ConfigStore presets for common configurations.
    Compatible with existing LLMConfiguration in config_manager.py.
    """
    
    # Hydra instantiation target
    _target_: str
    
    # Provider settings
    provider_type: ProviderType
    model_name: OpenAIModel
    
    # Prompt content
    num_articles: int
    num_books: int
    
    # Generation parameters
    temperature: float
    max_tokens: int
    
    # Output structure (from config_manager.py)
    chapters_min: int
    chapters_max: int
    
    # Quality control
    min_book_coverage: float
    
    # Prompt optimization
    use_structured_output: bool
