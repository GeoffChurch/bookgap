"""
LLM configuration for BookGap batch processing.

Defines model settings, provider configuration, and output constraints.
"""

from dataclasses import dataclass

from bookgap.llm.protocols import OpenAIModel


@dataclass
class LLMConfig:
    """Configuration for LLM parameters.
    
    All parameters are required - use ConfigStore presets for common configurations.
    Compatible with existing LLMConfiguration in config_manager.py.
    """
    
    # Hydra instantiation target
    _target_: str  # Target class path for Hydra instantiation
    
    # Provider settings
    model: OpenAIModel  # OpenAI model to use (GPT_4O, GPT_4O_MINI, GPT_5, etc.) 
    
    # Prompt content control - determines how much research context to include
    num_articles: int  # Number of gap articles to sample for each prompt (default: 10)
    num_books: int     # Number of covering books to sample for each prompt (default: 5)
    
    # Generation parameters - control LLM behavior and output quality
    temperature: float  # Sampling temperature 0.0-2.0, lower=more focused (default: 0.2)
    max_tokens: int    # Maximum tokens in generated response (default: 2000)
    
    # Response format control - enables structured JSON output parsing
    use_structured_output: bool  # Use JSON schema for structured responses (default: True)
