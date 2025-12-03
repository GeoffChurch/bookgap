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
    _target_: str = "bookgap.llm.generators.openai_batch.OpenAIBatchGenerator"  # Target class path for Hydra instantiation
    model: OpenAIModel = OpenAIModel.GPT_5  # OpenAI model to use (GPT_4O, GPT_4O_MINI, GPT_5, etc.) 
    num_articles: int = 10  # Number of gap articles to sample for each prompt (default: 10)
    num_books: int = 5      # Number of covering books to sample for each prompt (default: 5)
    temperature: float = 0.2  # Sampling temperature 0.0-2.0, lower=more focused (default: 0.2)
    max_tokens: int = 4096    # Maximum tokens in generated response (default: 4096)
    use_structured_output: bool = True  # Use JSON schema for structured responses (default: True)
