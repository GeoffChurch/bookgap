"""
LLM configuration for BookGap batch processing.

Defines model settings, provider configuration, and output constraints.
"""

from dataclasses import dataclass

from bookgap.llm.protocols import LLMBackend, OpenAIModel


@dataclass
class LLMConfig:
    """Configuration for LLM parameters.

    All parameters are required - use ConfigStore presets for common configurations.
    Compatible with existing LLMConfiguration in config_manager.py.
    """

    # Hydra instantiation target (used by the OpenAI batch path's
    # `generate_batch_requests`; the Ollama backend constructs its
    # generator directly from the fields below).
    _target_: str = "bookgap.llm.generators.openai_batch.OpenAIBatchGenerator"  # Target class path for Hydra instantiation

    # Backend selector. Defaults to the historical OpenAI Batch API path
    # so existing configs are byte-identical without override.
    backend: LLMBackend = LLMBackend.OPENAI_BATCH

    model: OpenAIModel = OpenAIModel.GPT_5  # OpenAI model to use (GPT_4O, GPT_4O_MINI, GPT_5, etc.)
    # Ollama model tag used when backend == OLLAMA_LOCAL (ignored otherwise).
    ollama_model: str = "gemma4:31b"

    num_articles: int = 10  # Number of gap articles to sample for each prompt (default: 10)
    num_books: int = 5      # Number of covering books to sample for each prompt (default: 5)
    temperature: float = 0.2  # Sampling temperature 0.0-2.0, lower=more focused (default: 0.2)
    max_tokens: int = 4096    # Maximum tokens in generated response (default: 4096)
    use_structured_output: bool = True  # Use JSON schema for structured responses (default: True)

    # Ollama-only knobs (ignored when backend == OPENAI_BATCH).
    # Context window passed to Ollama. Abstracts are uncapped (up to ~98K
    # characters in the AI subfield), so no window fits every possible prompt; a
    # prompt or prompt + output that does not fit raises ContextOverflowError
    # rather than being truncated (see bookgap.llm.generators.ollama_local).
    # 65536 holds the longest prompt seen, 29,133 tokens (T13062 at radius
    # fraction 1/2, ten ~10K-character spam abstracts) with room for the output.
    # A larger window does not change the output for a prompt that fits a smaller one.
    ollama_num_ctx: int = 65536
    ollama_num_predict: int = 8192   # Max generated tokens for Ollama
