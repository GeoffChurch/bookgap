"""LLM proposal generators package."""

from .openai_generator import OpenAIProposalGenerator
from .json_prompt_generator import JSONPromptProposalGenerator

__all__ = ["OpenAIProposalGenerator", "JSONPromptProposalGenerator"]
