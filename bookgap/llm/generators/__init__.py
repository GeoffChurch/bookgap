"""LLM proposal generators package."""

# New batch-first generators (Phase 1.2+)
from .batch_generator import BatchProposalGenerator
from .openai_batch import OpenAIBatchGenerator
from .ollama_local import OllamaProposalGenerator, OllamaProposalRecord
from .workflow_functions import generate_batch_requests, validate_batch_requests
from ..protocols import LLMBackend

# Legacy individual generators (to be deleted in Phase 4.1)
# from .openai_generator import OpenAIProposalGenerator
# from .json_prompt_generator import JSONPromptProposalGenerator

__all__ = [
    "BatchProposalGenerator",
    "LLMBackend",
    "OllamaProposalGenerator",
    "OllamaProposalRecord",
    "OpenAIBatchGenerator",
    "generate_batch_requests",
    "validate_batch_requests",
]
