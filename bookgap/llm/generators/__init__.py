"""LLM proposal generators package."""

from .openai_batch import OpenAIBatchGenerator
from .ollama_local import OllamaProposalGenerator, OllamaProposalRecord
from .workflow_functions import generate_batch_requests, validate_batch_requests
from ..protocols import LLMBackend

__all__ = [
    "LLMBackend",
    "OllamaProposalGenerator",
    "OllamaProposalRecord",
    "OpenAIBatchGenerator",
    "generate_batch_requests",
    "validate_batch_requests",
]
