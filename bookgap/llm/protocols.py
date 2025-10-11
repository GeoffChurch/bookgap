"""
Core protocols and data structures for LLM proposal generation.

This module defines the fundamental interfaces and data types used throughout
the LLM proposal generation system. No generator implementations are imported
here to avoid circular dependencies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Any, Literal
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from bookgap.gap_types import DistsAndIdxs
from bookgap.subcloud import SubCloud

class GapProtocol(Protocol):
    """Protocol defining the common interface for gap analysis objects.
    
    This eliminates the need for isinstance checks by defining the methods
    that both KballGap and VoronoiGap must implement.
    """
    
    def get_dists_idxs(self) -> DistsAndIdxs:
        """Return the distances and indices of articles in the gap."""
        ...
    
    def get_ambient(self) -> SubCloud:
        """Return the ambient space (articles)."""
        ...
        
    def get_landmark(self) -> SubCloud:  
        """Return the landmark space (books)."""
        ...
    
    def get_covering_book_indices(self) -> list[np.intp]:
        """Get the indices of books that cover articles in the gap."""
        ...
        
    def get_book_to_articles_mapping(self) -> dict[np.intp, DistsAndIdxs]:
        """Get mapping from book indices to their covered articles with distances."""
        ...

class OpenAIModel(Enum):
    """OpenAI model variants with their string representations."""
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_4 = "gpt-4"
    GPT_5 = "gpt-5"
    
    @property
    def supports_temperature(self) -> bool:
        """Whether this model supports temperature parameter."""
        return self != OpenAIModel.GPT_5
    
    @property
    def uses_max_completion_tokens(self) -> bool:
        """Whether this model uses max_completion_tokens instead of max_tokens."""
        return self == OpenAIModel.GPT_5


@dataclass(frozen=True)
class ChatMessage:
    """OpenAI chat message structure for batch requests."""
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Article:
    """Research article data structure."""
    display_name: str
    abstract: str


@dataclass(frozen=True)  
class Book:
    """Book data structure."""
    display_name: str
    abstract: str


@dataclass(frozen=True)
class PromptPair:
    """System and user prompts for batch request generation."""
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True) 
class ChatCompletionBody:
    """OpenAI chat completion request body for batch API."""
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None  # For structured outputs


@dataclass(frozen=True)
class BatchRequestData:
    """OpenAI batch API request structure."""
    custom_id: str
    body: ChatCompletionBody
    method: Literal["POST"] = "POST"
    url: Literal["/v1/chat/completions"] = "/v1/chat/completions"


@dataclass(frozen=True)
class ProposalCounts:
    """Counts of items used in proposal generation."""
    num_articles_for_llm: int
    num_books_for_llm: int


@dataclass(frozen=True)
class ProposalContext:
    """Context information for generating a textbook proposal."""
    topic_name: str
    topic_id: str
    gap_idx: int
    requested_num_prompt_articles: int
    requested_num_prompt_books: int
    sampled_articles: list[Article]
    covering_books: list[Book]
    counts: ProposalCounts


@dataclass(frozen=True)
class ProposalMetadata:
    """Metadata about the proposal generation process."""
    model: str
    implementation: str
    request_id: str | None = None
    usage: dict[str, Any] | None = None  # Keep as dict since it's LLM-provider specific


@dataclass(frozen=True)
class ProposalResult:
    """Result of a proposal generation operation."""
    json_path: Path  # validated JSON output (final artifact)
    prompt_path: Path  # text prompt written for audit
    raw_response_path: Path | None  # raw LLM response (may be None for structured outputs)
    counts: ProposalCounts
    metadata: ProposalMetadata


class ProposalGenerator(Protocol):
    """Protocol for LLM proposal generators."""
    
    def generate(self, context: ProposalContext) -> ProposalResult:
        """Generate a textbook proposal from the given context.
        
        Args:
            context: The proposal context containing gap analysis and configuration.
            
        Returns:
            ProposalResult with generated proposal and metadata.
        """
        ...


# # Pydantic models for structured output validation and data handling
# class Chapter(BaseModel):
#     """A chapter in a textbook proposal."""
#     chapter_title: str = Field(..., description="Chapter title")
#     # chapter_topics: list[str] = Field(..., description="Chapter topics")
#     # chapter_summary: str = Field(..., description="Chapter summary")


class Proposal(BaseModel):
    """A complete textbook proposal with title, abstract, and chapters."""
    title: str = Field(..., description="Textbook title")
    abstract: str = Field(..., description="Textbook abstract")
    # chapters: list[Chapter] = Field(..., min_length=10, max_length=15, description="Textbook chapters")


@dataclass(frozen=True)
class ParsedProposal:
    """Wrapper for a parsed proposal with metadata and embedding."""
    proposal: Proposal
    custom_id: str
    embedding: NDArray[np.float32]
