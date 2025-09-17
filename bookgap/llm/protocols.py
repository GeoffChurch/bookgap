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
from pydantic import BaseModel, Field


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


class ProposalGenerator(Protocol):
    """Protocol for LLM proposal generators."""
    
    def generate(self, context: 'ProposalContext') -> 'ProposalResult':
        """Generate a textbook proposal from the given context.
        
        Args:
            context: The proposal context containing gap analysis and configuration.
            
        Returns:
            ProposalResult with generated proposal and metadata.
            
        Raises:
            ProposalGenerationError: If generation fails for any reason.
        """
        ...


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
    provider: str
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


# Pydantic models for structured output validation
class Chapter(BaseModel):
    """A chapter in a textbook proposal."""
    chapter_title: str = Field(..., min_length=1, description="The title of the chapter")
    chapter_description: str = Field(..., min_length=1, description="Description of concepts covered in the chapter (1-2 sentences)")


def create_proposal_model(chapters_min: int = 10, chapters_max: int = 15) -> type[BaseModel]:
    """Create a Proposal model with configurable chapter constraints.
    
    Args:
        chapters_min: Minimum number of chapters
        chapters_max: Maximum number of chapters
        
    Returns:
        Dynamically created Proposal model class
    """
    class Proposal(BaseModel):
        """A complete textbook proposal with title, abstract, and chapters."""
        title: str = Field(..., min_length=1, description="The proposed title of the textbook")
        abstract: str = Field(..., min_length=1, description="Abstract describing the textbook's scope and approach")
        chapters: list[Chapter] = Field(
            ..., 
            min_length=chapters_min, 
            max_length=chapters_max, 
            description=f"List of {chapters_min}-{chapters_max} chapters for the textbook"
        )
    
    return Proposal


class ProposalGenerationError(Exception):
    """Exception raised when proposal generation fails."""
    
    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = context or {}
