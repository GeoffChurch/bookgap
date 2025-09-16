"""
Core protocols and data structures for LLM proposal generation.

This module defines the fundamental interfaces and data types used throughout
the LLM proposal generation system. No generator implementations are imported
here to avoid circular dependencies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Any
from pydantic import BaseModel, Field


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
class ProposalContext:
    """Context information for generating a textbook proposal."""
    topic_name: str
    topic_id: str
    gap_idx: int
    requested_num_prompt_articles: int
    requested_num_prompt_books: int
    sampled_articles: list[dict[str, Any]]
    covering_books: list[dict[str, Any]]
    counts: dict[str, int]  # {"num_articles_for_llm": int, "num_books_for_llm": int}


@dataclass(frozen=True)
class ProposalCounts:
    """Counts of items used in proposal generation."""
    num_articles_for_llm: int
    num_books_for_llm: int


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
