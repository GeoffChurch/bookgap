"""Tests for LLM protocols and data structures."""

import pytest
from pydantic import ValidationError

from nextbook.llm.protocols import (
    ProposalContext,
    ProposalResult,
    Chapter,
    Proposal,
    ProposalGenerationError
)


class TestProposalContext:
    """Tests for ProposalContext dataclass."""
    
    def test_basic_creation(self):
        """Test basic ProposalContext creation."""
        context = ProposalContext(
            topic_name="Test Topic",
            topic_id="test_topic",
            gap_idx=0,
            requested_num_prompt_articles=3,
            requested_num_prompt_books=5,
            sampled_articles=[{"title": "Test Article"}],
            covering_books=[{"title": "Test Book"}],
            counts={"num_articles_for_llm": 3, "num_books_for_llm": 5}
        )
        
        assert context.topic_name == "Test Topic"
        assert context.topic_id == "test_topic"
        assert context.gap_idx == 0
        assert context.requested_num_prompt_articles == 3
        assert context.requested_num_prompt_books == 5
        assert len(context.sampled_articles) == 1
        assert len(context.covering_books) == 1
        assert context.counts["num_articles_for_llm"] == 3
    
    def test_empty_samples(self):
        """Test ProposalContext with empty sample lists."""
        context = ProposalContext(
            topic_name="Test Topic",
            topic_id="test_topic",
            gap_idx=0,
            requested_num_prompt_articles=0,
            requested_num_prompt_books=0,
            sampled_articles=[],
            covering_books=[],
            counts={"num_articles_for_llm": 0, "num_books_for_llm": 0}
        )
        
        assert context.sampled_articles == []
        assert context.covering_books == []


class TestProposalResult:
    """Tests for ProposalResult dataclass."""
    
    def test_basic_creation(self):
        """Test basic ProposalResult creation."""
        from pathlib import Path
        
        context = ProposalContext(
            topic_name="Test Topic",
            topic_id="test_topic",
            gap_idx=0,
            requested_num_prompt_articles=3,
            requested_num_prompt_books=5,
            sampled_articles=[],
            covering_books=[],
            counts={"num_articles_for_llm": 3, "num_books_for_llm": 5}
        )
        
        result = ProposalResult(
            json_path=Path("/tmp/test.json"),
            prompt_path=Path("/tmp/test_prompt.txt"),
            raw_response_path=Path("/tmp/test_raw.txt"),
            counts={"num_articles_for_llm": 3, "num_books_for_llm": 5},
            metadata={"provider": "openai", "model": "gpt-4", "implementation": "structured"}
        )
        
        assert result.json_path == Path("/tmp/test.json")
        assert result.prompt_path == Path("/tmp/test_prompt.txt")
        assert result.raw_response_path == Path("/tmp/test_raw.txt")
        assert result.metadata["provider"] == "openai"


class TestChapter:
    """Tests for Chapter Pydantic model."""
    
    def test_valid_chapter(self):
        """Test valid chapter creation."""
        chapter = Chapter(
            chapter_title="Introduction to AI",
            chapter_description="This chapter covers the basics of artificial intelligence."
        )
        
        assert chapter.chapter_title == "Introduction to AI"
        assert chapter.chapter_description == "This chapter covers the basics of artificial intelligence."
    
    def test_empty_title_fails(self):
        """Test that empty title fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            Chapter(chapter_title="", chapter_description="Valid description")
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_empty_description_fails(self):
        """Test that empty description fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            Chapter(chapter_title="Valid Title", chapter_description="")
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_whitespace_only_passes_but_not_ideal(self):
        """Test that whitespace-only strings are technically valid but not ideal.
        
        Note: Pydantic's min_length only checks length, not content quality.
        In practice, the LLM should generate meaningful content.
        """
        # These will pass validation but are not ideal
        chapter1 = Chapter(chapter_title="   ", chapter_description="Valid description")
        chapter2 = Chapter(chapter_title="Valid Title", chapter_description="   ")
        
        assert chapter1.chapter_title == "   "
        assert chapter2.chapter_description == "   "


class TestProposal:
    """Tests for Proposal Pydantic model."""
    
    def test_valid_proposal(self):
        """Test valid proposal creation."""
        # Create 10 chapters (minimum required)
        chapters = [
            Chapter(chapter_title=f"Chapter {i}", chapter_description=f"Description for chapter {i}")
            for i in range(1, 11)
        ]
        
        proposal = Proposal(
            title="Test Book",
            abstract="A comprehensive test book covering important topics in the field.",
            chapters=chapters
        )
        
        assert proposal.title == "Test Book"
        assert proposal.abstract == "A comprehensive test book covering important topics in the field."
        assert len(proposal.chapters) == 10
        assert proposal.chapters[0].chapter_title == "Chapter 1"
    
    def test_empty_title_fails(self):
        """Test that empty title fails validation."""
        chapters = [
            Chapter(chapter_title=f"Chapter {i}", chapter_description=f"Description {i}")
            for i in range(1, 11)  # 10 chapters minimum
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            Proposal(title="", abstract="Valid abstract", chapters=chapters)
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_empty_description_fails(self):
        """Test that empty abstract fails validation."""
        chapters = [
            Chapter(chapter_title=f"Chapter {i}", chapter_description=f"Description {i}")
            for i in range(1, 11)  # 10 chapters minimum
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            Proposal(title="Valid Title", abstract="", chapters=chapters)
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_no_chapters_fails(self):
        """Test that proposal with no chapters fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            Proposal(
                title="Valid Title",
                abstract="Valid abstract",
                chapters=[]
            )
        
        assert "List should have at least 10 items" in str(exc_info.value)
    
    def test_too_few_chapters_fails(self):
        """Test that too few chapters fails validation."""
        chapters = [
            Chapter(chapter_title=f"Chapter {i}", chapter_description=f"Description {i}")
            for i in range(1, 10)  # 9 chapters (below minimum of 10)
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            Proposal(
                title="Too Small Book",
                abstract="A book with too few chapters",
                chapters=chapters
            )
        
        assert "List should have at least 10 items" in str(exc_info.value)
    
    def test_maximum_chapters(self):
        """Test proposal with maximum allowed chapters."""
        chapters = [
            Chapter(chapter_title=f"Chapter {i}", chapter_description=f"Description {i}")
            for i in range(1, 16)  # 15 chapters (max allowed)
        ]
        
        proposal = Proposal(
            title="Large Book",
            abstract="A book with many chapters covering extensive material.",
            chapters=chapters
        )
        
        assert len(proposal.chapters) == 15
    
    def test_too_many_chapters_fails(self):
        """Test that too many chapters fails validation."""
        chapters = [
            Chapter(chapter_title=f"Chapter {i}", chapter_description=f"Description {i}")
            for i in range(1, 17)  # 16 chapters (exceeds max of 15)
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            Proposal(
                title="Too Large Book",
                abstract="A book with too many chapters",
                chapters=chapters
            )
        
        assert "List should have at most 15 items" in str(exc_info.value)


class TestProposalGenerationError:
    """Tests for ProposalGenerationError exception."""
    
    def test_basic_error(self):
        """Test basic error creation."""
        error = ProposalGenerationError("Something went wrong")
        
        assert str(error) == "Something went wrong"
        assert error.context == {}
    
    def test_error_with_context(self):
        """Test error creation with context."""
        context = {"model": "gpt-4", "temperature": 0.7}
        error = ProposalGenerationError("API call failed", context=context)
        
        assert str(error) == "API call failed"
        assert error.context == context
    
    def test_error_inheritance(self):
        """Test that ProposalGenerationError inherits from Exception."""
        error = ProposalGenerationError("Test error")
        assert isinstance(error, Exception)
