"""Focused tests for LLM protocols - testing behavior, not field assignment."""

import pytest
from pydantic import ValidationError

from bookgap.llm.protocols import (
    # Pydantic models with validation logic
    Chapter,
    # New batch-oriented data structures with behavior
    OpenAIModel,
    ChatMessage,
    Article,
    Book,
    Proposal,
    ProposalCounts,
    PromptPair,
    ChatCompletionBody,
    BatchRequestData,
    ProposalContext,
    ProposalGenerationError,
)


class TestOpenAIModelBehavior:
    """Test OpenAIModel enum business logic."""
    
    def test_temperature_support_logic(self):
        """Test temperature support detection - actual business logic."""
        # Most models support temperature
        assert OpenAIModel.GPT_4O.supports_temperature is True
        assert OpenAIModel.GPT_4O_MINI.supports_temperature is True
        assert OpenAIModel.GPT_4.supports_temperature is True
        
        # GPT-5 is special case that doesn't support temperature
        assert OpenAIModel.GPT_5.supports_temperature is False
    
    def test_enum_completeness(self):
        """Prevent accidental omissions in the enum."""
        expected_models = {"gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-5"}
        actual_models = {model.value for model in OpenAIModel}
        assert actual_models == expected_models


class TestChapterValidation:
    """Test Chapter Pydantic validation rules."""

    def test_empty_title_allowed(self):
        """Empty titles are now permitted for in-progress outlines."""
        chapter = Chapter(chapter_title="")
        assert chapter.chapter_title == ""

    def test_empty_description_allowed(self):
        """Empty summaries are permitted when outlining chapter shells."""
        chapter = Chapter(chapter_title="Valid Title")
        assert chapter.chapter_title == "Valid Title"


class TestBatchRequestIntegration:
    """Test complex batch request construction and integration."""
    
    def test_complete_openai_batch_request(self):
        """Test building a complete OpenAI batch API request structure."""
        # Build a realistic batch request like the generator would
        system_msg = ChatMessage(role="system", content="You are an expert textbook author.")
        user_msg = ChatMessage(role="user", content="Generate a proposal for quantum computing.")
        
        body = ChatCompletionBody(
            model="gpt-4o-mini",
            messages=[system_msg, user_msg],
            temperature=0.7,
            max_tokens=4000,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "textbook_proposal",
                    "schema": {"type": "object", "properties": {"title": {"type": "string"}}}
                }
            }
        )
        
        request = BatchRequestData(custom_id="gap_5_topic_quantum", body=body)
        
        # Verify the structure matches OpenAI batch API expectations
        assert request.custom_id == "gap_5_topic_quantum"
        assert request.body.model == "gpt-4o-mini"
        assert len(request.body.messages) == 2
        assert request.body.messages[0].role == "system"
        assert request.body.messages[1].role == "user"
        assert request.body.response_format is not None
        assert request.body.response_format["type"] == "json_schema"
    
    def test_typed_context_integration(self):
        """Test that typed dataclasses work together properly."""
        # Create typed research data
        articles = [
            Article(display_name="Quantum Algorithms", abstract="Advanced quantum computing methods"),
            Article(display_name="Quantum Hardware", abstract="Physical quantum computer implementations")
        ]
        books = [
            Book(display_name="Quantum Computing Fundamentals", abstract="Basic quantum computing principles")
        ]
        counts = ProposalCounts(num_articles_for_llm=2, num_books_for_llm=1)
        
        # Create context with typed data
        context = ProposalContext(
            topic_name="Quantum Computing",
            topic_id="quantum_001",
            gap_idx=5,
            requested_num_prompt_articles=2,
            requested_num_prompt_books=1,
            sampled_articles=articles,
            covering_books=books,
            counts=counts
        )
        
        # Test that the typed integration works correctly
        assert len(context.sampled_articles) == 2
        assert isinstance(context.sampled_articles[0], Article)
        assert isinstance(context.covering_books[0], Book)
        assert isinstance(context.counts, ProposalCounts)
        assert context.counts.num_articles_for_llm == 2
        assert context.sampled_articles[0].display_name == "Quantum Algorithms"


class TestTemperatureEdgeCases:
    """Test temperature parameter edge cases that matter for API calls."""
    
    def test_temperature_boundaries(self):
        """Test temperature at API boundaries."""
        msg = ChatMessage(role="user", content="Test")
        
        # Test minimum temperature (deterministic)
        body_min = ChatCompletionBody(model="gpt-4o", messages=[msg], temperature=0.0)
        assert body_min.temperature == 0.0
        
        # Test maximum temperature (most creative)
        body_max = ChatCompletionBody(model="gpt-4o", messages=[msg], temperature=2.0)
        assert body_max.temperature == 2.0


class TestErrorHandling:
    """Test error handling behavior."""
    
    def test_error_with_context_preservation(self):
        """Test that error context is preserved properly."""
        context = {"model": "gpt-4", "temperature": 0.7}
        error = ProposalGenerationError("API call failed", context=context)
        
        assert str(error) == "API call failed"
        assert error.context == context
        assert error.context["model"] == "gpt-4"
    
    def test_error_inheritance(self):
        """Test that error inherits from Exception for proper exception handling."""
        error = ProposalGenerationError("Test error")
        assert isinstance(error, Exception)


class TestProposalValidation:
    """Test Proposal validation rules - these test actual business constraints."""
    
    def test_chapter_count_constraints(self):
        """Test chapter count validation with business rules."""
        # Test too few chapters fails
        few_chapters = [
            Chapter(chapter_title=f"Chapter {i}")
            for i in range(1, 10)  # 9 chapters (below minimum of 10)
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            Proposal(
                title="Too Small Book",
                abstract="A book with too few chapters",
                chapters=few_chapters
            )
        assert "List should have at least 10 items" in str(exc_info.value)
        
        # Test too many chapters fails
        many_chapters = [
            Chapter(chapter_title=f"Chapter {i}")
            for i in range(1, 17)  # 16 chapters (exceeds max of 15)
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            Proposal(
                title="Too Large Book",
                abstract="A book with too many chapters",
                chapters=many_chapters
            )
        assert "List should have at most 15 items" in str(exc_info.value)
        
        # Test valid chapter count succeeds
        valid_chapters = [
            Chapter(chapter_title=f"Chapter {i}")
            for i in range(1, 13)  # 12 chapters (within 10-15 range)
        ]
        
        proposal = Proposal(
            title="Valid Book",
            abstract="A book with appropriate number of chapters",
            chapters=valid_chapters
        )
        # Test that the model was created successfully
        assert len(proposal.chapters) == 12  # type: ignore
        assert proposal.title == "Valid Book"  # type: ignore
    
    def test_empty_proposal_fields_allowed(self):
        """Empty title/abstract are allowed when seeding drafts."""
        valid_chapters = [
            Chapter(chapter_title=f"Chapter {i}")
            for i in range(1, 11)
        ]

        proposal_with_empty_title = Proposal(title="", abstract="Valid abstract", chapters=valid_chapters)
        assert proposal_with_empty_title.title == ""

        proposal_with_empty_abstract = Proposal(title="Valid Title", abstract="", chapters=valid_chapters)
        assert proposal_with_empty_abstract.abstract == ""


class TestStructuredOutputHandling:
    """Test structured output configuration for batch requests."""
    
    def test_structured_output_configuration(self):
        """Test that structured output is properly configured for batch API."""
        msg = ChatMessage(role="user", content="Generate structured proposal.")
        
        # Test with structured output enabled
        body = ChatCompletionBody(
            model="gpt-4o",
            messages=[msg],
            temperature=0.3,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "textbook_proposal",
                    "schema": {"type": "object", "properties": {"title": {"type": "string"}}}
                }
            }
        )
        
        request = BatchRequestData(custom_id="structured_test", body=body)
        
        # Verify structured output configuration
        assert request.body.response_format is not None
        assert request.body.response_format["type"] == "json_schema"
        assert "json_schema" in request.body.response_format
        assert request.body.response_format["json_schema"]["name"] == "textbook_proposal"
        
        # Test without structured output (should be None)
        simple_body = ChatCompletionBody(
            model="gpt-4o-mini",
            messages=[msg],
            temperature=0.7
        )
        assert simple_body.response_format is None
