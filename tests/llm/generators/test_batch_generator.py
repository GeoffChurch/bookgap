"""Tests for batch proposal generators."""

import os
from unittest.mock import patch

import pytest

from bookgap.llm.generators.openai_batch import OpenAIBatchGenerator
from bookgap.llm.protocols import (
    Article,
    BatchRequestData,
    Book,
    OpenAIModel,
    PromptPair,
    ProposalContext,
    ProposalCounts,
    ProposalGenerationError,
)


class TestOpenAIBatchGenerator:
    """Tests for OpenAI batch generator."""
    
    @pytest.fixture
    def generator_kwargs(self):
        """Default keyword arguments for generator construction."""
        return {
            "model": OpenAIModel.GPT_4O_MINI,
            "num_articles": 2,
            "num_books": 1,
            "temperature": 0.7,
            "max_tokens": 4000,
            "use_structured_output": True,
        }
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample proposal context."""
        articles = [
            Article(display_name="Quantum Computing Basics", abstract="Introduction to quantum computing principles"),
            Article(display_name="Quantum Algorithms", abstract="Advanced quantum algorithm design")
        ]
        books = [
            Book(display_name="Quantum Physics Textbook", abstract="Comprehensive quantum physics reference")
        ]
        counts = ProposalCounts(num_articles_for_llm=2, num_books_for_llm=1)
        
        return ProposalContext(
            topic_name="Quantum Computing",
            topic_id="quantum_001",
            gap_idx=5,
            requested_num_prompt_articles=2,
            requested_num_prompt_books=1,
            sampled_articles=articles,
            covering_books=books,
            counts=counts
        )
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_initialization_success(self, generator_kwargs):
        """Test successful generator initialization."""
        generator = OpenAIBatchGenerator(**generator_kwargs)

        assert generator.model is OpenAIModel.GPT_4O_MINI
        assert generator.num_articles == 2
        assert generator.num_books == 1
        assert generator.temperature == 0.7
        assert generator.max_tokens == 4000
        assert generator.use_structured_output is True
    
    @patch.dict(os.environ, {}, clear=True)
    def test_initialization_missing_api_key(self, generator_kwargs):
        """Test initialization fails without API key."""
        with pytest.raises(ProposalGenerationError) as exc_info:
            OpenAIBatchGenerator(**generator_kwargs)

        assert "OPENAI_API_KEY environment variable is required" in str(exc_info.value)
        assert exc_info.value.context["provider"] == "openai"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_build_prompt(self, generator_kwargs, sample_context):
        """Test prompt building functionality."""
        generator = OpenAIBatchGenerator(**generator_kwargs)

        prompts = generator._build_prompt(sample_context)

        assert isinstance(prompts, PromptPair)
        assert "academic researcher and educator" in prompts.system_prompt
        assert "Quantum Computing" in prompts.user_prompt
        assert "Quantum Computing Basics" in prompts.user_prompt
        assert "Quantum Algorithms" in prompts.user_prompt
        assert "Quantum Physics Textbook" in prompts.user_prompt
        assert "Based on the research articles" in prompts.user_prompt
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_request_from_prompt(self, generator_kwargs, sample_context):
        """Test batch request creation from prompts."""
        generator = OpenAIBatchGenerator(**generator_kwargs)

        prompts = PromptPair(
            system_prompt="You are a helpful assistant.",
            user_prompt="Generate a textbook proposal."
        )
        
        request = generator._create_request_from_prompt(prompts, sample_context)

        assert isinstance(request, BatchRequestData)
        assert request.custom_id == "gap_5_topic_quantum_001"
        assert request.body.model == "gpt-4o-mini"
        assert len(request.body.messages) == 2
        assert request.body.messages[0].role == "system"
        assert request.body.messages[1].role == "user"
        assert request.body.temperature == 0.7  # GPT-4O-MINI supports temperature
        assert request.body.max_tokens == 4000
        assert request.body.response_format is not None  # Structured output enabled
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_temperature_handling_for_gpt5(self, generator_kwargs, sample_context):
        """Test that GPT-5 doesn't include temperature parameter."""
        kwargs = generator_kwargs.copy()
        kwargs["model"] = OpenAIModel.GPT_5
        generator = OpenAIBatchGenerator(**kwargs)

        prompts = PromptPair(
            system_prompt="You are a helpful assistant.",
            user_prompt="Generate a textbook proposal."
        )
        
        request = generator._create_request_from_prompt(prompts, sample_context)

        assert request.body.model == "gpt-5"
        assert request.body.temperature is None  # GPT-5 doesn't support temperature
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_structured_output_disabled(self, generator_kwargs, sample_context):
        """Test request creation with structured output disabled."""
        kwargs = generator_kwargs.copy()
        kwargs["use_structured_output"] = False
        generator = OpenAIBatchGenerator(**kwargs)

        prompts = PromptPair(
            system_prompt="You are a helpful assistant.",
            user_prompt="Generate a textbook proposal."
        )
        
        request = generator._create_request_from_prompt(prompts, sample_context)

        assert request.body.response_format is None  # No structured output
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_generate_batch_requests(self, generator_kwargs):
        """Test end-to-end batch request generation."""
        generator = OpenAIBatchGenerator(**generator_kwargs)

        # Create multiple contexts
        articles1 = [Article(display_name="AI Article 1", abstract="AI research")]
        articles2 = [Article(display_name="ML Article 1", abstract="ML research")]
        counts = ProposalCounts(num_articles_for_llm=1, num_books_for_llm=0)
        
        contexts = [
            ProposalContext(
                topic_name="AI", topic_id="ai_001", gap_idx=1,
                requested_num_prompt_articles=1, requested_num_prompt_books=0,
                sampled_articles=articles1, covering_books=[], counts=counts
            ),
            ProposalContext(
                topic_name="ML", topic_id="ml_001", gap_idx=2,
                requested_num_prompt_articles=1, requested_num_prompt_books=0,
                sampled_articles=articles2, covering_books=[], counts=counts
            )
        ]
        
        requests = generator.generate_batch_requests(contexts)

        assert len(requests) == 2
        assert requests[0].custom_id == "gap_1_topic_ai_001"
        assert requests[1].custom_id == "gap_2_topic_ml_001"
        assert all(isinstance(req, BatchRequestData) for req in requests)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_prompt_formatting_no_books(self, generator_kwargs):
        """Test prompt formatting when no books are provided."""
        generator = OpenAIBatchGenerator(**generator_kwargs)

        articles = [Article(display_name="Test Article", abstract="Test abstract")]
        counts = ProposalCounts(num_articles_for_llm=1, num_books_for_llm=0)
        
        context = ProposalContext(
            topic_name="Test Topic", topic_id="test_001", gap_idx=1,
            requested_num_prompt_articles=1, requested_num_prompt_books=0,
            sampled_articles=articles, covering_books=[], counts=counts
        )
        
        prompts = generator._build_prompt(context)
        
        # Should not mention books in the introduction
        assert " and related books" not in prompts.user_prompt
        # Should not have a books section
        assert "## Related Existing Books" not in prompts.user_prompt
        # Should still have articles section
        assert "## Sample Articles" in prompts.user_prompt
        assert "Test Article" in prompts.user_prompt
