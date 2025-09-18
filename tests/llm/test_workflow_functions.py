"""Tests for workflow functions."""

import pytest
from unittest.mock import Mock, patch

from bookgap.llm.generators.workflow_functions import (
    generate_batch_requests,
    validate_batch_requests,
    get_supported_models
)
from bookgap.llm.protocols import (
    ProposalContext,
    BatchRequestData,
    ChatCompletionBody,
    ChatMessage,
    Article,
    Book,
    ProposalCounts,
    ProposalGenerationError,
    OpenAIModel
)


class TestGenerateBatchRequests:
    """Test generate_batch_requests function."""
    
    def test_empty_contexts(self):
        """Test with empty contexts list."""
        config = Mock()
        result = generate_batch_requests([], config)
        assert result == []
    
    @patch('bookgap.llm.generators.workflow_functions.instantiate')
    def test_successful_generation(self, mock_instantiate):
        """Test successful batch request generation."""
        # Setup mock generator
        mock_generator = Mock()
        mock_requests = [
            BatchRequestData(
                custom_id="gap_0_topic_test",
                body=ChatCompletionBody(
                    model="gpt-4o-mini",
                    messages=[ChatMessage(role="user", content="Test")],
                    temperature=0.7
                )
            )
        ]
        mock_generator.generate_batch_requests.return_value = mock_requests
        mock_instantiate.return_value = mock_generator
        
        # Create test context
        contexts = [
            ProposalContext(
                topic_name="Test Topic",
                topic_id="test_topic",
                gap_idx=0,
                requested_num_prompt_articles=1,
                requested_num_prompt_books=1,
                sampled_articles=[Article(display_name="Test Article", abstract="Test abstract")],
                covering_books=[Book(display_name="Test Book", abstract="Test book abstract")],
                counts=ProposalCounts(num_articles_for_llm=1, num_books_for_llm=1)
            )
        ]
        
        config = Mock()
        config.llm = Mock()
        
        result = generate_batch_requests(contexts, config)
        
        # Verify results
        assert len(result) == 1
        assert result[0].custom_id == "gap_0_topic_test"
        mock_instantiate.assert_called_once_with(config.llm, _config=config)
        mock_generator.generate_batch_requests.assert_called_once_with(contexts)
    
    @patch('bookgap.llm.generators.workflow_functions.instantiate')
    def test_generation_error(self, mock_instantiate):
        """Test error handling during generation."""
        mock_instantiate.side_effect = Exception("Generator instantiation failed")
        
        # Create a real context to avoid type issues
        contexts = [
            ProposalContext(
                topic_name="Test Topic",
                topic_id="test_topic", 
                gap_idx=0,
                requested_num_prompt_articles=1,
                requested_num_prompt_books=1,
                sampled_articles=[Article(display_name="Test Article", abstract="Test abstract")],
                covering_books=[Book(display_name="Test Book", abstract="Test book abstract")],
                counts=ProposalCounts(num_articles_for_llm=1, num_books_for_llm=1)
            )
        ]
        config = Mock()
        
        with pytest.raises(ProposalGenerationError) as exc_info:
            generate_batch_requests(contexts, config)
        
        assert "Generator instantiation failed" in str(exc_info.value)
        # Check that error context contains provider info
        assert "provider" in exc_info.value.context


class TestValidateBatchRequests:
    """Test validate_batch_requests function."""
    
    def test_valid_requests(self):
        """Test validation of valid requests."""
        requests = [
            BatchRequestData(
                custom_id="gap_0_topic_test",
                body=ChatCompletionBody(
                    model="gpt-4o-mini",
                    messages=[
                        ChatMessage(role="system", content="You are helpful."),
                        ChatMessage(role="user", content="Generate a proposal.")
                    ],
                    temperature=0.7
                )
            )
        ]
        
        # Should not raise any exception
        validate_batch_requests(requests)
    
    def test_empty_requests(self):
        """Test validation of empty requests list."""
        with pytest.raises(ProposalGenerationError) as exc_info:
            validate_batch_requests([])
        
        assert "No batch requests provided" in str(exc_info.value)
    
    def test_duplicate_custom_ids(self):
        """Test validation catches duplicate custom IDs."""
        requests = [
            BatchRequestData(
                custom_id="duplicate_id",
                body=ChatCompletionBody(
                    model="gpt-4o-mini",
                    messages=[ChatMessage(role="user", content="Test 1")],
                    temperature=0.7
                )
            ),
            BatchRequestData(
                custom_id="duplicate_id",  # Duplicate ID
                body=ChatCompletionBody(
                    model="gpt-4o-mini",
                    messages=[ChatMessage(role="user", content="Test 2")],
                    temperature=0.7
                )
            )
        ]
        
        with pytest.raises(ProposalGenerationError) as exc_info:
            validate_batch_requests(requests)
        
        assert "Duplicate custom_id found: duplicate_id" in str(exc_info.value)


class TestGetSupportedModels:
    """Test get_supported_models function."""
    
    def test_returns_model_list(self):
        """Test that it returns a list of supported models."""
        models = get_supported_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt-4o-mini" in models
        assert "gpt-4o" in models
        assert "gpt-4" in models
        assert "gpt-5" in models
    
    def test_matches_enum_values(self):
        """Test that returned models match the OpenAIModel enum values."""
        models = get_supported_models()
        enum_values = [model.value for model in OpenAIModel]
        
        assert set(models) == set(enum_values)
