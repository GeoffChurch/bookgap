"""Comprehensive tests for workflow functions."""

import pytest
from unittest.mock import Mock, patch, mock_open

from bookgap.llm.generators.workflow_functions import (
    generate_batch_requests,
    validate_batch_requests,
    get_supported_models,
    find_gaps_for_config,
    build_contexts_for_gaps,
    submit_batch_job,
    monitor_and_download_results,
    run_batch_workflow,
    GapWithContext
)
from bookgap.llm.protocols import (
    ProposalContext,
    BatchRequestData,
    ChatCompletionBody,
    ChatMessage,
    ProposalGenerationError,
    OpenAIModel
)


class TestFindGapsForConfig:
    """Test find_gaps_for_config function."""
    
    @patch('bookgap.llm.generators.workflow_functions.build_cloud')
    @patch('pyalex.Subfields')  # Patch the actual module
    @patch('bookgap.llm.generators.workflow_functions.get_subclouds_for_topic')
    @patch('bookgap.llm.generators.workflow_functions.find_gaps_for_topic')
    def test_successful_gap_finding(self, mock_find_gaps, mock_get_subclouds, mock_subfields, mock_build_cloud):
        """Test successful gap finding with proper configuration."""
        # Setup mocks
        mock_config = Mock()
        mock_config.gap.denoiser.name = "NONE"
        mock_config.gap.drop_k = 0
        mock_config.data.max_works = 1000
        mock_config.data.deduplicator = None  # Simplified DataConfig
        mock_config.gap.radius_fraction = 0.5
        mock_config.gap.cell_type = Mock()
        mock_config.gap.index_type = Mock()
        mock_config.gap.hnsw_m = 16
        mock_config.gap.hnsw_efc = 200
        mock_config.gap.hnsw_efs = 200
        mock_config.batch.num_gaps = 2

        # Setup pyalex mock - hardcoded AI subfield (matches workflow_functions.py)
        mock_topic = {
            "display_name": "Machine Learning",
            "id": "https://openalex.org/T123"
        }
        mock_subfield_instance = {"topics": [mock_topic]}
        mock_subfields.return_value = {"1702": mock_subfield_instance}  # Hardcoded AI subfield

        # Setup cloud mocks
        mock_cloud = Mock()
        mock_build_cloud.return_value = mock_cloud
        
        mock_books = Mock()
        mock_articles = Mock()
        mock_books.embeddings.return_value = [1, 2, 3]  # 3 books
        mock_articles.embeddings.return_value = [1, 2, 3, 4, 5]  # 5 articles
        mock_get_subclouds.return_value = (mock_books, mock_articles)

        # Setup gap generator mock
        mock_gap1 = Mock()
        mock_gap1.dists_idxs.idxs = [0, 1, 2]
        mock_gap2 = Mock()
        mock_gap2.dists_idxs.idxs = [3, 4]
        mock_find_gaps.return_value = iter([mock_gap1, mock_gap2])

        # Call function
        result = find_gaps_for_config(mock_config)

        # Assertions
        assert len(result) == 2
        assert all(isinstance(gap, GapWithContext) for gap in result)
        assert result[0].topic_name == "Machine Learning"
        assert result[0].topic_id == "T123"  # ID after splitting on "/"
        assert result[0].gap_idx == 0
        assert result[1].gap_idx == 1
        
        # Verify all mocks were called correctly
        mock_build_cloud.assert_called_once()
        mock_get_subclouds.assert_called_once_with(mock_cloud, "T123")
        mock_find_gaps.assert_called_once()

    @patch('bookgap.llm.generators.workflow_functions.build_cloud')
    @patch('pyalex.Subfields')  # Patch the actual module
    @patch('bookgap.llm.generators.workflow_functions.get_subclouds_for_topic')
    def test_no_articles_found(self, mock_get_subclouds, mock_subfields, mock_build_cloud):
        """Test handling when no articles are found for a topic."""
        # Setup mocks
        mock_config = Mock()
        mock_config.gap.denoiser.name = "NONE"
        mock_config.data.max_works = 1000
        mock_config.data.deduplicator = None  # Simplified DataConfig
        mock_config.batch.num_gaps = 2

        # Setup pyalex mock - hardcoded AI subfield
        mock_topic = {
            "display_name": "Machine Learning", 
            "id": "https://openalex.org/T123"
        }
        mock_subfield_instance = {"topics": [mock_topic]}
        mock_subfields.return_value = {"1702": mock_subfield_instance}  # Hardcoded AI subfield_open

from bookgap.llm.generators.workflow_functions import (
    generate_batch_requests,
    validate_batch_requests,
    get_supported_models,
    find_gaps_for_config,
    build_contexts_for_gaps,
    submit_batch_job,
    monitor_and_download_results,
    run_batch_workflow,
    GapWithContext
)
from bookgap.llm.protocols import (
    ProposalContext,
    BatchRequestData,
    ChatCompletionBody,
    ChatMessage,
    ProposalGenerationError,
    OpenAIModel
)


class TestFindGapsForConfig:
    """Test find_gaps_for_config function."""
    
    @patch('bookgap.llm.generators.workflow_functions.build_cloud')
    @patch('pyalex.Subfields')  # Patch the actual module
    @patch('bookgap.llm.generators.workflow_functions.get_subclouds_for_topic')
    @patch('bookgap.llm.generators.workflow_functions.find_gaps_for_topic')
    def test_successful_gap_finding(self, mock_find_gaps, mock_get_subclouds, mock_subfields, mock_build_cloud):
        """Test successful gap finding with proper configuration."""
        # Setup mocks
        mock_config = Mock()
        mock_config.gap.denoiser.name = "NONE"
        mock_config.gap.drop_k = 0
        mock_config.data.max_works = 1000
        mock_config.data.deduplicator = None  # Simplified DataConfig
        mock_config.gap.radius_fraction = 0.5
        mock_config.gap.cell_type = Mock()
        mock_config.gap.index_type = Mock()
        mock_config.gap.hnsw_m = 16
        mock_config.gap.hnsw_efc = 200
        mock_config.gap.hnsw_efs = 200
        mock_config.batch.num_gaps = 2

        # Setup pyalex mock - hardcoded AI subfield (matches workflow_functions.py)
        mock_topic = {
            "display_name": "Machine Learning",
            "id": "https://openalex.org/T123"
        }
        mock_subfield_instance = {"topics": [mock_topic]}
        mock_subfields.return_value = {"1702": mock_subfield_instance}  # Hardcoded AI subfield

        # Setup cloud mocks
        mock_cloud = Mock()
        mock_build_cloud.return_value = mock_cloud
        
        mock_books = Mock()
        mock_articles = Mock()
        mock_books.embeddings.return_value = [1, 2, 3]  # 3 books
        mock_articles.embeddings.return_value = [1, 2, 3, 4, 5]  # 5 articles
        mock_get_subclouds.return_value = (mock_books, mock_articles)

        # Setup gap generator mock
        mock_gap1 = Mock()
        mock_gap1.dists_idxs.idxs = [0, 1, 2]
        mock_gap2 = Mock()
        mock_gap2.dists_idxs.idxs = [3, 4]
        mock_find_gaps.return_value = iter([mock_gap1, mock_gap2])

        # Call function
        result = find_gaps_for_config(mock_config)

        # Assertions
        assert len(result) == 2
        assert all(isinstance(gap, GapWithContext) for gap in result)
        assert result[0].topic_name == "Machine Learning"
        assert result[0].topic_id == "T123"  # ID after splitting on "/"
        assert result[0].gap_idx == 0
        assert result[1].gap_idx == 1
        
        # Verify all mocks were called correctly
        mock_build_cloud.assert_called_once()
        mock_get_subclouds.assert_called_once_with(mock_cloud, "T123")
        mock_find_gaps.assert_called_once()

    @patch('bookgap.llm.generators.workflow_functions.build_cloud')
    @patch('pyalex.Subfields')  # Patch the actual module
    @patch('bookgap.llm.generators.workflow_functions.get_subclouds_for_topic')
    def test_no_articles_found(self, mock_get_subclouds, mock_subfields, mock_build_cloud):
        """Test handling when no articles are found for a topic."""
        # Setup mocks
        mock_config = Mock()
        mock_config.gap.denoiser.name = "NONE"
        mock_config.data.subfield.value = "1702"
        mock_config.batch.num_gaps = 2

        # Setup pyalex mock
        mock_topic = {
            "display_name": "Empty Topic",
            "id": "https://openalex.org/T123"
        }
        mock_subfield_instance = {"topics": [mock_topic]}
        mock_subfields.return_value = {mock_config.data.subfield.value: mock_subfield_instance}

        # Setup cloud mocks - no articles
        mock_books = Mock()
        mock_articles = Mock()
        mock_books.embeddings.return_value = [1, 2]
        mock_articles.embeddings.return_value = []  # No articles
        mock_get_subclouds.return_value = (mock_books, mock_articles)

        # Call function
        result = find_gaps_for_config(mock_config)

        # Should return empty list when no articles
        assert result == []

    @patch('bookgap.llm.generators.workflow_functions.build_cloud')
    @patch('pyalex.Subfields')  # Patch the actual module
    def test_no_topics_found(self, mock_subfields, mock_build_cloud):
        """Test handling when no topics are found for a subfield."""
        # Setup mocks
        mock_config = Mock()
        mock_config.gap.denoiser.name = "NONE"
        mock_config.data.subfield.value = "1702"

        # Setup pyalex mock with no topics
        mock_subfield_instance = {"topics": []}
        mock_subfields.return_value = {mock_config.data.subfield.value: mock_subfield_instance}

        # Call function
        result = find_gaps_for_config(mock_config)

        # Should return empty list when no topics
        assert result == []


class TestBuildContextsForGaps:
    """Test build_contexts_for_gaps function."""
    
    def test_successful_context_building(self):
        """Test successful context building from gaps."""
        # Setup test data
        mock_gap = Mock()
        mock_dists_idxs = Mock()
        mock_dists_idxs.idxs = [0, 1, 2, 3, 4]  # 5 articles in gap
        mock_gap.get_dists_idxs.return_value = mock_dists_idxs
        
        mock_articles = Mock()
        mock_books = Mock()
        
        # Mock works data
        article_works = [
            {"display_name": f"Article {i}", "abstract": f"Abstract {i}"}
            for i in range(10)
        ]
        book_works = [
            {"display_name": f"Book {i}", "abstract": f"Book abstract {i}"}
            for i in range(5)
        ]
        
        mock_articles.works.return_value = article_works
        mock_books.works.return_value = book_works
        
        gap_with_context = GapWithContext(
            gap=mock_gap,
            topic_name="Test Topic",
            topic_id="123",
            gap_idx=0,
            books=mock_books,
            articles=mock_articles
        )
        
        mock_config = Mock()
        mock_config.llm.num_articles = 3
        mock_config.llm.num_books = 2
        
        # Call function
        with patch('random.sample') as mock_sample:
            # Mock random sampling to be deterministic
            mock_sample.side_effect = lambda population, k: population[:k]
            
            result = build_contexts_for_gaps([gap_with_context], mock_config)
        
        # Assertions
        assert len(result) == 1
        context = result[0]
        assert isinstance(context, ProposalContext)
        assert context.topic_name == "Test Topic"
        assert context.topic_id == "123"
        assert context.gap_idx == 0
        assert len(context.sampled_articles) == 3
        assert len(context.covering_books) == 2
        assert context.counts.num_articles_for_llm == 3
        assert context.counts.num_books_for_llm == 2

    def test_limited_by_gap_size(self):
        """Test context building when gap size is smaller than requested articles."""
        # Setup test data with small gap
        mock_gap = Mock()
        mock_dists_idxs = Mock()
        mock_dists_idxs.idxs = [0, 1]  # Only 2 articles in gap
        mock_gap.get_dists_idxs.return_value = mock_dists_idxs
        
        mock_articles = Mock()
        mock_books = Mock()
        
        article_works = [
            {"display_name": f"Article {i}", "abstract": f"Abstract {i}"}
            for i in range(10)
        ]
        book_works = [
            {"display_name": f"Book {i}", "abstract": f"Book abstract {i}"}
            for i in range(5)
        ]
        
        mock_articles.works.return_value = article_works
        mock_books.works.return_value = book_works
        
        gap_with_context = GapWithContext(
            gap=mock_gap,
            topic_name="Small Gap Topic",
            topic_id="456",
            gap_idx=1,
            books=mock_books,
            articles=mock_articles
        )
        
        mock_config = Mock()
        mock_config.llm.num_articles = 5  # Request more than available
        mock_config.llm.num_books = 2
        
        # Call function
        result = build_contexts_for_gaps([gap_with_context], mock_config)
        
        # Should be limited by gap size (2 articles)
        context = result[0]
        assert len(context.sampled_articles) == 2  # Limited by gap size
        assert len(context.covering_books) == 2
        assert context.counts.num_articles_for_llm == 2


class TestSubmitBatchJob:
    """Test submit_batch_job function."""
    
    @patch('bookgap.llm.generators.workflow_functions.OpenAI')
    @patch('tempfile.NamedTemporaryFile')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.unlink')
    def test_successful_batch_submission(self, mock_unlink, mock_file_open, mock_temp_file, mock_openai_class):
        """Test successful batch job submission."""
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_temp_file_handle = Mock()
        mock_temp_file_handle.name = "/tmp/test_batch.jsonl"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_handle
        
        mock_file = Mock()
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file
        
        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_client.batches.create.return_value = mock_batch
        
        # Test data
        batch_requests = [
            BatchRequestData(
                custom_id="test-1",
                body=ChatCompletionBody(
                    model="gpt-4o-mini",
                    messages=[
                        ChatMessage(role="system", content="You are a helpful assistant."),
                        ChatMessage(role="user", content="Test prompt.")
                    ],
                    max_tokens=1000
                )
            )
        ]
        
        mock_config = Mock()
        mock_config.batch.completion_window = "24h"
        
        # Call function
        result = submit_batch_job(batch_requests, mock_config)
        
        # Assertions
        assert result == "batch-456"
        mock_client.files.create.assert_called_once()
        mock_client.batches.create.assert_called_once()
        mock_unlink.assert_called_once_with("/tmp/test_batch.jsonl")

    def test_empty_requests_error(self):
        """Test error handling for empty batch requests."""
        mock_config = Mock()
        
        with pytest.raises(ValueError, match="No batch requests to submit"):
            submit_batch_job([], mock_config)

    @patch('bookgap.llm.generators.workflow_functions.OpenAI')
    @patch('tempfile.NamedTemporaryFile')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.unlink')
    def test_request_serialization(self, mock_unlink, mock_file_open, mock_temp_file, mock_openai_class):
        """Test that requests are properly serialized to JSONL format."""
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_temp_file_handle = Mock()
        mock_temp_file_handle.name = "/tmp/test_batch.jsonl"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_handle
        
        mock_file = Mock()
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file
        
        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_client.batches.create.return_value = mock_batch
        
        # Test data with optional fields
        batch_requests = [
            BatchRequestData(
                custom_id="test-with-temp",
                body=ChatCompletionBody(
                    model="gpt-4o",
                    messages=[ChatMessage(role="user", content="Test.")],
                    temperature=0.7,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
            )
        ]
        
        mock_config = Mock()
        mock_config.batch.completion_window = "24h"
        
        # Call function
        submit_batch_job(batch_requests, mock_config)
        
        # Verify file was written (mock_file_open should have been called)
        mock_temp_file.assert_called_once()


class TestMonitorAndDownloadResults:
    """Test monitor_and_download_results function."""
    
    @patch('bookgap.llm.generators.workflow_functions.OpenAI')
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_successful_monitoring_and_download(self, mock_sleep, mock_openai_class):
        """Test successful monitoring and result download."""
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock batch status progression
        mock_batch_in_progress = Mock()
        mock_batch_in_progress.status = "in_progress"
        
        mock_batch_completed = Mock()
        mock_batch_completed.status = "completed"
        mock_batch_completed.output_file_id = "file-output-123"
        mock_batch_completed.metadata = {"0": '{"test": "config"}'}
        mock_batch_completed.request_counts.total = 10
        mock_batch_completed.request_counts.completed = 10
        mock_batch_completed.request_counts.failed = 0
        
        mock_client.batches.retrieve.side_effect = [
            mock_batch_in_progress,  # First call: still in progress
            mock_batch_completed     # Second call: completed
        ]
        
        # Mock file download
        mock_file_response = Mock()
        mock_file_response.content = b'{"result": "test1"}\n{"result": "test2"}'
        mock_client.files.content.return_value = mock_file_response
        
        mock_config = Mock()
        
        # Call function
        result = monitor_and_download_results("batch-123", mock_config)
        
        # Assertions
        assert result["batch_id"] == "batch-123"
        assert result["status"] == "completed"
        assert len(result["results"]) == 2
        assert result["num_results"] == 2
        assert result["original_config"] == '{"test": "config"}'
        assert result["request_counts"]["total"] == 10
        assert result["request_counts"]["completed"] == 10
        assert result["request_counts"]["failed"] == 0
        
        # Verify sleep was called (polling)
        mock_sleep.assert_called()

    @patch('bookgap.llm.generators.workflow_functions.OpenAI')
    def test_batch_failure_handling(self, mock_openai_class):
        """Test handling of failed batch jobs."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_batch_failed = Mock()
        mock_batch_failed.status = "failed"
        mock_client.batches.retrieve.return_value = mock_batch_failed
        
        mock_config = Mock()
        
        # Should raise exception for failed batch
        with pytest.raises(Exception, match="Batch job failed with status: failed"):
            monitor_and_download_results("batch-failed", mock_config)

    @patch('bookgap.llm.generators.workflow_functions.OpenAI')
    @patch('time.sleep')
    def test_monitoring_progression(self, mock_sleep, mock_openai_class):
        """Test that monitoring progresses through different states."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock progression through different states
        statuses = ["validating", "in_progress", "finalizing", "completed"]
        mock_batches = []
        
        for status in statuses[:-1]:
            mock_batch = Mock()
            mock_batch.status = status
            mock_batches.append(mock_batch)
        
        # Final completed batch
        mock_batch_completed = Mock()
        mock_batch_completed.status = "completed"
        mock_batch_completed.output_file_id = "file-123"
        mock_batch_completed.metadata = {}
        mock_batch_completed.request_counts.total = 5
        mock_batch_completed.request_counts.completed = 5
        mock_batch_completed.request_counts.failed = 0
        mock_batches.append(mock_batch_completed)
        
        mock_client.batches.retrieve.side_effect = mock_batches
        
        # Mock file download
        mock_file_response = Mock()
        mock_file_response.content = b'{"result": "test"}'
        mock_client.files.content.return_value = mock_file_response
        
        mock_config = Mock()
        
        # Call function
        result = monitor_and_download_results("batch-progression", mock_config)
        
        # Should have called retrieve for each status
        assert mock_client.batches.retrieve.call_count == len(statuses)
        assert result["status"] == "completed"


class TestRunBatchWorkflow:
    """Test run_batch_workflow function."""
    
    @patch('bookgap.llm.generators.workflow_functions.find_gaps_for_config')
    @patch('bookgap.llm.generators.workflow_functions.build_contexts_for_gaps')
    @patch('bookgap.llm.generators.workflow_functions.generate_batch_requests')
    @patch('bookgap.llm.generators.workflow_functions.submit_batch_job')
    @patch('bookgap.llm.generators.workflow_functions.monitor_and_download_results')
    def test_complete_workflow(self, mock_monitor, mock_submit, mock_generate, 
                             mock_build_contexts, mock_find_gaps):
        """Test the complete workflow orchestration."""
        # Setup mocks
        mock_gaps = [Mock(), Mock()]
        mock_find_gaps.return_value = mock_gaps
        
        mock_contexts = [Mock(), Mock()]
        mock_build_contexts.return_value = mock_contexts
        
        mock_requests = [Mock(), Mock()]
        mock_generate.return_value = mock_requests
        
        mock_submit.return_value = "batch-123"
        
        mock_results = {
            "batch_id": "batch-123",
            "status": "completed",
            "results": [{"test": "result1"}, {"test": "result2"}],
            "num_results": 2
        }
        mock_monitor.return_value = mock_results
        
        mock_config = Mock()
        mock_config.batch.num_gaps = 2
        
        # Call function
        result = run_batch_workflow(mock_config)
        
        # Verify all steps were called in order
        mock_find_gaps.assert_called_once_with(mock_config)
        mock_build_contexts.assert_called_once_with(mock_gaps, mock_config)
        mock_generate.assert_called_once_with(mock_contexts, mock_config)
        mock_submit.assert_called_once_with(mock_requests, mock_config)
        mock_monitor.assert_called_once_with("batch-123", mock_config)
        
        # Verify result
        assert result == mock_results


class TestGenerateBatchRequests:
    """Test generate_batch_requests function."""
    
    def test_empty_contexts(self):
        """Test handling of empty contexts list."""
        mock_config = Mock()
        result = generate_batch_requests([], mock_config)
        assert result == []

    @patch('bookgap.llm.generators.workflow_functions.instantiate')
    def test_successful_generation(self, mock_instantiate):
        """Test successful batch request generation."""
        # Setup mocks
        mock_generator = Mock()
        mock_requests = [Mock(), Mock()]
        mock_generator.generate_batch_requests.return_value = mock_requests
        mock_instantiate.return_value = mock_generator
        
        mock_contexts = [Mock(), Mock()]
        mock_config = Mock()
        
        # Call function
        result = generate_batch_requests(mock_contexts, mock_config)
        
        # Assertions
        assert result == mock_requests
        mock_instantiate.assert_called_once_with(mock_config.llm, _config=mock_config)
        mock_generator.generate_batch_requests.assert_called_once_with(mock_contexts)

    @patch('bookgap.llm.generators.workflow_functions.instantiate')
    def test_generation_error(self, mock_instantiate):
        """Test error handling during generation."""
        # Setup mocks to raise exception
        mock_instantiate.side_effect = Exception("Test error")
        
        mock_contexts = [Mock()]
        mock_config = Mock()
        mock_config.llm.model_name = "gpt-4o"
        mock_config.llm._target_ = "test.target"
        
        # Should raise ProposalGenerationError
        with pytest.raises(ProposalGenerationError) as exc_info:
            generate_batch_requests(mock_contexts, mock_config)
        
        # Verify error context
        assert "Failed to generate batch requests" in str(exc_info.value)
        assert exc_info.value.context["num_contexts"] == 1


class TestValidateBatchRequests:
    """Test validate_batch_requests function."""
    
    def test_valid_requests(self):
        """Test validation of valid batch requests."""
        requests = [
            BatchRequestData(
                custom_id="req-1",
                body=ChatCompletionBody(
                    model="gpt-4o-mini",
                    messages=[ChatMessage(role="user", content="Test 1")]
                )
            ),
            BatchRequestData(
                custom_id="req-2", 
                body=ChatCompletionBody(
                    model="gpt-4o",
                    messages=[ChatMessage(role="user", content="Test 2")]
                )
            )
        ]
        
        # Should not raise any exception
        validate_batch_requests(requests)

    def test_empty_requests(self):
        """Test validation of empty requests list."""
        with pytest.raises(ProposalGenerationError, match="No batch requests provided"):
            validate_batch_requests([])

    def test_duplicate_custom_ids(self):
        """Test detection of duplicate custom IDs."""
        requests = [
            BatchRequestData(
                custom_id="duplicate",
                body=ChatCompletionBody(
                    model="gpt-4o-mini",
                    messages=[ChatMessage(role="user", content="Test 1")]
                )
            ),
            BatchRequestData(
                custom_id="duplicate",  # Same ID
                body=ChatCompletionBody(
                    model="gpt-4o",
                    messages=[ChatMessage(role="user", content="Test 2")]
                )
            )
        ]
        
        with pytest.raises(ProposalGenerationError) as exc_info:
            validate_batch_requests(requests)
        
        assert "Duplicate custom_id found: duplicate" in str(exc_info.value)


class TestGetSupportedModels:
    """Test get_supported_models function."""
    
    def test_returns_model_list(self):
        """Test that function returns a list of model names."""
        result = get_supported_models()
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(model, str) for model in result)

    def test_matches_enum_values(self):
        """Test that returned models match OpenAIModel enum values."""
        result = get_supported_models()
        expected_models = [model.value for model in OpenAIModel]
        assert result == expected_models
