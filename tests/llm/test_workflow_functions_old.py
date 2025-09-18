"""Tests for workflow functions."""

import pytest
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path

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
    Article,
    Book,
    ProposalCounts,
    ProposalGenerationError,
    OpenAIModel
)
from bookgap.configs.config import ExperimentConfig


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
        mock_config.data.subfield.value = "1702"
        mock_config.gap.radius_fraction = 0.5
        mock_config.gap.cell_type = Mock()
        mock_config.gap.index_type = Mock()
        mock_config.gap.hnsw_m = 16
        mock_config.gap.hnsw_efc = 200
        mock_config.gap.hnsw_efs = 200
        mock_config.batch.num_gaps = 2

        # Setup pyalex mock
        mock_subfield = Mock()
        mock_topic = Mock()
        mock_topic.display_name = "Machine Learning"
        mock_topic.id = "https://openalex.org/T123"
        mock_subfield_instance = {"topics": [mock_topic]}
        mock_subfields.return_value = {mock_config.data.subfield.value: mock_subfield_instance}

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
        assert result[0].topic_id == "123"
        assert result[0].gap_idx == 0
        assert result[1].gap_idx == 1
        
        # Verify all mocks were called correctly
        mock_build_cloud.assert_called_once()
        mock_get_subclouds.assert_called_once_with(mock_cloud, "123")
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
        mock_topic = Mock()
        mock_topic.display_name = "Empty Topic"
        mock_topic.id = "https://openalex.org/T123"
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
        mock_config.gap.index_type = Mock()
        mock_config.gap.hnsw_m = 16
        mock_config.gap.hnsw_efc = 200
        mock_config.gap.hnsw_efs = 200
        mock_config.batch.num_gaps = 2
        
        # Mock pyalex structure
        mock_topic = Mock()
        mock_topic.display_name = "Test Topic"
        mock_topic.id = "https://openalex.org/C41008148"
        
        mock_subfield = Mock()
        mock_subfield.topics = [mock_topic]
        
        mock_subfields = Mock()
        mock_subfields.__getitem__.return_value = mock_subfield
        
        mock_pyalex.Subfields.return_value = mock_subfields
        
        # Mock subclouds
        mock_books = Mock()
        mock_articles = Mock()
        mock_books.embeddings.return_value = [[1, 2, 3]] * 10  # 10 books
        mock_articles.embeddings.return_value = [[1, 2, 3]] * 100  # 100 articles
        mock_get_subclouds.return_value = (mock_books, mock_articles)
        
        # Mock gaps
        mock_gap1 = Mock()
        mock_gap1.dists_idxs.idxs = [1, 2, 3, 4, 5]  # Gap with 5 articles
        mock_gap2 = Mock()
        mock_gap2.dists_idxs.idxs = [6, 7, 8]  # Gap with 3 articles
        
        mock_find_gaps.return_value = iter([mock_gap1, mock_gap2])
        
        # Run function
        result = find_gaps_for_config(mock_config)
        
        # Verify results
        assert len(result) == 2
        assert isinstance(result[0], GapWithContext)
        assert result[0].topic_name == "Test Topic"
        assert result[0].topic_id == "C41008148"
        assert result[0].gap_idx == 0
        assert result[1].gap_idx == 1
        
        # Verify calls
        mock_build_cloud.assert_called_once()
        mock_get_subclouds.assert_called_once_with(mock_build_cloud.return_value, "C41008148")
        mock_find_gaps.assert_called_once()
    
    @patch('bookgap.llm.generators.workflow_functions.build_cloud')
    @patch('bookgap.llm.generators.workflow_functions.pyalex')
    @patch('bookgap.llm.generators.workflow_functions.get_subclouds_for_topic')
    def test_no_articles_found(self, mock_get_subclouds, mock_pyalex, mock_build_cloud):
        """Test handling when no articles are found for a topic."""
        mock_config = Mock()
        mock_config.gap.denoiser.name = "NONE"
        mock_config.data.subfield.value = "1702"
        mock_config.batch.num_gaps = 1
        
        # Mock pyalex with valid topic
        mock_topic = Mock()
        mock_topic.display_name = "Empty Topic"
        mock_topic.id = "https://openalex.org/C41008148"
        
        mock_subfield = Mock()
        mock_subfield.topics = [mock_topic]
        
        mock_subfields = Mock()
        mock_subfields.__getitem__.return_value = mock_subfield
        mock_pyalex.Subfields.return_value = mock_subfields
        
        # Mock empty articles
        mock_books = Mock()
        mock_articles = Mock()
        mock_books.embeddings.return_value = [[1, 2, 3]] * 10
        mock_articles.embeddings.return_value = []  # No articles
        mock_get_subclouds.return_value = (mock_books, mock_articles)
        
        result = find_gaps_for_config(mock_config)
        assert result == []
    
    @patch('bookgap.llm.generators.workflow_functions.build_cloud')
    @patch('bookgap.llm.generators.workflow_functions.pyalex')
    def test_no_topics_found(self, mock_pyalex, mock_build_cloud):
        """Test handling when no topics are found for a subfield."""
        mock_config = Mock()
        mock_config.gap.denoiser.name = "NONE"
        mock_config.data.subfield.value = "9999"  # Invalid subfield
        
        # Mock empty topics
        mock_subfield = Mock()
        mock_subfield.topics = []
        
        mock_subfields = Mock()
        mock_subfields.__getitem__.return_value = mock_subfield
        mock_pyalex.Subfields.return_value = mock_subfields
        
        result = find_gaps_for_config(mock_config)
        assert result == []


class TestBuildContextsForGaps:
    """Test build_contexts_for_gaps function."""
    
    def test_successful_context_building(self):
        """Test successful context building from gaps."""
        # Create mock gaps
        mock_gap = Mock()
        mock_gap.dists_idxs.idxs = [0, 1, 2, 3, 4]  # 5 articles in gap
        
        mock_articles = Mock()
        mock_books = Mock()
        
        # Mock work data
        article_works = [
            {'display_name': f'Article {i}', 'abstract': f'Abstract {i}'}
            for i in range(10)
        ]
        book_works = [
            {'display_name': f'Book {i}', 'abstract': f'Book abstract {i}'}
            for i in range(3)
        ]
        
        mock_articles.works.return_value = article_works
        mock_books.works.return_value = book_works
        
        gap_with_context = GapWithContext(
            gap=mock_gap,
            topic_name="Test Topic",
            topic_id="test_123",
            gap_idx=0,
            books=mock_books,
            articles=mock_articles
        )
        
        mock_config = Mock()
        mock_config.llm.num_articles = 3
        mock_config.llm.num_books = 2
        
        # Run function
        with patch('random.sample') as mock_sample:
            # Control random sampling for predictable results
            mock_sample.side_effect = [
                [0, 2, 4],  # Article indices
                [book_works[0], book_works[1]]  # Book samples
            ]
            
            result = build_contexts_for_gaps([gap_with_context], mock_config)
        
        # Verify results
        assert len(result) == 1
        context = result[0]
        
        assert context.topic_name == "Test Topic"
        assert context.topic_id == "test_123"
        assert context.gap_idx == 0
        assert context.requested_num_prompt_articles == 3
        assert context.requested_num_prompt_books == 2
        
        # Verify sampled articles (should have 3)
        assert len(context.sampled_articles) == 3
        assert all(isinstance(article, Article) for article in context.sampled_articles)
        
        # Verify covering books (should have 2)
        assert len(context.covering_books) == 2
        assert all(isinstance(book, Book) for book in context.covering_books)
        
        # Verify counts
        assert context.counts.num_articles_for_llm == 3
        assert context.counts.num_books_for_llm == 2
    
    def test_limited_by_gap_size(self):
        """Test context building when gap size is smaller than requested articles."""
        mock_gap = Mock()
        mock_gap.dists_idxs.idxs = [0, 1]  # Only 2 articles in gap
        
        mock_articles = Mock()
        mock_books = Mock()
        
        article_works = [
            {'display_name': f'Article {i}', 'abstract': f'Abstract {i}'}
            for i in range(10)
        ]
        mock_articles.works.return_value = article_works
        mock_books.works.return_value = []  # No books
        
        gap_with_context = GapWithContext(
            gap=mock_gap,
            topic_name="Small Gap Topic",
            topic_id="small_123",
            gap_idx=0,
            books=mock_books,
            articles=mock_articles
        )
        
        mock_config = Mock()
        mock_config.llm.num_articles = 5  # Request more than available
        mock_config.llm.num_books = 2
        
        result = build_contexts_for_gaps([gap_with_context], mock_config)
        
        context = result[0]
        # Should only get 2 articles (limited by gap size)
        assert len(context.sampled_articles) == 2
        assert len(context.covering_books) == 0  # No books available
        assert context.counts.num_articles_for_llm == 2
        assert context.counts.num_books_for_llm == 0


class TestSubmitBatchJob:
    """Test submit_batch_job function."""
    
    @patch('tempfile.NamedTemporaryFile')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.unlink')
    def test_successful_batch_submission(self, mock_unlink, mock_file_open, mock_temp_file):
        """Test successful batch job submission."""
        # Setup mocks
        mock_temp_file.return_value.__enter__.return_value.name = '/tmp/test_batch.jsonl'
        
        mock_client = Mock()
        mock_batch_input_file = Mock()
        mock_batch_input_file.id = "file_123"
        mock_client.files.create.return_value = mock_batch_input_file
        
        mock_batch = Mock()
        mock_batch.id = "batch_456"
        mock_client.batches.create.return_value = mock_batch
        
        # Create test requests
        requests = [
            BatchRequestData(
                custom_id="gap_0_topic_test",
                body=ChatCompletionBody(
                    model="gpt-4o-mini",
                    messages=[ChatMessage(role="user", content="Test prompt")],
                    temperature=0.7
                )
            )
        ]
        
        mock_config = Mock()
        mock_config.batch.completion_window = "24h"
        
        with patch('bookgap.llm.generators.workflow_functions.OpenAI', return_value=mock_client):
            result = submit_batch_job(requests, mock_config)
        
        assert result == "batch_456"
        
        # Verify API calls
        mock_client.files.create.assert_called_once()
        mock_client.batches.create.assert_called_once()
        
        # Verify batch creation arguments
        call_args = mock_client.batches.create.call_args
        assert call_args[1]['input_file_id'] == "file_123"
        assert call_args[1]['endpoint'] == "/v1/chat/completions"
        assert call_args[1]['completion_window'] == "24h"
        assert 'metadata' in call_args[1]
    
    def test_empty_requests_error(self):
        """Test error handling for empty requests."""
        mock_config = Mock()
        
        with pytest.raises(ValueError, match="No batch requests to submit"):
            submit_batch_job([], mock_config)
    
    @patch('tempfile.NamedTemporaryFile')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.unlink')
    def test_request_serialization(self, mock_unlink, mock_file_open, mock_temp_file):
        """Test that requests are properly serialized to JSONL."""
        mock_temp_file.return_value.__enter__.return_value.name = '/tmp/test_batch.jsonl'
        
        mock_client = Mock()
        mock_client.files.create.return_value.id = "file_123"
        mock_client.batches.create.return_value.id = "batch_456"
        
        requests = [
            BatchRequestData(
                custom_id="test_id",
                body=ChatCompletionBody(
                    model="gpt-4o-mini",
                    messages=[
                        ChatMessage(role="system", content="System prompt"),
                        ChatMessage(role="user", content="User prompt")
                    ],
                    temperature=0.5,
                    max_tokens=1000
                )
            )
        ]
        
        mock_config = Mock()
        mock_config.batch.completion_window = "24h"
        
        with patch('bookgap.llm.generators.workflow_functions.OpenAI', return_value=mock_client):
            submit_batch_job(requests, mock_config)
        
        # Verify file write calls contained proper JSONL format
        write_calls = mock_temp_file.return_value.__enter__.return_value.write.call_args_list
        assert len(write_calls) > 0
        
        # Parse the written JSONL content
        written_content = ''.join(call[0][0] for call in write_calls)
        lines = written_content.strip().split('\n')
        
        # Should have one line per request
        assert len(lines) == 1
        
        # Parse the JSONL line
        jsonl_data = json.loads(lines[0])
        assert jsonl_data['custom_id'] == "test_id"
        assert jsonl_data['method'] == "POST"
        assert jsonl_data['url'] == "/v1/chat/completions"
        assert jsonl_data['body']['model'] == "gpt-4o-mini"
        assert jsonl_data['body']['temperature'] == 0.5
        assert jsonl_data['body']['max_tokens'] == 1000
        assert len(jsonl_data['body']['messages']) == 2


class TestMonitorAndDownloadResults:
    """Test monitor_and_download_results function."""
    
    def test_successful_monitoring_and_download(self):
        """Test successful batch monitoring and result download."""
        mock_client = Mock()
        
        # Mock batch status progression
        completed_batch = Mock()
        completed_batch.status = "completed"
        completed_batch.output_file_id = "output_file_123"
        completed_batch.metadata = {"0": '{"test": "config"}'}
        completed_batch.request_counts.total = 10
        completed_batch.request_counts.completed = 9
        completed_batch.request_counts.failed = 1
        
        mock_client.batches.retrieve.return_value = completed_batch
        
        # Mock file download
        mock_response = Mock()
        result_content = '{"custom_id": "gap_0", "response": {"id": "resp_123"}}\n'
        mock_response.content = result_content.encode('utf-8')
        mock_client.files.content.return_value = mock_response
        
        mock_config = Mock()
        
        with patch('bookgap.llm.generators.workflow_functions.OpenAI', return_value=mock_client):
            result = monitor_and_download_results("batch_456", mock_config)
        
        # Verify result structure
        assert result['batch_id'] == "batch_456"
        assert result['status'] == "completed"
        assert result['num_results'] == 1
        assert len(result['results']) == 1
        assert result['results'][0]['custom_id'] == "gap_0"
        assert result['original_config'] == '{"test": "config"}'
        assert result['request_counts']['total'] == 10
        assert result['request_counts']['completed'] == 9
        assert result['request_counts']['failed'] == 1
    
    def test_batch_failure_handling(self):
        """Test handling of failed batch jobs."""
        mock_client = Mock()
        
        failed_batch = Mock()
        failed_batch.status = "failed"
        mock_client.batches.retrieve.return_value = failed_batch
        
        mock_config = Mock()
        
        with patch('bookgap.llm.generators.workflow_functions.OpenAI', return_value=mock_client):
            with pytest.raises(Exception, match="Batch job failed with status: failed"):
                monitor_and_download_results("batch_456", mock_config)
    
    @patch('time.sleep')
    def test_monitoring_progression(self, mock_sleep):
        """Test batch status monitoring with progression."""
        mock_client = Mock()
        
        # Simulate status progression: validating -> in_progress -> completed
        status_sequence = ["validating", "in_progress", "completed"]
        batches = []
        
        for status in status_sequence:
            batch = Mock()
            batch.status = status
            if status == "completed":
                batch.output_file_id = "output_file_123"
                batch.metadata = {}
                batch.request_counts.total = 5
                batch.request_counts.completed = 5
                batch.request_counts.failed = 0
            batches.append(batch)
        
        mock_client.batches.retrieve.side_effect = batches
        
        # Mock file download for completed batch
        mock_response = Mock()
        mock_response.content = b'{"custom_id": "test"}\n'
        mock_client.files.content.return_value = mock_response
        
        mock_config = Mock()
        
        with patch('bookgap.llm.generators.workflow_functions.OpenAI', return_value=mock_client):
            result = monitor_and_download_results("batch_456", mock_config)
        
        # Should have called sleep twice (for validating and in_progress)
        assert mock_sleep.call_count == 2
        assert result['status'] == "completed"


class TestRunBatchWorkflow:
    """Test the complete run_batch_workflow function."""
    
    @patch('bookgap.llm.generators.workflow_functions.find_gaps_for_config')
    @patch('bookgap.llm.generators.workflow_functions.build_contexts_for_gaps')
    @patch('bookgap.llm.generators.workflow_functions.generate_batch_requests')
    @patch('bookgap.llm.generators.workflow_functions.submit_batch_job')
    @patch('bookgap.llm.generators.workflow_functions.monitor_and_download_results')
    def test_complete_workflow(self, mock_monitor, mock_submit, mock_generate, 
                              mock_build_contexts, mock_find_gaps):
        """Test the complete batch workflow end-to-end."""
        # Setup mock returns
        mock_gaps = [Mock(), Mock()]  # 2 gaps
        mock_contexts = [Mock(), Mock()]  # 2 contexts
        mock_requests = [Mock(), Mock()]  # 2 requests
        mock_batch_id = "batch_789"
        mock_results = {
            'batch_id': mock_batch_id,
            'status': 'completed',
            'num_results': 2,
            'results': [{'custom_id': 'gap_0'}, {'custom_id': 'gap_1'}]
        }
        
        mock_find_gaps.return_value = mock_gaps
        mock_build_contexts.return_value = mock_contexts
        mock_generate.return_value = mock_requests
        mock_submit.return_value = mock_batch_id
        mock_monitor.return_value = mock_results
        
        mock_config = Mock()
        mock_config.batch.num_gaps = 2
        
        # Run complete workflow
        result = run_batch_workflow(mock_config)
        
        # Verify all steps were called in order
        mock_find_gaps.assert_called_once_with(mock_config)
        mock_build_contexts.assert_called_once_with(mock_gaps, mock_config)
        mock_generate.assert_called_once_with(mock_contexts, mock_config)
        mock_submit.assert_called_once_with(mock_requests, mock_config)
        mock_monitor.assert_called_once_with(mock_batch_id, mock_config)
        
        # Verify final result
        assert result == mock_results
        assert result['num_results'] == 2


# Original tests for existing functions
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
