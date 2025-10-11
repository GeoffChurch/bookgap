"""Comprehensive tests for workflow functions."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast, Optional

import numpy as np
import pytest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf, DictConfig

from bookgap.configs.config import ExperimentConfig
from bookgap.cloud import Embedder
from bookgap.llm.data_structures import BatchResults, GapWithContext
from bookgap.llm.generators.workflow_functions import (
    PendingProposal,
    _compute_proposal_embeddings,
    MetadataUtils,
    generate_batch_requests,
    validate_batch_requests,
    get_supported_models,
    find_gaps_for_config,
    build_contexts_for_gaps,
    submit_batch_job,
    monitor_and_download_results,
    run_batch_workflow,
)
from bookgap.llm.identifiers import build_gap_proposal_id
from bookgap.llm.protocols import (
    ProposalContext,
    BatchRequestData,
    ChatCompletionBody,
    ChatMessage,
    Proposal,
    # Chapter,
    OpenAIModel,
    ParsedProposal,
    ProposalCounts,
)


class TestFindGapsForConfig:
    """Test find_gaps_for_config function."""
    
    @patch('bookgap.llm.generators.workflow_functions.instantiate')
    @patch('pyalex.Subfields')  # Patch the actual module
    @patch('bookgap.llm.generators.workflow_functions.get_subclouds_for_topic')
    @patch('bookgap.llm.generators.workflow_functions.find_gaps_for_topic')
    def test_successful_gap_finding(self, mock_find_gaps, mock_get_subclouds, mock_subfields, mock_instantiate):
        """Test successful gap finding with proper configuration."""
        # Setup mocks
        mock_config = Mock()
        mock_config.cloud.denoiser = "NONE"
        mock_config.cloud.drop_k = 0
        mock_config.gap.radius_fraction = 0.5
        mock_config.gap.cell_type = Mock()
        mock_cell_type_value = Mock()
        mock_config.gap.cell_type.value = mock_cell_type_value
        mock_config.gap.index_type = Mock()
        mock_config.gap.hnsw_m = 16
        mock_config.gap.hnsw_efc = 200
        mock_config.gap.hnsw_efs = 200
        mock_config.batch.num_gaps = 2
        mock_config.embedder = Mock()
        mock_config.embedder.model_name = Embedder.ModelName.GEMMA_300M.value

        # Setup pyalex mock - hardcoded AI subfield (matches workflow_functions.py)
        mock_topic = {
            "display_name": "Machine Learning",
            "id": "https://openalex.org/T123"
        }
        mock_subfield_instance = {"topics": [mock_topic]}
        mock_subfields.return_value = {"1702": mock_subfield_instance}  # Hardcoded AI subfield

        # Setup cloud mocks
        mock_cloud = Mock()
        mock_instantiate.return_value = mock_cloud
        
        mock_books = Mock()
        mock_articles = Mock()
        mock_books.embeddings.return_value = [1, 2, 3]  # 3 books
        mock_articles.embeddings.return_value = [1, 2, 3, 4, 5]  # 5 articles
        mock_get_subclouds.return_value = (mock_books, mock_articles)

        # Setup gap generator mock
        mock_gap1 = Mock()
        mock_gap1.get_dists_idxs.return_value = Mock(idxs=[0, 1, 2])
        mock_gap2 = Mock()
        mock_gap2.get_dists_idxs.return_value = Mock(idxs=[3, 4])
        mock_find_gaps.return_value = iter([mock_gap1, mock_gap2])

        book_gap_mock1 = Mock()
        book_gap_mock1.get_dists_idxs.return_value = Mock(idxs=[0, 1])
        book_gap_mock2 = Mock()
        book_gap_mock2.get_dists_idxs.return_value = Mock(idxs=[0, 1, 2])
        mock_cell_type_value.find_gaps.return_value = [book_gap_mock1, book_gap_mock2]

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
        mock_instantiate.assert_called_once_with(mock_config.cloud)
        mock_get_subclouds.assert_called_once_with(mock_cloud, "T123")
        mock_find_gaps.assert_called_once()

    @patch('bookgap.llm.generators.workflow_functions.instantiate')
    @patch('pyalex.Subfields')  # Patch the actual module
    @patch('bookgap.llm.generators.workflow_functions.get_subclouds_for_topic')
    def test_no_articles_found(self, mock_get_subclouds, mock_subfields, mock_instantiate):
        """Test handling when no articles are found for a topic."""
        # Setup mocks
        mock_config = Mock()
        mock_config.cloud.denoiser = "NONE"
        mock_config.cloud.drop_k = 0
        mock_config.batch.num_gaps = 2
        mock_config.embedder = Mock()
        mock_config.embedder.model_name = Embedder.ModelName.GEMMA_300M.value

        # Setup pyalex mock - hardcoded AI subfield
        mock_topic = {
            "display_name": "Machine Learning", 
            "id": "https://openalex.org/T123"
        }
        mock_subfield_instance = {"topics": [mock_topic]}
        mock_subfields.return_value = {"1702": mock_subfield_instance}  # Hardcoded AI subfield_open

        mock_cloud = Mock()
        mock_instantiate.return_value = mock_cloud

        mock_books = Mock()
        mock_articles = Mock()
        mock_books.embeddings.return_value = []
        mock_articles.embeddings.return_value = []  # No articles for topic
        mock_get_subclouds.return_value = (mock_books, mock_articles)

        result = find_gaps_for_config(mock_config)

        assert result == []
        mock_instantiate.assert_called_once_with(mock_config.cloud)
        mock_get_subclouds.assert_called_once_with(mock_cloud, "T123")

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
        
        gap_size = len(mock_dists_idxs.idxs)
        gap_with_context = GapWithContext(
            gap=mock_gap,
            topic_name="Test Topic",
            topic_id="123",
            gap_idx=0,
            books=mock_books,
            articles=mock_articles,
            book_gap_sizes=[gap_size, gap_size + 1],
            gap_size=gap_size,
            gap_size_percentile=0.5,
            gap_zscore=0.0,
            shapiro_log10_pval=-1.0,
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
        
        gap_size = len(mock_dists_idxs.idxs)
        gap_with_context = GapWithContext(
            gap=mock_gap,
            topic_name="Small Gap Topic",
            topic_id="456",
            gap_idx=1,
            books=mock_books,
            articles=mock_articles,
            book_gap_sizes=[gap_size, gap_size + 2],
            gap_size=gap_size,
            gap_size_percentile=0.6,
            gap_zscore=0.1,
            shapiro_log10_pval=-0.5,
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
    
    @staticmethod
    def _make_config(max_completion_tokens: bool = False) -> DictConfig:
        return OmegaConf.create(
            {
                "batch": {"completion_window": "24h"},
                "llm": {
                    "model": {"uses_max_completion_tokens": max_completion_tokens}
                },
                "embedder": {"model_name": Embedder.ModelName.GEMMA_300M.value},
            }
        )

    @patch('bookgap.llm.generators.workflow_functions.OpenAI')
    def test_successful_batch_submission(self, mock_openai_class):
        """Test successful batch job submission."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_file = Mock()
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file

        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_client.batches.create.return_value = mock_batch

        batch_requests = [
            BatchRequestData(
                custom_id="test-1",
                body=ChatCompletionBody(
                    model="gpt-4o-mini",
                    messages=[
                        ChatMessage(role="system", content="You are a helpful assistant."),
                        ChatMessage(role="user", content="Test prompt."),
                    ],
                    max_tokens=1000,
                ),
            )
        ]

        config = cast(ExperimentConfig, self._make_config())

        with tempfile.TemporaryDirectory() as tmp_dir:
            hydra_runtime = SimpleNamespace(runtime=SimpleNamespace(output_dir=tmp_dir))
            with patch('bookgap.llm.generators.workflow_functions.HydraConfig.get', return_value=hydra_runtime):
                result = submit_batch_job(batch_requests, config)
                written_file = Path(tmp_dir) / "batch_input.jsonl"
                assert written_file.exists()
                assert written_file.read_text().strip() != ""

        assert result == "batch-456"
        mock_client.files.create.assert_called_once()
        mock_client.batches.create.assert_called_once()

    def test_empty_requests_error(self):
        """Test error handling for empty batch requests."""
        mock_config = cast(ExperimentConfig, self._make_config())

        with pytest.raises(ValueError, match="No batch requests to submit"):
            submit_batch_job([], mock_config)

    @patch('bookgap.llm.generators.workflow_functions.OpenAI')
    def test_request_serialization(self, mock_openai_class):
        """Test that requests are properly serialized to JSONL format."""
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

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
        
        config = cast(ExperimentConfig, self._make_config())

        with tempfile.TemporaryDirectory() as tmp_dir:
            hydra_runtime = SimpleNamespace(runtime=SimpleNamespace(output_dir=tmp_dir))
            with patch('bookgap.llm.generators.workflow_functions.HydraConfig.get', return_value=hydra_runtime):
                submit_batch_job(batch_requests, config)
                batch_input = Path(tmp_dir) / "batch_input.jsonl"
                assert batch_input.exists()
                lines = batch_input.read_text().strip().splitlines()
                assert len(lines) == 1
                payload = json.loads(lines[0])
                assert payload["body"].get("response_format") == {"type": "json_object"}
                assert payload["body"].get("temperature") == 0.7
 
class TestComputeProposalEmbeddings:
    """Tests for proposal embedding projection reuse."""

    @staticmethod
    def _make_pending(title_suffix: str) -> PendingProposal:
        # chapters = [Chapter(chapter_title=f"Chapter {i}") for i in range(10)]
        proposal = Proposal(title=f"Title {title_suffix}", abstract="Abstract")#, chapters=chapters)
        return PendingProposal(proposal=proposal, custom_id=f"id-{title_suffix}")

    @patch('bookgap.llm.generators.workflow_functions.instantiate')
    @patch('bookgap.llm.generators.workflow_functions.Embedder')
    def test_projection_and_normalization_applied(self, mock_embedder_cls, mock_instantiate):
        pending = [self._make_pending("A")]
        config = cast(ExperimentConfig, OmegaConf.create({
            "cloud": {"_target_": "mock.Cloud", "embedder_model": Embedder.ModelName.GEMMA_300M.value},
            "embedder": {"model_name": "mock-model"},
        }))

        raw_embeddings = np.array([[3.0, 4.0]], dtype=np.float32)

        class DummyEmbedder:
            def __init__(self, embeddings):
                self.embeddings = embeddings
                self.calls = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def embed_texts(self, _pairs):
                self.calls += 1
                return self.embeddings

        dummy_embedder = DummyEmbedder(raw_embeddings)
        mock_embedder_cls.return_value = dummy_embedder

        class FakeProjection:
            def __init__(self):
                self.seen = None

            def apply(self, embeddings):
                self.seen = embeddings
                return embeddings * 2

        projection = FakeProjection()
        mock_cloud = Mock()
        mock_cloud.denoiser_projection.return_value = projection
        mock_cloud.normalize_embeddings = True
        mock_instantiate.return_value = mock_cloud

        result = _compute_proposal_embeddings(pending, config)

        expected = raw_embeddings * 2
        expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)

        assert projection.seen is not None
        assert np.allclose(projection.seen, raw_embeddings)
        assert np.allclose(result, expected.astype(np.float32))
        mock_cloud.denoiser_projection.assert_called_once()
        assert dummy_embedder.calls == 1
        mock_instantiate.assert_called_once()


class TestMonitorAndDownloadResults:
    """Test monitor_and_download_results function."""

    @staticmethod
    def _make_proposal(title_suffix: str) -> Proposal:
        # chapters = [Chapter(chapter_title=f"Chapter {i}") for i in range(10)]
        return Proposal(title=f"Title {title_suffix}", abstract="Abstract")#, chapters=chapters)

    @staticmethod
    def _make_result_line(custom_id: str, finish_reason: str = "stop") -> str:
        return json.dumps({
            "custom_id": custom_id,
            "response": {
                "body": {
                    "choices": [
                        {
                            "finish_reason": finish_reason,
                            "message": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "dummy"
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        })

    @patch('bookgap.llm.generators.workflow_functions.OpenAI')
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_successful_monitoring_and_download(self, mock_sleep, mock_openai_class):
        """Test successful monitoring and result download."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_batch_in_progress = Mock()
        mock_batch_in_progress.status = "in_progress"

        mock_batch_completed = Mock()
        mock_batch_completed.status = "completed"
        mock_batch_completed.output_file_id = "file-output-123"
        mock_batch_completed.request_counts.total = 10
        mock_batch_completed.request_counts.completed = 10
        mock_batch_completed.request_counts.failed = 0

        mock_client.batches.retrieve.side_effect = [mock_batch_in_progress, mock_batch_completed]

        mock_file_response = Mock()
        mock_file_response.content = "\n".join([
            self._make_result_line("id-1"),
            self._make_result_line("id-2"),
        ]).encode('utf-8')
        mock_client.files.content.return_value = mock_file_response

        config_data = {
            "embedder": {"model_name": Embedder.ModelName.GEMMA_300M.value},
            "batch": {"completion_window": "24h"},
            "cloud": {"_target_": "mock.Cloud", "embedder_model": Embedder.ModelName.GEMMA_300M.value},
        }
        config = cast(ExperimentConfig, OmegaConf.create(config_data))
        expected_metadata = MetadataUtils.serialize_config(config)
        mock_batch_completed.metadata = expected_metadata

        with tempfile.TemporaryDirectory() as tmp_dir:
            hydra_runtime = SimpleNamespace(runtime=SimpleNamespace(output_dir=tmp_dir))
            with patch('bookgap.llm.generators.workflow_functions.HydraConfig.get', return_value=hydra_runtime), \
                 patch('bookgap.llm.generators.workflow_functions.parse_batch_result_to_proposal') as mock_parse, \
                 patch('bookgap.llm.generators.workflow_functions._compute_proposal_embeddings') as mock_embed:

                mock_parse.side_effect = [
                    PendingProposal(proposal=self._make_proposal("A"), custom_id="id-1"),
                    PendingProposal(proposal=self._make_proposal("B"), custom_id="id-2"),
                ]
                mock_embed.return_value = np.zeros((2, 3), dtype=np.float32)

                result = monitor_and_download_results("batch-123", config)

                output_path = Path(tmp_dir) / "batch_results.jsonl"
                assert output_path.exists()
                assert output_path.read_text().strip() != ""

        assert isinstance(result, BatchResults)
        assert len(result.proposals) == 2
        assert result.metadata is config
        mock_sleep.assert_called()

    @patch('bookgap.llm.generators.workflow_functions.instantiate')
    @patch('bookgap.llm.generators.workflow_functions.Embedder')
    @patch('bookgap.llm.generators.workflow_functions.OpenAI')
    @patch('time.sleep')
    def test_projection_applied_via_monitoring(self, mock_sleep, mock_openai_class, mock_embedder_cls, mock_instantiate):
        raw_embeddings = np.array([
            [3.0, 4.0],
            [0.0, 5.0],
        ], dtype=np.float32)

        class DummyEmbedder:
            def __init__(self, embeddings: np.ndarray):
                self.embeddings = embeddings
                self.calls = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def embed_texts(self, _pairs: list[tuple[str, str]]):
                self.calls += 1
                return self.embeddings

        dummy_embedder = DummyEmbedder(raw_embeddings)
        mock_embedder_cls.return_value = dummy_embedder

        class FakeProjection:
            def __init__(self):
                self.seen: Optional[np.ndarray] = None

            def apply(self, embeddings: np.ndarray) -> np.ndarray:
                self.seen = embeddings
                return embeddings * 2

        projection = FakeProjection()
        mock_cloud = Mock()
        mock_cloud.denoiser_projection.return_value = projection
        mock_cloud.normalize_embeddings = True
        mock_instantiate.return_value = mock_cloud

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_batch_in_progress = Mock()
        mock_batch_in_progress.status = "in_progress"

        mock_batch_completed = Mock()
        mock_batch_completed.status = "completed"
        mock_batch_completed.output_file_id = "file-output-123"
        mock_batch_completed.request_counts.total = 2
        mock_batch_completed.request_counts.completed = 2
        mock_batch_completed.request_counts.failed = 0

        config_data = {
            "embedder": {"model_name": Embedder.ModelName.GEMMA_300M.value},
            "batch": {"completion_window": "24h"},
            "cloud": {"_target_": "mock.Cloud", "embedder_model": Embedder.ModelName.GEMMA_300M.value},
        }
        config = cast(ExperimentConfig, OmegaConf.create(config_data))
        expected_metadata = MetadataUtils.serialize_config(config)
        mock_batch_completed.metadata = expected_metadata

        mock_client.batches.retrieve.side_effect = [mock_batch_in_progress, mock_batch_completed]

        proposal_lines = [
            self._make_result_line("id-1"),
            self._make_result_line("id-2"),
        ]
        mock_file_response = Mock()
        mock_file_response.content = "\n".join(proposal_lines).encode('utf-8')
        mock_client.files.content.return_value = mock_file_response

        with tempfile.TemporaryDirectory() as tmp_dir:
            hydra_runtime = SimpleNamespace(runtime=SimpleNamespace(output_dir=tmp_dir))
            with patch('bookgap.llm.generators.workflow_functions.HydraConfig.get', return_value=hydra_runtime), \
                 patch('bookgap.llm.generators.workflow_functions.parse_batch_result_to_proposal') as mock_parse:

                mock_parse.side_effect = [
                    PendingProposal(proposal=self._make_proposal("A"), custom_id="id-1"),
                    PendingProposal(proposal=self._make_proposal("B"), custom_id="id-2"),
                ]

                result = monitor_and_download_results("batch-xyz", config)

        mock_parse.assert_called()

        assert len(result.proposals) == 2
        assert dummy_embedder.calls == 1
        assert projection.seen is not None
        assert np.allclose(projection.seen, raw_embeddings)

        embeddings = np.stack([proposal.embedding for proposal in result.proposals])
        expected_projected = raw_embeddings * 2
        expected_normalized = expected_projected / np.linalg.norm(expected_projected, axis=1, keepdims=True)
        assert np.allclose(embeddings, expected_normalized.astype(np.float32))

        mock_client.batches.retrieve.assert_called()
        mock_cloud.denoiser_projection.assert_called()
        mock_sleep.assert_called()
    @patch('bookgap.llm.generators.workflow_functions.OpenAI')
    def test_batch_failure_handling(self, mock_openai_class):
        """Test handling of failed batch jobs."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_batch_failed = Mock()
        mock_batch_failed.status = "failed"
        mock_client.batches.retrieve.return_value = mock_batch_failed

        config_data = {
            "embedder": {"model_name": Embedder.ModelName.GEMMA_300M.value},
            "batch": {"completion_window": "24h"},
        }
        config = cast(ExperimentConfig, OmegaConf.create(config_data))

        with pytest.raises(Exception, match="Batch job failed with status: failed"):
            monitor_and_download_results("batch-failed", config)

    @patch('bookgap.llm.generators.workflow_functions.OpenAI')
    @patch('time.sleep')
    def test_monitoring_progression(self, mock_sleep, mock_openai_class):
        """Test that monitoring progresses through different states."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        statuses = ["validating", "in_progress", "finalizing", "completed"]
        mock_batches = [Mock(status=status) for status in statuses[:-1]]

        mock_batch_completed = Mock()
        mock_batch_completed.status = "completed"
        mock_batch_completed.output_file_id = "file-123"
        mock_batch_completed.request_counts.total = 5
        mock_batch_completed.request_counts.completed = 5
        mock_batch_completed.request_counts.failed = 0
        mock_batches.append(mock_batch_completed)

        mock_client.batches.retrieve.side_effect = mock_batches

        mock_file_response = Mock()
        mock_file_response.content = self._make_result_line("id-1").encode('utf-8')
        mock_client.files.content.return_value = mock_file_response

        config_data = {
            "embedder": {"model_name": Embedder.ModelName.GEMMA_300M.value},
            "batch": {"completion_window": "24h"},
        }
        config = cast(ExperimentConfig, OmegaConf.create(config_data))
        mock_batch_completed.metadata = MetadataUtils.serialize_config(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            hydra_runtime = SimpleNamespace(runtime=SimpleNamespace(output_dir=tmp_dir))
            with patch('bookgap.llm.generators.workflow_functions.HydraConfig.get', return_value=hydra_runtime), \
                 patch('bookgap.llm.generators.workflow_functions.parse_batch_result_to_proposal') as mock_parse, \
                 patch('bookgap.llm.generators.workflow_functions._compute_proposal_embeddings') as mock_embed:

                mock_parse.return_value = PendingProposal(proposal=self._make_proposal("Single"), custom_id="id-1")
                mock_embed.return_value = np.zeros((1, 3), dtype=np.float32)

                result = monitor_and_download_results("batch-progression", config)

        assert mock_client.batches.retrieve.call_count == len(statuses)
        assert isinstance(result, BatchResults)
        assert len(result.proposals) == 1
        assert result.metadata is config


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
        
        proposal = ParsedProposal(
            proposal=Proposal(
                title="Title",
                abstract="Abstract",
                # chapters=[Chapter(chapter_title=f"Chapter {i}") for i in range(10)]
            ),
            custom_id=build_gap_proposal_id(gap_idx=0, topic_id="topic-1"),
            embedding=np.zeros(3, dtype=np.float32),
        )
        metadata = cast(ExperimentConfig, OmegaConf.create({"batch": {"num_gaps": 2}}))
        mock_monitor.return_value = BatchResults(proposals=[proposal], metadata=metadata)
        
        mock_config = Mock()
        mock_config.batch.num_gaps = 2
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            hydra_runtime = SimpleNamespace(runtime=SimpleNamespace(output_dir=tmp_dir))
            with patch('bookgap.llm.generators.workflow_functions.HydraConfig.get', return_value=hydra_runtime), \
                 patch('bookgap.llm.generators.workflow_functions.generate_all_contexts') as mock_generate_contexts, \
                 patch('bookgap.llm.generators.workflow_functions.generate_gap_plots_for_results') as mock_generate_plots, \
                 patch('bookgap.llm.generators.workflow_functions.generate_gaps_analysis_csv') as mock_generate_csv:

                context_paths = [Path(tmp_dir) / "context.json"]
                plot_paths = [Path(tmp_dir) / "plot.png"]
                results_csv = Path(tmp_dir) / "results.csv"

                mock_generate_contexts.return_value = context_paths
                mock_generate_plots.return_value = plot_paths
                mock_generate_csv.return_value = results_csv

                # Call function
                result = run_batch_workflow(mock_config)
        
        # Verify all steps were called in order
        mock_find_gaps.assert_called_once_with(mock_config)
        mock_build_contexts.assert_called_once_with(mock_gaps, mock_config)
        mock_generate.assert_called_once_with(mock_contexts, mock_config)
        mock_submit.assert_called_once_with(mock_requests, mock_config)
        mock_monitor.assert_called_once_with("batch-123", mock_config)
        mock_generate_contexts.assert_called_once_with(mock_gaps, Path(tmp_dir))
        mock_generate_plots.assert_called_once()
        plot_args, plot_kwargs = mock_generate_plots.call_args
        assert plot_args == (mock_gaps, result.proposals, Path(tmp_dir))
        assert plot_kwargs.get("proposals_by_gap") == {0: [proposal]}

        mock_generate_csv.assert_called_once()
        csv_args, csv_kwargs = mock_generate_csv.call_args
        assert csv_args == (mock_gaps, result.proposals, Path(tmp_dir), mock_config, mock_contexts)
        assert list(csv_kwargs.get("proposals_grouped", {}).keys()) == [("topic-1", 0)]
        assert csv_kwargs.get("ordered_proposals")
        
        # Verify result
        assert isinstance(result, BatchResults)
        assert result.proposals == [proposal]
        assert result.metadata == metadata
        assert result.artifact_paths is not None
        assert result.artifact_paths.context_files == context_paths
        assert result.artifact_paths.plots == plot_paths
        assert result.artifact_paths.results_csv == results_csv


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
        
        counts = ProposalCounts(num_articles_for_llm=0, num_books_for_llm=0)
        context = ProposalContext(
            topic_name="Topic",
            topic_id="id",
            gap_idx=0,
            requested_num_prompt_articles=2,
            requested_num_prompt_books=1,
            sampled_articles=[],
            covering_books=[],
            counts=counts,
        )
        mock_contexts = [context, context]
        mock_config = cast(
            ExperimentConfig,
            SimpleNamespace(llm=SimpleNamespace(model="gpt-4o-mini", _target_="dummy.generator")),
        )
        
        # Call function
        result = generate_batch_requests(mock_contexts, mock_config)
        
        # Assertions
        assert result == mock_requests
        mock_instantiate.assert_called_once_with(mock_config.llm)
        mock_generator.generate_batch_requests.assert_called_once_with(mock_contexts)

    @patch('bookgap.llm.generators.workflow_functions.instantiate')
    def test_generation_error(self, mock_instantiate):
        """Test error handling during generation."""
        # Setup mocks to raise exception
        mock_instantiate.side_effect = Exception("Test error")
        
        counts = ProposalCounts(num_articles_for_llm=0, num_books_for_llm=0)
        context = ProposalContext(
            topic_name="Topic",
            topic_id="id",
            gap_idx=0,
            requested_num_prompt_articles=2,
            requested_num_prompt_books=1,
            sampled_articles=[],
            covering_books=[],
            counts=counts,
        )
        mock_contexts = [context]
        mock_config = cast(
            ExperimentConfig,
            SimpleNamespace(llm=SimpleNamespace(model="gpt-4o", _target_="test.target")),
        )
        
        with pytest.raises(Exception, match=r"^Test error$"):
            generate_batch_requests(mock_contexts, mock_config)

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
        with pytest.raises(ValueError, match=r"^No batch requests to validate$"):
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
        
        with pytest.raises(ValueError, match=r"^Batch request validation failed:\nDuplicate custom_id found: duplicate$"):
            validate_batch_requests(requests)


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
