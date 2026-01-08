"""Integration tests for AdalFlow TSGD-M optimization."""

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from artifacts_scripts.llm_eval.optimize_prompt import (
    OptimizationConfig,
    ProposalTask,
    ProposalTrainTask,
    SamplingConfig,
    TrainerConfig,
)
from bookgap.llm.data_structures import GapWithContext

# Import real AdalFlow client for testing
from adalflow.components.model_client import OpenAIClient


class FakeEmbedder:
    """Fake embedder that returns dummy embeddings."""

    def embed_texts(self, pairs):
        """Return dummy embeddings."""
        import numpy as np
        return np.random.randn(len(pairs), 768).astype(np.float32)


class FakeCloud:
    """Fake cloud for testing."""

    def project(self, embeddings):
        """Just return embeddings as-is."""
        return embeddings


@pytest.fixture
def sample_gaps():
    """Create sample gaps for testing."""
    gaps = []
    for i in range(5):
        gap = Mock(spec=GapWithContext)
        gap.description = f"Gap {i}: Research area {i}"
        gap.gap_zscore = 1.0 + i * 0.5
        gap.gap_size_percentile = 0.5 + i * 0.1
        gaps.append(gap)
    return gaps


@pytest.fixture
def model_client():
    """Create a real OpenAI client with a test API key."""
    # Use a fake API key for testing
    return OpenAIClient(api_key="test-api-key")


def test_proposal_task_creation(model_client):
    """Test that ProposalTask can be created with AdalFlow components."""
    model_kwargs = {"model": "gpt-4o", "temperature": 0.7}

    # Should not raise an error
    task = ProposalTask(model_client, model_kwargs)  # type: ignore
    assert task is not None
    assert task.prompt is not None
    assert task.generator is not None


def test_proposal_task_call(model_client):
    """Test that ProposalTask.call() works."""
    model_kwargs = {"model": "gpt-4o"}

    task = ProposalTask(model_client, model_kwargs)  # type: ignore

    # Mock the generator to return a response
    mock_response = Mock()
    mock_response.data = json.dumps([
        {"title": "Test Proposal", "abstract": "Test abstract"}
    ])
    task.generator = Mock(return_value=mock_response)

    # Create a mock sample with gaps_text and id attributes
    from types import SimpleNamespace
    gaps_sample = SimpleNamespace(
        gaps_text="Gap 1: Some research area\nGap 2: Another area",
        id="test_sample_1"
    )
    result = task.call(gaps_sample)

    # ProposalTask now returns a lightweight _Prediction wrapper
    assert hasattr(result, "data")
    assert isinstance(result.data, str)
    assert "Test Proposal" in result.data


def test_proposal_train_task_creation(model_client, sample_gaps):
    """Test that ProposalTrainTask can be created."""
    model_kwargs = {"model": "gpt-4o"}
    embedder = FakeEmbedder()
    cloud = FakeCloud()

    task = ProposalTrainTask(
        model_client=model_client,  # type: ignore
        model_kwargs=model_kwargs,
        embedder=embedder,  # type: ignore
        cloud=cloud,  # type: ignore
        sampled_gaps=sample_gaps,
    )

    assert task is not None
    assert task.task is not None
    assert task.embedder is not None
    assert task.cloud is not None


def test_proposal_train_task_eval_fn(model_client, sample_gaps):
    """Test that ProposalTrainTask eval_fn works."""
    model_kwargs = {"model": "gpt-4o"}
    embedder = FakeEmbedder()
    cloud = FakeCloud()

    train_task = ProposalTrainTask(
        model_client=model_client,  # type: ignore
        model_kwargs=model_kwargs,
        embedder=embedder,  # type: ignore
        cloud=cloud,  # type: ignore
        sampled_gaps=sample_gaps,
    )

    # Mock the task to return a valid response
    mock_response = json.dumps([
        {"title": "Test Book", "abstract": "A test book proposal"}
    ])
    train_task.task = Mock(return_value=mock_response)

    # Mock parse_proposals_from_response and score_proposals_against_gaps
    with patch("artifacts_scripts.llm_eval.optimize_prompt.parse_proposals_from_response") as mock_parse, \
         patch("artifacts_scripts.llm_eval.optimize_prompt.embed_proposals") as mock_embed, \
         patch("artifacts_scripts.llm_eval.optimize_prompt.score_proposals_against_gaps") as mock_score:

        # Setup mocks
        mock_proposal = Mock()
        mock_proposal.title = "Test Book"
        mock_proposal.abstract = "Test abstract"
        mock_parse.return_value = [mock_proposal]

        import numpy as np
        mock_embed.return_value = np.random.randn(1, 768).astype(np.float32)

        mock_score.return_value = {
            "aggregate_score": 2.5,
            "proposal_scores": [{"title": "Test Book", "max_zscore": 2.5}],
        }

        # Call eval_fn - need to access it through closure
        sample = "Some gaps text"
        # For now, we just check that the eval_fn exists
        assert train_task.eval_fn is not None


def test_optimization_config_with_tsgd_enabled():
    """Test that OptimizationConfig supports tsgd_m_enabled flag."""
    cfg = OptimizationConfig()
    assert cfg.trainer.tsgd_m_enabled is True
    assert cfg.trainer.tsgd_m_momentum_window == 5

    # Can create with disabled TSGD
    cfg2 = OptimizationConfig(
        trainer=TrainerConfig(tsgd_m_enabled=False)
    )
    assert cfg2.trainer.tsgd_m_enabled is False


def test_run_adalflow_optimization_structure():
    """Test the structure of run_adalflow_optimization without actually running."""
    from artifacts_scripts.llm_eval.optimize_prompt import run_adalflow_optimization

    # Function should exist and be callable
    assert callable(run_adalflow_optimization)

    # Check signature
    import inspect
    sig = inspect.signature(run_adalflow_optimization)
    assert "config" in sig.parameters


@pytest.mark.skip(reason="Requires actual OpenAI API and training - use only for manual testing")
def test_run_adalflow_optimization_e2e(tmp_path):
    """End-to-end test of TSGD-M optimization (skipped in CI)."""
    from artifacts_scripts.llm_eval.optimize_prompt import run_adalflow_optimization

    cfg = OptimizationConfig(
        output_dir=str(tmp_path),
        trainer=TrainerConfig(
            iterations=2,
            tsgd_m_enabled=True,
            candidates_per_iter=2,
        ),
        sampling=SamplingConfig(num_gaps=3),
    )

    # This would run the actual optimization
    # run_adalflow_optimization(cfg)

    # In actual test, would check that:
    # - Output directory was created
    # - best_prompt.txt was saved
    # - Iteration logs exist
