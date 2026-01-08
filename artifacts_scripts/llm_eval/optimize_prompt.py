#!/usr/bin/env python3
"""Prompt optimization script for maximizing proposal gap quality.

Uses AdalFlow's TSGD-M optimizer with Gumbel exploration to optimize an LLM prompt
that generates high-quality textbook proposals. The objective is to maximize the
gap z-score (with percentile as fallback) by iteratively improving the prompt based
on computed gap metrics.

Usage:
    python optimize_prompt.py [--config-name=NAME] [hydra overrides...]
    
Example:
    python optimize_prompt.py \
        --config-name=prompt_opt \
        trainer.iterations=50 \
        trainer.candidates_per_iter=5 \
        sampling.num_gaps=16
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any



import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
from hydra.utils import instantiate
from numpy.typing import NDArray
from openai import OpenAI

from bookgap.cloud import Cloud, Embedder
from bookgap.configs.config import ExperimentConfig
from bookgap.configs.config_store import register_configs
from bookgap.configs.gap.gap_config import GapConfig
from bookgap.configs.llm.llm_config import LLMConfig
from bookgap.configs.batch.batch_config import BatchConfig
from bookgap.configs.output.output_config import OutputConfig
from bookgap.configs.cloud.cloud_config import CloudConfig
from bookgap.gaps import compute_gaps_for_config
from bookgap.gap_metrics import compute_gap_metrics_for_embeddings
from bookgap.llm.data_structures import GapWithContext
from bookgap.llm.protocols import Proposal
from bookgap.util import LOGGER, setup_logging

from adalflow import Component
from adalflow.eval.base import EvaluationResult
from adalflow.core import Generator
from adalflow.core.model_client import ModelClient
from adalflow.optim import Trainer, AdalComponent
from adalflow.optim.parameter import ComponentTrace, Parameter
from adalflow.optim.loss_component import LossComponent
from adalflow.optim.types import ParameterType


@dataclass
class ProposalGeneratorConfig:
    """Configuration for the proposal generator component."""
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2048
    initial_system_prompt: str = field(default_factory=lambda: """You are an expert commissioning editor for a leading academic publisher.
Your task is to generate high-quality textbook proposals for underserved research areas.

**ABSOLUTE REQUIREMENT: You MUST ONLY output valid JSON.** No markdown, no explanations, no headings, nothing else whatsoever.

Generate exactly ONE proposal as a JSON array with this exact structure:
[
  {
    "title": "Textbook Title Here",
    "abstract": "Comprehensive abstract explaining the book's scope, target audience, and specific gap it addresses. 2-3 sentences minimum."
  }
]

**CRITICAL CONSTRAINT:** Output ONLY the JSON array above. Not a single character before or after it. Your entire response must be parseable as valid JSON.""")


@dataclass
class SamplingConfig:
    """Configuration for gap sampling during optimization."""
    num_gaps: int = 16  # Number of gaps to sample for scoring
    strategy: str = "top_zscore"  # "top_zscore", "top_percentile", or "random"
    random_seed: int = 42
    top_k_books: int = 3  # How many top gaps' books (by book_gap_sizes descending) to surface per gap


@dataclass
class TrainerConfig:
    """Configuration for the TSGD-M trainer."""
    iterations: int = 30
    candidates_per_iter: int = 5
    history_update_strategy: str = "always"  # "always" or "improvement_only"
    exploration_epsilon: float = 0.2
    confidence_threshold: float = 0.1
    tsgd_m_enabled: bool = True
    tsgd_m_momentum_window: int = 5
    random_seed: int = 42


@dataclass
class OptimizationConfig:
    """Top-level configuration for prompt optimization."""
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(
            gap=GapConfig(),
            llm=LLMConfig(),
            batch=BatchConfig(),
            output=OutputConfig(),
            cloud=CloudConfig(),
        )
    )
    output_dir: str = ""  # Where to load/save gaps and logs (empty = auto-derive)
    generator: ProposalGeneratorConfig = field(default_factory=ProposalGeneratorConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


@dataclass
class GapSample:
    """Lightweight sample wrapper with id and gaps text."""
    id: str
    gaps_text: str


class TaskParameter(Parameter):
    """Parameter subclass that allows generator outputs and loss values."""

    allowed_types = {
        ParameterType.NONE,
        ParameterType.PROMPT,
        ParameterType.DEMOS,
        ParameterType.HYPERPARAM,
        ParameterType.INPUT,
        ParameterType.OUTPUT,
        ParameterType.GENERATOR_OUTPUT,
        ParameterType.LOSS_OUTPUT,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure debug graph utilities can access a component_trace
        if not hasattr(self, "component_trace") or self.component_trace is None:
            self.component_trace = ComponentTrace()


def sample_gaps(
    gaps: list[GapWithContext],
    config: SamplingConfig,
) -> list[GapWithContext]:
    """Sample a subset of gaps for optimization scoring.
    
    Args:
        gaps: All available gaps
        config: Sampling configuration
        
    Returns:
        Sampled subset of gaps
    """
    rng = random.Random(config.random_seed)
    
    if config.strategy == "random":
        k = min(config.num_gaps, len(gaps))
        return rng.sample(gaps, k)
    elif config.strategy == "top_zscore":
        # Sort by z-score descending, take top-k
        sorted_gaps = sorted(gaps, key=lambda g: g.gap_zscore, reverse=True)
        return sorted_gaps[:config.num_gaps]
    elif config.strategy == "top_percentile":
        # Sort by percentile descending, take top-k
        sorted_gaps = sorted(gaps, key=lambda g: g.gap_size_percentile, reverse=True)
        return sorted_gaps[:config.num_gaps]
    else:
        raise ValueError(f"Unknown sampling strategy: {config.strategy}")


def parse_proposals_from_response(response_text: str) -> tuple[list[Proposal], str | None]:
    """Parse JSON proposals from LLM response.

    Attempts to handle malformed JSON gracefully with multiple parsing strategies,
    including raw markdown format where proposals are marked with ### Proposal headings.

    Args:
        response_text: Raw text response from LLM
        
    Returns:
        (proposals, error_reason) where proposals may be empty and error_reason is a brief
        string when parsing fails.
    """
    import re
    
    if not response_text or not response_text.strip():
        return [], "empty response"
    
    cleaned_text = response_text.strip()
    
    # Strategy 1: Extract JSON from markdown code blocks
    if "```" in cleaned_text:
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', cleaned_text)
        if json_match:
            cleaned_text = json_match.group(1).strip()
    
    # Strategy 2: Try to parse as JSON directly
    if cleaned_text.lstrip().startswith('[') or cleaned_text.lstrip().startswith('{'):
        try:
            data = json.loads(cleaned_text)
            if isinstance(data, dict):
                proposals = [Proposal.model_validate(data)]
            elif isinstance(data, list):
                proposals = [Proposal.model_validate(item) for item in data]
            else:
                LOGGER.warning("JSON is valid but not dict/list, skipping direct parse")
                proposals = []
            
            if proposals:
                return proposals, None
        except json.JSONDecodeError:
            pass  # Fall through to other strategies
    
    # Strategy 3: Extract JSON from text if present
    if '[' in cleaned_text or '{' in cleaned_text:
        json_start = cleaned_text.find('[')
        if json_start == -1:
            json_start = cleaned_text.find('{')
        if json_start != -1:
            json_text = cleaned_text[json_start:]
            
            # Fix unbalanced brackets
            if json_text.count('[') > json_text.count(']'):
                last_bracket = json_text.rfind(']')
                if last_bracket != -1:
                    json_text = json_text[:last_bracket + 1]
            if json_text.count('{') > json_text.count('}'):
                last_brace = json_text.rfind('}')
                if last_brace != -1:
                    json_text = json_text[:last_brace + 1]
            
            try:
                data = json.loads(json_text)
                if isinstance(data, dict):
                    proposals = [Proposal.model_validate(data)]
                elif isinstance(data, list):
                    proposals = [Proposal.model_validate(item) for item in data]
                else:
                    proposals = []
                
                if proposals:
                    return proposals, None
            except json.JSONDecodeError:
                pass  # Fall through to markdown parsing
    
    # Strategy 4: Parse markdown format with ### Proposal headings
    # Format: ### Proposal: "title"\n\n#### Abstract:\nabstract text
    LOGGER.debug(f"Checking for markdown format in response (len={len(cleaned_text)})")
    if "### Proposal:" in cleaned_text:
        LOGGER.debug("Found '### Proposal:' marker, attempting markdown extraction")
        proposals = []
        
        # Split by proposal markers
        proposal_blocks = re.split(r'###\s+Proposal:\s*', cleaned_text)
        LOGGER.debug(f"Split into {len(proposal_blocks)} blocks")
        
        for block_idx, block in enumerate(proposal_blocks[1:]):  # Skip first empty split
            LOGGER.debug(f"Processing block {block_idx}, first 100 chars: {block[:100]}")
            
            # Extract title (quoted string after ###)
            title_match = re.match(r'"([^"]+)"\s*\n', block)
            if not title_match:
                # Try alternative: title without quotes
                title_match = re.match(r'([^\n]+)\s*\n', block)
            
            if title_match:
                title = title_match.group(1).strip()
                LOGGER.debug(f"Extracted title: {title[:60]}")
                
                # Extract abstract after #### Abstract:
                abstract_match = re.search(r'####\s+Abstract:\s*([\s\S]+?)(?=###|$)', block)
                if abstract_match:
                    abstract = abstract_match.group(1).strip()
                    LOGGER.debug(f"Extracted abstract (len={len(abstract)})")
                    
                    try:
                        proposal = Proposal.model_validate({
                            "title": title,
                            "abstract": abstract
                        })
                        proposals.append(proposal)
                        LOGGER.info(f"✓ Parsed markdown proposal: {title[:50]}...")
                    except Exception as e:
                        LOGGER.warning(f"✗ Failed to validate markdown-parsed proposal: {e}")
                else:
                    LOGGER.warning(f"No abstract match for block {block_idx}")
            else:
                LOGGER.warning(f"No title match for block {block_idx}")
        
        if proposals:
            LOGGER.info(f"✓ Successfully parsed {len(proposals)} proposal(s) from markdown")
            return proposals, None
        else:
            LOGGER.warning(f"✗ Markdown format detected but no proposals extracted")
            return [], "markdown format detected but extraction failed"
    
    # All strategies failed
    LOGGER.error(f"✗ Failed to parse response with any strategy (len={len(response_text)})")
    LOGGER.error(f"First 200 chars: {response_text[:200]}")
    return [], f"parse error: no valid format detected"


def embed_proposals(
    proposals: list[Proposal],
    embedder: Embedder,
    cloud: Cloud,
) -> NDArray[np.float32]:
    """Embed proposals and project into the cloud space.
    
    Args:
        proposals: Proposals to embed
        embedder: Initialized embedder (reused across calls)
        cloud: Cloud instance for projection
        
    Returns:
        Projected and normalized embeddings (N, D)
    """
    if not proposals:
        return np.empty((0, 0), dtype=np.float32)
    
    # Create title+abstract pairs
    title_abstract_pairs = [(p.title, p.abstract) for p in proposals]
    
    # Embed
    raw_embeddings = embedder.embed_texts(title_abstract_pairs)
    
    # Project via denoiser
    projection = cloud.denoiser_projection()
    projected = projection.apply(raw_embeddings)
    
    # Normalize if configured
    if cloud.normalize_embeddings and projected.shape[0] > 0:
        projected = projected / (np.linalg.norm(projected, axis=1, keepdims=True) + 1e-12)
    
    return projected.astype(np.float32)


def score_proposals_against_gaps(
    proposals: list[Proposal],
    embeddings: NDArray[np.float32],
    gaps: list[GapWithContext],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Score each proposal against all sampled gaps.
    
    Args:
        proposals: Proposals to score
        embeddings: Corresponding embeddings
        gaps: Gaps to evaluate against
        config: Experiment configuration
        
    Returns:
        Dict with scoring results including per-proposal and aggregate metrics
    """
    if len(proposals) == 0:
        return {
            "num_proposals": 0,
            "aggregate_score": float("-inf"),
            "proposal_scores": [],
            "per_gap_metrics": [],
        }
    
    # Score each proposal against each gap
    proposal_scores = []
    per_gap_metrics = []
    
    for i, (proposal, embedding) in enumerate(zip(proposals, embeddings)):
        # Collect z-scores across all gaps for this proposal
        gap_zscores = []
        gap_percentiles = []
        
        for gap in gaps:
            # Compute gap metrics for this single embedding
            stats_list = compute_gap_metrics_for_embeddings(
                gap=gap,
                embeddings=embedding.reshape(1, -1),
                config=config,
            )
            
            if stats_list:
                stats = stats_list[0]
                gap_zscores.append(stats.gap_zscore)
                gap_percentiles.append(stats.gap_size_percentile)
        
        # Aggregate: use max z-score as primary metric
        # TODO: Consider other aggregation strategies (mean, top-k mean, etc.)
        max_zscore = max(gap_zscores) if gap_zscores else float("-inf")
        max_percentile = max(gap_percentiles) if gap_percentiles else 0.0
        
        proposal_scores.append({
            "proposal_idx": i,
            "title": proposal.title,
            "max_zscore": max_zscore,
            "max_percentile": max_percentile,
            "mean_zscore": float(np.mean(gap_zscores)) if gap_zscores else float("-inf"),
            "gap_zscores": gap_zscores,
        })
        
        per_gap_metrics.append({
            "proposal_idx": i,
            "zscores": gap_zscores,
            "percentiles": gap_percentiles,
        })
    
    # Aggregate across proposals: use best max_zscore
    best_max_zscore = max(s["max_zscore"] for s in proposal_scores)

    # Optional verbose logging when the gap set is small
    if len(gaps) <= 20:
        for prop_idx, score in enumerate(proposal_scores):
            LOGGER.info(
                "Per-gap scores for proposal %d (%s)",
                prop_idx,
                score["title"][:80],
            )
            gap_metrics = per_gap_metrics[prop_idx]
            for gap, z, pct in zip(
                gaps, gap_metrics["zscores"], gap_metrics["percentiles"]
            ):
                LOGGER.info(
                    "  Gap %s (%s): z=%.4f pct=%.2f",
                    gap.gap_idx,
                    gap.topic_name[:60],
                    z,
                    pct,
                )

    return {
        "num_proposals": len(proposals),
        "aggregate_score": best_max_zscore,
        "proposal_scores": proposal_scores,
        "per_gap_metrics": per_gap_metrics,
    }


def format_gaps_for_prompt(
    gaps: list[GapWithContext], top_k_books: int
) -> str:
    """Format gaps into text for LLM prompt without exposing metrics.
    
    Args:
        gaps: List of gaps with context
        top_k_books: Number of highest-gap books (by book_gap_sizes) to surface
        
    Returns:
        Formatted string describing all gaps
    """
    gaps_texts = []
    for gap in gaps:
        assert top_k_books >= 0
        if top_k_books == 0:
            gaps_texts.append(f"Topic: {gap.topic_name}")
            continue

        # Avoid materializing full works (expensive); pull titles/abstracts separately
        titles = gap.books.work_titles()
        abstracts = gap.books.work_abstracts()
        gap_sizes = gap.book_gap_sizes
        if len(titles) != len(gap_sizes) or len(abstracts) != len(gap_sizes):
            LOGGER.error(
                "book titles/abstracts and book_gap_sizes length mismatch for topic %s: %d titles, %d abstracts vs %d gap_sizes."
                " First titles: %s; first gap sizes: %s",
                gap.topic_name,
                len(titles),
                len(abstracts),
                len(gap_sizes),
                titles[:3],
                gap_sizes[:3],
            )
            raise ValueError(
                f"book metadata and book_gap_sizes length mismatch for topic {gap.topic_name}: "
                f"{len(titles)} titles, {len(abstracts)} abstracts vs {len(gap_sizes)} gap_sizes"
            )

        books_with_scores = list(zip(titles, abstracts, gap_sizes))
        top_books = sorted(
            books_with_scores,
            key=lambda pair: pair[2],
            reverse=True,
        )[:top_k_books]

        book_lines = []
        for title, abstract, _size in top_books:
            book_lines.append(f"- Title: {title}\n  Abstract: {abstract}")

        actual_count = len(top_books)
        if not book_lines:
            book_lines.append("- No related books available")

        gap_text = (
            f"Topic: {gap.topic_name}\n"
            f"Top {actual_count} related books:\n" + "\n".join(book_lines)
        )
        gaps_texts.append(gap_text)
    return "\n\n".join(gaps_texts)


class ProposalTask(Component):
    """Task component that generates proposals given gaps."""
    
    def __init__(self, model_client: ModelClient, model_kwargs: dict):
        super().__init__()
        # Create trainable prompt parameter
        self.prompt = Parameter(
            data="Generate compelling textbook proposals that would fill gaps in the provided research areas. "
                 "Each proposal should include a clear title and detailed abstract explaining the book's scope, "
                 "target audience, and how it addresses the research gap.",
            param_type=ParameterType.PROMPT,
            role_desc="System prompt for proposal generation",
            requires_opt=True,
        )
        # Don't modify model_kwargs with response_format since it's not supported by Responses API
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template="{{system_prompt}}\n\nGaps:\n{{gaps}}\n\nRESPOND ONLY WITH VALID JSON: [{\"title\": \"...\", \"abstract\": \"...\"}]",
            prompt_kwargs={"system_prompt": self.prompt, "gaps": None},
        )
    
    class _Prediction:
        """Lightweight prediction wrapper with id and data (non-Parameter)."""

        def __init__(self, sample_id: str, data: Any):
            self.id = sample_id
            self.data = data

    def call(self, sample) -> Any:
        """Generate proposals for given gaps."""
        gaps_text = sample.gaps_text
        sample_id = sample.id
        result = self.generator(prompt_kwargs={"gaps": gaps_text})
        
        # Extract text from GeneratorOutput, handling nested structures
        data = result
        max_depth = 10
        depth = 0
        
        # Recursively extract from GeneratorOutput objects and Parameters
        while depth < max_depth:
            if isinstance(data, str):
                break
            
            # Try to extract .data attribute
            inner_data = getattr(data, 'data', None)
            if inner_data is not None:
                data = inner_data
                depth += 1
            else:
                break
        
        # Ensure we have a string
        if not isinstance(data, str):
            LOGGER.warning(f"Unexpected final data type after extraction: {type(data).__name__}, converting to string")
            data = str(data)
        
        LOGGER.debug(f"Extracted proposal text (depth={depth}, len={len(data)}): {data[:100]}...")
        
        if self.training:
            return TaskParameter(
                data=data,
                data_id=sample_id,
                requires_opt=False,
                role_desc="Generated proposals output",
                param_type=ParameterType.OUTPUT,
                name="proposal_output",
                eval_input=data,
            )
        return self._Prediction(sample_id=sample_id, data=data)

    def forward(self, *args, **kwargs):  # type: ignore[override]
        """AdalFlow calls forward during training; delegate to call."""
        return self.call(*args, **kwargs)


class ProposalTrainTask(AdalComponent):
    """Training task that wraps proposal generation with evaluation."""

    class GapLoss(LossComponent):
        """Loss computed from proposal quality scores."""

        def __init__(self, parent: "ProposalTrainTask"):
            super().__init__()
            # Avoid registering parent as a child component to prevent recursion
            object.__setattr__(self, "parent", parent)

        def set_backward_engine(self, backward_engine, *args, **kwargs):
            self.backward_engine = backward_engine

        def forward(self, sample, y_pred):
            if not isinstance(y_pred, Parameter):
                raise ValueError("y_pred must be a TaskParameter for loss computation")
            text = y_pred.data
            score = self.parent.compute_score(sample, text)
            loss_val = -score  # minimize negative score
            reason = self.parent.last_eval_failure_reason
            role_desc = "Proposal quality loss"
            if reason:
                role_desc += f" (parse failure: {reason})"
            return TaskParameter(
                data=loss_val,
                requires_opt=True,
                role_desc=role_desc,
                param_type=ParameterType.LOSS_OUTPUT,
                eval_input=loss_val,
                score=loss_val,
            )

        def backward(self, *args, **kwargs):
            # Text gradients handled by AdalFlow components
            return None
    
    def __init__(
        self,
        model_client: ModelClient,
        model_kwargs: dict,
        embedder: Embedder,
        cloud: Cloud,
        sampled_gaps: list[GapWithContext],
    ):
        # Create the task component
        task = ProposalTask(model_client, model_kwargs)
        
        # Store for loss computation
        self.embedder = embedder
        self.cloud = cloud
        self.sampled_gaps = sampled_gaps
        self.last_eval_failure_reason: str | None = None
        gap_loss = self.GapLoss(self)
        super().__init__(
            task=task,
            eval_fn=self._placeholder_eval_fn,
            backward_engine_model_config={
                "model_client": model_client,
                "model_kwargs": model_kwargs,
            },
            text_optimizer_model_config={
                "model_client": model_client,
                "model_kwargs": model_kwargs,
            },
            loss_fn=gap_loss,
        )
        self.loss_component = gap_loss

    @staticmethod
    def _placeholder_eval_fn(*args, **kwargs):
        """Placeholder eval_fn to satisfy AdalFlow serialization."""
        return 0.0

    def compute_score(self, sample, response_text: str) -> float:
        """Compute mean gap z-score for proposals in a response.

        Records parse failures in self.last_eval_failure_reason so gradients can cite them.
        """
        if not isinstance(response_text, str):
            raise ValueError("response_text must be a string for scoring")

        proposals, parse_err = parse_proposals_from_response(response_text)
        self.last_eval_failure_reason = parse_err
        if not proposals:
            return -1000.0

        embeddings = embed_proposals(proposals, self.embedder, self.cloud)
        from bookgap.configs.config import ExperimentConfig
        from bookgap.configs.gap.gap_config import GapConfig
        from bookgap.configs.llm.llm_config import LLMConfig
        from bookgap.configs.batch.batch_config import BatchConfig
        from bookgap.configs.output.output_config import OutputConfig
        from bookgap.configs.cloud.cloud_config import CloudConfig
        temp_config = ExperimentConfig(
            gap=GapConfig(),
            llm=LLMConfig(),
            batch=BatchConfig(),
            output=OutputConfig(),
            cloud=CloudConfig(),
        )
        scores = score_proposals_against_gaps(
            proposals, embeddings, self.sampled_gaps, config=temp_config
        )
        all_zscores: list[float] = []
        for prop_score in scores["proposal_scores"]:
            all_zscores.extend(prop_score["gap_zscores"])
        mean_zscore = float(np.mean(all_zscores)) if all_zscores else -1000.0
        self.last_eval_failure_reason = None
        return mean_zscore

    def to_dict(self):  # type: ignore[override]
        """Avoid deep serialization loops by returning minimal state."""
        return {"prompt": self.task.prompt.data}
    
    def prepare_task(self, sample, y_gt=None):
        """Prepare a single training sample."""
        return self.task, {"sample": sample}
    
    def prepare_eval(self, sample, y_pred):  # type: ignore[override]
        """Prepare evaluation inputs."""
        LOGGER.info(f"=+=Type debugging=+= {type(y_pred)=}")
        if isinstance(y_pred, Parameter):
            response_text = y_pred.data
        elif hasattr(y_pred, "data"):
            LOGGER.info(f"=+=Type debugging=+= {type(y_pred.data)=}")
            response_text = y_pred.data
        else:
            response_text = y_pred
        if not isinstance(response_text, str):
            response_text = str(response_text)
        return self.compute_score, {"sample": sample, "response_text": response_text}
    
    def prepare_loss(self, sample, y_pred, *args, **kwargs):
        """Compute loss for backward pass."""
        return self.loss_component.forward, {"sample": sample, "y_pred": y_pred}

    def evaluate_samples(self, samples, y_preds, num_workers: int = 1, use_loss_eval_fn: bool = False):  # type: ignore[override]
        """AdalFlow hook: compute evaluation scores for a batch."""
        scores: list[float] = []
        for sample, y_pred in zip(samples, y_preds):
            LOGGER.info(f"=+=Type debugging=+= {type(y_pred)=}")
            if isinstance(y_pred, Parameter):
                response_text = y_pred.data
            elif hasattr(y_pred, "data"):
                LOGGER.info(f"=+=Type debugging=+= {type(y_pred.data)=}")
                response_text = y_pred.data
            else:
                response_text = y_pred
            if not isinstance(response_text, str):
                response_text = str(response_text)
            scores.append(self.compute_score(sample, response_text))
        avg_score = float(np.mean(scores)) if scores else 0.0
        return EvaluationResult(avg_score=avg_score, per_item_scores=scores)


def run_simple_optimization(config: OptimizationConfig) -> None:
    """Run simple hill-climbing optimization without AdalFlow.
    
    This is a fallback implementation that doesn't require AdalFlow installation.
    Implements basic prompt perturbation and greedy selection.
    
    Args:
        config: Optimization configuration
    """
    LOGGER.info("Running simple hill-climbing optimization (no AdalFlow)")
    
    # Convert output_dir to Path
    output_dir = Path(config.output_dir)
    LOGGER.info(f"Output directory: {output_dir}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("Loading gaps...")
    gaps = compute_gaps_for_config(config.experiment)
    LOGGER.info(f"Loaded {len(gaps)} gaps")
    
    # Sample gaps for optimization
    LOGGER.info(f"Sampling {config.sampling.num_gaps} gaps with strategy: {config.sampling.strategy}")
    sampled_gaps = sample_gaps(gaps, config.sampling)
    LOGGER.info(f"Selected {len(sampled_gaps)} gaps for optimization")
    
    # Initialize cloud and embedder once
    LOGGER.info("Initializing cloud and embedder...")
    cloud: Cloud = instantiate(config.experiment.cloud)
    
    # Create OpenAI client
    client = OpenAI()
    
    # Tracking for best prompt
    current_prompt = config.generator.initial_system_prompt
    best_prompt = current_prompt
    best_score = float("-inf")
    
    # Initialize embedder context manager
    with Embedder(model_name=config.experiment.cloud.embedder_model) as embedder:
        LOGGER.info("Embedder initialized")
        
        # Optimization loop
        for iteration in range(config.trainer.iterations):
            LOGGER.info(f"\n=== Iteration {iteration + 1}/{config.trainer.iterations} ===")
            
            # Generate candidates
            # For simplicity, we generate one batch with current prompt
            # TODO: Implement prompt perturbation/mutation strategy
            # Generate N candidate proposals
            user_prompt = format_gaps_for_prompt(sampled_gaps, top_k_books=config.sampling.top_k_books)
            LOGGER.info(f"User prompt length: {len(user_prompt)} chars")
            LOGGER.debug(f"User prompt: {user_prompt[:500]}...")
            
            try:
                response = client.chat.completions.create(
                    model=config.generator.model_name,
                    messages=[
                        {"role": "system", "content": current_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=config.generator.temperature,
                    max_tokens=config.generator.max_tokens,
                )
                
                response_text = response.choices[0].message.content or ""
                LOGGER.info(f"LLM response length: {len(response_text)} chars")
                LOGGER.info(f"LLM response preview: {response_text[:200]}...")
                LOGGER.debug(f"Full LLM response: {response_text}")
                
            except Exception as e:
                LOGGER.error(f"Failed to generate proposals: {e}")
                continue
            
            # Parse proposals
            proposals, parse_err = parse_proposals_from_response(response_text)
            LOGGER.info(f"Parsed {len(proposals)} proposals")
            if parse_err:
                LOGGER.warning("No valid proposals generated (parse failure: %s)", parse_err)
            if not proposals:
                LOGGER.warning("No valid proposals generated, skipping iteration")
                continue
            
            # Embed proposals
            embeddings = embed_proposals(proposals, embedder, cloud)
            
            # Score proposals
            results = score_proposals_against_gaps(
                proposals=proposals,
                embeddings=embeddings,
                gaps=sampled_gaps,
                config=config.experiment,
            )
            
            # Log results
            LOGGER.info(f"Aggregate score: {results['aggregate_score']:.4f}")
            for score_dict in results["proposal_scores"]:
                LOGGER.info(
                    f"  - {score_dict['title'][:50]}: "
                    f"max_z={score_dict['max_zscore']:.4f}, "
                    f"mean_z={score_dict['mean_zscore']:.4f}"
                )
            
            # Update best if improved
            if results["aggregate_score"] > best_score:
                best_score = results["aggregate_score"]
                best_prompt = current_prompt
                LOGGER.info(f"New best score: {best_score:.4f}")
            
            # Save iteration results
            iteration_log = {
                "iteration": iteration + 1,
                "prompt": current_prompt,
                "results": results,
                "parse_error": parse_err,
            }
            log_file = output_dir / f"iteration_{iteration + 1:03d}.json"
            with open(log_file, "w") as f:
                json.dump(iteration_log, f, indent=2)
            
            # TODO: Implement prompt mutation/perturbation
            # For now, we just keep the same prompt (no optimization happening)
            # Real implementation would:
            # 1. Generate a textual gradient or feedback based on scores
            # 2. Use LLM to mutate the prompt based on feedback
            # 3. Decide whether to accept new prompt (e.g., always accept if improved)
            
            LOGGER.info(f"Iteration {iteration + 1} complete")
    
    # Save final best prompt
    best_prompt_file = output_dir / "best_prompt.txt"
    with open(best_prompt_file, "w") as f:
        f.write(best_prompt)
    
    LOGGER.info(f"\n=== Optimization Complete ===")
    LOGGER.info(f"Best score: {best_score:.4f}")
    LOGGER.info(f"Best prompt saved to: {best_prompt_file}")


def run_adalflow_optimization(config: OptimizationConfig) -> None:
    """Run optimization using AdalFlow's TSGD-M trainer.
    
    Args:
        config: Optimization configuration
    """
    LOGGER.info("Running AdalFlow TSGD-M optimization")
    
    # Convert output_dir to Path
    output_dir = Path(config.output_dir)
    LOGGER.info(f"Output directory: {output_dir}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("Loading gaps...")
    gaps = compute_gaps_for_config(config.experiment)
    LOGGER.info(f"Loaded {len(gaps)} gaps")
    
    # Sample gaps for optimization
    LOGGER.info(f"Sampling {config.sampling.num_gaps} gaps with strategy: {config.sampling.strategy}")
    sampled_gaps = sample_gaps(gaps, config.sampling)
    LOGGER.info(f"Selected {len(sampled_gaps)} gaps for optimization")
    
    # Create gaps text for training samples
    combined_gaps = format_gaps_for_prompt(sampled_gaps, top_k_books=config.sampling.top_k_books)
    train_dataset = [GapSample(id="sample0", gaps_text=combined_gaps)]  # Single sample with all gaps
    
    # Initialize cloud and embedder
    LOGGER.info("Initializing cloud and embedder...")
    cloud: Cloud = instantiate(config.experiment.cloud)
    
    # Initialize embedder context manager
    with Embedder(model_name=config.experiment.cloud.embedder_model) as embedder:
        LOGGER.info("Embedder initialized")
        
        # Create model client
        from adalflow.components.model_client import OpenAIClient
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key not found in environment variable OPENAI_API_KEY")
        
        model_client = OpenAIClient(api_key=api_key)
        model_kwargs = {
            "model": config.generator.model_name,
            "temperature": config.generator.temperature,
            "max_tokens": config.generator.max_tokens,
        }
        
        # Create training task
        LOGGER.info("Creating training task...")
        train_task = ProposalTrainTask(
            model_client=model_client,
            model_kwargs=model_kwargs,
            embedder=embedder,
            cloud=cloud,
            sampled_gaps=sampled_gaps,
        )
        LOGGER.debug("Initial trainable prompt: %s", train_task.task.prompt.data)
        
        # Configure trainer with TGD optimizer
        LOGGER.info("Initializing TSGD-M trainer...")
        # Pass kwargs directly to TGDOptimizer; keep only supported arguments
        text_optimizer_config = {
            "max_past_history": config.trainer.tsgd_m_momentum_window,
        }
        
        trainer = Trainer(
            adaltask=train_task,
            max_steps=config.trainer.iterations,
            text_optimizers_config_kwargs=text_optimizer_config,
            debug=True,
        )
        
        # Run training
        LOGGER.info(f"Starting optimization for {config.trainer.iterations} iterations...")
        trainer.fit(
            train_dataset=train_dataset,
            train_loader=None,
            val_dataset=train_dataset,
        )
        
        # Get optimized prompt
        optimized_prompt = train_task.task.prompt.data
        if not isinstance(train_task.task.prompt, Parameter):
            raise ValueError(f"Expected train_task.task.prompt to be Parameter, got {type(train_task.task.prompt)}")
        else:
            # Print out all fields
            for field_name in vars(train_task.task.prompt):
                field_value = getattr(train_task.task.prompt, field_name)
                LOGGER.info(f"Prompt field {field_name}: {field_value}")
        LOGGER.info(f"\n=== Optimization Complete ===")
        LOGGER.info(f"Optimized prompt: {optimized_prompt}")
        LOGGER.info("Prompt optimizer trace: %s", train_task.task.prompt.tgd_optimizer_trace)
        
        # Save best prompt
        best_prompt_file = output_dir / "best_prompt.txt"
        with open(best_prompt_file, "w") as f:
            f.write(optimized_prompt)
        
        LOGGER.info(f"Best prompt saved to: {best_prompt_file}")


@hydra.main(version_base=None, config_path=None, config_name="prompt_opt")
def main(cfg: OptimizationConfig) -> None:
    """Main entry point for prompt optimization.
    
    Args:
        cfg: Optimization configuration (provided by Hydra)
    """
    # Initialize project-standard logging (file + console)
    setup_logging(log_name="optimize_prompt", log_level="DEBUG")
    # Resolve output_dir
    output_dir = Path(cfg.output_dir) if cfg.output_dir else Path("outputs/prompt_opt")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("=" * 60)
    LOGGER.info("PROMPT OPTIMIZATION")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info(f"Iterations: {cfg.trainer.iterations}")
    LOGGER.info(f"Candidates per iteration: {cfg.trainer.candidates_per_iter}")
    LOGGER.info(f"Gap sampling: {cfg.sampling.strategy} (n={cfg.sampling.num_gaps})")
    LOGGER.info(f"Model: {cfg.generator.model_name}")
    
    # Update the config's output_dir to the resolved path
    cfg.output_dir = str(output_dir)
    
    # Run optimization based on mode
    if cfg.trainer.tsgd_m_enabled:
        run_adalflow_optimization(cfg)
    else:
        run_simple_optimization(cfg)


# Register configs with Hydra ConfigStore
register_configs()  # Register base bookgap configs

cs = ConfigStore.instance()
cs.store(name="prompt_opt", node=OptimizationConfig())


if __name__ == "__main__":
    main()
