"""Workflow functions for batch proposal generation.

Pure functions for the batch processing pipeline using Hydra's instantiation system.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import logging
import random
import time
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .ollama_local import OllamaProposalRecord

from hydra.utils import instantiate
from omegaconf import OmegaConf
from openai import OpenAI

import numpy as np
from numpy.typing import NDArray

from bookgap.configs.config import ExperimentConfig
from bookgap.cloud import Cloud, Embedder
from bookgap.configs.gap.gap_config import CellType
from bookgap.gap_metrics import compute_gap_metrics_for_embeddings
from bookgap.llm.artifacts.common import group_gap_articles_by_covering_books
from bookgap.llm.data_structures import (
    BatchResults,
    GapWithContext,
    ProposalAnalyticsStore,
    ProposalGapStats,
    ProposalOutrankCounts,
)
from bookgap.llm.generators.batch_generator import BatchProposalGenerator
from bookgap.subcloud import SubCloud
from bookgap.util import LOGGER, create_unicode_safe_filename
from ..protocols import (
    ProposalContext, BatchRequestData,
    Article, Book, LLMBackend, OpenAIModel, ProposalCounts, ParsedProposal, Proposal
)
from ..identifiers import build_gap_proposal_id, parse_gap_proposal_id, GapProposalId
from bookgap.index import range_search_radii_with
from bookgap.llm.artifacts.gap_plots import generate_gap_plots_for_results
from bookgap.llm.artifacts.context_export import generate_all_contexts
from bookgap.llm.artifacts.results_export import generate_gaps_analysis_csv
from .run_state import RunStateManager, RunStep
from bookgap.gaps import compute_gaps_for_config


@dataclass(frozen=True)
class PendingProposal:
    """Temporary holder for parsed proposal data before embeddings are attached."""
    proposal: Proposal
    custom_id: str


@dataclass(frozen=True)
class ProposalGrouping:
    """Indexed proposal collections for downstream artifact generation."""
    by_gap_idx: dict[int, list[ParsedProposal]]
    by_topic_gap: dict[tuple[str, int], list[ParsedProposal]]
    ordered: list[tuple[ParsedProposal, GapProposalId]]


def find_gaps_for_topic(
    books: SubCloud,
    articles: SubCloud,
    radius_fraction: float,
    cell_type: CellType,
    k: int = 1,
    is_intra_cloud: bool = False,
    deflation_strategy=None,
):
    """Return gaps for the given books/articles subclouds and parameters.

    ``deflation_strategy`` defaults to None (greedy hard-subtraction,
    byte-identical to prior behaviour); pass
    ``DeflationStrategy.FL_BOOKDEMAND_RESID`` to opt into resid.
    """
    return cell_type.value.find_gaps(
        ambient=articles,
        landmark=books,
        query=articles,
        radius_fraction=radius_fraction,
        k=k,
        is_intra_cloud=is_intra_cloud,
        deflation_strategy=deflation_strategy,
    )

def _compute_proposal_embeddings(
    pending: list[PendingProposal],
    config: ExperimentConfig,
) -> NDArray[np.float32]:
    """Compute embeddings for pending proposals projected into the cloud space."""

    title_abstract_pairs = [
        (proposal.proposal.title, proposal.proposal.abstract)
        for proposal in pending
    ]

    with Embedder(model_name=config.cloud.embedder.model_name) as embedder:
        raw_embeddings = embedder.embed_texts(title_abstract_pairs)

    cloud: Cloud = instantiate(config.cloud)
    projection = cloud.denoiser_projection()
    projected_embeddings = projection.apply(raw_embeddings)

    if cloud.normalize_embeddings and projected_embeddings.shape[0] > 0:
        projected_embeddings = projected_embeddings / np.linalg.norm(projected_embeddings, axis=1, keepdims=True)

    return projected_embeddings.astype(np.float32, copy=False)


def _index_proposals(proposals: list[ParsedProposal]) -> ProposalGrouping:
    """Build indexed views of proposals keyed by gap identifiers."""
    by_gap_idx: dict[int, list[ParsedProposal]] = defaultdict(list)
    by_topic_gap: dict[tuple[str, int], list[ParsedProposal]] = defaultdict(list)
    ordered: list[tuple[ParsedProposal, GapProposalId]] = []

    for proposal in proposals:
        try:
            parsed = parse_gap_proposal_id(proposal.custom_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning(
                "Skipping proposal with unparseable custom_id %s: %s",
                proposal.custom_id,
                exc,
            )
            continue

        by_gap_idx[parsed.gap_idx].append(proposal)
        key = (parsed.topic_id, parsed.gap_idx)
        by_topic_gap[key].append(proposal)
        ordered.append((proposal, parsed))

    return ProposalGrouping(
        by_gap_idx=dict(by_gap_idx),
        by_topic_gap=dict(by_topic_gap),
        ordered=ordered,
    )


def _compute_proposal_outrank_counts(
    gaps: list[GapWithContext],
    proposals_by_topic_gap: dict[tuple[str, int], list[ParsedProposal]],
    analytics_store: ProposalAnalyticsStore,
    proposal_id_lookup: dict[str, GapProposalId],
    *,
    epsilon: float = float(np.finfo(np.float32).eps * 8),
) -> None:
    """Compute per-article outrank counts and record them in the analytics store."""

    if not gaps:
        return

    gap_lookup = {(gap.topic_id, gap.gap_idx): gap for gap in gaps}

    for key, gap in gap_lookup.items():
        proposals = proposals_by_topic_gap.get(key)
        if not proposals:
            continue

        gap_article_indices = gap.gap.get_dists_idxs().idxs
        if len(gap_article_indices) == 0:
            continue

        article_embeddings_all = gap.articles.embeddings()
        try:
            gap_article_vecs = np.ascontiguousarray(
                article_embeddings_all[np.asarray(gap_article_indices, dtype=int)],
                dtype=np.float32,
            )
        except IndexError:
            LOGGER.warning(
                "Gap %s has invalid article indices; skipping coverage computation",
                key,
            )
            continue

        if gap_article_vecs.size == 0:
            continue

        try:
            book_index = gap.books.index()
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to build book index for gap %s: %s", key, exc)
            continue

        for proposal in proposals:
            proposal_vec = np.asarray(proposal.embedding, dtype=np.float32)

            diffs = gap_article_vecs - proposal_vec
            sq_dists = np.einsum("ij,ij->i", diffs, diffs, dtype=np.float32)

            counts = np.zeros(len(gap_article_vecs), dtype=np.int32)

            def kernel_fn(article_idx: int, neighbor_dists, _neighbor_ids) -> None:
                threshold = float(sq_dists[article_idx]) - epsilon
                if threshold <= 0.0:
                    strictly_closer = 0
                else:
                    strictly_closer = int(np.count_nonzero(neighbor_dists < threshold))
                counts[article_idx] = strictly_closer

            range_search_radii_with(
                book_index,
                query=gap_article_vecs,
                radii=sq_dists,
                kernel_fn=kernel_fn,
                show_progress=False,
            )

            proposal_id = proposal_id_lookup.get(proposal.custom_id)
            if proposal_id is None:
                LOGGER.warning(
                    "Unhandled proposal custom_id %s while recording outrank counts",
                    proposal.custom_id,
                )
                continue

            analytics_store.add_outrank_counts(
                proposal.custom_id,
                ProposalOutrankCounts(counts=counts),
            )

def _compute_proposal_gap_stats(
    gaps: list[GapWithContext],
    proposals_by_topic_gap: dict[tuple[str, int], list[ParsedProposal]],
    config: ExperimentConfig,
    analytics_store: ProposalAnalyticsStore,
    proposal_id_lookup: dict[str, GapProposalId],
) -> None:
    """Compute hypothetical gap sizes and record them in the analytics store."""

    if not gaps:
        return

    gap_lookup = {(gap.topic_id, gap.gap_idx): gap for gap in gaps}
    algorithm = config.gap.cell_type.value

    for key, proposals in proposals_by_topic_gap.items():
        if not proposals:
            continue

        gap = gap_lookup[key]
        embeddings = np.stack(
            [np.asarray(proposal.embedding, dtype=np.float32) for proposal in proposals],
            axis=0,
        )

        gap_stats = compute_gap_metrics_for_embeddings(
            gap=gap,
            embeddings=embeddings,
            config=config,
        )

        for proposal, stats in zip(proposals, gap_stats, strict=True):
            proposal_id = proposal_id_lookup.get(proposal.custom_id)
            if proposal_id is None:
                LOGGER.warning(
                    "Unhandled proposal custom_id %s while recording gap stats",
                    proposal.custom_id,
                )
                continue

            analytics_store.add_gap_stats(
                proposal.custom_id,
                ProposalGapStats(
                    gap_size=stats.gap_size,
                    gap_size_percentile=stats.gap_size_percentile,
                    gap_zscore=stats.gap_zscore,
                ),
            )

class MetadataUtils:
    """Static utilities for metadata chunking and parsing."""
    
    CHUNK_SIZE = 512  # OpenAI batch API metadata value limit
    
    @staticmethod
    def serialize_config(config: Any) -> dict[str, str]:
        """Convert config to chunked metadata dict for batch API.
        
        Args:
            config: Configuration object
            
        Returns:
            Dictionary with chunked metadata (keys are string indices "0", "1", etc.)
        """
        if not OmegaConf.is_config(config):
            config = OmegaConf.structured(config)
        config_dict = OmegaConf.to_container(config, resolve=True)
        config_json = json.dumps(config_dict, default=str, indent=None, separators=(',', ':'))
        
        # Split into chunks for metadata (512 char limit per key)
        metadata = {}
        for i, chunk_start in enumerate(range(0, len(config_json), MetadataUtils.CHUNK_SIZE)):
            chunk = config_json[chunk_start:chunk_start + MetadataUtils.CHUNK_SIZE]
            metadata[str(i)] = chunk
        
        return metadata




def build_contexts_for_gaps(gaps: list[GapWithContext], config: ExperimentConfig) -> list[ProposalContext]:
    """Build proposal contexts from gaps.
    
    Samples articles and books for each gap to create the context needed for proposal generation.
    
    Args:
        gaps: List of GapWithContext objects
        config: Experiment configuration with LLM settings
        
    Returns:
        List of ProposalContext objects
    """
    contexts = []
    
    for gap_with_context in gaps:
        gap_idxs = gap_with_context.gap.get_dists_idxs().idxs
        gap_size = len(gap_idxs)

        if gap_size == 0:
            raise ValueError(
                f"Gap {gap_with_context.gap_idx} for topic {gap_with_context.topic_id} contains no articles"
            )

        # Deterministic RNG seeded by topic and gap
        seed_string = f"{gap_with_context.topic_id}_{gap_with_context.gap_idx}"
        seed_hash = hashlib.md5(seed_string.encode()).hexdigest()
        rng = random.Random(int(seed_hash[:8], 16))

        # Get work metadata
        article_works = gap_with_context.articles.works()
        book_works = gap_with_context.books.works()

        # Sample articles from the gap (up to configured limit)
        requested_articles = config.llm.num_articles
        sample_size = min(requested_articles, gap_size)
        sampled_positions = rng.sample(range(gap_size), sample_size)
        sampled_article_indices = [int(gap_idxs[pos]) for pos in sampled_positions]
        sampled_article_dicts = [article_works[idx] for idx in sampled_article_indices]

        sampled_articles: list[Article] = []
        for i, work in enumerate(sampled_article_dicts):
            display_name = work.get('display_name') if isinstance(work, dict) else getattr(work, 'display_name', None)
            abstract = work.get('abstract') if isinstance(work, dict) else getattr(work, 'abstract', None)
            if not display_name or not abstract:
                raise ValueError(
                    f"Sampled article {i} for gap {gap_with_context.gap_idx} missing display_name or abstract"
                )
            sampled_articles.append(Article(display_name=display_name, abstract=abstract))

        # Select covering books ordered by decreasing coverage, deterministic tie-breaker
        covering_books: list[Book] = []
        try:
            book_coverage = group_gap_articles_by_covering_books(gap_with_context.gap)
        except (AttributeError, TypeError):
            book_coverage = {}

        if book_coverage:
            requested_books = config.llm.num_books
            sorted_books = sorted(
                book_coverage.items(),
                key=lambda item: (-len(item[1]), int(item[0]))
            )
            for book_idx, _ in sorted_books:
                idx = int(book_idx)
                if idx < 0 or idx >= len(book_works):
                    continue
                book_work = book_works[idx]
                display_name = book_work.get('display_name') if isinstance(book_work, dict) else getattr(book_work, 'display_name', None)
                abstract = book_work.get('abstract') if isinstance(book_work, dict) else getattr(book_work, 'abstract', None)
                if not display_name or not abstract:
                    continue
                covering_books.append(Book(display_name=display_name, abstract=abstract))
                if len(covering_books) >= requested_books:
                    break
        else:
            requested_books = config.llm.num_books
            for book_work in book_works[:requested_books]:
                display_name = book_work.get('display_name') if isinstance(book_work, dict) else getattr(book_work, 'display_name', None)
                abstract = book_work.get('abstract') if isinstance(book_work, dict) else getattr(book_work, 'abstract', None)
                if not display_name or not abstract:
                    continue
                covering_books.append(Book(display_name=display_name, abstract=abstract))
                if len(covering_books) >= requested_books:
                    break
        
        # Create counts using correct field names
        counts = ProposalCounts(
            num_articles_for_llm=len(sampled_articles),
            num_books_for_llm=len(covering_books)
        )
        
        # Build context
        context = ProposalContext(
            topic_name=gap_with_context.topic_name,
            topic_id=gap_with_context.topic_id,
            gap_idx=gap_with_context.gap_idx,
            requested_num_prompt_articles=config.llm.num_articles,
            requested_num_prompt_books=config.llm.num_books,
            sampled_articles=sampled_articles,
            covering_books=covering_books,
            counts=counts
        )
        
        contexts.append(context)
        LOGGER.info(f"Built context for gap {gap_with_context.gap_idx}: {len(sampled_articles)} articles, {len(covering_books)} books")
    
    return contexts


def save_batch_input_jsonl(batch_requests: list[BatchRequestData], batch_input_filename: Path, config: ExperimentConfig) -> None:
    """Save batch requests to JSONL file for OpenAI batch API."""
    if not batch_requests:
        raise ValueError("No batch requests to submit")
    
    # Create JSONL content for batch API
    jsonl_lines = []
    
    # Determine the correct token parameter based on the model from config
    if config.llm.model.uses_max_completion_tokens:
        token_param = "max_completion_tokens"
    else:
        token_param = "max_tokens"

    for request in batch_requests:
        jsonl_line = {
            "custom_id": request.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": request.body.model,
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in request.body.messages
                ],
                token_param: request.body.max_tokens,
            }
        }
        
        if request.body.temperature is not None:
            jsonl_line["body"]["temperature"] = request.body.temperature  # type: ignore
        
        if request.body.response_format is not None:
            jsonl_line["body"]["response_format"] = request.body.response_format  # type: ignore
        
        jsonl_lines.append(json.dumps(jsonl_line))
    
    # Write JSONL to persistent file in Hydra output directory
    with open(batch_input_filename, 'w') as f:
        for line in jsonl_lines:
            f.write(line + '\n')
    LOGGER.info(f"Batch input file saved as: {batch_input_filename}")

def submit_batch_job(batch_input_filename: Path, config: ExperimentConfig) -> str:
    """Submit batch job to OpenAI API.

    Returns:
        Batch job ID string
    """
    
    # Initialize OpenAI client
    client = OpenAI()  # Uses OPENAI_API_KEY environment variable
    
    
    # Upload file
    LOGGER.info(f"Uploading {batch_input_filename}")
    with open(batch_input_filename, 'rb') as file:
        batch_input_file = client.files.create(
            file=file,
            purpose="batch"
        )
    
    # Create batch job
    LOGGER.info(f"Creating batch job with completion window: {config.batch.completion_window}")
    batch = client.batches.create(
        input_file_id = batch_input_file.id,
        endpoint = "/v1/chat/completions",
        completion_window = config.batch.completion_window,  # type: ignore
        metadata = MetadataUtils.serialize_config(config)    
    )
    
    LOGGER.info(f"Batch job created successfully: {batch.id}")
    return batch.id


def parse_batch_result_to_proposal(result: dict[str, Any]) -> PendingProposal:
    """Parse OpenAI batch result into a pending proposal without embeddings.
    
    Args:
        result: Raw batch result dict from OpenAI API
        
    Returns:
        PendingProposal ready for embedding computation
    """
    # Extract the JSON content from OpenAI batch result structure
    content = result["response"]["body"]["choices"][0]["message"]["content"]
    
    # Parse using Pydantic model for validation
    proposal = Proposal.model_validate_json(content)
    
    # Wrap with metadata
    return PendingProposal(
        proposal=proposal,
        custom_id=result["custom_id"]
    )

def _parse_batch_results_content(result_content: str, config: ExperimentConfig) -> tuple[list[ParsedProposal], dict[str, int]]:
    """Convert raw JSONL batch output into parsed proposals and statistics."""

    raw_results = result_content.splitlines()
    pending_proposals: list[PendingProposal] = []
    num_empty = 0
    num_errors = 0
    num_stop_reason_length = 0

    for line in raw_results:
        if not line.strip():
            num_empty += 1
            continue

        try:
            line_json = json.loads(line)
        except json.JSONDecodeError as exc:
            num_errors += 1
            LOGGER.error("Failed to decode batch result line: %s", exc)
            continue

        try:
            finish_reason = line_json["response"]["body"]["choices"][0]["finish_reason"]
        except (KeyError, IndexError, TypeError):
            finish_reason = None

        if finish_reason == "length":
            num_stop_reason_length += 1
            continue

        try:
            pending = parse_batch_result_to_proposal(line_json)
            pending_proposals.append(pending)
        except Exception as exc:  # pragma: no cover - defensive logging
            num_errors += 1
            LOGGER.error("Failed to parse line in batch results: %s", exc)

    parsed_proposals: list[ParsedProposal] = []
    if pending_proposals:
        embeddings = _compute_proposal_embeddings(pending_proposals, config=config)
        parsed_proposals = [
            ParsedProposal(
                proposal=pending.proposal,
                custom_id=pending.custom_id,
                embedding=embedding,
            )
            for pending, embedding in zip(pending_proposals, embeddings, strict=True)
        ]

    stats = {
        "raw_count": len(raw_results),
        "parsed_count": len(parsed_proposals),
        "empty_count": num_empty,
        "error_count": num_errors,
        "length_stop_count": num_stop_reason_length,
    }

    return parsed_proposals, stats


def monitor_and_download_results(
    batch_job_id: str,
    config: ExperimentConfig,
    output_dir: Path,
    *,
    reuse_existing: bool = True,
    state_manager: Optional[RunStateManager] = None,
) -> BatchResults:
    """Monitor batch job and download results when complete.
    
    Args:
        batch_job_id: Batch job ID to monitor
        config: Experiment configuration
        output_dir: Destination directory for outputs and artifacts
        reuse_existing: When True, re-use an existing results file if present
        
    Returns:
        BatchResults frozen dataclass containing proposals and parsed config
        
    Raises:
        Exception: If monitoring or download fails
    """
    # Suppress verbose httpx logging during batch polling
    logging.getLogger("httpx").setLevel(logging.WARNING)

    client = OpenAI()
    results_path = output_dir / "batch_results.jsonl"

    LOGGER.info(f"Monitoring batch job: {batch_job_id}")
    
    # Poll for completion
    while True:
        batch = client.batches.retrieve(batch_job_id)
        status = batch.status
        
        LOGGER.info(f"Batch status: {status}")
        
        if status == "completed":
            LOGGER.info("Batch completed successfully")
            break
        elif status in ["failed", "expired", "cancelled"]:
            LOGGER.warning("Batch %s reached terminal status %s", batch_job_id, status)
            if state_manager is not None:
                state_manager.mark_ready_for_resubmission(status=status)
            raise Exception(f"Batch job failed with status: {status}")
        elif status in ["validating", "in_progress", "finalizing"]:
            # Still processing, wait and check again
            time.sleep(30)  # Wait 30 seconds before checking again
        else:
            LOGGER.warning(f"Unknown batch status: {status}")
            time.sleep(30)
    expected_metadata = MetadataUtils.serialize_config(config)
    if batch.metadata != expected_metadata:
        LOGGER.error("Batch metadata does not match expected serialized config.")
        LOGGER.error(f"Expected metadata: {expected_metadata}")
        LOGGER.error(f"Actual metadata: {batch.metadata}")
        raise ValueError("Batch metadata does not match expected serialized config.")

    if reuse_existing and results_path.exists():
        LOGGER.info(f"Reusing existing batch results from {results_path}")
        result_content = results_path.read_text(encoding="utf-8")
    else:
        if not batch.output_file_id:
            LOGGER.error("Batch completed but no output file available. Batch info:")
            LOGGER.error(f"  Batch ID: {batch.id}")
            LOGGER.error(f"  Status: {batch.status}")
            LOGGER.error(f"  Input file ID: {batch.input_file_id}")
            LOGGER.error(f"  Output file ID: {batch.output_file_id}")
            LOGGER.error(f"  Error file ID: {getattr(batch, 'error_file_id', 'N/A')}")
            if batch.request_counts:
                LOGGER.error(
                    "  Request counts: total=%s, completed=%s, failed=%s",
                    batch.request_counts.total,
                    batch.request_counts.completed,
                    batch.request_counts.failed,
                )

            error_file_id = getattr(batch, "error_file_id", None)
            if error_file_id:
                LOGGER.error("Downloading error file for details...")
                try:
                    error_file_response = client.files.content(error_file_id)
                    error_content = error_file_response.content.decode("utf-8")
                    LOGGER.error(f"Error file content: {error_content}")
                except Exception as exc:
                    LOGGER.error("Failed to download error file: %s", exc)

            raise Exception(
                f"Batch {batch_job_id} completed but no output file available. Status: {batch.status}"
            )

        LOGGER.info("Downloading batch results")
        result_file_response = client.files.content(batch.output_file_id)
        result_content = result_file_response.content.decode("utf-8")

        with open(results_path, "w", encoding="utf-8") as f:
            f.write(result_content)

    parsed_proposals, stats = _parse_batch_results_content(result_content, config)

    LOGGER.info(
        "Parsed %s proposals out of %s results (%s empty, %s errors, %s stopped by length)",
        stats["parsed_count"],
        stats["raw_count"],
        stats["empty_count"],
        stats["error_count"],
        stats["length_stop_count"],
    )

    if parsed_proposals:
        LOGGER.info("Embedded %s proposals for downstream artifacts", len(parsed_proposals))

    return BatchResults(
        proposals=parsed_proposals,
        metadata=config,
    )
    



def run_batch_workflow(
    config: ExperimentConfig,
    output_dir: Path,
    state_manager: RunStateManager,
    *,
    reuse_existing_results: bool = True,
) -> BatchResults:
    """Run the complete proposal workflow from gap finding to result ingestion.

    Dispatches on ``config.llm.backend``:

    - ``LLMBackend.OPENAI_BATCH`` (default): OpenAI Batch API path
      (:func:`_run_openai_batch_path`); byte-identical to historical
      behaviour when no overrides are supplied.
    - ``LLMBackend.OLLAMA_LOCAL``: synchronous local Ollama path
      (:func:`_run_ollama_local_path`).

    Both paths share the same gap-finding, context-building, and
    downstream artifact-generation code, so the only divergence is the
    proposal-generation step itself.
    """
    LOGGER.info("Starting batch workflow with %s gaps per topic", config.batch.num_gaps)

    # Step 1: Find gaps (shared across backends)
    gaps = compute_gaps_for_config(config)
    if state_manager is not None:
        state_manager.mark_gaps_ready(gap_count=len(gaps))
    LOGGER.info("Found %s gaps", len(gaps))

    # ARTIFACTS: Generate context files (shared)
    context_paths = generate_all_contexts(gaps, output_dir)

    # Step 2: Build contexts (shared)
    contexts = build_contexts_for_gaps(gaps, config)
    LOGGER.info("Built %s proposal contexts", len(contexts))

    backend = _resolve_backend(config)
    if backend is LLMBackend.OPENAI_BATCH:
        results = _run_openai_batch_path(
            config=config,
            output_dir=output_dir,
            state_manager=state_manager,
            gaps=gaps,
            contexts=contexts,
            context_paths=context_paths,
            reuse_existing_results=reuse_existing_results,
        )
    elif backend is LLMBackend.OLLAMA_LOCAL:
        results = _run_ollama_local_path(
            config=config,
            output_dir=output_dir,
            state_manager=state_manager,
            gaps=gaps,
            contexts=contexts,
            context_paths=context_paths,
        )
    else:  # pragma: no cover - exhaustively handled above
        raise ValueError(f"Unknown LLM backend: {backend!r}")

    _generate_downstream_artifacts(
        gaps=gaps,
        contexts=contexts,
        results=results,
        config=config,
        output_dir=output_dir,
    )

    return results


def _resolve_backend(config: ExperimentConfig) -> LLMBackend:
    """Coerce ``config.llm.backend`` to an :class:`LLMBackend` member.

    OmegaConf may surface the enum as either the enum value itself or its
    string value (e.g. ``"openai_batch"``) depending on whether the
    config was constructed via ``OmegaConf.structured`` or built ad-hoc
    in tests. Accept both forms.
    """
    raw = getattr(config.llm, "backend", LLMBackend.OPENAI_BATCH)
    if isinstance(raw, LLMBackend):
        return raw
    if isinstance(raw, str):
        try:
            return LLMBackend(raw)
        except ValueError:
            try:
                return LLMBackend[raw]
            except KeyError as exc:
                raise ValueError(f"Unknown LLM backend string: {raw!r}") from exc
    raise TypeError(f"Unexpected LLMBackend type {type(raw)!r}: {raw!r}")


def _run_openai_batch_path(
    *,
    config: ExperimentConfig,
    output_dir: Path,
    state_manager: RunStateManager,
    gaps: list[GapWithContext],
    contexts: list[ProposalContext],
    context_paths: Any,
    reuse_existing_results: bool,
) -> BatchResults:
    """Execute the OpenAI Batch API proposal-generation path.

    Behavior identical to the pre-dispatch implementation: generates
    batch requests, writes the JSONL input file, submits the batch job,
    monitors it, downloads results, and parses them into
    :class:`ParsedProposal` objects.
    """
    # Step 3: Generate batch requests
    requests = generate_batch_requests(contexts, config)
    LOGGER.info("Generated %s batch requests", len(requests))

    embedder_model = str(config.cloud.embedder.model_name)
    denoiser_name = str(config.cloud.denoiser)
    drop_k_value = str(config.cloud.drop_k)
    num_articles_value = str(config.llm.num_articles)
    num_books_value = str(config.llm.num_books)

    raw_filename = (
        f"{embedder_model}__{denoiser_name}__drop_{drop_k_value}__"
        f"articles_{num_articles_value}__books_{num_books_value}.jsonl"
    )
    safe_filename = create_unicode_safe_filename(raw_filename).replace(" ", "_")
    batch_input_filename = output_dir / safe_filename

    save_batch_input_jsonl(requests, batch_input_filename, config)

    state_manager.mark_ready_to_submit(
        gap_count=len(gaps),
        context_file_count=len(context_paths),
        request_count=len(requests),
        embedder_model=embedder_model,
        denoiser=denoiser_name,
        drop_k=drop_k_value,
    )

    reuse_existing_results_flag = state_manager.state.step == RunStep.COMPLETED

    # Step 4: Submit batch job (if needed)
    if state_manager.state.step in {RunStep.SUBMITTED, RunStep.COMPLETED}:
        batch_job_id = state_manager.require_batch_job_id()
        LOGGER.info("Using recorded batch job id: %s", batch_job_id)
    else:
        batch_job_id = submit_batch_job(batch_input_filename, config)
        state_manager.mark_submitted(batch_job_id, request_count=len(requests))
        LOGGER.info("Submitted batch job: %s", batch_job_id)
        reuse_existing_results_flag = False

    # Step 5: Monitor and download results
    results = monitor_and_download_results(
        batch_job_id,
        config,
        output_dir,
        reuse_existing=reuse_existing_results_flag,
        state_manager=state_manager,
    )
    state_manager.mark_completed(batch_job_id, proposal_count=len(results.proposals))
    LOGGER.info("Completed! Generated %s proposals", len(results.proposals))
    return results


def _run_ollama_local_path(
    *,
    config: ExperimentConfig,
    output_dir: Path,
    state_manager: RunStateManager,
    gaps: list[GapWithContext],
    contexts: list[ProposalContext],
    context_paths: Any,
) -> BatchResults:
    """Execute the synchronous local-Ollama proposal-generation path.

    Iterates :class:`ProposalContext` objects one at a time, calls
    :class:`OllamaProposalGenerator.generate`, appends each successful
    record to ``ollama_results.jsonl`` as it is produced (for crash
    resumability), and finally embeds the parsed proposals to produce
    the same :class:`BatchResults` shape the OpenAI path returns.

    Resumability: on restart, already-written custom IDs are skipped so
    a SIGINT'd run can pick up where it left off without re-paying for
    completed generations. JSON-parse failures (``record.proposal is
    None``) are logged and skipped — they do not block the run.
    """
    # Local import to keep the `ollama` dependency optional at
    # import-time of `workflow_functions` (only required when the user
    # actually opts into the Ollama backend).
    from .ollama_local import OllamaProposalGenerator, OllamaProposalRecord

    results_path = output_dir / "ollama_results.jsonl"

    embedder_model = str(config.cloud.embedder.model_name)
    denoiser_name = str(config.cloud.denoiser)
    drop_k_value = str(config.cloud.drop_k)

    state_manager.mark_ready_to_submit(
        gap_count=len(gaps),
        context_file_count=len(context_paths),
        request_count=len(contexts),
        embedder_model=embedder_model,
        denoiser=denoiser_name,
        drop_k=drop_k_value,
    )

    already_done: set[str] = _read_completed_custom_ids(results_path)
    if already_done:
        LOGGER.info(
            "Resuming Ollama run: %s of %s contexts already on disk",
            len(already_done),
            len(contexts),
        )

    generator = OllamaProposalGenerator(
        model=str(config.llm.ollama_model),
        num_articles=int(config.llm.num_articles),
        num_books=int(config.llm.num_books),
        temperature=float(config.llm.temperature),
        num_ctx=int(config.llm.ollama_num_ctx),
        num_predict=int(config.llm.ollama_num_predict),
    )

    pseudo_batch_id = f"ollama:{config.llm.ollama_model}"

    with open(results_path, "a", encoding="utf-8") as results_fp:
        for i, context in enumerate(contexts, start=1):
            custom_id = build_gap_proposal_id(
                gap_idx=context.gap_idx, topic_id=context.topic_id,
            )
            if custom_id in already_done:
                continue

            LOGGER.info(
                "[ollama %d/%d] generating proposal for %s",
                i, len(contexts), custom_id,
            )
            record = generator.generate(context)
            record_dict = _ollama_record_to_jsonl_dict(record, custom_id=custom_id)
            results_fp.write(json.dumps(record_dict) + "\n")
            results_fp.flush()
            already_done.add(custom_id)

    # Parse the full on-disk results file (including any pre-existing
    # entries from prior runs) into ParsedProposal objects.
    raw = results_path.read_text(encoding="utf-8") if results_path.exists() else ""
    parsed_proposals, stats = _parse_ollama_results_content(raw, config)
    LOGGER.info(
        "Parsed %s ollama proposals out of %s records (%s parse failures, %s errors)",
        stats["parsed_count"], stats["raw_count"],
        stats["parse_failure_count"], stats["error_count"],
    )

    state_manager.mark_completed(
        pseudo_batch_id, proposal_count=len(parsed_proposals),
    )

    return BatchResults(proposals=parsed_proposals, metadata=config)


def _ollama_record_to_jsonl_dict(
    record: OllamaProposalRecord,
    *,
    custom_id: str,
) -> dict[str, Any]:
    """Serialise an :class:`OllamaProposalRecord` to a JSONL line.

    Stores the parsed proposal (if any), the raw content, and provenance
    metadata. The schema is private to the Ollama path —
    :func:`_parse_ollama_results_content` is the only consumer.
    """
    proposal_dict = (
        record.proposal.model_dump() if record.proposal is not None else None
    )
    return {
        "custom_id": custom_id,
        "proposal": proposal_dict,
        "raw_content": record.raw_content,
        "thinking": record.thinking,
        "metadata": record.metadata,
    }


def _read_completed_custom_ids(results_path: Path) -> set[str]:
    """Return the set of ``custom_id`` values already persisted in
    ``ollama_results.jsonl``, used for crash-resume.

    Tolerates partial-write trailing lines (the last entry, if any, will
    just be regenerated — cheap on a local model). Lines without a valid
    JSON body or without a ``custom_id`` field are simply skipped.
    """
    if not results_path.exists():
        return set()
    done: set[str] = set()
    with open(results_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = entry.get("custom_id")
            if isinstance(cid, str) and cid:
                done.add(cid)
    return done


def _parse_ollama_results_content(
    raw: str, config: ExperimentConfig,
) -> tuple[list[ParsedProposal], dict[str, int]]:
    """Convert ``ollama_results.jsonl`` content into ParsedProposal list.

    Mirrors the shape of :func:`_parse_batch_results_content` so the
    downstream pipeline (analytics, plots, CSV export) doesn't care
    which backend produced the proposals.
    """
    raw_lines = raw.splitlines() if raw else []
    pending_proposals: list[PendingProposal] = []
    num_empty = 0
    num_errors = 0
    num_parse_failures = 0

    for line in raw_lines:
        if not line.strip():
            num_empty += 1
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            num_errors += 1
            LOGGER.error("Failed to decode ollama result line: %s", exc)
            continue

        proposal_dict = entry.get("proposal")
        if proposal_dict is None:
            num_parse_failures += 1
            LOGGER.warning(
                "Skipping ollama record %s: model produced unparseable JSON",
                entry.get("custom_id"),
            )
            continue

        try:
            proposal = Proposal.model_validate(proposal_dict)
        except Exception as exc:  # pragma: no cover - defensive logging
            num_errors += 1
            LOGGER.error("Failed to validate ollama proposal: %s", exc)
            continue

        custom_id = entry.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            num_errors += 1
            LOGGER.error("Ollama record missing custom_id; skipping")
            continue

        pending_proposals.append(
            PendingProposal(proposal=proposal, custom_id=custom_id)
        )

    parsed_proposals: list[ParsedProposal] = []
    if pending_proposals:
        embeddings = _compute_proposal_embeddings(pending_proposals, config=config)
        parsed_proposals = [
            ParsedProposal(
                proposal=pending.proposal,
                custom_id=pending.custom_id,
                embedding=embedding,
            )
            for pending, embedding in zip(
                pending_proposals, embeddings, strict=True,
            )
        ]

    stats = {
        "raw_count": len(raw_lines),
        "parsed_count": len(parsed_proposals),
        "empty_count": num_empty,
        "error_count": num_errors,
        "parse_failure_count": num_parse_failures,
    }
    return parsed_proposals, stats


def _generate_downstream_artifacts(
    *,
    gaps: list[GapWithContext],
    contexts: list[ProposalContext],
    results: BatchResults,
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    """Run the post-generation artifact pipeline shared by both backends.

    Computes outrank counts and per-proposal gap stats, then writes gap
    plots and the gaps-analysis CSV. Extracted unchanged from the
    pre-dispatch implementation so the two backends produce identical
    downstream artifacts.
    """
    proposal_grouping = _index_proposals(results.proposals)

    proposal_id_lookup = {
        proposal.custom_id: parsed
        for proposal, parsed in proposal_grouping.ordered
    }
    analytics_store = ProposalAnalyticsStore()

    _compute_proposal_outrank_counts(
        gaps,
        proposal_grouping.by_topic_gap,
        analytics_store,
        proposal_id_lookup,
    )

    _compute_proposal_gap_stats(
        gaps,
        proposal_grouping.by_topic_gap,
        config,
        analytics_store,
        proposal_id_lookup,
    )

    _plot_paths = generate_gap_plots_for_results(
        gaps,
        results.proposals,
        output_dir,
        proposals_by_gap=proposal_grouping.by_gap_idx,
    )

    _results_csv = generate_gaps_analysis_csv(
        gaps,
        results.proposals,
        output_dir,
        config,
        contexts,
        proposals_grouped=proposal_grouping.by_topic_gap,
        ordered_proposals=proposal_grouping.ordered,
        proposal_analytics=analytics_store,
    )


def generate_batch_requests(contexts: list[ProposalContext], config: ExperimentConfig) -> list[BatchRequestData]:
    if not contexts:
        return []
    
    generator: BatchProposalGenerator = instantiate(config.llm)
    
    # Generate batch requests
    return generator.generate_batch_requests(contexts)



def validate_batch_requests(requests: list[BatchRequestData]) -> None:
    """Validate that batch requests are properly formatted.
    
    Args:
        requests: List of batch request data structures
    """
    if not requests:
        raise ValueError("No batch requests to validate")
    
    errors = []
    custom_ids = set()
    
    for i, request in enumerate(requests):
        # Check for duplicate custom IDs
        if request.custom_id in custom_ids:
            errors.append(f"Duplicate custom_id found: {request.custom_id}")
        custom_ids.add(request.custom_id)
        
        # Check custom_id is not empty
        if not request.custom_id or not request.custom_id.strip():
            errors.append(f"Empty custom_id found at request {i}")
        
        # Check model is specified
        if not request.body.model:
            errors.append(f"Request {i}: model is not specified")
        
        # Check messages exist
        if not request.body.messages:
            errors.append(f"Request {i}: no messages provided")
        
        # Check message format
        for j, message in enumerate(request.body.messages):
            if message.role not in ["system", "user", "assistant"]:
                errors.append(f"Request {i}, Message {j}: invalid role '{message.role}'")
    
    if errors:
        raise ValueError("Batch request validation failed:\n" + "\n".join(errors))


def get_supported_models() -> list[str]:
    """Get list of supported model names.
    
    Returns:
        List of supported OpenAI model names
    """
    return [model.value for model in OpenAIModel]
