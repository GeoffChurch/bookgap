"""Workflow functions for batch proposal generation.

Pure functions for the batch processing pipeline using Hydra's instantiation system.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
import json
import logging
import random
import time
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, IO, Optional

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
from bookgap.llm.generators.openai_batch import OpenAIBatchGenerator
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
from bookgap.llm.prompt_record import (
    ArticleEntry, BookEntry, PromptRecord, write_prompt_record,
)
from bookgap.llm.prompt_render import render_prompts
from .run_state import RunStateManager, RunStep
from bookgap.gaps import compute_gaps_for_config


# Suppress per-request httpx INFO logging from both the OpenAI client
# (batch-job polling) and the Ollama client (per-proposal `chat` calls).
# Applied at module load so it affects every code path that issues HTTP
# traffic through httpx, not just the OpenAI batch polling loop where
# the call was historically nested.
logging.getLogger("httpx").setLevel(logging.WARNING)


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


# ---------------------------------------------------------------------------
# Streaming-metrics path (Ollama only)
# ---------------------------------------------------------------------------

_STREAMING_CSV_FILENAME = "coverage_progress.csv"
_STREAMING_CSV_FIELDNAMES = [
    "topic_id",
    "topic_name",
    "gap_size",
    "coverage_pct",
    "proposal_gap_percentile",
    "proposal_title",
]


def _stream_one_proposal_metrics(
    *,
    gap: GapWithContext,
    proposal: Proposal,
    embedder: Any,
    projection: Any,
    normalize_embeddings: bool,
    config: ExperimentConfig,
    epsilon: float = float(np.finfo(np.float32).eps * 8),
) -> dict[str, Any] | None:
    """Compute the streaming-CSV row for a single (gap, proposal) pair.

    Mirrors the per-gap subset of :func:`_compute_proposal_outrank_counts`
    (for coverage) and :func:`_compute_proposal_gap_stats` (for the
    gap-size percentile), but operates on a single proposal at a time so
    the operator can `tail -F` `coverage_progress.csv` during a long
    Ollama run. Returns ``None`` if the row cannot be computed (empty
    gap or index-build failure) — the caller skips writing in that case.
    """
    # Embed the proposal's (title, abstract) into the cloud space.
    raw = embedder.embed_texts([(proposal.title, proposal.abstract)])
    projected = projection.apply(raw)
    if normalize_embeddings and projected.shape[0] > 0:
        projected = projected / np.linalg.norm(projected, axis=1, keepdims=True)
    proposal_vec = np.asarray(projected[0], dtype=np.float32)

    # --- Coverage %: per-gap outrank logic on a single proposal. -----------
    gap_article_indices = gap.gap.get_dists_idxs().idxs
    if len(gap_article_indices) == 0:
        return None

    article_embeddings_all = gap.articles.embeddings()
    try:
        gap_article_vecs = np.ascontiguousarray(
            article_embeddings_all[np.asarray(gap_article_indices, dtype=int)],
            dtype=np.float32,
        )
    except IndexError:
        LOGGER.warning(
            "Streaming: gap (%s, %d) has invalid article indices; "
            "skipping streaming row",
            gap.topic_id, gap.gap_idx,
        )
        return None

    if gap_article_vecs.size == 0:
        return None

    try:
        book_index = gap.books.index()
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning(
            "Streaming: failed to build book index for gap (%s, %d): %s",
            gap.topic_id, gap.gap_idx, exc,
        )
        return None

    diffs = gap_article_vecs - proposal_vec
    sq_dists = np.einsum("ij,ij->i", diffs, diffs, dtype=np.float32)
    counts = np.zeros(len(gap_article_vecs), dtype=np.int32)

    def _kernel_fn(article_idx: int, neighbor_dists, _neighbor_ids) -> None:
        threshold = float(sq_dists[article_idx]) - epsilon
        if threshold <= 0.0:
            counts[article_idx] = 0
        else:
            counts[article_idx] = int(np.count_nonzero(neighbor_dists < threshold))

    range_search_radii_with(
        book_index,
        query=gap_article_vecs,
        radii=sq_dists,
        kernel_fn=_kernel_fn,
        show_progress=False,
    )

    outrank = ProposalOutrankCounts(counts=counts)
    coverage_pct = 100.0 * outrank.coverage_fraction()

    # --- Gap-size percentile: per-gap subset of gap-stats logic. -----------
    proposal_gap_percentile: float
    try:
        gap_stats_list = compute_gap_metrics_for_embeddings(
            gap=gap,
            embeddings=proposal_vec[np.newaxis, :].astype(np.float32),
            config=config,
        )
        proposal_gap_percentile = float(gap_stats_list[0].gap_size_percentile)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning(
            "Streaming: failed to compute gap percentile for (%s, %d): %s",
            gap.topic_id, gap.gap_idx, exc,
        )
        proposal_gap_percentile = float("nan")

    return {
        "topic_id": gap.topic_id,
        "topic_name": gap.topic_name,
        "gap_size": int(len(gap_article_indices)),
        "coverage_pct": coverage_pct,
        "proposal_gap_percentile": proposal_gap_percentile,
        "proposal_title": proposal.title,
    }


def _touch_streaming_csv(output_dir: Path) -> None:
    """Create ``coverage_progress.csv`` with just a header row if missing.

    Called early in :func:`run_batch_workflow` (Ollama backend only) so
    operators can ``tail -F`` the file from the start of the workflow,
    before gap detection + context-building complete. Subsequent
    :func:`_open_streaming_csv` calls preserve the file (the file already
    has a header, so the append-mode branch skips the writeheader step).
    """
    csv_path = output_dir / _STREAMING_CSV_FILENAME
    if csv_path.exists():
        return
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=_STREAMING_CSV_FIELDNAMES)
        writer.writeheader()
        fp.flush()


def _open_streaming_csv(
    output_dir: Path,
    *,
    truncate: bool = False,
) -> tuple[IO[str], csv.DictWriter]:
    """Open ``coverage_progress.csv`` and return ``(fp, writer)``.

    When ``truncate=True`` (used on resume so the CSV is regenerated
    coherently from the JSONL), the file is wiped first. Otherwise the
    file is opened for append; the header is written only if the file
    is empty or new.

    Caller is responsible for closing the returned file handle.
    """
    csv_path = output_dir / _STREAMING_CSV_FILENAME
    if truncate and csv_path.exists():
        csv_path.unlink()
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    fp = open(csv_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fp, fieldnames=_STREAMING_CSV_FIELDNAMES)
    if write_header:
        writer.writeheader()
        fp.flush()
    return fp, writer


def _write_streaming_row(
    writer: csv.DictWriter,
    fp: IO[str],
    row: dict[str, Any],
) -> None:
    """Append one row + flush so ``tail -F`` sees it immediately."""
    writer.writerow(row)
    fp.flush()


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


def build_prompt_records(
    contexts: list[ProposalContext], config: ExperimentConfig,
) -> list[PromptRecord]:
    """Build one PromptRecord per ProposalContext, populated with the
    backend name, model id, and backend-specific generation params from
    the ExperimentConfig. Returns a list parallel to ``contexts``.

    The in-memory ProposalContext doesn't carry per-book coverage stats
    (they live in the ContextRecord on disk). For PromptRecord we record
    articles_covered=0 / fraction_covered=0.0 for the books — the
    rendered prompt doesn't use those fields, and downstream consumers
    that need coverage look up the ContextRecord by (topic_id, gap_idx).
    """
    backend = _resolve_backend(config)
    if backend is LLMBackend.OLLAMA_LOCAL:
        backend_name = LLMBackend.OLLAMA_LOCAL.value
        model = str(config.llm.ollama_model)
        params = {
            "temperature": float(config.llm.temperature),
            "num_ctx": int(config.llm.ollama_num_ctx),
            "num_predict": int(config.llm.ollama_num_predict),
        }
    elif backend is LLMBackend.OPENAI_BATCH:
        backend_name = LLMBackend.OPENAI_BATCH.value
        m = config.llm.model
        model = m.value if hasattr(m, "value") else str(m)
        params = {
            "temperature": float(config.llm.temperature),
            "max_tokens": int(config.llm.max_tokens),
            "use_structured_output": bool(config.llm.use_structured_output),
        }
    else:  # pragma: no cover
        raise ValueError(f"Unknown LLM backend: {backend!r}")

    records = []
    for ctx in contexts:
        records.append(PromptRecord(
            custom_id=build_gap_proposal_id(
                gap_idx=ctx.gap_idx, topic_id=ctx.topic_id,
            ),
            topic_id=ctx.topic_id,
            topic_name=ctx.topic_name,
            gap_idx=ctx.gap_idx,
            requested_num_prompt_articles=ctx.requested_num_prompt_articles,
            requested_num_prompt_books=ctx.requested_num_prompt_books,
            sampled_articles=[
                ArticleEntry(openalex_id="", display_name=a.display_name,
                             abstract=a.abstract)
                for a in ctx.sampled_articles
            ],
            covering_books=[
                BookEntry(openalex_id="", display_name=b.display_name,
                          abstract=b.abstract,
                          articles_covered=0, fraction_covered=0.0)
                for b in ctx.covering_books
            ],
            backend=backend_name,
            model=model,
            generation_params=params,
        ))
    return records


def generate_all_prompts(
    prompt_records: list[PromptRecord], output_dir: Path,
) -> list[Path]:
    """Write each PromptRecord to ``<output_dir>/prompts/`` and return
    the list of written paths."""
    prompts_dir = output_dir / "prompts"
    return [write_prompt_record(prompts_dir, r) for r in prompt_records]


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

    # Touch the Ollama streaming CSV before gap detection so operators can
    # `tail -F coverage_progress.csv` from the very start of the workflow
    # and see the header before any rows arrive (gap detection +
    # context-building can take ~5-10 minutes on large topic sets). No-op
    # for the OpenAI backend.
    if _resolve_backend(config) is LLMBackend.OLLAMA_LOCAL:
        _touch_streaming_csv(output_dir)

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

    # ARTIFACTS: Write one PromptRecord per context (shared across backends).
    # Backends do NOT read these files at run-time — they're the on-disk
    # audit trail used by downstream consumers and the appendix generator.
    prompt_records = build_prompt_records(contexts, config)
    prompt_paths = generate_all_prompts(prompt_records, output_dir)
    LOGGER.info("Wrote %s prompt records to %s/prompts/",
                len(prompt_paths), output_dir)

    backend = _resolve_backend(config)
    if backend is LLMBackend.OPENAI_BATCH:
        results = _run_openai_batch_path(
            config=config,
            output_dir=output_dir,
            state_manager=state_manager,
            gaps=gaps,
            contexts=contexts,
            prompt_records=prompt_records,
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
            prompt_records=prompt_records,
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
    prompt_records: list[PromptRecord],
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
    requests = generate_batch_requests(contexts, prompt_records, config)
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
    prompt_records: list[PromptRecord],
    context_paths: Any,
) -> BatchResults:
    """Execute the synchronous local-Ollama proposal-generation path.

    Iterates :class:`ProposalContext` objects one at a time, calls
    :class:`OllamaProposalGenerator.generate`, appends each successful
    record to ``ollama_results.jsonl`` as it is produced (for crash
    resumability), and finally embeds the parsed proposals to produce
    the same :class:`BatchResults` shape the OpenAI path returns.

    In parallel with the JSONL persistence, this path runs a *streaming*
    metrics pass: after each successful generation it embeds the
    proposal's (title, abstract) through a hot embedder + the cloud's
    denoiser projection, computes the per-gap coverage and percentile
    metrics, and appends one row to ``coverage_progress.csv``. This
    second output gives the operator live visibility into headline
    numbers during a multi-hour run (`tail -F coverage_progress.csv`)
    while the canonical ``gaps_analysis_results.csv`` is still produced
    by the post-loop batched pipeline.

    Resumability: on restart, already-written custom IDs are skipped so
    a SIGINT'd run can pick up where it left off without re-paying for
    completed generations. The streaming CSV does NOT skip those — on
    resume, rows for already-completed records are emitted from the
    JSONL so the CSV remains a complete view of the run. JSON-parse
    failures (``record.proposal is None``) are logged and skipped — they
    do not block the run and they do not produce a streaming row.
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

    # Build a (topic_id, gap_idx) -> GapWithContext index for the
    # streaming pass. Cheap; gaps is at most a few hundred entries.
    gap_lookup: dict[tuple[str, int], GapWithContext] = {
        (gap.topic_id, gap.gap_idx): gap for gap in gaps
    }

    # Resolve the cloud's projection ONCE so the streaming loop reuses
    # the cached projection matrix instead of recomputing per iteration.
    cloud = instantiate(config.cloud)
    projection = cloud.denoiser_projection()
    normalize_embeddings = bool(cloud.normalize_embeddings)

    # On resume, emit streaming rows for pre-existing JSONL records so
    # the CSV stays a complete view. Loaded BEFORE we open the hot
    # embedder context so the resume path doesn't pay for a model load
    # if everything is already done.
    resumed_records = _load_existing_ollama_records(
        results_path, already_done,
    )

    # Open the hot embedder once for the duration of the loop. Embedding
    # one (title, abstract) per iteration costs ~1-2s; the model load is
    # what's expensive, and we amortise it across all generations.
    with Embedder(model_name=embedder_model) as embedder, \
            open(results_path, "a", encoding="utf-8") as results_fp:
        # On resume (already_done non-empty), truncate-and-rewrite the
        # streaming CSV so it stays a complete + duplicate-free view of
        # every record on disk. Fresh runs just create a new file.
        stream_fp, stream_writer = _open_streaming_csv(
            output_dir, truncate=bool(resumed_records),
        )
        try:
            # First, replay resumed records into the streaming CSV.
            for resumed_cid, resumed_proposal in resumed_records:
                gap_key = _gap_key_from_custom_id(resumed_cid)
                _maybe_write_stream_row(
                    gap_lookup=gap_lookup,
                    gap_key=gap_key,
                    proposal=resumed_proposal,
                    embedder=embedder,
                    projection=projection,
                    normalize_embeddings=normalize_embeddings,
                    config=config,
                    stream_writer=stream_writer,
                    stream_fp=stream_fp,
                    custom_id=resumed_cid,
                )

            # Then the normal generation loop.
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
                # i is 1-indexed via enumerate(..., start=1); prompt_records
                # is parallel to contexts (same length, same order).
                prompts = render_prompts(prompt_records[i - 1])
                record = generator.generate(context, prompts)
                record_dict = _ollama_record_to_jsonl_dict(
                    record, custom_id=custom_id,
                )
                results_fp.write(json.dumps(record_dict) + "\n")
                results_fp.flush()
                already_done.add(custom_id)

                # Streaming-CSV: skip rows when the model produced
                # unparseable JSON (same skip semantics as the post-loop
                # canonical pipeline).
                if record.proposal is None:
                    LOGGER.warning(
                        "Streaming: skipping CSV row for %s "
                        "(unparseable proposal)",
                        custom_id,
                    )
                    continue

                _maybe_write_stream_row(
                    gap_lookup=gap_lookup,
                    gap_key=(context.topic_id, context.gap_idx),
                    proposal=record.proposal,
                    embedder=embedder,
                    projection=projection,
                    normalize_embeddings=normalize_embeddings,
                    config=config,
                    stream_writer=stream_writer,
                    stream_fp=stream_fp,
                    custom_id=custom_id,
                )
        finally:
            stream_fp.close()

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


def _load_existing_ollama_records(
    results_path: Path, custom_ids: set[str],
) -> list[tuple[str, Proposal]]:
    """Reload (custom_id, parsed proposal) pairs from a partially-written
    ``ollama_results.jsonl``.

    Used on resume to feed the streaming-CSV pass for records that
    completed in a prior run. Records whose ``proposal`` is None (the
    model produced unparseable JSON last time) are skipped to mirror
    the live-loop skip semantics.

    Returns the list in JSONL order so the resumed streaming rows arrive
    in the same order they were originally generated.
    """
    if not results_path.exists() or not custom_ids:
        return []

    out: list[tuple[str, Proposal]] = []
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
            if not isinstance(cid, str) or cid not in custom_ids:
                continue
            proposal_dict = entry.get("proposal")
            if proposal_dict is None:
                continue
            try:
                proposal = Proposal.model_validate(proposal_dict)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning(
                    "Streaming: skipping resumed record %s (invalid "
                    "proposal payload: %s)",
                    cid, exc,
                )
                continue
            out.append((cid, proposal))
    return out


def _gap_key_from_custom_id(custom_id: str) -> tuple[str, int] | None:
    """Convert an ollama-results custom_id back to ``(topic_id, gap_idx)``.

    Returns ``None`` if the id is malformed (we log + skip the row in
    that case rather than crashing the run).
    """
    try:
        parsed = parse_gap_proposal_id(custom_id)
    except Exception as exc:
        LOGGER.warning(
            "Streaming: cannot parse custom_id %s for streaming row: %s",
            custom_id, exc,
        )
        return None
    return (parsed.topic_id, parsed.gap_idx)


def _maybe_write_stream_row(
    *,
    gap_lookup: dict[tuple[str, int], GapWithContext],
    gap_key: tuple[str, int] | None,
    proposal: Proposal,
    embedder: Any,
    projection: Any,
    normalize_embeddings: bool,
    config: ExperimentConfig,
    stream_writer: csv.DictWriter,
    stream_fp: IO[str],
    custom_id: str,
) -> None:
    """Compute and write one streaming-CSV row, logging + skipping on
    any expected non-fatal error (missing gap, computation failure)."""
    if gap_key is None:
        return
    gap = gap_lookup.get(gap_key)
    if gap is None:
        LOGGER.warning(
            "Streaming: no gap found for %s; skipping streaming row",
            custom_id,
        )
        return
    row = _stream_one_proposal_metrics(
        gap=gap,
        proposal=proposal,
        embedder=embedder,
        projection=projection,
        normalize_embeddings=normalize_embeddings,
        config=config,
    )
    if row is None:
        return
    _write_streaming_row(stream_writer, stream_fp, row)


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


def generate_batch_requests(
    contexts: list[ProposalContext],
    prompt_records: list[PromptRecord],
    config: ExperimentConfig,
) -> list[BatchRequestData]:
    """Build OpenAI batch requests from already-rendered prompts.

    ``prompt_records`` is parallel to ``contexts`` (typically the output
    of :func:`build_prompt_records`).
    """
    if not contexts:
        return []

    generator = OpenAIBatchGenerator(
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        use_structured_output=config.llm.use_structured_output,
    )
    pairs = [
        (ctx, render_prompts(rec))
        for ctx, rec in zip(contexts, prompt_records, strict=True)
    ]
    return generator.build_requests(pairs)



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
