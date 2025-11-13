"""Workflow functions for batch proposal generation.

Pure functions for the batch processing pipeline using Hydra's instantiation system.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
import json
import random
import time
import hashlib
from pathlib import Path
from typing import Optional, Any, Union

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from openai import OpenAI
import pyalex

import numpy as np
from numpy.typing import NDArray

from bookgap.configs.config import ExperimentConfig
from bookgap.cloud import Cloud, Embedder
from bookgap.configs.gap.gap_config import CellType
from bookgap.gap_statistics import compute_gap_statistics
from bookgap.llm.artifacts.common import group_gap_articles_by_covering_books
from bookgap.llm.data_structures import (
    BatchResults,
    GapWithContext,
    ProposalAnalyticsStore,
    ProposalGapStats,
    ProposalOutrankCounts,
)
from bookgap.llm.generators.batch_generator import BatchProposalGenerator
from bookgap.subcloud import SubCloud, get_subclouds_for_topic
from bookgap.util import LOGGER
from ..protocols import (
    ProposalContext, BatchRequestData,
    Article, Book, OpenAIModel, ProposalCounts, ParsedProposal, Proposal
)
from ..identifiers import parse_gap_proposal_id, GapProposalId
from bookgap.index import range_search_radii_with
from bookgap.llm.artifacts.gap_plots import generate_gap_plots_for_results
from bookgap.llm.artifacts.context_export import generate_all_contexts
from bookgap.llm.artifacts.results_export import generate_gaps_analysis_csv

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
):
    """Return gaps for the given books/articles subclouds and parameters."""
    return cell_type.value.find_gaps(
        ambient=articles,
        landmark=books,
        query=articles,
        radius_fraction=radius_fraction,
        k=k,
        is_intra_cloud=is_intra_cloud
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

    with Embedder(model_name=config.cloud.embedder_model) as embedder:
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

        gap_sizes = algorithm.compute_gap_sizes_for_embeddings(
            ambient=gap.articles,
            landmark=gap.books,
            embeddings=embeddings,
            radius_fraction=config.gap.radius_fraction,
            k=1,
            is_intra_cloud=False,
        )

        book_gap_sizes = gap.book_gap_sizes

        for proposal, gap_size in zip(proposals, gap_sizes, strict=True):
            proposal_id = proposal_id_lookup.get(proposal.custom_id)
            if proposal_id is None:
                LOGGER.warning(
                    "Unhandled proposal custom_id %s while recording gap stats",
                    proposal.custom_id,
                )
                continue

            percentile, zscore, _ = compute_gap_statistics(gap_size, book_gap_sizes)

            analytics_store.add_gap_stats(
                proposal.custom_id,
                ProposalGapStats(
                    gap_size=gap_size,
                    gap_size_percentile=percentile,
                    gap_zscore=zscore,
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
            raise ValueError("Config must be an OmegaConf DictConfig or similar object")
        config_dict = OmegaConf.to_container(config, resolve=True)
        config_json = json.dumps(config_dict, default=str, indent=None, separators=(',', ':'))
        
        # Split into chunks for metadata (512 char limit per key)
        metadata = {}
        for i, chunk_start in enumerate(range(0, len(config_json), MetadataUtils.CHUNK_SIZE)):
            chunk = config_json[chunk_start:chunk_start + MetadataUtils.CHUNK_SIZE]
            metadata[str(i)] = chunk
        
        return metadata


def find_gaps_for_config(config: ExperimentConfig) -> list[GapWithContext]:
    """Find gaps using the provided configuration.
    
    Uses the data and gap configuration to build cloud and find gaps for each topic.
    
    Args:
        config: Experiment configuration with gap detection settings
        
    Returns:
        List of GapWithContext objects found across all topics
        
    Raises:
        Exception: If gap finding fails
    """
    cloud = instantiate(config.cloud)
    
    gaps: list[GapWithContext] = []
    
    # Analyze all topics from the AI subfield
    
    subfields = pyalex.Subfields()
    subfield_obj = subfields["1702"]  # Hardcoded AI subfield (matches build_cloud)
    topics: Any = subfield_obj["topics"]  # type: ignore # pyalex typing issue
    
    if not topics:
        LOGGER.warning(f"No topics found for AI subfield")
        return gaps
    
    # Analyze all topics
    for i, topic in enumerate(topics):
        topic_name = topic["display_name"]
        topic_id = topic["id"].split("/")[-1]
        
        LOGGER.info(f"Analyzing topic {i + 1}/{len(topics)}: {topic_name} ({topic['id']})")
        
        # Get subclouds for this topic
        books, articles = get_subclouds_for_topic(cloud, topic_id)
        num_books = len(books.embeddings())
        num_articles = len(articles.embeddings())
        
        LOGGER.info(f"Topic {topic_name}: {num_books} books, {num_articles} articles")
        
        if num_articles == 0:
            LOGGER.warning(f"No articles found for topic {topic_id}")
            continue  # Skip to next topic instead of returning
        
        # Compute book gap sizes for statistics
        # Note: For book gaps, we call the algorithm directly with query=books
        book_gap_sizes = [
            len(g.get_dists_idxs().idxs)
            for g in config.gap.cell_type.value.find_gaps(
                ambient=articles,
                landmark=books,
                query=books,  # Key difference: query=books for book gaps
                radius_fraction=config.gap.radius_fraction,
                k=1,
                is_intra_cloud=True,  # Key difference: intra_cloud=True for book gaps
            )
        ]
        
        # Find gaps using configured parameters
        gap_generator = find_gaps_for_topic(
            books=books,
            articles=articles,
            radius_fraction=config.gap.radius_fraction,
            cell_type=config.gap.cell_type,
            k=1,
            is_intra_cloud=False
        )
        
        # Take the requested number of gaps for this topic using itertools.islice
        topic_gaps: list[GapWithContext] = []
        for gap in islice(gap_generator, config.batch.num_gaps):
            gap_size = len(gap.get_dists_idxs().idxs)
            LOGGER.info(f"Found gap of size {gap_size} articles")
            gap_size_percentile, gap_zscore, shapiro_log10_pval = compute_gap_statistics(gap_size, book_gap_sizes)

            topic_gaps.append(
                GapWithContext(
                    gap=gap,
                    topic_name=topic_name,
                    topic_id=topic_id,
                    gap_idx=len(gaps) + len(topic_gaps),  # Use global gap index across all topics
                    books=books,
                    articles=articles,
                    book_gap_sizes=book_gap_sizes,
                    gap_size=gap_size,
                    gap_size_percentile=gap_size_percentile,
                    gap_zscore=gap_zscore,
                    shapiro_log10_pval=shapiro_log10_pval,
                )
            )

        if len(topic_gaps) == 0:
            LOGGER.warning(f"No gaps found for topic {topic_name}")
        
        gaps.extend(topic_gaps)
 
    LOGGER.info(f"Found {len(gaps)} gaps total")
    return gaps


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

def monitor_and_download_results(batch_job_id: str, config: ExperimentConfig) -> BatchResults:
    """Monitor batch job and download results when complete.
    
    Args:
        batch_job_id: Batch job ID to monitor
        config: Experiment configuration
        
    Returns:
        BatchResults frozen dataclass containing proposals and parsed config
        
    Raises:
        Exception: If monitoring or download fails
    """
    client = OpenAI()
    
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
            raise Exception(f"Batch job failed with status: {status}")
        elif status in ["validating", "in_progress", "finalizing"]:
            # Still processing, wait and check again
            time.sleep(30)  # Wait 30 seconds before checking again
        else:
            LOGGER.warning(f"Unknown batch status: {status}")
            time.sleep(30)
    
    # Download results
    if batch.output_file_id:
        LOGGER.info("Downloading batch results")
        result_file_response = client.files.content(batch.output_file_id)
        result_content = result_file_response.content.decode('utf-8')

        # Write results to Hydra output directory
        with open(Path(HydraConfig.get().runtime.output_dir) / f"batch_results.jsonl", "w") as f:
            f.write(result_content)
        
        # Parse JSONL results into PendingProposal instances
        pending_proposals: list[PendingProposal] = []

        raw_results = result_content.strip().split('\n')

        num_errors = 0
        num_empty = 0
        num_stop_reason_length = 0

        for line in raw_results:
            if not line.strip():
                num_empty += 1
                continue
            line_json = json.loads(line)
            if line_json["response"]["body"]["choices"][0]["finish_reason"] == "length":
                num_stop_reason_length += 1
                continue
            try:
                pending = parse_batch_result_to_proposal(line_json)
                LOGGER.info(f"Parsed proposal title: {pending.proposal.title}")
                pending_proposals.append(pending)
            except Exception as e:
                num_errors += 1
                LOGGER.error(f"Failed to parse line in results: {e}. Line:\n{line}")
        
        LOGGER.info(f"Downloaded and parsed {len(pending_proposals)} proposals out of {len(raw_results)} results ({num_empty} empty results, {num_errors} errors, {num_stop_reason_length} with stop_reason='length')")

        # Compute embeddings once for all proposals and materialize ParsedProposal instances
        embeddings = _compute_proposal_embeddings(pending_proposals, config=config)

        parsed_proposals = [
            ParsedProposal(
                proposal=pending.proposal,
                custom_id=pending.custom_id,
                embedding=embedding,
            )
            for pending, embedding in zip(pending_proposals, embeddings, strict=True)
        ]

        LOGGER.info(f"Embedded {len(parsed_proposals)} proposals for downstream artifacts")
        
        expected_metadata = MetadataUtils.serialize_config(config)
        if batch.metadata != expected_metadata:
            LOGGER.error("Batch metadata does not match expected serialized config.")
            LOGGER.error(f"Expected metadata: {expected_metadata}")
            LOGGER.error(f"Actual metadata: {batch.metadata}")
            raise ValueError("Batch metadata does not match expected serialized config.")

        return BatchResults(
            proposals=parsed_proposals,
            metadata=config
        )
    else:
        # Log detailed batch information for debugging
        LOGGER.error(f"Batch completed but no output file available. Batch info:")
        LOGGER.error(f"  Batch ID: {batch.id}")
        LOGGER.error(f"  Status: {batch.status}")
        LOGGER.error(f"  Input file ID: {batch.input_file_id}")
        LOGGER.error(f"  Output file ID: {batch.output_file_id}")
        LOGGER.error(f"  Error file ID: {getattr(batch, 'error_file_id', 'N/A')}")
        if batch.request_counts:
            LOGGER.error(f"  Request counts: total={batch.request_counts.total}, completed={batch.request_counts.completed}, failed={batch.request_counts.failed}")
        
        # Check if there's an error file available
        if hasattr(batch, 'error_file_id') and batch.error_file_id:
            LOGGER.error("Downloading error file for details...")
            try:
                error_file_response = client.files.content(batch.error_file_id)
                error_content = error_file_response.content.decode('utf-8')
                LOGGER.error(f"Error file content: {error_content}")
            except Exception as e:
                LOGGER.error(f"Failed to download error file: {e}")
        
        raise Exception(f"Batch {batch_job_id} completed but no output file available. Status: {batch.status}")
    
    


def run_batch_workflow(config: ExperimentConfig) -> BatchResults:
    """Run the complete batch workflow from gap finding to results.
    
    This is the main function that coordinates the entire pipeline.
    
    Args:
        config: Experiment configuration
        
    Returns:
        BatchResults frozen dataclass containing proposals, metadata, and artifact paths
    """
    LOGGER.info(f"Starting batch workflow with {config.batch.num_gaps} gaps per topic")
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    
    # Step 1: Find gaps
    gaps = find_gaps_for_config(config)
    LOGGER.info(f"Found {len(gaps)} gaps")
    
    # ARTIFACTS: Generate context files
    context_paths = generate_all_contexts(gaps, output_dir)

    # Step 2: Build contexts
    contexts = build_contexts_for_gaps(gaps, config)
    LOGGER.info(f"Built {len(contexts)} proposal contexts")
    

    # Step 3: Generate batch requests
    requests = generate_batch_requests(contexts, config)
    LOGGER.info(f"Generated {len(requests)} batch requests")

    batch_input_filename = Path(HydraConfig.get().runtime.output_dir) / f"batch_input.jsonl"

    save_batch_input_jsonl(requests, batch_input_filename, config)

    # Step 4: Submit batch job
    batch_job_id = submit_batch_job(batch_input_filename, config)
    LOGGER.info(f"Submitted batch job: {batch_job_id}")
    
    # Step 5: Monitor and download results
    results = monitor_and_download_results(batch_job_id, config)
    LOGGER.info(f"Completed! Generated {len(results.proposals)} proposals")

    # Index proposals once for downstream artifact generation
    proposal_grouping = _index_proposals(results.proposals)

    proposal_id_lookup = {
        proposal.custom_id: parsed
        for proposal, parsed in proposal_grouping.ordered
    }
    analytics_store = ProposalAnalyticsStore()

    # Compute proposal outrank counts for coverage metrics
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
    
    # ARTIFACTS: Generate plots
    _plot_paths = generate_gap_plots_for_results(
        gaps,
        results.proposals,
        output_dir,
        proposals_by_gap=proposal_grouping.by_gap_idx,
    )

    # ARTIFACTS: Generate simple results CSV (Step 2.2 minimal)
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

    # Return enhanced results with artifacts
    return BatchResults(
        proposals=results.proposals,
        metadata=results.metadata,
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
