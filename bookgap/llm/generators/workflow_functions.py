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
from typing import List, Optional, Any, Dict, Tuple

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from openai import OpenAI
import pyalex

import numpy as np
from numpy.typing import NDArray

from bookgap.configs.config import ExperimentConfig
from bookgap.cloud import build_cloud, get_subclouds_for_topic, Embedder
from bookgap.deduplicators import Deduplicator
from bookgap.gap_finder import find_gaps_for_topic, group_gap_articles_by_covering_books
from bookgap.gap_statistics import compute_gap_statistics
from bookgap.llm.data_structures import BatchResults, GapWithContext, ArtifactPaths
from bookgap.util import LOGGER
from ..protocols import (
    ProposalContext, BatchRequestData, ProposalGenerationError, 
    Article, Book, OpenAIModel, ProposalCounts, ParsedProposal, Proposal
)
from ..identifiers import parse_gap_proposal_id, GapProposalId
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
    by_gap_idx: Dict[int, List[ParsedProposal]]
    by_topic_gap: Dict[Tuple[str, int], List[ParsedProposal]]
    ordered: List[Tuple[ParsedProposal, GapProposalId]]

def _compute_proposal_embeddings(pending: List[PendingProposal]) -> NDArray[np.float32]:
    """Compute embeddings for pending proposals in a single batch pass."""
    title_abstract_pairs = [
        (proposal.proposal.title, proposal.proposal.abstract)
        for proposal in pending
    ]

    with Embedder() as embedder:
        return embedder.embed_texts(title_abstract_pairs)


def _index_proposals(proposals: List[ParsedProposal]) -> ProposalGrouping:
    """Build indexed views of proposals keyed by gap identifiers."""
    by_gap_idx: Dict[int, List[ParsedProposal]] = defaultdict(list)
    by_topic_gap: Dict[Tuple[str, int], List[ParsedProposal]] = defaultdict(list)
    ordered: List[Tuple[ParsedProposal, GapProposalId]] = []

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

class MetadataUtils:
    """Static utilities for metadata chunking and parsing."""
    
    CHUNK_SIZE = 512  # OpenAI batch API metadata value limit
    
    @staticmethod
    def serialize_config(config: Any) -> Dict[str, str]:
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
    
    @staticmethod
    def deserialize_config(metadata: Dict[str, str]) -> ExperimentConfig:
        """Parse chunked metadata dict back to ExperimentConfig object.
        
        Args:
            metadata: Dictionary with chunked metadata from batch API
            
        Returns:
            Reconstructed ExperimentConfig object
        """
        # Reconstruct config JSON from chunks
        config_chunks = []
        i = 0
        while str(i) in metadata:
            config_chunks.append(metadata[str(i)])
            i += 1
        
        if not config_chunks:
            raise ValueError("No config chunks found in metadata")
        
        config_json = ''.join(config_chunks)
        metadata_dict = json.loads(config_json)
        
        # Reconstruct the config using hydra's instantiate
        return instantiate(metadata_dict)


def find_gaps_for_config(config: ExperimentConfig) -> List[GapWithContext]:
    """Find gaps using the provided configuration.
    
    Uses the data and gap configuration to build cloud and find gaps for each topic.
    
    Args:
        config: Experiment configuration with gap detection settings
        
    Returns:
        List of GapWithContext objects found across all topics
        
    Raises:
        Exception: If gap finding fails
    """
    LOGGER.info(f"Building cloud with denoiser={config.gap.denoiser}")
    
    deduplicator: Optional[Deduplicator] = None if config.data.deduplicator is None else instantiate(config.data.deduplicator)
    assert deduplicator is None or isinstance(deduplicator, Deduplicator), f"deduplicator must be a Deduplicator instance or None. Got {deduplicator} of type {type(deduplicator)}"

    # Build cloud from data configuration
    cloud = build_cloud(
        drop_k=config.gap.drop_k,
        max_works=config.data.max_works,
        deduplicator=deduplicator,
        denoiser=config.gap.denoiser.name,  # Convert enum to string using .name
        index_type=config.gap.index_type,
        hnsw_m=config.gap.hnsw_m,
        hnsw_efc=config.gap.hnsw_efc,
        hnsw_efs=config.gap.hnsw_efs
    )
    
    gaps: List[GapWithContext] = []
    
    # Analyze all topics from the AI subfield (matches gap_finder.main)
    
    subfields = pyalex.Subfields()
    subfield_obj = subfields["1702"]  # Hardcoded AI subfield (matches build_cloud)
    topics: Any = subfield_obj["topics"]  # type: ignore # pyalex typing issue
    
    if not topics:
        LOGGER.warning(f"No topics found for AI subfield")
        return gaps
    
    # Analyze all topics (matches gap_finder.main behavior)
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
        
        # Compute book gap sizes for statistics (matches gap_finder.write_gaps)
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


def build_contexts_for_gaps(gaps: List[GapWithContext], config: ExperimentConfig) -> List[ProposalContext]:
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

        sampled_articles: List[Article] = []
        for i, work in enumerate(sampled_article_dicts):
            display_name = work.get('display_name') if isinstance(work, dict) else getattr(work, 'display_name', None)
            abstract = work.get('abstract') if isinstance(work, dict) else getattr(work, 'abstract', None)
            if not display_name or not abstract:
                raise ValueError(
                    f"Sampled article {i} for gap {gap_with_context.gap_idx} missing display_name or abstract"
                )
            sampled_articles.append(Article(display_name=display_name, abstract=abstract))

        # Select covering books ordered by decreasing coverage, deterministic tie-breaker
        covering_books: List[Book] = []
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


def submit_batch_job(batch_requests: List[BatchRequestData], config: ExperimentConfig) -> str:
    """Submit batch job to OpenAI API.
    
    Args:
        batch_requests: List of formatted batch request data
        config: Experiment configuration with batch settings
        
    Returns:
        Batch job ID string
        
    Raises:
        Exception: If batch submission fails
    """
    if not batch_requests:
        raise ValueError("No batch requests to submit")
    
    # Initialize OpenAI client
    client = OpenAI()  # Uses OPENAI_API_KEY environment variable
    
    # Create JSONL content for batch API
    jsonl_lines = []
    
    # Prepare batch metadata with config using utility
    metadata = MetadataUtils.serialize_config(config)
    
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
        
        # Add optional fields
        if hasattr(request.body, 'temperature') and request.body.temperature is not None:
            jsonl_line["body"]["temperature"] = request.body.temperature  # type: ignore
        
        if hasattr(request.body, 'response_format') and request.body.response_format is not None:
            jsonl_line["body"]["response_format"] = request.body.response_format  # type: ignore
        
        jsonl_lines.append(json.dumps(jsonl_line))
    
    # Write JSONL to persistent file in Hydra output directory
    batch_input_filename = Path(HydraConfig.get().runtime.output_dir) / f"batch_input.jsonl"
    with open(batch_input_filename, 'w') as f:
        for line in jsonl_lines:
            f.write(line + '\n')
    
    # Upload file
    LOGGER.info(f"Uploading batch file with {len(batch_requests)} requests from {batch_input_filename}")
    with open(batch_input_filename, 'rb') as file:
        batch_input_file = client.files.create(
            file=file,
            purpose="batch"
        )
    
    # Create batch job
    LOGGER.info(f"Creating batch job with completion window: {config.batch.completion_window}")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window=config.batch.completion_window,  # type: ignore
        metadata=metadata
    )
    
    LOGGER.info(f"Batch job created successfully: {batch.id}")
    LOGGER.info(f"Batch input file saved as: {batch_input_filename}")
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
        embeddings = _compute_proposal_embeddings(pending_proposals)

        if len(embeddings) != len(pending_proposals):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(pending_proposals)}, got {len(embeddings)}"
            )

        parsed_proposals = [
            ParsedProposal(
                proposal=pending.proposal,
                custom_id=pending.custom_id,
                embedding=embedding,
            )
            for pending, embedding in zip(pending_proposals, embeddings)
        ]

        LOGGER.info(f"Embedded {len(parsed_proposals)} proposals for downstream artifacts")
        
        # Parse original config from metadata using MetadataUtils
        metadata = batch.metadata or {}
        try:
            parsed_config = MetadataUtils.deserialize_config(metadata)
        except (ProposalGenerationError, KeyError):
            # Fall back to the provided config if metadata parsing fails
            LOGGER.warning("Failed to parse metadata, using provided config")
            parsed_config = config
        
        return BatchResults(
            proposals=parsed_proposals,
            metadata=parsed_config
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
    
    # Step 4: Submit batch job
    batch_job_id = submit_batch_job(requests, config)
    LOGGER.info(f"Submitted batch job: {batch_job_id}")
    
    # Step 5: Monitor and download results
    results = monitor_and_download_results(batch_job_id, config)
    LOGGER.info(f"Completed! Generated {len(results.proposals)} proposals")

    # Index proposals once for downstream artifact generation
    proposal_grouping = _index_proposals(results.proposals)
    
    # ARTIFACTS: Generate plots
    plot_paths = generate_gap_plots_for_results(
        gaps,
        results.proposals,
        output_dir,
        proposals_by_gap=proposal_grouping.by_gap_idx,
    )

    # ARTIFACTS: Generate simple results CSV (Step 2.2 minimal)
    results_csv = generate_gaps_analysis_csv(
        gaps,
        results.proposals,
        output_dir,
        config,
        contexts,
        proposals_grouped=proposal_grouping.by_topic_gap,
        ordered_proposals=proposal_grouping.ordered,
    )
    
    
    artifact_paths = ArtifactPaths(plots=plot_paths, context_files=context_paths, results_csv=results_csv)
    
    # Return enhanced results with artifacts
    return BatchResults(
        proposals=results.proposals,
        metadata=results.metadata,
        artifact_paths=artifact_paths
    )


def generate_batch_requests(contexts: List[ProposalContext], config: ExperimentConfig) -> List[BatchRequestData]:
    """Generate batch API requests from proposal contexts.
    
    Uses Hydra's _target_ pattern to instantiate the appropriate generator for the provider.
    
    Args:
        contexts: List of proposal contexts
        config: Experiment configuration with LLM settings
        
    Returns:
        List of batch API request data structures
        
    Raises:
        ProposalGenerationError: If generator instantiation or generation fails
        
    Example:
        ```python
        from hydra import compose, initialize
        from bookgap.llm.generators.workflow_functions import generate_batch_requests
        
        # Using Hydra config
        with initialize(config_path="../configs"):
            config = compose(config_name="production")
            requests = generate_batch_requests(contexts, config)
        
        # Manual config (for testing)
        config = ExperimentConfig(
            llm=LLMConfig(
                _target_="bookgap.llm.generators.openai_batch.OpenAIBatchGenerator",
                model_name=OpenAIModel.GPT_4O_MINI,
                # ... other required fields
            ),
            # ... other config fields
        )
        requests = generate_batch_requests(contexts, config)
        ```
    """
    if not contexts:
        return []
    
    try:
        generator = instantiate(config.llm)
        
        # Generate batch requests
        return generator.generate_batch_requests(contexts)
        
    except Exception as e: # TODO: Why don't we just let the exception propagate? Do we need ProposalGenerationError at all?
        # Wrap any errors in ProposalGenerationError with context
        error_context = {
            "num_contexts": len(contexts),
            "model": str(config.llm.model),
            "target": config.llm._target_
        }
        
        # If it's already a ProposalGenerationError, preserve the original
        if isinstance(e, ProposalGenerationError):
            # Add workflow context to existing error context
            e.context.update(error_context)
            raise e
        else:
            # Wrap other exceptions
            raise ProposalGenerationError(
                f"Failed to generate batch requests: {str(e)}",
                context=error_context
            ) from e


def validate_batch_requests(requests: List[BatchRequestData]) -> None:
    """Validate that batch requests are properly formatted.
    
    Args:
        requests: List of batch request data structures
        
    Raises:
        ProposalGenerationError: If validation fails
    """
    if not requests:
        raise ProposalGenerationError(
            "No batch requests provided",
            context={"validation_step": "empty_check"}
        )
    
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
        raise ProposalGenerationError(
            f"Batch request validation failed: {'; '.join(errors)}",
            context={
                "num_requests": len(requests),
                "num_errors": len(errors),
                "validation_errors": errors
            }
        )


def get_supported_models() -> List[str]:
    """Get list of supported model names.
    
    Returns:
        List of supported OpenAI model names
    """
    return [model.value for model in OpenAIModel]
