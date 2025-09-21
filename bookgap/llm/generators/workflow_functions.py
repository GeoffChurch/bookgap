"""Workflow functions for batch proposal generation.

Pure functions for the batch processing pipeline using Hydra's instantiation system.
"""

from dataclasses import dataclass, asdict
from itertools import islice
import json
import os
import random
import time
import tempfile
from typing import List, Any, Dict, Optional

from hydra.utils import instantiate
from openai import OpenAI
import pyalex

from bookgap.configs.config import ExperimentConfig
from bookgap.cloud import build_cloud, get_subclouds_for_topic, SubCloud
from bookgap.deduplicators import Deduplicator
from bookgap.gap_finder import find_gaps_for_topic, GapProtocol
from bookgap.llm.generators.openai_batch import OpenAIBatchGenerator
from bookgap.util import LOGGER
from pydantic import ValidationError
from ..protocols import (
    ProposalContext, BatchRequestData, ProposalGenerationError, 
    Article, Book, OpenAIModel, ProposalCounts, ParsedProposal, Proposal
)


@dataclass(frozen=True)
class BatchResults:
    """Immutable container for batch processing results."""
    proposals: List[ParsedProposal]
    metadata: ExperimentConfig


class MetadataUtils:
    """Static utilities for metadata chunking and parsing."""
    
    CHUNK_SIZE = 512  # OpenAI batch API metadata value limit
    
    @staticmethod
    def serialize_config(config: ExperimentConfig) -> Dict[str, str]:
        """Convert ExperimentConfig to chunked metadata dict for batch API.
        
        Args:
            config: ExperimentConfig to serialize
            
        Returns:
            Dictionary with chunked metadata (keys are string indices "0", "1", etc.)
        """
        config_dict = asdict(config)
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
            
        Raises:
            ProposalGenerationError: If parsing fails
        """
        try:
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
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            LOGGER.error(f"Failed to parse metadata: {e}")
            raise ProposalGenerationError(f"Invalid metadata format: {e}")

@dataclass(frozen=True)
class GapWithContext:
    """Gap with additional context for workflow processing."""
    gap: GapProtocol
    topic_name: str
    topic_id: str
    gap_idx: int
    books: SubCloud
    articles: SubCloud


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
        topic_gaps = [
            GapWithContext(
                gap=gap,
                topic_name=topic_name,
                topic_id=topic_id,
                gap_idx=len(gaps),  # Use global gap index across all topics
                books=books,
                articles=articles
            )
            for gap in islice(gap_generator, config.batch.num_gaps)]
        
        
        if len(topic_gaps) == 0:
            LOGGER.warning(f"No gaps found for topic {topic_name}")
        else:
            LOGGER.info(f"Found {len(topic_gaps)} gaps for topic {topic_name}")
        
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
        gap = gap_with_context.gap
        gap_size = len(gap.get_dists_idxs().idxs)
        
        # Get work metadata
        article_works = gap_with_context.articles.works()
        book_works = gap_with_context.books.works()
        
        # Sample random articles from the gap (up to configured limit)
        sample_size = min(config.llm.num_articles, gap_size)
        sampled_indices = random.sample(range(gap_size), sample_size)
        sampled_article_works = [article_works[gap.get_dists_idxs().idxs[i]] for i in sampled_indices]
        
        # Convert to Article dataclasses
        sampled_articles = [
            Article(
                display_name=work.get('display_name', 'Unknown'),
                abstract=work.get('abstract', '')
            )
            for work in sampled_article_works
        ]
        
        # Find covering books (simplified logic for now)
        covering_books = []
        if len(book_works) > 0:
            # Take up to requested number of books
            num_books_to_sample = min(config.llm.num_books, len(book_works))
            sampled_book_works = random.sample(book_works, num_books_to_sample)
            
            covering_books = [
                Book(
                    display_name=work.get('display_name', 'Unknown'),
                    abstract=work.get('abstract', '')
                )
                for work in sampled_book_works
            ]
        
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
    
    # Write JSONL to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for line in jsonl_lines:
            f.write(line + '\n')
        temp_file_path = f.name
    
    try:
        # Upload file
        LOGGER.info(f"Uploading batch file with {len(batch_requests)} requests")
        with open(temp_file_path, 'rb') as file:
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
        return batch.id
        
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)


def parse_batch_result_to_proposal(result: dict[str, Any]) -> ParsedProposal:
    """Parse OpenAI batch result into ParsedProposal instance.
    
    Args:
        result: Raw batch result dict from OpenAI API
        
    Returns:
        ParsedProposal instance or None if parsing fails
    """
    # Extract the JSON content from OpenAI batch result structure
    content = result["response"]["body"]["choices"][0]["message"]["content"]
    
    # Parse using Pydantic model for validation
    proposal = Proposal.model_validate_json(content)
    
    # Wrap with metadata
    parsed_proposal = ParsedProposal(
        proposal=proposal,
        custom_id=result["custom_id"]
    )
    
    return parsed_proposal

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
        
        # Parse JSONL results into ParsedProposal instances
        results = []
        
        for line in result_content.strip().split('\n'):
            if line:
                # Parse into ParsedProposal instance
                parsed_proposal = parse_batch_result_to_proposal(json.loads(line))
                results.append(parsed_proposal)
        
        LOGGER.info(f"Downloaded and parsed {len(results)} proposals")
        
        # Parse original config from metadata using MetadataUtils
        metadata = batch.metadata or {}
        try:
            parsed_config = MetadataUtils.deserialize_config(metadata)
        except (ProposalGenerationError, KeyError):
            # Fall back to the provided config if metadata parsing fails
            LOGGER.warning("Failed to parse metadata, using provided config")
            parsed_config = config
        
        return BatchResults(
            proposals=results,
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
        BatchResults frozen dataclass containing proposals and metadata
    """
    LOGGER.info(f"Starting batch workflow with {config.batch.num_gaps} gaps per topic")
    
    # Step 1: Find gaps
    gaps = find_gaps_for_config(config)
    LOGGER.info(f"Found {len(gaps)} gaps")
    
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
    
    return results


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
