#!/usr/bin/env python3
"""
Shared batch operations for OpenAI Batch API.

This module provides common functionality for monitoring, downloading, and processing
OpenAI batch jobs, used by both send_prompts_to_openai.py and check_batch_status.py.
"""

import json
import time
import pandas as pd
import os
import numpy as np
from pathlib import Path
from typing import Any, Optional, Callable
from openai import OpenAI
from pylatexenc.latexencode import unicode_to_latex
from dataclasses import dataclass

from bookgap.cloud import get_embedding_model, Cloud, get_subclouds_for_topic
from bookgap.cloud_index import compute_proposal_rankings_for_gap
from bookgap.util import LOGGER
from bookgap.gap_finder import find_gaps_for_topic, save_gap_plot_with_proposals, save_gap_books_vs_proposals_plot
from bookgap.gap_directory_manager import GapDirectoryManager
from bookgap.results_manager import DataFrameResults

# Import centralized configuration management
from bookgap.config_manager import (
    AnalysisConfiguration,
    GapConfiguration,
    LLMConfiguration,
    CustomIdComponents,
    CustomIdManager, 
    TopicPathManager, 
    ConfigPathParser
)

# Use centralized functions instead of local definitions
build_custom_id = CustomIdManager.build_custom_id
parse_custom_id = CustomIdManager.parse_custom_id  
extract_topic_name_from_filename = TopicPathManager.extract_topic_name_from_filename

# Configuration - now managed by GapDirectoryManager
ARTIFACTS_DIR = GapDirectoryManager.get_base_artifacts_dir()
GAPS_DIR = GapDirectoryManager.get_gaps_base_dir()


def validate_batch_results_dataframe(df: pd.DataFrame, operation_name: str = "batch_results") -> None:
    """
    Validate batch results DataFrame using unified framework validation patterns.
    
    Args:
        df: DataFrame to validate
        operation_name: Name of the operation for error reporting
        
    Raises:
        ValidationError: If DataFrame doesn't meet expected criteria (raised by DataFrameResults.validate_data)
    """
    # Use DataFrameResults for validation - creates temporary instance for validation
    results = DataFrameResults(df, name=operation_name)
    results.validate_data(operation_name)


def process_dataframe_numeric_columns(df: pd.DataFrame, significant_figures: int = 4) -> pd.DataFrame:
    """
    Process DataFrame numeric columns using unified framework patterns.
    
    This uses DataFrameResults.process_numeric_columns() for consistent processing.
    
    Args:
        df: DataFrame to process
        significant_figures: Number of significant figures to maintain
        
    Returns:
        Processed DataFrame with truncated floating-point values
    """
    # Use DataFrameResults for processing - creates temporary instance for processing
    results = DataFrameResults(df, name="numeric_processing")
    processed_results = results.process_numeric_columns(significant_figures=significant_figures)
    return processed_results.get_df()


def export_dataframe_with_unified_patterns(df: pd.DataFrame, base_path: Path, 
                                           formats: list[str] = ['csv', 'json', 'parquet']) -> list[Path]:
    """
    Export DataFrame using unified framework patterns.
    
    This uses DataFrameResults.export_multiple_formats() for consistent export.
    
    Args:
        df: DataFrame to export
        base_path: Base path for output files (extension will be replaced)
        formats: List of formats to export ('csv', 'json', 'parquet')
        
    Returns:
        List of exported file paths
    """
    # Use DataFrameResults for export - creates temporary instance for export
    results = DataFrameResults(df, name="batch_export")
    exported_paths_list = results.export_multiple_formats(base_path, formats)
    
    return exported_paths_list





def get_config_dir_from_custom_id(custom_id: str) -> Path:
    """
    Get the configuration directory from a custom_id using centralized config management.
    
    Args:
        custom_id: Custom ID string to parse
        
    Returns:
        Path to the configuration directory
        
    Raises:
        ValueError: If custom_id cannot be parsed
    """
    parsed_id = parse_custom_id(custom_id)
    # Create AnalysisConfiguration using structured approach
    gap_configuration = GapConfiguration.from_directory_name(parsed_id.gap_config)
    llm_configuration = LLMConfiguration.from_subdir_name(parsed_id.llm_subdir)
    analysis_config = AnalysisConfiguration.from_components(gap_configuration, llm_configuration)
    return GapDirectoryManager.get_config_dir(analysis_config)


def get_config_dir_from_results_file(results_file: Path) -> Path:
    """
    Get the configuration directory from a batch results file.
    
    Args:
        results_file: Path to the JSONL batch results file
        
    Returns:
        Path to the configuration directory
        
    Raises:
        ValueError: If results file is empty or custom_id cannot be parsed
        FileNotFoundError: If results file doesn't exist
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if not first_line:
            LOGGER.error(f"Results file is empty: {results_file}")
            raise ValueError(f"Results file is empty: {results_file}")
        
        result = json.loads(first_line)
        return get_config_dir_from_custom_id(result['custom_id'])


def find_prompt_files() -> list[Path]:
    """Find all _prompt.txt files in the gaps directory."""
    prompt_files = []
    if not GAPS_DIR.exists():
        return prompt_files
    
    for root, _dirs, files in os.walk(GAPS_DIR):
        for file in files:
            if file.endswith("_prompt.txt"):
                prompt_files.append(Path(root) / file)
    return sorted(prompt_files)


def read_prompt_content(prompt_file: Path) -> str:
    """Read the content of a prompt file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()


def create_batch_requests_from_prompts(prompt_files: list[Path]) -> list[dict[str, Any]]:
    """
    Create batch request objects for OpenAI API from prompt files.
    
    Args:
        prompt_files: List of paths to prompt files
        
    Returns:
        List of batch request dictionaries
    """
    requests = []
    
    for prompt_file in prompt_files:
        # Use ConfigPathParser to extract all configuration information
        parsed_info = ConfigPathParser.parse_prompt_file_path(str(prompt_file), str(GAPS_DIR))
        
        gap_config = parsed_info['gap_config']
        llm_config = parsed_info['llm_config'] 
        topic_name = parsed_info['topic_name']
        topic_id = parsed_info['topic_id']
        gap_idx = parsed_info['gap_idx']

        # Create AnalysisConfiguration from parsed information using structured approach
        gap_configuration = GapConfiguration.from_directory_name(gap_config)
        llm_configuration = LLMConfiguration.from_subdir_name(llm_config)
        analysis_config = AnalysisConfiguration.from_components(gap_configuration, llm_configuration)
        
        # Read prompt content
        try:
            prompt_content = read_prompt_content(prompt_file)
        except Exception as e:
            LOGGER.error(f"Error reading {prompt_file}: {e}")
            continue
        
        # Create request body with proper message structure
        body = {
            "model": llm_configuration.model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a publishing expert and academic researcher."
                },
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
            "max_completion_tokens": llm_configuration.max_tokens  # Updated to max_completion_tokens
        }
        
        # Add temperature only for non-GPT-5 models
        if not llm_configuration.model.startswith("gpt-5"):
            body["temperature"] = llm_configuration.temperature
        
        # Add structured output format if using STRUCTURED implementation
        if llm_configuration.implementation.value == "structured":
            from openai.lib._pydantic import to_strict_json_schema
            from bookgap.llm.protocols import create_proposal_model
            
            # Create dynamic proposal model with configuration-specific constraints
            ProposalModel = create_proposal_model(llm_configuration.chapters_min, llm_configuration.chapters_max)
            schema = to_strict_json_schema(ProposalModel)
            
            # Wrap in the format expected by batch API
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "textbook_proposal",
                    "strict": True,
                    "schema": schema
                }
            }
        
        request = {
            "custom_id": build_custom_id(analysis_config, topic_name, topic_id, gap_idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }
        requests.append(request)
    
    return requests


def create_batch_file_from_requests(requests: list[dict[str, Any]], config: AnalysisConfiguration) -> Path:
    """
    Create a JSONL file with all batch requests.
    
    Args:
        requests: List of batch request dictionaries
        config: Analysis configuration for determining output location
        
    Returns:
        Path to the created batch file
    """
    timestamp = int(time.time())
    batch_file = GapDirectoryManager.get_batch_request_file(config, timestamp)
    
    with open(batch_file, 'w', encoding='utf-8') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    
    LOGGER.info(f"Created batch file: {batch_file}")
    LOGGER.info(f"Batch file size: {batch_file.stat().st_size / 1024:.1f} KB")
    return batch_file


def upload_and_submit_batch(batch_file: Path, description: str = "Textbook proposals for research gaps") -> str:
    """
    Upload batch file and submit batch job to OpenAI.
    
    Args:
        batch_file: Path to the batch file to upload
        description: Description for the batch job
        
    Returns:
        Batch ID
        
    Raises:
        Exception: If upload or submission fails
    """
    # Upload file
    LOGGER.info(f"Uploading batch file to OpenAI...")
    with open(batch_file, 'rb') as f:
        file_response = get_client().files.create(
            file=f,
            purpose="batch"
        )
    
    file_id = file_response.id
    LOGGER.info(f"Uploaded file ID: {file_id}")
    
    # Create batch job
    LOGGER.info("Creating batch job...")
    batch_response = get_client().batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",  # Complete within 24 hours
        metadata={
            "description": description,
            "project": "bookgap"
        }
    )
    
    batch_id = batch_response.id
    LOGGER.info(f"Created batch job ID: {batch_id}")
    return batch_id


def estimate_batch_cost(prompt_lengths: list[int]) -> None:
    """
    Estimate the cost of the batch job based on actual prompt lengths.
    
    Args:
        prompt_lengths: List of prompt lengths in characters
    """
    # Convert character counts to approximate token counts (rough estimate: 4 chars per token)
    prompt_tokens = [length // 4 for length in prompt_lengths]
    
    # Calculate statistics
    prompt_tokens_array = np.array(prompt_tokens)
    num_requests = len(prompt_tokens)
    
    if num_requests == 0:
        LOGGER.info("No requests to estimate cost for.")
        return
    
    min_tokens = np.min(prompt_tokens_array)
    max_tokens = np.max(prompt_tokens_array)
    mean_tokens = np.mean(prompt_tokens_array)
    
    # Calculate percentiles
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    percentile_values = np.percentile(prompt_tokens_array, percentiles)
    
    LOGGER.info(f"\nPrompt length analysis ({num_requests} prompts):")
    LOGGER.info(f"Min tokens: {min_tokens:,}")
    for p, val in zip(percentiles, percentile_values):
        LOGGER.info(f"{p}th percentile: {val:,.0f} tokens")
    LOGGER.info(f"Max tokens: {max_tokens:,}")
    LOGGER.info(f"Mean tokens: {mean_tokens:,.0f}")
    
    # GPT-4.1 pricing: $2.00 per 1M input tokens, $8.00 per 1M output tokens
    # Estimate ~800 output tokens per response (for textbook proposals)
    total_input_tokens = sum(prompt_tokens)
    estimated_output_tokens = num_requests * 800
    
    input_cost = (total_input_tokens / 1_000_000) * 2.00
    output_cost = (estimated_output_tokens / 1_000_000) * 8.00
    total_cost = input_cost + output_cost
    
    LOGGER.info(f"\nCost estimate (GPT-4.1):")
    LOGGER.info(f"Total input tokens: {total_input_tokens:,}")
    LOGGER.info(f"Estimated output tokens: {estimated_output_tokens:,}")
    LOGGER.info(f"Estimated input cost: ${input_cost:.3f}")
    LOGGER.info(f"Estimated output cost: ${output_cost:.3f}")
    LOGGER.info(f"Estimated total cost: ${total_cost:.3f}")
    LOGGER.info(f"Note: Batch API provides 50% discount, so actual cost: ${total_cost * 0.5:.3f}")


def extract_proposal_data_from_result(result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Extract proposal data (title, abstract) and parsed custom_id from a batch result.
    
    Args:
        result: Single result from batch results JSONL
        
    Returns:
        Dictionary with proposal data and custom_id components, or None if failed
    """
    custom_id = result['custom_id']
    parsed_id = parse_custom_id(custom_id)
    
    if not parsed_id:
        return None
    
    proposal_data = {
        'topic_name': parsed_id.topic_name,
        'topic_id': parsed_id.topic_id,
        'denoiser': parsed_id.denoiser,
        'drop_k': parsed_id.drop_k,
        'gap_idx': parsed_id.gap_idx,
        'proposal_title': '',
        'proposal_abstract': ''
    }
    
    # Extract title and abstract if response was successful
    if result.get('response') and result['response'].get('status_code') == 200:
        try:
            raw_message = result['response']['body']['choices'][0]['message']['content']
            cleaned_json = clean_json_content(raw_message)
            json_data = json.loads(cleaned_json)
            
            proposal_data['proposal_title'] = json_data.get('title', '')
            proposal_data['proposal_abstract'] = json_data.get('abstract', '')
            
        except (json.JSONDecodeError, KeyError, TypeError):
            # Keep empty strings for failed parsing
            pass
    
    return proposal_data

def find_batch_request_files_for_config(config: AnalysisConfiguration) -> list[Path]:
    """Find batch request files for a specific configuration."""
    batch_requests_dir = GapDirectoryManager.get_batch_requests_dir(config)
    
    if not batch_requests_dir.exists():
        return []
    
    pattern = "batch_request_*.jsonl"
    batch_files = list(batch_requests_dir.glob(pattern))
    return sorted(batch_files, key=lambda x: x.stat().st_mtime)

def find_batch_results_files_for_config(config: AnalysisConfiguration) -> list[Path]:
    """Find batch results files for a specific configuration."""
    batch_results_dir = GapDirectoryManager.get_batch_results_dir(config)
    
    if not batch_results_dir.exists():
        return []
    
    pattern = "results_*.jsonl"
    results_files = list(batch_results_dir.glob(pattern))
    return sorted(results_files, key=lambda x: x.stat().st_mtime)

def find_prompt_files_for_config(config: AnalysisConfiguration) -> list[Path]:
    """Find prompt files for a specific configuration."""
    config_dir = GapDirectoryManager.get_config_dir(config)
    prompt_files = []
    
    if not config_dir.exists():
        return prompt_files
    
    # Search for _prompt.txt files in the LLM-specific prompt directories
    llm_subdir_name = config.to_llm_subdir_name()
    for gap_dir in config_dir.glob("gap_*"):
        if gap_dir.is_dir():
            prompts_llm_dir = gap_dir / "prompts" / llm_subdir_name
            if prompts_llm_dir.exists():
                for file in prompts_llm_dir.glob("*_prompt.txt"):
                    prompt_files.append(file)
    
    return sorted(prompt_files)

def find_all_batch_request_files() -> list[Path]:
    """Find all batch request files across all configurations and LLM settings."""
    batch_files = []
    if not GAPS_DIR.exists():
        return batch_files
    
    for config_dir in GAPS_DIR.iterdir():
        if config_dir.is_dir():
            # Look for LLM subdirectories (e.g., articles_10_books_5)
            for llm_dir in config_dir.iterdir():
                if llm_dir.is_dir():# and llm_dir.name.startswith('articles_'):
                    if not llm_dir.name.startswith('articles_'):
                        raise ValueError(f"Unexpected LLM directory name: {llm_dir.name}")
                    batch_requests_dir = llm_dir / "batch_requests"
                    if batch_requests_dir.exists():
                        pattern = "batch_request_*.jsonl"
                        batch_files.extend(batch_requests_dir.glob(pattern))
    
    return sorted(batch_files, key=lambda x: x.stat().st_mtime)

def find_all_batch_results_files() -> list[Path]:
    """Find all batch results files across all configurations and LLM settings."""
    results_files = []
    if not GAPS_DIR.exists():
        return results_files
    
    for config_dir in GAPS_DIR.iterdir():
        if config_dir.is_dir():
            # Look for LLM subdirectories (e.g., articles_10_books_5)
            for llm_dir in config_dir.iterdir():
                if llm_dir.is_dir():# and llm_dir.name.startswith('articles_'):
                    if not llm_dir.name.startswith('articles_'):
                        raise ValueError(f"Unexpected LLM directory name: {llm_dir.name}")
                    batch_results_dir = llm_dir / "batch_results"
                    if batch_results_dir.exists():
                        pattern = "results_*.jsonl"
                        results_files.extend(batch_results_dir.glob(pattern))
    
    return sorted(results_files, key=lambda x: x.stat().st_mtime)

# OpenAI client singleton
_client = None

def get_client() -> OpenAI:
    """Get or create the OpenAI client singleton."""
    global _client
    if _client is None:
        _client = OpenAI()  # Requires OPENAI_API_KEY environment variable
    return _client

def monitor_batch_job(batch_id: str, poll_interval: int = 30) -> Any:
    """
    Monitor a batch job until completion.
    
    Args:
        batch_id: The batch job ID to monitor
        poll_interval: Seconds to wait between status checks
        
    Returns:
        The final batch status object
        
    Raises:
        Exception: If monitoring fails repeatedly
    """
    LOGGER.info(f"Monitoring batch job {batch_id}...")
    
    while True:
        batch_status = get_client().batches.retrieve(batch_id)
        status = batch_status.status
        
        LOGGER.info(f"Batch status: {status}")
        
        if status == "completed":
            LOGGER.info("Batch completed successfully!")
            return batch_status
        elif status == "failed":
            LOGGER.info("Batch failed!")
            if batch_status.errors:
                LOGGER.error(f"Error details: {batch_status.errors}")
            return batch_status
        elif status == "cancelled":
            LOGGER.info("Batch was cancelled!")
            return batch_status
        elif status in ["validating", "in_progress", "finalizing"]:
            LOGGER.info(f"Batch is {status}... waiting {poll_interval} seconds")
            time.sleep(poll_interval)
        else:
            LOGGER.warning(f"Unknown status: {status}")
            time.sleep(poll_interval)

def download_batch_results(batch_status: Any, results_file_path: Optional[Path] = None) -> Path:
    """
    Download the results file from a completed batch job.
    
    Args:
        batch_status: The batch status object from OpenAI
        results_file_path: Optional path where to save the results. If not provided,
                          will attempt to determine proper location from custom_id in results.
        
    Returns:
        Path to the downloaded results file
        
    Raises:
        ValueError: If batch is not completed or no output file ID
        RuntimeError: If unable to determine proper location from results
    """
    if batch_status.status != "completed":
        raise ValueError(f"Batch not completed. Status: {batch_status.status}")
    
    output_file_id = batch_status.output_file_id
    if not output_file_id:
        raise ValueError("No output file ID found")
    
    LOGGER.info(f"Downloading results from file ID: {output_file_id}")
    
    try:
        # Download the results file
        file_response = get_client().files.content(output_file_id)
        
        if results_file_path is None:
            # Attempt to determine proper location from first JSON record
            content_str = file_response.content.decode('utf-8')
            lines = content_str.strip().split('\n')
            
            first_result = json.loads(lines[0])
            custom_id = first_result["custom_id"]
            
            # Parse custom_id to get configuration information
            parsed_id = parse_custom_id(custom_id)
            if not parsed_id:
                raise RuntimeError(f"Unable to parse custom_id: {custom_id}")
            
            # Create AnalysisConfiguration from parsed components
            gap_configuration = GapConfiguration.from_directory_name(parsed_id.gap_config)
            llm_configuration = LLMConfiguration.from_subdir_name(parsed_id.llm_subdir)
            analysis_config = AnalysisConfiguration.from_components(gap_configuration, llm_configuration)
            
            # Get the proper results file path
            results_file_path = GapDirectoryManager.get_batch_results_file(analysis_config, batch_status.id)
            
            LOGGER.info(f"Determined results location from custom_id: {results_file_path}")
                
        else:
            GapDirectoryManager.ensure_directory(results_file_path.parent)
        
        # Save the results file
        with open(results_file_path, 'wb') as f:
            f.write(file_response.content)
        
        LOGGER.info(f"Results saved to: {results_file_path}")
        return results_file_path
        
    except Exception as e:
        LOGGER.error(f"Error downloading results: {e}")
        raise

def clean_json_content(content: str) -> str:
    """
    Clean JSON content by removing leading and trailing code block markers.
    
    Args:
        content: Raw content from OpenAI response
        
    Returns:
        Cleaned JSON content with leading/trailing code block markers removed
    """
    content = content.strip()
    
    # Remove various code block markers
    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    elif content.startswith("```"):
        content = content[len("```"):].strip()

    if content.endswith("```"):
        content = content[:-len("```")].strip()
    
    return content.strip()

def json_to_latex(json_data: dict) -> str:
    """
    Convert JSON proposal data to LaTeX format.
    
    Args:
        json_data: Dictionary containing title, abstract, and chapters
        
    Returns:
        LaTeX formatted string
    """
    latex_content = []
    
    # Title using custom command
    title = json_data.get('title', 'Untitled Textbook')
    escaped_title = unicode_to_latex(title)
    latex_content.append(f"\\proposalTitle{{{escaped_title}}}")
    
    # Abstract using custom command
    abstract = json_data.get('abstract', 'No abstract provided.')
    escaped_abstract = unicode_to_latex(abstract)
    latex_content.append(f"\\proposalAbstract{{{escaped_abstract}}}")
    
    # Chapters using the simpler approach with manual formatting
    chapters = json_data.get('chapters', [])
    if chapters:
        # Build the chapter items manually
        chapter_items = []
        for chapter in chapters:
            chapter_title = chapter.get('chapter_title', 'Untitled Chapter')
            chapter_desc = chapter.get('chapter_description', 'No description provided.')
            escaped_title = unicode_to_latex(chapter_title)
            escaped_desc = unicode_to_latex(chapter_desc)
            chapter_items.append(f"\\chapterItem{{{escaped_title}}}{{{escaped_desc}}}")
        
        # Create the command call with formatted items
        chapters_content = '\n        '.join(chapter_items)
        latex_content.append(f"\\proposalChapters{{\n        {chapters_content}\n    }}")
    else:
        # Handle case with no chapters
        latex_content.append("\\proposalChapters{}")
    
    return '\n\n'.join(latex_content) + '\n'

def clean_latex_content(content: str) -> str:
    """
    Clean LaTeX content by removing leading and trailing code block markers.
    
    Args:
        content: Raw content from OpenAI response
        
    Returns:
        Cleaned LaTeX content with leading/trailing code block markers removed
    """
    content = content.strip()
    
    if content.startswith("```latex"):
        content = content[len("```latex"):].strip()
    
    if content.startswith("```"):
        content = content[len("```"):].strip()

    if content.endswith("```"):
        content = content[:-len("```")].strip()
    
    return content.strip()

def create_gap_analysis_with_proposals_csv(results_file: Path, cloud: Cloud, config: AnalysisConfiguration,
                                           generate_latex_table: bool = True,
                                           config_dir: Optional[Path] = None,
                                           precomputed_proposals_df: Optional[pd.DataFrame] = None) -> Path:
    """
    Create gap_analysis_with_proposals.csv by joining gap analysis with proposals.
    If precomputed_proposals_df is provided, avoid re-parsing and re-embedding.
    """
    if config_dir is None:
        config_dir = get_config_dir_from_results_file(results_file)
    
    # Find gap_analysis.csv using LLM-specific location
    gap_analysis_csv = _get_gap_analysis_csv_path(results_file)
    if not gap_analysis_csv.exists():
        raise FileNotFoundError(f"gap_analysis.csv not found at {gap_analysis_csv}")
    
    # Use LLM-specific directory for output (same as gap_analysis.csv location)
    llm_specific_dir = gap_analysis_csv.parent
    output_path = llm_specific_dir / "gap_analysis_with_proposals.csv"
    return _merge_gap_analysis_with_proposals(gap_analysis_csv, results_file, output_path, cloud, config, generate_latex_table, precomputed_proposals_df)

def _get_gap_analysis_csv_path(results_file: Path) -> Path:
    """
    Get the correct gap_analysis.csv path from batch results file.
    
    Extracts configuration from results file to determine LLM-specific location.
    """
    with open(results_file, 'r') as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError(f"Empty results file: {results_file}")
        
        data = json.loads(first_line)
        custom_id = data.get('custom_id', '')
        if not custom_id:
            raise ValueError(f"No custom_id found in results file: {results_file}")
        
        # Parse custom_id to get configuration
        components = CustomIdManager.parse_custom_id(custom_id)
        
        # Parse gap_config string into GapConfiguration
        gap_config = GapConfiguration.from_directory_name(components.gap_config)
        
        # Parse llm_subdir into LLMConfiguration
        llm_config = LLMConfiguration.from_subdir_name(components.llm_subdir)
        
        analysis_config = AnalysisConfiguration.from_components(gap_config, llm_config)
        
        return GapDirectoryManager.get_gap_analysis_file(analysis_config)

def create_latex_table_from_csv(csv_path: Path, columns: Optional[list[str]] = None, 
                               max_rows: Optional[int] = None,# caption: Optional[str] = None, 
                               generate_standalone: bool = True, column_renames: Optional[dict[str, str]] = None,
                               transform_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                               column_alignment: Optional[str] = None, label: Optional[str] = None) -> Path:
    """
    Create a LaTeX table from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        columns: Optional list of column names to include. If None, includes all columns
        max_rows: Optional maximum number of rows to include. If None, includes all rows
        generate_standalone: If True, also generate a standalone .tex file with preamble
        column_renames: Optional dictionary mapping original column names to display names
        transform_fn: Optional function to transform the DataFrame after loading, e.g., for sorting
        column_alignment: Optional column alignment string (e.g., "lrcr"). If None, auto-detects based on data types
        label: Optional table label for referencing (e.g., "tab:my_table")
        
    Returns:
        Path to the includable LaTeX file (without preamble)
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If specified columns don't exist in CSV
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    LOGGER.info(f"Creating LaTeX table from: {csv_path}")
    
    # Load the CSV data
    try:
        df = pd.read_csv(csv_path)
        LOGGER.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")
    
    # Filter rows
    original_count = len(df)
    min_articles = 10_000
    min_books = 100
    df = df[(df['num_articles'] >= min_articles) & (df['num_books'] >= min_books)]
    filtered_count = len(df)
    LOGGER.info(f"Filtered: {original_count} -> {filtered_count} rows")
    
    # Apply transformation function if provided
    if transform_fn is not None:
        try:
            df = transform_fn(df)
            LOGGER.info(f"Applied transformation function, result: {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            raise ValueError(f"Error applying transformation function: {e}")
    
    # Filter columns if specified
    if columns is not None:
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in CSV: {missing_cols}")
        df = df[columns]
        LOGGER.info(f"Using {len(columns)} specified columns")
    else:
        LOGGER.info(f"Using all {len(df.columns)} columns")
    
    # Limit rows if specified
    if max_rows is not None and max_rows < len(df):
        df = df.head(max_rows)
        LOGGER.info(f"Limited to first {max_rows} rows")
    
    # Generate LaTeX table content (without document structure)
    table_lines = []
    
    # Table setup with intelligent column alignment
    num_cols = len(df.columns)
    
    # Determine column alignment
    if column_alignment is not None:
        col_spec = column_alignment
        if len(col_spec) != num_cols:
            LOGGER.warning(f"Column alignment string length ({len(col_spec)}) doesn't match number of columns ({num_cols}). Using left alignment.")
            col_spec = "l" * num_cols
    else:
        # Auto-detect alignment based on data types
        col_spec = ""
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_spec += "r"  # Right-align numerical columns
            else:
                col_spec += "l"  # Left-align text columns
        LOGGER.info(f"Auto-detected column alignment: {col_spec}")
    
    table_lines.append("\\begin{longtable}{" + col_spec + "}")
    
    # Caption and label (if provided)
    caption = (f"Per-topic gap analysis of {len(df)} topics with at least {min_articles} articles and {min_books} books. "
                #    f"\"$\\log_{{10}}$ Shapiro\" is the base-10 logarithm of the Shapiro-Wilk p-value for normality of the gap distribution. "
                    f"Gap z is the z-score of the gap center's 1-ball coverage, relative to the 1-ball coverages of existing books. "
                    f"Topics are sorted by descending Gap z.")
    if caption is not None:
        # table_lines.append(f"\\caption{{{caption}}} \\\\")
        # if label is not None:
        #     table_lines.append(f"\\label{{{label}}}")
        line = f"\\caption{{{unicode_to_latex(caption)}}}"
        if label is not None:
            line += f"\\label{{{label}}}"
        table_lines.append(line + " \\\\")
        table_lines.append("\\toprule")
    else:
        table_lines.append("\\toprule")
    
    # Header row - use renamed columns if provided
    header_names = []
    for col in df.columns:
        # Don't use unicode_to_latex on columns (may already contain LaTeX formatting)
        col_display = col
        if column_renames and col in column_renames:
            col_display = column_renames[col]
        # Make headers bold
        header_names.append(f"\\textbf{{{col_display}}}")
    
    header_line = " & ".join(header_names) + " \\\\"
    table_lines.append(header_line)
    table_lines.append("\\midrule")
    table_lines.append("\\endfirsthead")
    
    # Continuation header for long tables
    table_lines.append("")
    table_lines.append("\\multicolumn{" + str(num_cols) + "}{c}")
    table_lines.append("{\\tablename\\ \\thetable{} -- \\textit{Continued from previous page}} \\\\")
    table_lines.append("\\toprule")
    table_lines.append(header_line)
    table_lines.append("\\midrule")
    table_lines.append("\\endhead")
    
    # Footer for continuation
    table_lines.append("")
    table_lines.append("\\midrule")
    table_lines.append("\\multicolumn{" + str(num_cols) + "}{r}{\\textit{Continued on next page}} \\\\")
    table_lines.append("\\endfoot")
    table_lines.append("")
    table_lines.append("\\bottomrule")
    table_lines.append("\\endlastfoot")
    table_lines.append("")
    
    # Data rows
    for _, row in df.iterrows():
        escaped_values = []
        for col_name, value in row.items():
            if pd.isna(value):
                escaped_values.append("")
            else:
                # Handle different data types with custom formatting
                if isinstance(value, (int, float)):
                    if pd.isna(value):
                        escaped_values.append("")
                    elif isinstance(value, float) and value.is_integer():
                        escaped_values.append(str(int(value)))
                    else:
                        # Special formatting for gap z-score column
                        col_name_str = str(col_name)
                        if 'gap_zscore' == col_name_str:
                            escaped_values.append(f"{float(value):.0f}")
                        elif 'shapiro_log10_pval' == col_name_str:
                            escaped_values.append(f"{float(value):.2f}")
                        else:
                            escaped_values.append(str(value))
                else:
                    # Handle text columns with truncation for Topic column
                    str_value = str(value)
                    col_name_str = str(col_name)
                    
                    # Truncate Topic column entries that are too long
                    if 'topic_name' == col_name_str:
                        max_topic_length = 25  # Adjust as needed
                        if len(str_value) > max_topic_length:
                            str_value = str_value[:max_topic_length]
                    if 'proposal_title' == col_name_str:
                        max_title_length = 25  # Adjust as needed
                        if len(str_value) > max_title_length:
                            str_value = str_value[:max_title_length]
                    
                    # if was_truncated:
                    #     escaped_values.append(f"\\small\\texttt{{{unicode_to_latex(str_value)}\\ldots}}")
                    # else:
                    #     escaped_values.append(f"\\small\\texttt{{{unicode_to_latex(str_value)}}}")
                    escaped_values.append(f"\\small\\texttt{{{unicode_to_latex(str_value)}}}") # Omit ellipsis
        
        data_line = " & ".join(escaped_values) + " \\\\"
        table_lines.append(data_line)
    
    # Close table
    table_lines.append("\\end{longtable}")
    
    # Create includable LaTeX file (table only)
    includable_filename = csv_path.stem + "_table.tex"
    includable_path = csv_path.parent / includable_filename
    
    with open(includable_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(table_lines))
    
    LOGGER.info(f"Created includable LaTeX table: {includable_path}")
    
    # Create standalone LaTeX file if requested
    if generate_standalone:
        standalone_lines: list[str] = []
        
        # Document structure
        standalone_lines.append("\\documentclass{article}")
        standalone_lines.append("\\usepackage[utf8]{inputenc}")
        standalone_lines.append("\\usepackage{booktabs}")
        standalone_lines.append("\\usepackage{longtable}")
        standalone_lines.append("\\usepackage{array}")
        standalone_lines.append("\\usepackage[margin=1in]{geometry}")
        standalone_lines.append("\\begin{document}")
        standalone_lines.append("")
        
        # Include the table content
        standalone_lines.extend(table_lines)
        
        # Close document
        standalone_lines.append("\\end{document}")
        
        # Create standalone file
        standalone_filename = csv_path.stem + "_table_standalone.tex"
        standalone_path = csv_path.parent / standalone_filename
        
        with open(standalone_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(standalone_lines))
        
        LOGGER.info(f"Created standalone LaTeX table: {standalone_path}")
    
    LOGGER.info(f"Table contains {len(df)} rows and {len(df.columns)} columns")
    
    return includable_path


def extract_textbook_proposals(results_file: Path, config: AnalysisConfiguration, cloud: Cloud, extract_responses: bool = True) -> dict[str, Any]:
    """
    Extract and save individual textbook proposals from batch results to the gaps directory structure.
    
    Validates that all results in the file match the expected configuration, ensuring data consistency
    and providing clear error messages for any mismatches.
    
    Args:
        results_file: Path to the JSONL results file
        config: Expected AnalysisConfiguration that all results must match
        cloud: Cloud object for ranking
        extract_responses: If True, extract proposals to responses/ subdirectories
        
    Returns:
        Dictionary with extraction statistics
        
    Raises:
        FileNotFoundError: If results file doesn't exist or config directory not found
        ValueError: If results file is empty or contains results with mismatched configurations
    """
    # Get configuration directory from expected config
    config_dir = GapDirectoryManager.get_config_dir(config)
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory {config_dir} does not exist.")

    LOGGER.info(f"Extracting textbook proposals for config: {config_dir.name}")
    LOGGER.info(f"Expected configuration: {config.to_string()}")

    successful_count = 0
    failed_count = 0

    grouped_proposals: dict[tuple[str, int], list[tuple[str, str]]] = {}
    grouped_topic_name: dict[tuple[str, int], str] = {}
    embedding_model = get_embedding_model()
    precomputed_rows: list[dict[str, Any]] = []

    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line.strip())
            custom_id = result['custom_id']
            parsed_id = parse_custom_id(custom_id)
            
            if not parsed_id:
                failed_count += 1
                continue
            
            # Validate that all results match the expected configuration
            result_config = AnalysisConfiguration.from_string(parsed_id.gap_config)
            if result_config != config:
                raise ValueError(f"Configuration mismatch in results file. Expected: {config.to_string()}, Found: {result_config.to_string()}")
            
            if result.get('response') and result['response'].get('status_code') == 200:
                raw_message = result['response']['body']['choices'][0]['message']['content']
                cleaned_json = clean_json_content(raw_message)
                try:
                    json_data = json.loads(cleaned_json)
                    latex_content = json_to_latex(json_data)

                    key = (parsed_id.topic_id, parsed_id.gap_idx)
                    title = json_data.get('title', '')
                    abstract = json_data.get('abstract', '')
                    grouped_proposals.setdefault(key, []).append((title, abstract))
                    grouped_topic_name.setdefault(key, parsed_id.topic_name)

                    text = f"{title} {abstract}".strip()
                    if text:
                        vec = embedding_model.encode(text)
                        precomputed_rows.append({
                            'custom_id': custom_id,
                            'topic_name': parsed_id.topic_name,
                            'topic_id': parsed_id.topic_id,
                            'denoiser': parsed_id.denoiser,
                            'drop_k': parsed_id.drop_k,
                            'gap_idx': parsed_id.gap_idx,
                            'radius_fraction': parsed_id.radius_fraction,
                            'proposal_title': title,
                            'proposal_abstract': abstract,
                            'embedding': vec,
                        })
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    LOGGER.warning(f"Warning: Failed to parse JSON for {custom_id}: {e}")
                    LOGGER.info(f"Raw content: {raw_message[:200]}...")
                    latex_content = f"\\textbf{{Error parsing JSON for {custom_id}}}\n\n" + f"\\texttt{{{cleaned_json}}}\n"
                
                config_dir = get_config_dir_from_custom_id(custom_id)
                # Create analysis configuration using structured approach
                gap_configuration = GapConfiguration.from_directory_name(parsed_id.gap_config)
                llm_configuration = LLMConfiguration.from_subdir_name(parsed_id.llm_subdir)
                parsed_analysis_config = AnalysisConfiguration.from_components(gap_configuration, llm_configuration)
                gap_dir = GapDirectoryManager.get_gap_dir(parsed_analysis_config, parsed_id.gap_idx)
                
                if extract_responses:
                    responses_dir = GapDirectoryManager.ensure_directory(gap_dir / "responses")
                    proposal_file = responses_dir / f"{parsed_id.topic_name}_id={parsed_id.topic_id}_gap_{parsed_id.gap_idx}_proposal.tex"
                else:
                    proposal_file = gap_dir / f"{parsed_id.topic_name}_id={parsed_id.topic_id}_gap_{parsed_id.gap_idx}_proposal.tex"
                
                with open(proposal_file, 'w', encoding='utf-8') as pf:
                    pf.write(latex_content)
                successful_count += 1
            else:
                LOGGER.warning(f"Failed request for {custom_id}")
                if result.get('error'):
                    LOGGER.error(f"Error: {result['error']}")
                failed_count += 1
                config_dir = get_config_dir_from_custom_id(custom_id)
                # Create analysis configuration using structured approach
                gap_configuration = GapConfiguration.from_directory_name(parsed_id.gap_config)
                llm_configuration = LLMConfiguration.from_subdir_name(parsed_id.llm_subdir)
                parsed_analysis_config = AnalysisConfiguration.from_components(gap_configuration, llm_configuration)
                gap_dir = GapDirectoryManager.get_gap_dir(parsed_analysis_config, parsed_id.gap_idx)
                errors_dir = GapDirectoryManager.ensure_directory(gap_dir / "errors")
                error_file = errors_dir / f"failed_{custom_id}.json"
                with open(error_file, 'w') as ef:
                    json.dump(result, ef, indent=2)

    # Build lookup for overlays
    precomputed_by_key: dict[tuple[str, int], list[np.ndarray]] = {}
    labels_by_key: dict[tuple[str, int], list[str]] = {}
    for r in precomputed_rows:
        key = (r['topic_id'], r['gap_idx'])
        precomputed_by_key.setdefault(key, []).append(r['embedding'])
        labels_by_key.setdefault(key, []).append(r['proposal_title'] if r['proposal_title'] else 'Proposal')

    # Generate PCA overlays (LPP optional later)
    if precomputed_by_key:
        for (topic_id, gap_idx), embeddings_list in precomputed_by_key.items():
            books, articles = get_subclouds_for_topic(cloud, topic_id)
            topic_name = grouped_topic_name.get((topic_id, gap_idx), topic_id)
            gaps_iter = find_gaps_for_topic(books, articles, config.gap_config.radius_fraction, cell_type=config.gap_config.cell_type)
            gap = next(g for i, g in enumerate(gaps_iter) if i == gap_idx)
            args = {
                'gap': gap,
                'gap_idx': gap_idx,
                'topic_name': topic_name,
                'topic_id': topic_id,
                'analysis_config': config,
                'books': books,
                'articles': articles,
                'method': 'PCA',
                'proposal_embeddings': embeddings_list,
                'proposal_labels': None,  # no title annotations
            }
            # Standard PCA overlay with proposals
            save_gap_plot_with_proposals(**args)
            # New books-vs-proposals classification plot
            save_gap_books_vs_proposals_plot(**args)

    stats = {
        'successful_count': successful_count,
        'failed_count': failed_count,
        'csv_path': None,
        'latex_table_paths': [],
        'all_tex_count': 0,
    }

    if extract_responses and successful_count > 0:
        all_tex_count = create_all_tex_files_in_gaps()
        stats['all_tex_count'] = all_tex_count
        if all_tex_count > 0:
            LOGGER.info(f"Created {all_tex_count} all.tex compilation files")
    
    LOGGER.info(f"\nExtraction summary:")
    LOGGER.info(f"Successful proposals: {successful_count}")
    LOGGER.info(f"Failed requests: {failed_count}")
    LOGGER.info(f"Proposals saved to gaps directory structure")
    if extract_responses and successful_count > 0:
        LOGGER.info("Use all.tex files to include in existing documents, or standalone.tex files for direct compilation")
    
    # Create the gap_analysis_with_proposals.csv
    LOGGER.info("\nCreating gap_analysis_with_proposals.csv...")
    precomputed_df = pd.DataFrame(precomputed_rows) if precomputed_rows else None
    csv_path = create_gap_analysis_with_proposals_csv(results_file, cloud, config, precomputed_proposals_df=precomputed_df)
    if csv_path:
        LOGGER.info(f"Gap analysis with proposals CSV created at: {csv_path}")
        stats['csv_path'] = csv_path
        latex_includable = csv_path.parent / f"{csv_path.stem}_table.tex"
        latex_standalone = csv_path.parent / f"{csv_path.stem}_table_standalone.tex"
        if latex_includable.exists():
            stats['latex_table_paths'].append(latex_includable)
        if latex_standalone.exists():
            stats['latex_table_paths'].append(latex_standalone)

    return stats

def create_all_tex_files_in_gaps() -> int:
    """
    Create all.tex files in gap response subdirectories that include all individual proposals.
    
    Returns:
        Number of all.tex files created
    """
    LOGGER.info("Creating all.tex files for gap response subdirectories...")
    
    all_tex_created = 0
    
    if not GAPS_DIR.exists():
        return all_tex_created
    
    # Iterate through config directories (e.g., NONE_k=0_radiusfrac=0.75)
    for config_dir in GAPS_DIR.iterdir():
        if not config_dir.is_dir():
            continue
            
        # Look for gap subdirectories (gap_0, gap_1, etc.)
        gap_dirs = [d for d in config_dir.iterdir() if d.is_dir() and d.name.startswith('gap_')]
        
        for gap_dir in gap_dirs:
            # Look for responses subdirectory
            responses_dir = gap_dir / "responses"
            if not responses_dir.exists():
                continue
                
            # Find all .tex files (excluding all.tex if it already exists)
            tex_files = [f for f in responses_dir.glob("*.tex") if f.name not in ["all.tex", "all_with_path.tex", "standalone.tex"]]
            
            if not tex_files:
                continue
                
            # Sort files for consistent ordering
            tex_files.sort(key=lambda f: f.name)
            
            # Create all.tex file (for use within the responses directory)
            all_tex_path = responses_dir / "all.tex"
            with open(all_tex_path, 'w', encoding='utf-8') as f:
                # Include the custom proposal commands
                f.write("\\input{proposal_commands}\n\n")
                # Include each proposal on a separate page
                for i, tex_file in enumerate(tex_files):
                    topic_name = extract_topic_name_from_filename(tex_file.name)
                    
                    f.write(f"\\subsection{{{topic_name}}}\n")
                    f.write(f"\\input{{{tex_file.stem}}}\n")  # Local reference, no path
                    
                    # Add page break between proposals (except for the last one)
                    if i < len(tex_files) - 1:
                        f.write("\\clearpage\n\n")
                    else:
                        f.write("\n")
            
            # Create all_with_path.tex file (for including from parent directories)
            all_with_path = responses_dir / "all_with_path.tex"
            with open(all_with_path, 'w', encoding='utf-8') as f:
                # Include the custom proposal commands
                f.write("\\input{proposal_commands}\n\n")
                # Include each proposal on a separate page with relative paths
                for i, tex_file in enumerate(tex_files):
                    topic_name = extract_topic_name_from_filename(tex_file.name)
                    
                    f.write(f"\\subsection{{{topic_name}}}\n")
                    f.write(f"\\input{{{gap_dir.name}/responses/{tex_file.stem}}}\n")  # Include gap_X/responses/ prefix
                    
                    # Add page break between proposals (except for the last one)
                    if i < len(tex_files) - 1:
                        f.write("\\clearpage\n\n")
                    else:
                        f.write("\n")
            
            # Create standalone.tex file (complete document for direct compilation)
            standalone_path = responses_dir / "standalone.tex"
            with open(standalone_path, 'w', encoding='utf-8') as f:
                f.write("\\documentclass{article}\n")
                f.write("\\usepackage[utf8]{inputenc}\n")
                f.write("\\usepackage[T1]{fontenc}\n")
                f.write("\\usepackage{geometry}\n")
                f.write("\\usepackage{amsmath}\n")
                f.write("\\usepackage{amsfonts}\n")
                f.write("\\usepackage{amssymb}\n")
                f.write("\\usepackage{graphicx}\n")
                f.write("\\usepackage{booktabs}\n")
                f.write("\\usepackage{hyperref}\n")
                f.write("\\usepackage{xcolor}\n")
                f.write("\\usepackage{enumitem}\n")
                f.write("\\usepackage{ragged2e}\n")
                f.write("\\usepackage{alphalph}\n")
                f.write("\\geometry{margin=1in}\n\n")
                
                f.write("\\title{Textbook Proposals Collection}\n")
                f.write(f"\\author{{Gap Analysis: {unicode_to_latex(gap_dir.name)}}}\n")
                f.write("\\date{\\today}\n\n")
                
                f.write("\\begin{document}\n")
                f.write("\\maketitle\n")
                f.write("\\tableofcontents\n")
                f.write("\\clearpage\n\n")
                
                # Include the local all.tex content
                f.write("\\input{all}\n")
                
                f.write("\\end{document}\n")
            
            LOGGER.info(f"Created {all_tex_path} (local), {all_with_path} (with paths), and {standalone_path} (standalone) with {len(tex_files)} proposals")
            all_tex_created += 1
    
    return all_tex_created

def list_batch_jobs(limit: int = 20) -> None:
    """List all batch jobs with their status information."""
    LOGGER.info("Listing all batch jobs...")
    
    try:
        batches = get_client().batches.list(limit=limit)
        
        if not batches.data:
            LOGGER.info("No batch jobs found.")
            return
        
        for batch in batches.data:
            LOGGER.info(f"\nBatch ID: {batch.id}")
            LOGGER.info(f"Status: {batch.status}")
            LOGGER.info(f"Created: {batch.created_at}")
            LOGGER.info(f"Endpoint: {batch.endpoint}")
            if batch.metadata:
                LOGGER.info(f"Description: {batch.metadata.get('description', 'N/A')}")
            if batch.status == "completed":
                LOGGER.info(f"Completed at: {batch.completed_at}")
                LOGGER.info(f"Request counts: {batch.request_counts}")
            elif batch.status == "failed":
                LOGGER.info(f"Failed at: {batch.failed_at}")
                if batch.errors:
                    LOGGER.error(f"Errors: {batch.errors}")
    except Exception as e:
        LOGGER.error(f"Error listing batch jobs: {e}")

def get_batch_status(batch_id: str) -> Any:
    """
    Get the status of a specific batch job.
    
    Args:
        batch_id: The batch job ID to check
        
    Returns:
        The batch status object
        
    Raises:
        Exception: If retrieval fails
    """
    batch = get_client().batches.retrieve(batch_id)
    LOGGER.info(f"\nBatch ID: {batch.id}")
    LOGGER.info(f"Status: {batch.status}")
    LOGGER.info(f"Created: {batch.created_at}")
    if batch.status == "completed":
        LOGGER.info(f"Completed at: {batch.completed_at}")
        LOGGER.info(f"Request counts: {batch.request_counts}")
        LOGGER.info(f"Output file ID: {batch.output_file_id}")
    elif batch.status == "failed":
        LOGGER.info(f"Failed at: {batch.failed_at}")
        if batch.errors:
            LOGGER.error(f"Errors: {batch.errors}")
    elif batch.status in ["validating", "in_progress", "finalizing"]:
        LOGGER.info(f"Batch is still {batch.status}")
    
    return batch

def _merge_gap_analysis_with_proposals(gap_analysis_csv_path: Path, results_jsonl_path: Path, output_path: Path, cloud: Cloud, config: AnalysisConfiguration, generate_latex_table: bool = True,
                                       precomputed_proposals_df: Optional[pd.DataFrame] = None) -> Path:
    """
    Shared function to merge gap analysis CSV with batch results and create the combined CSV.
    
    Args:
        gap_analysis_csv_path: Path to gap_analysis.csv file
        results_jsonl_path: Path to batch results JSONL file
        output_path: Path where to save the merged CSV
        cloud: Cloud object for ranking
        config: Analysis configuration
        generate_latex_table: If True, also generate a LaTeX table from the CSV
        
    Returns:
        Path to the created CSV file
        
    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If columns are missing or no proposals found
    """
    # Load the existing gap analysis data
    gap_df = pd.read_csv(gap_analysis_csv_path)
    LOGGER.info(f"Loaded gap analysis data: {len(gap_df)} rows")
    
    # Build proposals_df (either precomputed fast-path or by parsing JSONL)
    if precomputed_proposals_df is not None and len(precomputed_proposals_df) > 0:
        proposals_df = precomputed_proposals_df.copy()
        LOGGER.info(f"Using precomputed proposals: {len(proposals_df)} entries")
    else:
        proposals_data = []
        LOGGER.info("Extracting proposal titles and abstracts from batch results...")
        with open(results_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line.strip())
                proposal_data = extract_proposal_data_from_result(result)
                if proposal_data:
                    proposals_data.append(proposal_data)
                else:
                    LOGGER.warning(f"Warning: Failed to parse custom_id: {result['custom_id']}")
        if not proposals_data:
            raise ValueError("No proposal data extracted from batch results")
        proposals_df = pd.DataFrame(proposals_data)
        proposals_df['radius_fraction'] = config.gap_config.radius_fraction
        proposals_df['embedding'] = None
        LOGGER.info(f"Extracted {len(proposals_df)} proposal entries")

    # Merge with gap analysis data on key columns
    key_columns = ['topic_name', 'topic_id', 'denoiser', 'drop_k', 'gap_idx']
    missing_columns = [col for col in key_columns if col not in gap_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in gap analysis data: {missing_columns}. Available columns: {list(gap_df.columns)}")
    merged_df = gap_df.merge(proposals_df[key_columns + ['proposal_title', 'proposal_abstract']], on=key_columns, how='left')

    merged_df['proposal_title'] = merged_df['proposal_title'].fillna('')
    merged_df['proposal_abstract'] = merged_df['proposal_abstract'].fillna('')

    LOGGER.info("Computing proposal rankings...")
    rankings_df = compute_proposal_rankings(results_jsonl_path, cloud, config, precomputed_proposals_df=proposals_df)
    merged_df = merged_df.merge(rankings_df, on=key_columns, how='left')

    merged_df['proposal_book_rank_fraction'] = merged_df.apply(
        lambda row: row['proposal_book_rank'] / row['num_books'] if row['proposal_book_rank'] >= 0 and row['num_books'] > 0 else -1,
        axis=1
    )
    merged_df['proposal_article_rank_fraction'] = merged_df.apply(
        lambda row: row['proposal_article_rank'] / row['num_articles'] if row['proposal_article_rank'] >= 0 and row['num_articles'] > 0 else -1,
        axis=1
    )

    last_cols = ['proposal_title', 'proposal_abstract']
    other_cols = [col for col in merged_df.columns if col not in last_cols]
    merged_df = merged_df[other_cols + last_cols]

    # Process numeric columns using unified framework patterns
    merged_df = process_dataframe_numeric_columns(merged_df, significant_figures=4)

    # Export using unified framework patterns
    LOGGER.info(f"Exporting gap_analysis_with_proposals using unified framework patterns")
    
    # Validate DataFrame using unified framework validation patterns
    validate_batch_results_dataframe(merged_df, "gap_analysis_with_proposals")
    
    exported_paths = export_dataframe_with_unified_patterns(
        merged_df, 
        output_path, 
        formats=['csv', 'json', 'parquet']
    )
    
    LOGGER.info(f"Successfully exported gap_analysis_with_proposals with {len(merged_df)} rows in {len(exported_paths)} formats")

    with_proposals = (merged_df['proposal_title'] != '').sum()
    without_proposals = (merged_df['proposal_title'] == '').sum()
    LOGGER.info(f"Rows with proposals: {with_proposals}")
    LOGGER.info(f"Rows without proposals: {without_proposals}")

    if generate_latex_table:
        available_columns = [col for col in merged_df.columns if col not in {
            'topic_id','denoiser', 'radius_fraction', 'drop_k','gap_idx','gap_size_percentile',
            'proposal_title', 'proposal_abstract','shapiro_log10_pval','gap_size'
        }]
        column_renames = {
            'topic_name': 'Topic',
            'proposal_title': 'Proposed Title',
            'proposal_abstract': 'Proposed Abstract',
            'proposal_book_rank': 'PBR',
            'proposal_article_rank': 'PAR',
            'gap_idx': 'Gap Index (0-based)',
            'denoiser': 'Denoiser',
            'drop_k': 'Drop K',
            'radius_fraction': 'Radius Fraction',
            'proposal_book_rank_fraction': 'PBRF',
            'proposal_article_rank_fraction': 'PARF',
            'topic_id': 'Topic ID',
            'num_books': 'Books',
            'num_articles': 'Articles',
            'num_books_for_llm': 'Books for LLM',
            'num_articles_for_llm': 'Articles for LLM',
            'gap_size': 'Gap Size',
            'gap_size_percentile': unicode_to_latex(f'Gap %ile'),
            'gap_zscore': 'Gap z',
            'shapiro_log10_pval': '$\\log_{10}$ Shapiro'
        }
        latex_path = create_latex_table_from_csv(
            output_path,
            columns=available_columns,
            max_rows=50,
            column_renames=column_renames,
            transform_fn=lambda df: df.sort_values('proposal_article_rank_fraction', ascending=False) if 'gap_zscore' in df.columns else df,
            label='tab:gap_analysis_with_proposals'
        )
        LOGGER.info(f"LaTeX table created: {latex_path}")

    return output_path

@dataclass(frozen=True)
class ProposalRanking:
    book_rank: int
    article_rank: int

def compute_proposal_rankings(results_file: Path, cloud: Cloud, config: AnalysisConfiguration, precomputed_proposals_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute proposal rankings using FAISS range searches.
    
    Args:
        results_file: Path to results file (unused in current implementation)
        cloud: Cloud object for ranking computations
        config: Analysis configuration (required)
        precomputed_proposals_df: DataFrame with precomputed proposals (required)
        
    Returns:
        DataFrame with proposal rankings
        
    Raises:
        ValueError: If precomputed_proposals_df is None or empty
    """
    if precomputed_proposals_df is None or len(precomputed_proposals_df) == 0:
        raise ValueError("precomputed_proposals_df is required and cannot be empty")
    
    LOGGER.info("Computing proposal rankings using FAISS range searches...")
    records = []

    radius_fraction = config.gap_config.radius_fraction

    def resolve_gap(parsed_id: CustomIdComponents):
        if abs(parsed_id.radius_fraction - radius_fraction) > 1e-6:
            raise ValueError(f"radius_fraction mismatch: parsed_id={parsed_id.radius_fraction}, expected={radius_fraction}")
        cell_type = config.gap_config.cell_type
        if cell_type != GapConfiguration.from_directory_name(parsed_id.gap_config).cell_type:
            raise ValueError(f"cell_type mismatch: parsed_id={parsed_id.gap_config}, expected={config.gap_config.cell_type}")
        books, articles = get_subclouds_for_topic(cloud, parsed_id.topic_id)
        topic_gaps = find_gaps_for_topic(books, articles, radius_fraction, cell_type=cell_type)
        gap = next(g for i, g in enumerate(topic_gaps) if i == parsed_id.gap_idx)
        return books, articles, gap

    grouped = precomputed_proposals_df.groupby(['topic_name','topic_id','denoiser','drop_k','gap_idx'], sort=False)
    for (topic_name, topic_id, denoiser, drop_k, gap_idx), df_group in grouped:
        # Extract LLM subdir from first row if available, otherwise use default
        llm_subdir = df_group.get('llm_subdir', pd.Series(['DEFAULT'])).iloc[0]
        
        # Use centralized configuration management instead of manual string building
        actual_radius_fraction = float(df_group['radius_fraction'].iloc[0] if 'radius_fraction' in df_group.columns else radius_fraction)
        
        # Extract configuration from the first row's custom_id if available in data
        first_row = df_group.iloc[0]
        # Parse complete configuration from custom_id if available
        parsed_id = parse_custom_id(first_row['custom_id'])
        full_config = AnalysisConfiguration.from_string(parsed_id.gap_config)
        analysis_config = full_config

        
        
        parsed_id = CustomIdComponents(
            gap_config=analysis_config.to_string(),  # Use centralized string conversion
            topic_name=topic_name,
            topic_id=topic_id,
            gap_idx=int(gap_idx),
            denoiser=denoiser,
            drop_k=int(drop_k),
            radius_fraction=actual_radius_fraction,
            llm_subdir=llm_subdir
        )
        books, articles, gap = resolve_gap(parsed_id)

        # Gap center
        center_idx = int(gap.dists_idxs.idxs[0])
        center_emb = articles.embeddings()[center_idx:center_idx+1]
        assert isinstance(center_emb, np.ndarray)
        assert center_emb.dtype == np.float32, f"center_emb must be float32, got {center_emb.dtype}"

        # Collect proposal vectors (float32) in this group
        vecs = []
        metas = []
        for _, row in df_group.iterrows():
            title = row.get('proposal_title', '')
            abstract = row.get('proposal_abstract', '')
            vec = row['embedding']
            if not title or not abstract or vec is None:
                records.append({
                    'topic_name': topic_name,
                    'topic_id': topic_id,
                    'denoiser': denoiser,
                    'drop_k': int(drop_k),
                    'gap_idx': int(gap_idx),
                    'proposal_book_rank': -1,
                    'proposal_article_rank': -1
                })
                continue
            vv = np.asarray(vec)
            assert vv.dtype == np.float32, f"proposal embedding must be float32, got {vv.dtype}"
            vecs.append(vv)
            metas.append((topic_name, topic_id, denoiser, int(drop_k), int(gap_idx)))

        if not vecs:
            continue

        V = np.vstack(vecs)
        assert V.dtype == np.float32
        assert center_emb.shape[0] == 1 and center_emb.shape[1] == V.shape[1]

        # Use consolidated function to compute proposal rankings
        ranking_result = compute_proposal_rankings_for_gap(books, articles, center_emb, V)
        book_ranks = ranking_result.book_ranks
        article_ranks = ranking_result.article_ranks

        for (tn, tid, den, dk, gi), br, ar in zip(metas, book_ranks, article_ranks):
            records.append({
                'topic_name': tn,
                'topic_id': tid,
                'denoiser': den,
                'drop_k': dk,
                'gap_idx': gi,
                'proposal_book_rank': int(br),
                'proposal_article_rank': int(ar)
            })

    LOGGER.info(f"Processed {len(records)} proposals (precomputed) with FAISS range search")
    return pd.DataFrame(records)
