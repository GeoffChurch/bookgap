#!/usr/bin/env python3
"""
Interactive batch workflow for textbook proposal generation.

This script provides a guided workflow for:
1. Generating prompts using gap_finder
2. Creating batch request files from existing prompts  
3. Sending batch requests to OpenAI
4. Monitoring batch job until completion
5. Downloading completed batch results
6. Generating LaTeX files from downloaded batch results

The user can jump to any step and will be guided through subsequent steps.
"""

import json
import os
import time
import openai
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass

from bookgap.cloud import build_cloud
from bookgap.config_manager import AnalysisConfiguration, GapConfiguration, LLMConfiguration
from bookgap.gap_finder import main as gap_finder_main
from bookgap.batch_utils.batch_operations import (
    GAPS_DIR, get_client, estimate_batch_cost,
    extract_textbook_proposals, 
    find_batch_request_files_for_config, find_batch_results_files_for_config, find_prompt_files_for_config,
    get_batch_status, download_batch_results,
    create_batch_requests_from_prompts, 
    create_batch_file_from_requests, upload_and_submit_batch,
)
from bookgap.util import LOGGER

@dataclass(frozen=True)
class InteractiveLLMConfiguration:
    """Configuration for LLM processing parameters collected during interactive workflow."""
    requested_num_prompt_articles: int
    requested_num_prompt_books: int


def get_llm_configuration() -> tuple[InteractiveLLMConfiguration, str]:
    """
    Prompt the user for LLM-specific configuration parameters.
    
    Returns:
        Tuple of (LLM configuration, cell_type)
    """
    LOGGER.info("\n" + "="*50)
    LOGGER.info("LLM Configuration Setup")
    LOGGER.info("="*50)
    LOGGER.info("The following parameters are needed for batch LLM requests:")
    
    # Get cell type
    cell_types: Tuple[str, ...] = ('kball', 'voronoi')
    default_cell_type: str = 'kball'
    cell_type: str
    while True:
        LOGGER.info("\nCell type determines the gap detection algorithm:")
        LOGGER.info("  - kball: Uses k-nearest neighbors with radius-based ball cells")  
        LOGGER.info("  - voronoi: Uses Voronoi diagram partitioning")
        cell_type = input(f"Enter cell type ({'/'.join(cell_types)}, default: {default_cell_type}): ").lower().strip() or default_cell_type
        if cell_type in cell_types:
            break
        LOGGER.warning(f"Please enter one of: {', '.join(cell_types)}.")
    
    # Get number of articles
    default_num_articles: int = 10
    num_articles: int
    while True:
        try:
            articles_input: str = input(f"Number of articles to include in prompt (default: {default_num_articles}): ").strip()
            num_articles = default_num_articles if not articles_input else int(articles_input)
            if num_articles >= 0:
                break
            LOGGER.warning("Number of articles must be non-negative.")
        except ValueError:
            LOGGER.warning("Please enter a valid number.")
    
    default_num_books: int = 5
    num_books: int
    while True:
        try:
            books_input: str = input(f"Number of books to include in prompt (default: {default_num_books}): ").strip()
            num_books = default_num_books if not books_input else int(books_input)
            if num_books >= 0:
                break
            LOGGER.warning("Number of books must be non-negative.")
        except ValueError:
            LOGGER.warning("Please enter a valid number.")

    # # Get minimum book coverage for LLM prompts
    # default_coverage: float = 0.5
    # prompt_book_min_coverage: float
    # while True:
    #     try:
    #         coverage_input: str = input(f"Minimum book coverage for LLM prompts (0.0-1.0, default: {default_coverage}): ").strip()
    #         prompt_book_min_coverage = default_coverage if not coverage_input else float(coverage_input)
    #         if 0.0 <= prompt_book_min_coverage <= 1.0:
    #             break
    #         LOGGER.warning("Book coverage must be between 0.0 and 1.0.")
    #     except ValueError:
    #         LOGGER.warning("Please enter a valid number between 0.0 and 1.0.")
    

    LOGGER.info(f"\n✓ LLM Configuration:")
    LOGGER.info(f"  Cell type: {cell_type}")
    LOGGER.info(f"  Articles: {num_articles}")
    LOGGER.info(f"  Books: {num_books}")
    # LOGGER.info(f"  Book coverage: {prompt_book_min_coverage:.2f}")
    
    return InteractiveLLMConfiguration(
        requested_num_prompt_articles=num_articles,
        requested_num_prompt_books=num_books,
        # prompt_book_min_coverage=prompt_book_min_coverage,
    ), cell_type


@dataclass(frozen=True)
class DownloadBatchResults:
    """Result of downloading batch results."""
    success: bool
    results_file: Optional[Path] = None

def list_available_batches() -> List[dict]:
    """List all available batch jobs from OpenAI."""
    try:
        batches = get_client().batches.list(limit=20)
        batch_list = []
        for batch in batches.data:
            batch_info = {
                'id': batch.id,
                'status': batch.status,
                'created_at': batch.created_at,
                'request_counts': getattr(batch, 'request_counts', None)
            }
            batch_list.append(batch_info)
        return batch_list
    except Exception as e:
        LOGGER.error(f"Error listing batches: {e}")
        return []


def step_1_generate_prompts(drop_k: int, max_works: Optional[int], deduplicator, radius_fraction: float, denoiser: str,
         index_type, hnsw_m: int, hnsw_efc: int, hnsw_efs: int, cell_type: str, llm_config: InteractiveLLMConfiguration, 
         config: AnalysisConfiguration):
    """Step 1: Generate prompts using gap_finder."""
    LOGGER.info("\n" + "="*50)
    LOGGER.info("STEP 1: Generate Gap Analysis Prompts")
    LOGGER.info("="*50)
    existing_prompts = find_prompt_files_for_config(config)
    if existing_prompts:
        LOGGER.info(f"Found {len(existing_prompts)} existing prompt files for current configuration.")
        response = input("Do you want to regenerate prompts? This will create new files. (y/N): ")
        if response.lower() != 'y':
            LOGGER.info("Skipping prompt generation.")
            return True
    # Ask about plot generation
    LOGGER.info("\nGap visualization plots can be time-consuming to generate.")
    plot_response = input("Generate gap visualization plots? (Y/n): ").lower() or 'y'
    generate_plots = plot_response == 'y'
    LOGGER.info("Running gap_finder to generate prompts...")
    LOGGER.info("This may take several minutes depending on your configuration.")
    gap_finder_main(
        drop_k=drop_k,
        max_works=max_works,
        deduplicator=deduplicator,
        radius_fraction=radius_fraction,
        denoiser=denoiser,
        index_type=index_type,
        hnsw_m=hnsw_m,
        hnsw_efc=hnsw_efc,
        hnsw_efs=hnsw_efs,
        generate_plots=generate_plots,
        requested_num_prompt_articles=llm_config.requested_num_prompt_articles,
        requested_num_prompt_books=llm_config.requested_num_prompt_books,
        cell_type=cell_type
    )
    new_prompts = find_prompt_files_for_config(config)
    LOGGER.info(f"✓ Successfully generated {len(new_prompts)} prompt files for current configuration")
    return True


def step_2_create_batch_file(config: AnalysisConfiguration):
    """Step 2: Create batch request file from existing prompts."""
    LOGGER.info("\n" + "="*50)
    LOGGER.info("STEP 2: Create Batch Request File")
    LOGGER.info("="*50)
    
    prompt_files = find_prompt_files_for_config(config)
    if not prompt_files:
        LOGGER.error("✗ No prompt files found for current configuration. Please run Step 1 first.")
        return False, None
    
    LOGGER.info(f"Found {len(prompt_files)} prompt files for current configuration")
    
    # Check for existing batch files for this config
    existing_batch_files = find_batch_request_files_for_config(config)
    if existing_batch_files:
        LOGGER.info(f"Found {len(existing_batch_files)} existing batch files for current configuration:")
        for bf in existing_batch_files:
            LOGGER.info(f"  - {bf.name}")
        LOGGER.info("These will be overwritten if you proceed.")
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            LOGGER.info("Operation cancelled.")
            return False, None
    
    LOGGER.info("Creating batch request file...")
    requests = create_batch_requests_from_prompts(prompt_files)
    if not requests:
        LOGGER.error("No valid requests could be created from prompt files!")
        return False, None
    
    batch_file = create_batch_file_from_requests(requests, config)
    LOGGER.info(f"✓ Created batch file with {len(requests)} requests")
    
    return True, batch_file


def step_3_send_batch_request(config: AnalysisConfiguration, batch_file: Optional[Path] = None) -> Tuple[bool, Optional[str]]:
    """Step 3: Send batch request to OpenAI."""
    LOGGER.info("\n" + "="*50)
    LOGGER.info("STEP 3: Send Batch Request to OpenAI")
    LOGGER.info("="*50)
    
    # # Check API key
    # if not os.getenv("OPENAI_API_KEY"):
    #     LOGGER.error("✗ OPENAI_API_KEY environment variable not set!")
    #     LOGGER.info("Please set your OpenAI API key:")
    #     LOGGER.info("export OPENAI_API_KEY='your-api-key-here'")
    #     return False, None
    while not os.getenv("OPENAI_API_KEY"):
        api_key: str = input("Please enter your OpenAI API key: ")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    
    if batch_file is None:
        batch_files: List[Path] = find_batch_request_files_for_config(config)
        if not batch_files:
            LOGGER.error("✗ No batch request files found for current configuration. Please run Step 2 first.")
            return False, None
        
        if len(batch_files) == 1:
            batch_file = batch_files[0]
            LOGGER.info(f"Using batch file: {batch_file.name}")
        else:
            LOGGER.info("Multiple batch files found:")
            for i, bf in enumerate(batch_files):
                LOGGER.info(f"  {i+1}. {bf.name}")
            
            while True:
                try:
                    choice: int = int(input("Select batch file (number): ")) - 1
                    if 0 <= choice < len(batch_files):
                        batch_file = batch_files[choice]
                        break
                    else:
                        LOGGER.warning("Invalid choice. Please try again.")
                except ValueError:
                    LOGGER.warning("Please enter a number.")
    
    LOGGER.info(f"Submitting batch file: {batch_file.name}")
    
    # Show cost estimate
    try:
        with open(batch_file, 'r') as f:
            requests: List[Dict[str, Any]] = [json.loads(line) for line in f]
            prompt_lengths: List[int] = [len(req['body']['messages'][0]['content']) for req in requests]
            estimate_batch_cost(prompt_lengths)
        response: str = input("Proceed with batch submission? (y/N): ")
        if response.lower() != 'y':
            LOGGER.info("Batch submission cancelled.")
            return False, None
    except Exception as e:
        LOGGER.warning(f"Could not estimate cost: {e}")
        response = input("Proceed anyway? (y/N): ")
        if response.lower() != 'y':
            return False, None
    batch_id: Optional[str] = upload_and_submit_batch(batch_file)
    if batch_id:
        LOGGER.info(f"✓ Successfully submitted batch job: {batch_id}")
        return True, batch_id
    else:
        LOGGER.error("✗ Failed to submit batch job")
        return False, None


def step_4_monitor_batch_job(batch_id: Optional[str] = None) -> bool:
    """Step 4: Monitor batch job until completion."""
    LOGGER.info("\n" + "="*50)
    LOGGER.info("STEP 4: Monitor Batch Job")
    LOGGER.info("="*50)
    
    # Check API key
    # if not os.getenv("OPENAI_API_KEY"):
    #     print("✗ OPENAI_API_KEY environment variable not set!")
    #     return False
    if not os.getenv("OPENAI_API_KEY"):
        api_key: str = input("Please enter your OpenAI API key: ")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    
    if batch_id is None:
        LOGGER.info("Fetching available batch jobs...")
        available_batches: List[Dict[str, Any]] = list_available_batches()
        
        if not available_batches:
            LOGGER.error("✗ No batch jobs found")
            return False
        
        # Show all batches (including in-progress ones)
        LOGGER.info(f"Found {len(available_batches)} batch jobs:")
        for i, batch in enumerate(available_batches):
            request_counts: openai.types.BatchRequestCounts = batch['request_counts']
            if request_counts:
                total: int = request_counts.total
                completed: int = request_counts.completed
                LOGGER.info(f"  {i+1}. {batch['id']}: {batch['status']} ({completed}/{total} requests)")
            else:
                LOGGER.info(f"  {i+1}. {batch['id']}: {batch['status']}")
        
        # Let user select batch
        while True:
            try:
                choice: int = int(input("Select batch to monitor (number): ")) - 1
                if 0 <= choice < len(available_batches):
                    selected_batch: Dict[str, Any] = available_batches[choice]
                    batch_id = selected_batch['id']
                    break
                else:
                    LOGGER.warning("Invalid choice. Please try again.")
            except ValueError:
                LOGGER.warning("Please enter a number.")
    
    poll_interval: int = 10
    LOGGER.info(f"Polling batch job {batch_id} every {poll_interval} seconds until completion.")
    LOGGER.info("Press Ctrl+C to stop monitoring and return to the workflow.")
    
    try:
        # Import here to use the monitoring function
        from bookgap.batch_utils.batch_operations import monitor_batch_job
        assert type(batch_id) is str, f"Expected str, got {type(batch_id)}"
        final_status: Optional[Any] = monitor_batch_job(batch_id, poll_interval=poll_interval)
        
        if final_status and final_status.status == "completed":
            LOGGER.info(f"✓ Batch job completed successfully!")
            LOGGER.info("  Run Step 5 to download the completed results.")
            return True
        elif final_status and final_status.status == "failed":
            LOGGER.error(f"✗ Batch job failed")
            if hasattr(final_status, 'errors') and final_status.errors:
                LOGGER.error(f"  Error details: {final_status.errors}")
            return False
        else:
            LOGGER.error(f"✗ Batch monitoring ended with status: {final_status.status if final_status else 'Unknown'}")
            return False
    except KeyboardInterrupt:
        LOGGER.info("\n⏸️ Monitoring interrupted by user.")
        LOGGER.info("  The batch job is still running in the background.")
        LOGGER.info("  You can resume monitoring later or proceed to Step 5 when ready.")
        return True


def step_5_download_batch_results(batch_id: Optional[str] = None) -> DownloadBatchResults:
    """Step 5: Download completed batch results."""
    LOGGER.info("\n" + "="*50)
    LOGGER.info("STEP 5: Download Batch Results")
    LOGGER.info("="*50)
    
    while not os.getenv("OPENAI_API_KEY"):
        api_key: str = input("Please enter your OpenAI API key: ")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    
    LOGGER.info("Fetching available batch jobs...")
    available_batches: List[Dict[str, Any]] = list_available_batches()
    
    if not available_batches:
        LOGGER.error("✗ No batch jobs found")
        return DownloadBatchResults(success=False)
    
    # Filter for completed batches
    completed_batches: List[Dict[str, Any]] = [b for b in available_batches if b['status'] == 'completed']
    
    LOGGER.info(f"Found {len(available_batches)} total batches, {len(completed_batches)} completed")
    
    if not completed_batches:
        LOGGER.info("No completed batches available for download.")
        in_progress: List[Dict[str, Any]] = [b for b in available_batches if b['status'] in ['validating', 'in_progress', 'finalizing']]
        if in_progress:
            LOGGER.info("\nIn-progress batches:")
            for batch in in_progress:
                LOGGER.info(f"  {batch['id']}: {batch['status']}")
        return DownloadBatchResults(success=False)

    # If batch_id is provided, use it directly
    selected_batch: Optional[Dict[str, Any]] = None
    if batch_id:
        selected_batch = next((b for b in completed_batches if b['id'] == batch_id), None)
        if not selected_batch:
            LOGGER.error(f"✗ Provided batch_id {batch_id} not found among completed batches.")
            raise ValueError(f"Batch ID {batch_id} not found.")
        LOGGER.info(f"Selected batch: {selected_batch['id']} (created_at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(selected_batch['created_at']))})")
    else:
        LOGGER.info("\nCompleted batches available for download:")
        for i, batch in enumerate(completed_batches):
            request_counts: openai.types.BatchRequestCounts = batch['request_counts']
            created_str: str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(batch['created_at']))
            if request_counts:
                total: int = request_counts.total
                completed: int = request_counts.completed
                LOGGER.info(f"  {i+1}. {batch['id']}: {completed}/{total} requests completed | created_at: {created_str}")
            else:
                LOGGER.info(f"  {i+1}. {batch['id']}: Status unknown | created_at: {created_str}")
        if len(completed_batches) == 1:
            selected_batch = completed_batches[0]
            LOGGER.info(f"Auto-selecting: {selected_batch['id']} (created_at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(selected_batch['created_at']))})")
        else:
            while True:
                try:
                    choice: int = int(input("Select batch to download (number): ")) - 1
                    if 0 <= choice < len(completed_batches):
                        selected_batch = completed_batches[choice]
                        break
                    else:
                        LOGGER.warning("Invalid choice. Please try again.")
                except ValueError:
                    LOGGER.warning("Please enter a number.")
    
    LOGGER.info(f"Downloading results for batch: {selected_batch['id']}")
    try:
        batch_status = get_batch_status(selected_batch['id'])
        if not batch_status:
            LOGGER.error("✗ Failed to get batch status")
            return DownloadBatchResults(success=False)
        if batch_status.status != "completed":
            LOGGER.error(f"✗ Batch is not completed. Current status: {batch_status.status}")
            return DownloadBatchResults(success=False)
        results_file: Path = download_batch_results(batch_status)
        LOGGER.info(f"✓ Successfully downloaded batch results to: {results_file}")
        LOGGER.info("  Run Step 6 to extract proposals from the downloaded results.")
        return DownloadBatchResults(success=True, results_file=results_file)
    except Exception as e:
        LOGGER.error(f"✗ Error downloading batch results: {e}")
        raise


def step_6_generate_tex_files(downloaded_results_file: Optional[Path], config: AnalysisConfiguration, max_works: Optional[int], deduplicator, index_type, hnsw_m: int, hnsw_efc: int, hnsw_efs: int) -> bool:
    """Step 6: Generate LaTeX files from downloaded batch results using explicit configuration."""
    LOGGER.info("\n" + "="*50)
    LOGGER.info("STEP 6: Generate LaTeX Files")
    LOGGER.info("="*50)

    # Use the provided explicit configuration instead of trying to extract from files
    LOGGER.info(f"Using explicit configuration: {config.to_string()}")
    
    # Build the expected config directory path from the explicit configuration
    config_dir_name: str = config.to_directory_name()
    config_dir: Path = GAPS_DIR / config_dir_name
    
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory {config_dir} does not exist. Make sure gap analysis has been run with this configuration.")

    # Use the new config-specific finder function
    results_files: List[Path] = find_batch_results_files_for_config(config)
    if downloaded_results_file:
        if downloaded_results_file.exists():
            # Move the downloaded file to the front of the list
            if downloaded_results_file in results_files:
                results_files.remove(downloaded_results_file)
            results_files.insert(0, downloaded_results_file)
            LOGGER.info(f"Will prioritize newly downloaded file: {downloaded_results_file.name}")
        else:
            raise FileNotFoundError(f"Downloaded results file {downloaded_results_file.name} does not exist.")

    if not results_files:
        LOGGER.error("✗ No batch results files found for selected config.")
        return False

    LOGGER.info(f"Found {len(results_files)} batch results files for config {config_dir.name}:")
    for i, results_file in enumerate(results_files):
        file_age: float = time.time() - results_file.stat().st_mtime
        age_str: str = f"{file_age/3600:.1f} hours ago" if file_age < 86400 else f"{file_age/86400:.1f} days ago"
        LOGGER.info(f"  {i+1}. {results_file.name} ({age_str})")

    selected_files: List[Path] = results_files
    total_successful: int = 0
    aggregated_stats: Dict[str, Any] = {
        'csv_path': None,
        'latex_table_paths': [],
        'all_tex_count': 0,
    }


    cloud = build_cloud(
        drop_k=config.gap_config.drop_k,
        max_works=max_works,
        deduplicator=deduplicator,
        denoiser=config.gap_config.denoiser,
        index_type=index_type,
        hnsw_m=hnsw_m,
        hnsw_efc=hnsw_efc,
        hnsw_efs=hnsw_efs
    )

    for results_file in selected_files:
        LOGGER.info(f"\nProcessing: {results_file.name}")
        
        # Use the explicit configuration instead of extracting from results file
        stats: Dict[str, Any] = extract_textbook_proposals(results_file, config, cloud)
        total_successful += stats['successful_count']
        total_failed: int = stats['failed_count']
        if total_failed > 0:
            LOGGER.warning(f"  {total_failed} failed extractions.")
            raise RuntimeError(f"Extraction failed for {total_failed} proposals in {results_file.name}.")
        if stats.get('csv_path'):
            aggregated_stats['csv_path'] = stats['csv_path']
        if stats.get('latex_table_paths'):
            aggregated_stats['latex_table_paths'].extend(stats['latex_table_paths'])
        aggregated_stats['all_tex_count'] += stats.get('all_tex_count', 0)
        LOGGER.info(f"  Extracted {stats['successful_count']} proposals")
        if stats['failed_count'] > 0:
            LOGGER.warning(f"  {stats['failed_count']} failed proposals")

    LOGGER.info(f"\n✓ Processing complete!")
    LOGGER.info(f"  Total successful proposals: {total_successful}")
    if total_successful > 0:
        _show_extraction_summary(aggregated_stats)
    return True

def _show_existing_files_summary(extracted_proposals):
    """Show summary of existing extracted files."""
    from bookgap.batch_utils.batch_operations import GAPS_DIR
    
    # Check if all.tex files exist
    all_tex_files = []
    if GAPS_DIR.exists():
        for config_dir in GAPS_DIR.iterdir():
            if config_dir.is_dir():
                for gap_dir in config_dir.glob("gap_*"):
                    if gap_dir.is_dir():
                        responses_dir = gap_dir / "responses"
                        if responses_dir.exists():
                            all_tex = responses_dir / "all.tex"
                            if all_tex.exists():
                                all_tex_files.append(all_tex)
    
    if all_tex_files:
        LOGGER.info(f"Found {len(all_tex_files)} compilation files (all.tex) ready for use.")
    else:
        LOGGER.info("Compilation files missing. Regenerating...")
        from bookgap.batch_utils.batch_operations import create_all_tex_files_in_gaps
        all_tex_count = create_all_tex_files_in_gaps()
        if all_tex_count > 0:
            LOGGER.info(f"✓ Created {all_tex_count} compilation files")
        else:
            LOGGER.error("✗ Failed to create compilation files")
            return False
    _show_extraction_summary()
    return True


def _show_extraction_summary(stats: Optional[Dict[str, Any]] = None) -> None:
    """Show summary of available extracted files.
    
    Args:
        stats: Optional dictionary with extraction statistics including:
               - csv_path: Path to generated CSV file
               - latex_table_paths: List of generated LaTeX table files
               - all_tex_count: Number of all.tex compilation files created
    """
    from bookgap.batch_utils.batch_operations import GAPS_DIR
    
    # Show information from extraction stats if provided
    if stats:
        if stats.get('csv_path'):
            LOGGER.info(f"✓ Gap analysis CSV: {stats['csv_path']}")
        if stats.get('latex_table_paths'):
            LOGGER.info("✓ Generated LaTeX tables:")
            for latex_path in stats['latex_table_paths']:
                if 'standalone' in latex_path.name:
                    LOGGER.info(f"  - Standalone: {latex_path}")
                else:
                    LOGGER.info(f"  - Includable: {latex_path}")
        if stats.get('all_tex_count', 0) > 0:
            LOGGER.info(f"✓ Created {stats['all_tex_count']} all.tex compilation files")
    else:
        gap_analysis_csv: Path = GAPS_DIR / "gap_analysis_with_proposals.csv"
        if gap_analysis_csv.exists():
            LOGGER.info(f"Gap analysis with proposals CSV available at: {gap_analysis_csv}")
            latex_includable: Path = gap_analysis_csv.parent / f"{gap_analysis_csv.stem}_table.tex"
            latex_standalone: Path = gap_analysis_csv.parent / f"{gap_analysis_csv.stem}_table_standalone.tex"
            if latex_includable.exists() or latex_standalone.exists():
                LOGGER.info("LaTeX tables available:")
                if latex_includable.exists():
                    LOGGER.info(f"  - Includable: {latex_includable}")
                if latex_standalone.exists():
                    LOGGER.info(f"  - Standalone: {latex_standalone}")
    LOGGER.info("\nAvailable files:")
    LOGGER.info("- Individual proposal files: Located in artifacts/gaps/*/gap_*/responses/")
    LOGGER.info("- Compilation files: all.tex (for inclusion) and standalone.tex (for direct compilation)")
    LOGGER.info("- CSV summary: gap_analysis_with_proposals.csv")
    if stats and stats.get('latex_table_paths'):
        LOGGER.info("- LaTeX tables: Ready for inclusion in documents or standalone compilation")


def interactive_workflow(drop_k: int, max_works: Optional[int], deduplicator, radius_fraction: float, denoiser: str, index_type, hnsw_m: int, hnsw_efc: int, hnsw_efs: int) -> None:
    """Run the interactive batch workflow."""
    LOGGER.info("=" * 60)
    LOGGER.info("     BOOKGAP BATCH WORKFLOW")
    LOGGER.info("     Automated Textbook Proposal Generation")
    LOGGER.info("=" * 60)

    # Build complete configuration by combining CLI args with user prompts
    llm_config, cell_type = get_llm_configuration()
    
    # Create structured configuration objects
    gap_config = GapConfiguration(
        denoiser=denoiser,
        drop_k=drop_k,
        radius_fraction=radius_fraction,
        cell_type=cell_type
    )
    
    config_llm_config = LLMConfiguration(
        requested_num_prompt_articles=llm_config.requested_num_prompt_articles,
        requested_num_prompt_books=llm_config.requested_num_prompt_books
    )
    
    config: AnalysisConfiguration = AnalysisConfiguration.from_components(gap_config, config_llm_config)
    
    LOGGER.info(f"\n✓ Complete Analysis Configuration: {config.to_string()}")
    
    # Check current state and suggest starting point - use config-specific lookups
    results_files: List[Path] = find_batch_results_files_for_config(config)
    batch_files: List[Path] = find_batch_request_files_for_config(config)
    prompt_files: List[Path] = find_prompt_files_for_config(config)
    
    if results_files:
        suggested_step: int = 6
        LOGGER.info(f"Found {len(results_files)} downloaded batch results for current configuration.")
        LOGGER.info("Suggested starting point: Step 6 (Generate LaTeX files)")
    elif batch_files:
        suggested_step = 3
        LOGGER.info(f"Found {len(batch_files)} batch request files for current configuration.")
        LOGGER.info("Suggested starting point: Step 3 (Send to OpenAI)")
    elif prompt_files:
        suggested_step = 2
        LOGGER.info(f"Found {len(prompt_files)} prompt files for current configuration.")
        LOGGER.info("Suggested starting point: Step 2 (Create batch file)")
    else:
        suggested_step = 1
        LOGGER.info("No existing files found for current configuration.")
        LOGGER.info("Suggested starting point: Step 1 (Generate prompts)")
    
    # Let user choose starting step
    current_step: int
    while True:
        try:
            response: str = input(f"\nStart at which step? (1-6, default {suggested_step}): ").strip()
            if not response:
                current_step = suggested_step
                break
            if 1 <= (current_step := int(response)) <= 6:
                break
            else:
                LOGGER.warning("Please enter a number between 1 and 6.")
        except ValueError:
            LOGGER.warning("Please enter a valid number.")
    
    # Execute workflow starting from chosen step
    batch_file: Optional[Path] = None
    batch_id: Optional[str] = None
    downloaded_results_file: Optional[Path] = None

    if current_step <= 1:
        success: bool = step_1_generate_prompts(
            drop_k=drop_k,
            max_works=max_works,
            deduplicator=deduplicator,
            radius_fraction=radius_fraction,
            denoiser=denoiser,
            index_type=index_type,
            hnsw_m=hnsw_m,
            hnsw_efc=hnsw_efc,
            hnsw_efs=hnsw_efs,
            cell_type=gap_config.cell_type,
            llm_config=llm_config,
            config=config
        )
        if not success:
            LOGGER.error("Workflow stopped due to error in Step 1.")
            return
        current_step = 2
    
    if current_step <= 2:
        success, batch_file = step_2_create_batch_file(config)
        if not success:
            LOGGER.error("Workflow stopped due to error in Step 2.")
            return
        current_step = 3
    
    if current_step <= 3:
        success, batch_id = step_3_send_batch_request(config, batch_file)
        if not success:
            LOGGER.error("Workflow stopped due to error in Step 3.")
            return
        current_step = 4
    
    if current_step <= 4:
        success = step_4_monitor_batch_job(batch_id)
        if not success:
            LOGGER.error("Workflow stopped due to error in Step 4.")
            return
        current_step = 5
    
    if current_step <= 5:
        download_result: DownloadBatchResults = step_5_download_batch_results()
        if not download_result.success:
            LOGGER.error("Workflow stopped due to error in Step 5.")
            return
        downloaded_results_file = download_result.results_file
        current_step = 6
    
    if current_step <= 6:
        success = step_6_generate_tex_files(downloaded_results_file, config, max_works, deduplicator, index_type, hnsw_m, hnsw_efc, hnsw_efs)
        if success:
            LOGGER.info("\n🎉 Workflow completed successfully!")
            LOGGER.info("Your textbook proposals are ready for use in LaTeX documents.")


def main(drop_k: int, max_works: Optional[int], deduplicator, radius_fraction: float, denoiser: str,
         index_type, hnsw_m: int, hnsw_efc: int, hnsw_efs: int) -> None:
    """Main function for the batch workflow."""
    interactive_workflow(
        drop_k=drop_k,
        max_works=max_works,
        deduplicator=deduplicator,
        radius_fraction=radius_fraction,
        denoiser=denoiser,
        index_type=index_type,
        hnsw_m=hnsw_m,
        hnsw_efc=hnsw_efc,
        hnsw_efs=hnsw_efs
    )


if __name__ == "__main__":
    from bookgap.script_utils import run_script_main
    run_script_main(main, __file__)
