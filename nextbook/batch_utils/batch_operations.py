#!/usr/bin/env python3
"""
Shared batch operations for OpenAI Batch API.

This module provides common functionality for monitoring, downloading, and processing
OpenAI batch jobs, used by both send_prompts_to_openai.py and check_batch_status.py.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional
from openai import OpenAI

# Configuration
ARTIFACTS_DIR = Path("artifacts")
BATCH_DIR = ARTIFACTS_DIR / "batch_results"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI client
client = OpenAI()

def monitor_batch_job(batch_id: str, poll_interval: int = 30) -> Optional[Any]:
    """
    Monitor a batch job until completion.
    
    Args:
        batch_id: The batch job ID to monitor
        poll_interval: Seconds to wait between status checks
        
    Returns:
        The final batch status object, or None if monitoring failed
    """
    print(f"Monitoring batch job {batch_id}...")
    
    while True:
        try:
            batch_status = client.batches.retrieve(batch_id)
            status = batch_status.status
            
            print(f"Batch status: {status}")
            
            if status == "completed":
                print("Batch completed successfully!")
                return batch_status
            elif status == "failed":
                print("Batch failed!")
                if batch_status.errors:
                    print(f"Error details: {batch_status.errors}")
                return batch_status
            elif status == "cancelled":
                print("Batch was cancelled!")
                return batch_status
            elif status in ["validating", "in_progress", "finalizing"]:
                print(f"Batch is {status}... waiting {poll_interval} seconds")
                time.sleep(poll_interval)
            else:
                print(f"Unknown status: {status}")
                time.sleep(poll_interval)
                
        except Exception as e:
            print(f"Error checking batch status: {e}")
            time.sleep(poll_interval)

def download_batch_results(batch_status: Any) -> Path:
    """
    Download the results file from a completed batch job.
    
    Args:
        batch_status: The batch status object from OpenAI
        
    Returns:
        Path to the downloaded results file
        
    Raises:
        ValueError: If batch is not completed or no output file ID
    """
    if batch_status.status != "completed":
        raise ValueError(f"Batch not completed. Status: {batch_status.status}")
    
    output_file_id = batch_status.output_file_id
    if not output_file_id:
        raise ValueError("No output file ID found")
    
    print(f"Downloading results from file ID: {output_file_id}")
    
    try:
        # Download the results file
        file_response = client.files.content(output_file_id)
        
        # Save the results
        results_file = BATCH_DIR / f"results_{batch_status.id}.jsonl"
        with open(results_file, 'wb') as f:
            f.write(file_response.content)
        
        print(f"Results saved to: {results_file}")
        return results_file
        
    except Exception as e:
        print(f"Error downloading results: {e}")
        raise

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

def create_all_tex_files(output_dir: Path) -> int:
    """
    Create all.tex files in each gap subdirectory that include all individual proposals.
    
    Args:
        output_dir: The base output directory containing config directories
        
    Returns:
        Number of all.tex files created
    """
    print("Creating all.tex files for gap subdirectories...")
    
    all_tex_created = 0
    
    # Iterate through config directories (e.g., NONE_k=0_radiusfrac=0.75)
    for config_dir in output_dir.iterdir():
        if not config_dir.is_dir():
            continue
            
        # Look for gap subdirectories (gap_0, gap_1, etc.)
        gap_dirs = [d for d in config_dir.iterdir() if d.is_dir() and d.name.startswith('gap_')]
        
        for gap_dir in gap_dirs:
            # Find all .tex files (excluding all.tex if it already exists)
            tex_files = [f for f in gap_dir.glob("*.tex") if f.name != "all.tex"]
            
            if not tex_files:
                continue
                
            # Sort files for consistent ordering
            tex_files.sort(key=lambda f: f.name)
            
            # Create all.tex file (for use within the gap directory)
            all_tex_path = gap_dir / "all.tex"
            with open(all_tex_path, 'w', encoding='utf-8') as f:
                # Include each proposal on a separate page
                for i, tex_file in enumerate(tex_files):
                    # Extract topic name from filename (remove _gap_X_proposal.tex suffix)
                    topic_name = tex_file.stem
                    if topic_name.endswith('_proposal'):
                        # Remove _gap_X_proposal suffix
                        parts = topic_name.split('_gap_')
                        if len(parts) > 1:
                            topic_name = parts[0]
                    
                    f.write(f"\\section{{{topic_name.replace('_', ' ')}}}\n")
                    f.write(f"\\input{{{tex_file.name[:-4]}}}\n")  # Local reference, no path
                    
                    # Add page break between proposals (except for the last one)
                    if i < len(tex_files) - 1:
                        f.write("\\clearpage\n\n")
                    else:
                        f.write("\n")
            
            # Create all_with_path.tex file (for including from parent directories)
            all_with_path = gap_dir / "all_with_path.tex"
            with open(all_with_path, 'w', encoding='utf-8') as f:
                # Include each proposal on a separate page with relative paths
                for i, tex_file in enumerate(tex_files):
                    # Extract topic name from filename (remove _gap_X_proposal.tex suffix)
                    topic_name = tex_file.stem
                    if topic_name.endswith('_proposal'):
                        # Remove _gap_X_proposal suffix
                        parts = topic_name.split('_gap_')
                        if len(parts) > 1:
                            topic_name = parts[0]
                    
                    f.write(f"\\section{{{topic_name.replace('_', ' ')}}}\n")
                    f.write(f"\\input{{{gap_dir.name}/{tex_file.name[:-4]}}}\n")  # Include gap_X/ prefix
                    
                    # Add page break between proposals (except for the last one)
                    if i < len(tex_files) - 1:
                        f.write("\\clearpage\n\n")
                    else:
                        f.write("\n")
            
            # Create standalone.tex file (complete document for direct compilation)
            standalone_path = gap_dir / "standalone.tex"
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
                f.write("\\geometry{margin=1in}\n\n")
                
                f.write("\\title{Textbook Proposals Collection}\n")
                f.write(f"\\author{{Gap Analysis: {gap_dir.name}}}\n")
                f.write("\\date{\\today}\n\n")
                
                f.write("\\begin{document}\n")
                f.write("\\maketitle\n")
                f.write("\\tableofcontents\n")
                f.write("\\clearpage\n\n")
                
                # Include the local all.tex content
                f.write("\\input{all}\n")
                
                f.write("\\end{document}\n")
            
            print(f"Created {all_tex_path} (local), {all_with_path} (with paths), and {standalone_path} (standalone) with {len(tex_files)} proposals")
            all_tex_created += 1
    
    return all_tex_created

def extract_textbook_proposals(results_file: Path, output_dir: Optional[Path] = None, batch_id: Optional[str] = None, split_by_gap_idx: bool = True) -> Dict[str, Any]:
    """
    Extract and save individual textbook proposals from batch results.
    
    Args:
        results_file: Path to the JSONL results file
        output_dir: Directory to save proposals (optional, will create unique dir if not provided)
        batch_id: Batch ID for creating unique directory structure
        split_by_gap_idx: If True, create subdirectories for each gap index (gap_0/, gap_1/, etc.)
        
    Returns:
        Dictionary with extraction statistics
    """
    if output_dir is None:
        if batch_id:
            output_dir = BATCH_DIR / f"batch_{batch_id}" / "textbook_proposals"
        else:
            # Extract batch ID from results filename if possible
            filename = results_file.name
            if filename.startswith("results_") and filename.endswith(".jsonl"):
                extracted_batch_id = filename[8:-6]  # Remove "results_" and ".jsonl"
                output_dir = BATCH_DIR / f"batch_{extracted_batch_id}" / "textbook_proposals"
            else:
                output_dir = BATCH_DIR / "textbook_proposals"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful_count = 0
    failed_count = 0
    
    print("Extracting textbook proposals...")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line.strip())
            custom_id = result['custom_id']
            
            if result.get('response') and result['response'].get('status_code') == 200:
                # Successful response
                raw_message = result['response']['body']['choices'][0]['message']['content']
                
                # Clean the LaTeX content
                cleaned_message = clean_latex_content(raw_message)
                
                # Parse custom_id to get file structure
                # Format: "gap_config___topic_name___gap_X"
                parts = custom_id.split("___")
                if len(parts) >= 3:
                    gap_config = parts[0]
                    topic_name = parts[1]
                    gap_info = parts[2]  # "gap_X"
                    
                    # Create directory structure
                    config_dir = output_dir / gap_config
                    
                    if split_by_gap_idx:
                        # Extract gap index from gap_info (e.g., "gap_0" -> "gap_0")
                        gap_dir = config_dir / gap_info
                        gap_dir.mkdir(parents=True, exist_ok=True)
                        # Save the proposal in the gap subdirectory
                        proposal_file = gap_dir / f"{topic_name}_{gap_info}_proposal.tex"
                    else:
                        # Save directly in config directory
                        config_dir.mkdir(exist_ok=True)
                        proposal_file = config_dir / f"{topic_name}_{gap_info}_proposal.tex"
                    
                    with open(proposal_file, 'w', encoding='utf-8') as pf:
                        pf.write(cleaned_message)
                    
                    successful_count += 1
                    print(f"Saved proposal: {proposal_file}")
                else:
                    print(f"Warning: Invalid custom_id format: {custom_id}")
                    failed_count += 1
                
            else:
                # Failed response
                print(f"Failed request for {custom_id}")
                if result.get('error'):
                    print(f"Error: {result['error']}")
                failed_count += 1
                
                # Save error details
                error_file = output_dir / f"failed_{custom_id}.json"
                with open(error_file, 'w') as ef:
                    json.dump(result, ef, indent=2)
    
    stats = {
        'successful_count': successful_count,
        'failed_count': failed_count,
        'output_dir': output_dir
    }
    
    # Create all.tex files for gap subdirectories when split_by_gap_idx is enabled
    if split_by_gap_idx and successful_count > 0:
        all_tex_count = create_all_tex_files(output_dir)
        if all_tex_count > 0:
            print(f"Created {all_tex_count} all.tex compilation files")
    
    print(f"\nExtraction summary:")
    print(f"Successful proposals: {successful_count}")
    print(f"Failed requests: {failed_count}")
    print(f"Proposals saved to: {output_dir}")
    if split_by_gap_idx and successful_count > 0:
        print("Use all.tex files to include in existing documents, or standalone.tex files for direct compilation")
    
    return stats

def list_batch_jobs(limit: int = 20) -> None:
    """List all batch jobs with their status information."""
    print("Listing all batch jobs...")
    
    try:
        batches = client.batches.list(limit=limit)
        
        if not batches.data:
            print("No batch jobs found.")
            return
        
        for batch in batches.data:
            print(f"\nBatch ID: {batch.id}")
            print(f"Status: {batch.status}")
            print(f"Created: {batch.created_at}")
            print(f"Endpoint: {batch.endpoint}")
            if batch.metadata:
                print(f"Description: {batch.metadata.get('description', 'N/A')}")
            
            if batch.status == "completed":
                print(f"Completed at: {batch.completed_at}")
                print(f"Request counts: {batch.request_counts}")
            elif batch.status == "failed":
                print(f"Failed at: {batch.failed_at}")
                if batch.errors:
                    print(f"Errors: {batch.errors}")
                    
    except Exception as e:
        print(f"Error listing batch jobs: {e}")

def get_batch_status(batch_id: str) -> Optional[Any]:
    """
    Get the status of a specific batch job.
    
    Args:
        batch_id: The batch job ID to check
        
    Returns:
        The batch status object, or None if retrieval failed
    """
    try:
        batch = client.batches.retrieve(batch_id)
        print(f"\nBatch ID: {batch.id}")
        print(f"Status: {batch.status}")
        print(f"Created: {batch.created_at}")
        
        if batch.status == "completed":
            print(f"Completed at: {batch.completed_at}")
            print(f"Request counts: {batch.request_counts}")
            print(f"Output file ID: {batch.output_file_id}")
        elif batch.status == "failed":
            print(f"Failed at: {batch.failed_at}")
            if batch.errors:
                print(f"Errors: {batch.errors}")
        elif batch.status in ["validating", "in_progress", "finalizing"]:
            print(f"Batch is still {batch.status}")
        
        return batch
    except Exception as e:
        print(f"Error checking batch status: {e}")
        return None

def process_completed_batch(batch_id: str, output_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Download and process results for a completed batch job.
    
    Args:
        batch_id: The batch job ID to process
        output_dir: Directory to save proposals (optional)
        
    Returns:
        Dictionary with processing statistics, or None if failed
    """
    batch = get_batch_status(batch_id)
    
    if not batch:
        return None
    
    if batch.status != "completed":
        print(f"Batch is not completed. Current status: {batch.status}")
        return None
    
    try:
        # Download results
        results_file = download_batch_results(batch)
        
        # Extract proposals with batch-specific directory
        stats = extract_textbook_proposals(results_file, output_dir, batch_id)
        
        return stats
        
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None
