#!/usr/bin/env python3
"""
Script to send all gap analysis prompts to OpenAI using their Batch API.

This script:
1. Finds all _prompt.txt files in the artifacts/gaps directory
2. Creates a JSONL file with batch requests for OpenAI
3. Uploads the batch file to OpenAI
4. Creates and submits a batch job
5. Monitors the batch job until completion
6. Downloads and saves the results
"""

import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import openai
from openai import OpenAI
from . import batch_operations

# Configuration
ARTIFACTS_DIR = Path("artifacts")
GAPS_DIR = ARTIFACTS_DIR / "gaps"
BATCH_DIR = ARTIFACTS_DIR / "batch_results"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI client
client = OpenAI()  # Requires OPENAI_API_KEY environment variable

def find_prompt_files() -> List[Path]:
    """Find all _prompt.txt files in the gaps directory."""
    prompt_files = []
    for root, dirs, files in os.walk(GAPS_DIR):
        for file in files:
            if file.endswith("_prompt.txt"):
                prompt_files.append(Path(root) / file)
    return sorted(prompt_files)

def read_prompt_content(prompt_file: Path) -> str:
    """Read the content of a prompt file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

def create_batch_requests(prompt_files: List[Path]) -> List[Dict[str, Any]]:
    """Create batch request objects for OpenAI API."""
    requests = []
    
    GAP_IDX_PAT = re.compile(r'gap_(\d+)')
    for i, prompt_file in enumerate(prompt_files):
        # Extract identifying information from the file path
        relative_path = prompt_file.relative_to(GAPS_DIR)
        gap_config = relative_path.parts[0]  # e.g., "NONE_k=0_radiusfrac=0.75"
        
        # Handle new 3-level directory structure: config/gap_X/topic_gap_X_prompt.txt
        if len(relative_path.parts) >= 3:
            gap_dir = relative_path.parts[1]     # e.g., "gap_0"
            filename = relative_path.parts[2]    # e.g., "AI and Multimedia in Education_gap_0_prompt.txt"
            
            # Extract gap index from directory name (gap_X)
            # assert gap_dir.startswith("gap_")
            # gap_idx = gap_dir[4:]  # Remove "gap_" prefix
            match = GAP_IDX_PAT.match(gap_dir)
            assert match
            gap_idx = match.group(1)
                
        else:
            # Fallback for old 2-level structure
            filename = relative_path.parts[1] if len(relative_path.parts) > 1 else relative_path.name
            gap_idx = None
        
        # Extract topic name from filename
        # Format: "Topic_Name_gap_X_prompt.txt"
        base_name = filename.replace("_prompt.txt", "")
        parts = base_name.split("_gap_")
        topic_name = parts[0] if len(parts) > 1 else base_name
        
        # Use gap_idx from directory name if available, otherwise from filename
        if len(relative_path.parts) < 3 and len(parts) > 1:
            gap_idx = parts[1]
        
        prompt_content = read_prompt_content(prompt_file)
        
        request = {
            "custom_id": f"{gap_config}___{topic_name}___gap_{gap_idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4.1",  # Best model for complex academic synthesis
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_content
                    }
                ],
                "max_tokens": 2000,  # Sufficient for textbook proposals
                "temperature": 0.7   # Some creativity for textbook proposals
            }
        }
        requests.append(request)
    
    print(f"Created {len(requests)} batch requests")
    return requests

def create_batch_file(requests: List[Dict[str, Any]]) -> Path:
    """Create a JSONL file with all batch requests."""
    batch_file = BATCH_DIR / f"textbook_proposals_batch_{int(time.time())}.jsonl"
    
    with open(batch_file, 'w', encoding='utf-8') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    
    print(f"Created batch file: {batch_file}")
    print(f"Batch file size: {batch_file.stat().st_size / 1024:.1f} KB")
    return batch_file

def upload_batch_file(batch_file: Path) -> str:
    """Upload the batch file to OpenAI and return the file ID."""
    print(f"Uploading batch file to OpenAI...")
    
    with open(batch_file, 'rb') as f:
        file_response = client.files.create(
            file=f,
            purpose="batch"
        )
    
    file_id = file_response.id
    print(f"Uploaded file ID: {file_id}")
    return file_id

def create_batch_job(file_id: str) -> str:
    """Create a batch job and return the batch ID."""
    print("Creating batch job...")
    
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",  # Complete within 24 hours
        metadata={
            "description": "Textbook proposals for research gaps",
            "project": "nextbook"
        }
    )
    
    batch_id = batch_response.id
    print(f"Created batch job ID: {batch_id}")
    return batch_id

def estimate_cost(prompt_lengths: List[int]) -> None:
    """Estimate the cost of the batch job based on actual prompt lengths."""
    # Convert character counts to approximate token counts (rough estimate: 4 chars per token)
    prompt_tokens = [length // 4 for length in prompt_lengths]
    
    # Calculate statistics
    prompt_tokens_array = np.array(prompt_tokens)
    num_requests = len(prompt_tokens)
    
    min_tokens = np.min(prompt_tokens_array)
    max_tokens = np.max(prompt_tokens_array)
    mean_tokens = np.mean(prompt_tokens_array)
    
    # Calculate percentiles
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    percentile_values = np.percentile(prompt_tokens_array, percentiles)
    
    print(f"\nPrompt length analysis ({num_requests} prompts):")
    print(f"Min tokens: {min_tokens:,}")
    for p, val in zip(percentiles, percentile_values):
        print(f"{p}th percentile: {val:,.0f} tokens")
    print(f"Max tokens: {max_tokens:,}")
    print(f"Mean tokens: {mean_tokens:,.0f}")
    
    # GPT-4.1 pricing: $2.00 per 1M input tokens, $8.00 per 1M output tokens
    # Estimate ~800 output tokens per response (for textbook proposals)
    total_input_tokens = sum(prompt_tokens)
    estimated_output_tokens = num_requests * 800
    
    input_cost = (total_input_tokens / 1_000_000) * 2.00
    output_cost = (estimated_output_tokens / 1_000_000) * 8.00
    total_cost = input_cost + output_cost
    
    print(f"\nCost estimate (GPT-4.1):")
    print(f"Total input tokens: {total_input_tokens:,}")
    print(f"Estimated output tokens: {estimated_output_tokens:,}")
    print(f"Estimated input cost: ${input_cost:.3f}")
    print(f"Estimated output cost: ${output_cost:.3f}")
    print(f"Estimated total cost: ${total_cost:.3f}")
    print(f"Note: Batch API provides 50% discount, so actual cost: ${total_cost * 0.5:.3f}")
    print(f"This is for high-quality academic synthesis - excellent value for research!")

def main():
    """Main function to run the batch processing."""
    print("Starting OpenAI Batch API processing for textbook proposals...")
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Step 1: Find all prompt files
        prompt_files = find_prompt_files()
        if not prompt_files:
            print("No prompt files found in the gaps directory!")
            return
        
        print(f"Found {len(prompt_files)} prompt files")
        
        # Step 2: Calculate prompt lengths and estimate cost
        print("Analyzing prompt lengths...")
        prompt_lengths = []
        for prompt_file in prompt_files:
            content = read_prompt_content(prompt_file)
            prompt_lengths.append(len(content))
        
        estimate_cost(prompt_lengths)
        
        # Ask for confirmation
        response = input("\nDo you want to proceed with the batch job? (y/N): ")
        if response.lower() != 'y':
            print("Batch job cancelled.")
            return
        
        # Step 3: Create batch requests
        requests = create_batch_requests(prompt_files)
        
        # Step 4: Create batch file
        batch_file = create_batch_file(requests)
        
        # Step 5: Upload batch file
        file_id = upload_batch_file(batch_file)
        
        # Step 6: Create batch job
        batch_id = create_batch_job(file_id)
        
        # Step 7: Monitor batch job using shared module
        batch_status = batch_operations.monitor_batch_job(batch_id)
        
        # Step 8: Download and process results using shared module
        if batch_status and batch_status.status == "completed":
            results_file = batch_operations.download_batch_results(batch_status)
            batch_operations.extract_textbook_proposals(results_file)
            print("\nBatch processing completed successfully!")
        else:
            status = batch_status.status if batch_status else "unknown"
            print(f"Batch processing failed with status: {status}")
    
    except Exception as e:
        print(f"Error during batch processing: {e}")
        raise

if __name__ == "__main__":
    main()
