#!/usr/bin/env python3
"""
Script to extract textbook proposals from all existing batch results files.

This script finds all results_batch_*.jsonl files in the batch_results directory
and extracts the textbook proposals from each one into organized directories.
"""

import argparse
import glob
import os
from pathlib import Path
from batch_operations import extract_textbook_proposals, BATCH_DIR

def find_results_files(file_pattern: str | None = None) -> list[Path]:
    """Find existing batch results files based on pattern or default search."""
    results_files = []
    
    if not BATCH_DIR.exists():
        print(f"Batch directory {BATCH_DIR} does not exist.")
        return results_files
    
    if file_pattern:
        # User specified a pattern or file
        pattern_path = Path(file_pattern)
        
        if pattern_path.is_absolute():
            # Absolute path provided
            if pattern_path.exists() and pattern_path.suffix == '.jsonl':
                results_files.append(pattern_path)
            else:
                # Try as glob pattern
                matches = glob.glob(file_pattern)
                for match in matches:
                    match_path = Path(match)
                    if match_path.suffix == '.jsonl':
                        results_files.append(match_path)
        else:
            # Relative path - look in BATCH_DIR
            if '*' in file_pattern or '?' in file_pattern:
                # It's a glob pattern
                for file_path in BATCH_DIR.glob(file_pattern):
                    if file_path.is_file() and file_path.suffix == '.jsonl':
                        results_files.append(file_path)
            else:
                # Direct filename
                file_path = BATCH_DIR / file_pattern
                if file_path.exists() and file_path.suffix == '.jsonl':
                    results_files.append(file_path)
                else:
                    # Maybe it's in current directory
                    file_path = Path(file_pattern)
                    if file_path.exists() and file_path.suffix == '.jsonl':
                        results_files.append(file_path)
    else:
        # Default behavior - find all results files
        # Look for results_batch_*.jsonl files
        pattern = "results_batch_*.jsonl"
        for file_path in BATCH_DIR.glob(pattern):
            if file_path.is_file():
                results_files.append(file_path)
        
        # Also look for results_*.jsonl files (broader pattern)
        pattern2 = "results_*.jsonl"
        for file_path in BATCH_DIR.glob(pattern2):
            if file_path.is_file() and file_path not in results_files:
                results_files.append(file_path)
    
    return sorted(results_files)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract textbook proposals from OpenAI batch results files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Extract from all results files
  %(prog)s results_batch_*.jsonl              # Extract from specific pattern
  %(prog)s results_batch_12345.jsonl          # Extract from specific file
  %(prog)s artifacts/batch_results/results_*.jsonl  # Extract with full path pattern
  %(prog)s --split-by-gap-idx                 # Extract with gap index subdirectories
  %(prog)s results_batch_*.jsonl --split-by-gap-idx  # Combine pattern with gap subdirs
        """
    )
    
    parser.add_argument(
        'files',
        nargs='?',
        help='Specific batch results file or glob pattern to process. '
             'If not provided, processes all results_*.jsonl files in batch_results directory.'
    )
    
    parser.add_argument(
        '--split-by-gap-idx',
        action='store_true',
        help='Create subdirectories for each gap index (gap_0/, gap_1/, etc.). '
             'When enabled, proposals will be organized in nested directories by gap index, '
             'and two compilation files will be created: all.tex (includable) and standalone.tex (complete document).'
    )
    
    return parser.parse_args()

def main():
    """Extract proposals from batch results."""
    args = parse_arguments()
    
    if args.files:
        print(f"Extracting textbook proposals from: {args.files}")
        results_files = find_results_files(args.files)
    else:
        print("Extracting textbook proposals from all existing batch results...")
        results_files = find_results_files()
    
    if args.split_by_gap_idx:
        print("Gap index splitting enabled: proposals will be organized in gap_X/ subdirectories")
    
    if not results_files:
        if args.files:
            print(f"No batch results files found matching: {args.files}")
            print(f"Looking in: {BATCH_DIR}")
            print("Make sure the file exists and has a .jsonl extension.")
        else:
            print("No batch results files found.")
            print(f"Looking in: {BATCH_DIR}")
            print("Expected pattern: results_batch_*.jsonl or results_*.jsonl")
        return
    
    print(f"Found {len(results_files)} results files:")
    for file_path in results_files:
        print(f"  - {file_path.name}")
    
    total_successful = 0
    total_failed = 0
    
    for results_file in results_files:
        print(f"\nProcessing: {results_file.name}")
        try:
            stats = extract_textbook_proposals(results_file, split_by_gap_idx=args.split_by_gap_idx)
            total_successful += stats['successful_count']
            total_failed += stats['failed_count']
            print(f"  Extracted {stats['successful_count']} proposals")
            if stats['failed_count'] > 0:
                print(f"  Failed: {stats['failed_count']} requests")
        except Exception as e:
            print(f"  Error processing {results_file.name}: {e}")
            total_failed += 1
    
    print(f"\n" + "="*50)
    print(f"SUMMARY:")
    print(f"Total successful proposals: {total_successful}")
    print(f"Total failed extractions: {total_failed}")
    print(f"Proposals organized in: {BATCH_DIR}")
    
    # List the batch directories created
    batch_dirs = [d for d in BATCH_DIR.iterdir() if d.is_dir() and d.name.startswith('batch_')]
    if batch_dirs:
        print(f"\nBatch directories created:")
        for batch_dir in sorted(batch_dirs):
            proposals_dir = batch_dir / "textbook_proposals"
            if proposals_dir.exists():
                config_dirs = [d for d in proposals_dir.iterdir() if d.is_dir()]
                total_files = sum(len(list(d.glob("*.tex"))) for d in config_dirs)
                print(f"  {batch_dir.name}: {total_files} proposal files")

if __name__ == "__main__":
    main()
