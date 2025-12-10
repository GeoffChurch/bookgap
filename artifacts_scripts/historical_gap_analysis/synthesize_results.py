#!/usr/bin/env python3
"""Synthesize historical gap analysis results.

Combines all summary CSV files from the results directory into a single CSV file.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from bookgap.util import setup_logging, LOGGER, ARTIFACTS_DIR

def synthesize_results(results_dir: Path, output_file: Path) -> None:
    """Combine all summary CSVs in results_dir into output_file."""
    if not results_dir.exists():
        LOGGER.error(f"Results directory not found: {results_dir}")
        return

    # Find all summary CSVs
    summary_files = list(results_dir.glob("historical_gap_analysis_summary_*.csv"))
    
    if not summary_files:
        LOGGER.warning(f"No summary CSV files found in {results_dir}")
        return

    LOGGER.info(f"Found {len(summary_files)} summary files")

    dfs = []
    for csv_file in summary_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            LOGGER.error(f"Error reading {csv_file}: {e}")

    if not dfs:
        LOGGER.warning("No data loaded")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by year and other parameters for better readability
    sort_cols = [c for c in ['year', 'p', 'embedder_model', 'denoiser', 'drop_k'] if c in combined_df.columns]
    if sort_cols:
        combined_df = combined_df.sort_values(by=sort_cols)

    LOGGER.info(f"Combined data shape: {combined_df.shape}")
    
    combined_df.to_csv(output_file, index=False)
    LOGGER.info(f"Written combined results to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Synthesize historical gap analysis results")
    parser.add_argument(
        "--results-dir", 
        type=Path, 
        default=Path(__file__).parent / "results",
        help="Directory containing result CSVs"
    )
    parser.add_argument(
        "--output-file", 
        type=Path, 
        default=None,
        help="Output CSV file path (default: results_dir/historical_gap_analysis_combined.csv)"
    )
    
    args = parser.parse_args()
    
    setup_logging(log_name="synthesize_results", log_level="INFO", include_time=True)
    
    output_file = args.output_file
    if output_file is None:
        output_dir = ARTIFACTS_DIR / "historical_gap_analysis" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "historical_gap_analysis_combined.csv"
        
    synthesize_results(args.results_dir, output_file)

if __name__ == "__main__":
    main()
