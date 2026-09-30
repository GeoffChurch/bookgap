#!/usr/bin/env python3
"""Plot synthesized historical gap analysis results.

Reads the combined results CSV and generates plots showing the evolution of gap sizes over years.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import cast

from bookgap.util import setup_logging, LOGGER, create_unicode_safe_filename, ARTIFACTS_DIR
from artifacts_scripts.common import friendly_embedder_name, friendly_denoiser_name, latex_provenance

# =============================================================================
# Configuration
# =============================================================================

# Which work types to include in plots
SHOW_PRIOR_BOOKS = True
SHOW_NEW_BOOKS = True
SHOW_NEW_ARTICLES = False

# Whether to show uncertainty band shading
SHOW_SHADING = False

# Percentiles for uncertainty bands (lower, upper)
BAND_PERCENTILE_LOW = 50
BAND_PERCENTILE_HIGH = 80

# Year range filter
MIN_YEAR = 1999
MAX_YEAR = None

# Whether to show the title in LaTeX output
SHOW_LATEX_TITLE = False

def _format_config_desc(group_cols: list[str], values: tuple | str, latex: bool = False) -> str:
    """Format configuration description with friendly names."""
    if not isinstance(values, tuple):
        values = (values,)
    
    parts = []
    for k, v in zip(group_cols, values):
        if k == 'embedder_model':
            parts.append(f"embedder={friendly_embedder_name(v)}")
        elif k == 'denoiser':
            parts.append(f"denoiser={friendly_denoiser_name(v, latex=latex)}")
        elif k == 'drop_k':
            parts.append(f"drop {v}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def generate_latex_plot(
    group_df: pd.DataFrame,
    series_config: list[tuple[str, str, str, str, str]],
    config_desc: str,
) -> str:
    """Generate TikZ/PGFPlots code for the gap evolution plot."""
    # Escape underscores in config_desc for LaTeX if they aren't already in math mode
    # Simple heuristic: if it doesn't have $, escape underscores
    if "$" not in config_desc:
        safe_config_desc = config_desc.replace("_", r"\_")
    else:
        # More careful escaping for mixed content
        safe_config_desc = config_desc.replace("_", r"\_") # Still probably safe
    
    latex_lines = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=\linewidth, height=0.7\linewidth,",
        r"    font=\footnotesize,",
        r"    xlabel={Year},",
        r"    ylabel={Mean Gap Size},",
        r"    legend pos=north west,",
        r"    grid=major,",
        r"    grid style={dashed,gray!30},",
        r"    xtick distance=5,",
        r"    xticklabel={\pgfmathprintnumber[fixed, precision=0, 1000 sep={}]{\tick}},",
    ]
    
    if SHOW_LATEX_TITLE:
        latex_lines.append(r"    title={" + safe_config_desc + "},")
        
    latex_lines.extend([
        r"]"
    ])

    for prefix, label, color, mean_style, _ in series_config:
        mean_col = f"{prefix}_mean"
        if mean_col not in group_df.columns:
            continue
            
        # Map matplotlib styles to pgfplots styles
        pgf_color = color
        pgf_style = f"color={pgf_color}"
        
        if 'o' in mean_style:
            pgf_style += ", mark=*"
        elif 's' in mean_style:
            pgf_style += ", mark=square*"
        elif '^' in mean_style:
            pgf_style += ", mark=triangle*"
        elif 'x' in mean_style:
            pgf_style += ", mark=x"
        elif 'd' in mean_style:
            pgf_style += ", mark=diamond*"
        elif 'v' in mean_style:
            pgf_style += ", mark=triangle*, mark options={rotate=180}"
        else:
            pgf_style += ", mark=*"

        if '--' in mean_style:
            pgf_style += ", dashed"
        elif ':' in mean_style:
            pgf_style += ", dotted"
        elif '-.' in mean_style:
            pgf_style += ", dashdotted"
        else:
            pgf_style += ", solid"

        latex_lines.append(rf"    \addplot[{pgf_style}, line width=0.6pt, mark size=1.5pt] coordinates {{")
        for _, row in group_df.iterrows():
            val = row[mean_col]
            if not pd.isna(val):
                latex_lines.append(f"        ({int(row['plot_year'])}, {val:.4f})")
        latex_lines.append(r"    };")
        latex_lines.append(rf"    \addlegendentry{{{label}}}")

    latex_lines.append(r"\end{axis}")
    latex_lines.append(r"\end{tikzpicture}")
    
    return "\n".join(latex_lines)


def plot_results(
    input_file: Path, 
    output_dir: Path,
) -> None:
    """Generate plots from the combined results CSV.
    
    Args:
        input_file: Path to the combined results CSV.
        output_dir: Directory to save plots.
    """
    if not input_file.exists():
        LOGGER.error(f"Input file not found: {input_file}")
        return

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        LOGGER.error(f"Error reading {input_file}: {e}")
        return

    if df.empty:
        LOGGER.warning("Input dataframe is empty")
        return

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter by year range
    if MIN_YEAR is not None:
        df = df[df['year'] >= MIN_YEAR]
    if MAX_YEAR is not None:
        df = df[df['year'] <= MAX_YEAR]
        
    if df.empty:
        LOGGER.warning(f"No data found in year range {MIN_YEAR}-{MAX_YEAR}")
        return

    # Shift display years forward so plots align with publication year of test sets
    df = df.copy()
    df['plot_year'] = df['year'] + 1

    # Group by configuration parameters
    group_cols = ['p', 'embedder_model', 'denoiser', 'drop_k']
    # Filter to columns that actually exist in the dataframe
    group_cols = [c for c in group_cols if c in df.columns]
    
    if not group_cols:
        # If none of the config columns exist, just plot everything as one group (unlikely)
        grouped = [('default', df)]
    else:
        # Convert to list to allow multiple iterations (generators exhaust after one pass)
        grouped = list(df.groupby(group_cols))

    for name, group in grouped:
        # Prepare the group name for filename
        if isinstance(name, tuple):
            name_str = "_".join(str(x) for x in name)
            config_desc = _format_config_desc(group_cols, name)
            config_desc_latex = _format_config_desc(group_cols, name, latex=True)
        else:
            name_str = str(name)
            config_desc = _format_config_desc(group_cols, name)
            config_desc_latex = _format_config_desc(group_cols, name, latex=True)
            
        safe_name = create_unicode_safe_filename(name_str)
        
        # Sort by year to ensure correct line plotting
        group_df = cast(pd.DataFrame, group).sort_values(by='plot_year')
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot series - filter based on global show flags
        series_config = []
        if SHOW_PRIOR_BOOKS:
            series_config.append(('landmark_gap_size', 'Prior Books', 'gray', 'o-', 'x-'))
        if SHOW_NEW_BOOKS:
            series_config.append(('book_gap_size', 'New Books', 'red', 's-', 'd-'))
        if SHOW_NEW_ARTICLES:
            series_config.append(('article_gap_size', 'New Articles', 'blue', '^:', 'v:'))
        
        for prefix, label, color, mean_style, median_style in series_config:
            mean_col = f"{prefix}_mean"
            std_col = f"{prefix}_std"
            band_low_col = f"{prefix}_p{BAND_PERCENTILE_LOW}"
            band_high_col = f"{prefix}_p{BAND_PERCENTILE_HIGH}"
            
            # Plot mean line
            if mean_col in group_df.columns:
                plt.plot(group_df['plot_year'], group_df[mean_col], mean_style, label=label, 
                        color=color, linewidth=2, markersize=8)
            
            # Fill uncertainty band - prefer percentile band if available, else use std
            if SHOW_SHADING:
                if band_low_col in group_df.columns and band_high_col in group_df.columns:
                    plt.fill_between(
                        group_df['plot_year'], 
                        group_df[band_low_col], 
                        group_df[band_high_col], 
                        color=color, 
                        alpha=0.15,
                        label=f"{label} (p{BAND_PERCENTILE_LOW}-p{BAND_PERCENTILE_HIGH})"
                    )
                elif std_col in group_df.columns and mean_col in group_df.columns:
                    lower = group_df[mean_col] - group_df[std_col]
                    upper = group_df[mean_col] + group_df[std_col]
                    plt.fill_between(
                        group_df['plot_year'], 
                        lower, 
                        upper, 
                        color=color, 
                        alpha=0.15
                    )
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Mean Gap Size', fontsize=12)
        plt.title('Evolution of Gap Sizes Over Time', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Use integer ticks for years, formatted as last two digits
        unique_years = sorted(group_df['plot_year'].unique())
        plt.xticks(unique_years, [str(y)[-2:] for y in unique_years], rotation=45)
        
        output_path = output_dir / f"gap_evolution_{safe_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        LOGGER.info(f"Saved plot to {output_path}")

        # Generate and save LaTeX version
        latex_code = generate_latex_plot(group_df, series_config, config_desc_latex)
        latex_path = output_dir / f"gap_evolution_{safe_name}.tex"
        with open(latex_path, "w") as f:
            f.write(latex_code + latex_provenance(__spec__.name))
        LOGGER.info(f"Saved LaTeX plot to {latex_path}")

    # --- New Plotting Logic: Compare configurations for each metric ---
    
    # Filter metrics based on global show flags
    metrics = []
    if SHOW_PRIOR_BOOKS:
        metrics.append(('landmark_gap_size', 'Prior Books (Landmarks)'))
    if SHOW_NEW_BOOKS:
        metrics.append(('book_gap_size', 'New Books'))
    if SHOW_NEW_ARTICLES:
        metrics.append(('article_gap_size', 'New Articles'))
    
    # Define a color palette for different configurations
    # Using a qualitative palette with enough colors
    palette = sns.color_palette("husl", len(grouped))
    
    for metric_prefix, metric_label in metrics:
        mean_col = f"{metric_prefix}_mean"
        median_col = f"{metric_prefix}_p50"
        std_col = f"{metric_prefix}_std"
        band_low_col = f"{metric_prefix}_p{BAND_PERCENTILE_LOW}"
        band_high_col = f"{metric_prefix}_p{BAND_PERCENTILE_HIGH}"
        
        # Need at least mean or median to plot
        if mean_col not in df.columns and median_col not in df.columns:
            continue
            
        # Create two figures: one for mean, one for median
        for stat_type, stat_col in [('Mean', mean_col), ('Median', median_col)]:
            if stat_col not in df.columns:
                continue
                
            plt.figure(figsize=(12, 8))
            
            # Iterate through each configuration group
            for (name, group), color in zip(grouped, palette):
                # Prepare label for the legend
                label = _format_config_desc(group_cols, name)
                
                # Sort by year
                group_df = cast(pd.DataFrame, group).sort_values(by='plot_year')
                
                plt.plot(group_df['plot_year'], group_df[stat_col], 'o-', label=label, color=color, linewidth=2, markersize=6)
                
                # Fill uncertainty band - prefer percentile band if available, else use std
                if SHOW_SHADING:
                    if band_low_col in group.columns and band_high_col in group.columns:
                        plt.fill_between(
                            group_df['plot_year'], 
                            group_df[band_low_col], 
                            group_df[band_high_col], 
                            color=color, 
                            alpha=0.1
                        )
                    elif std_col in group.columns and mean_col in group.columns:
                        lower = group_df[mean_col] - group_df[std_col]
                        upper = group_df[mean_col] + group_df[std_col]
                        plt.fill_between(
                            group_df['plot_year'], 
                            lower, 
                            upper, 
                            color=color, 
                            alpha=0.1
                        )
        
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Gap Size (Number of Ambient Articles)', fontsize=12)
            plt.title(f'Comparison of {metric_label} ({stat_type}) Across Configurations', fontsize=14)
            plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3, which="both")
            plt.yscale('log')
            
            # Use integer ticks for years, formatted as last two digits
            unique_years = sorted(df['plot_year'].unique())
            plt.xticks(unique_years, [str(y)[-2:] for y in unique_years], rotation=45)
            
            safe_metric_name = create_unicode_safe_filename(metric_label)
            output_path = output_dir / f"config_comparison_{safe_metric_name}_{stat_type.lower()}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            LOGGER.info(f"Saved comparison plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot historical gap analysis results")
    parser.add_argument(
        "--input-file", 
        type=Path, 
        default=ARTIFACTS_DIR / "historical_gap_analysis" / "results" / "historical_gap_analysis_combined.csv",
        help="Path to the combined results CSV"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=None,
        help="Directory to save plots (default: same as input file directory)"
    )
    
    args = parser.parse_args()
    
    setup_logging(log_name="plot_results", log_level="INFO", include_time=True)
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.input_file.parent
        
    plot_results(args.input_file, output_dir)

if __name__ == "__main__":
    main()
