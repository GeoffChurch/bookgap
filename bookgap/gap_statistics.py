"""
Gap analysis statistics utilities for computing and visualizing gap size statistics.

This module consolidates statistical analysis functions for gap sizes, including:
- Gap size percentile and z-score computation
- Shapiro-Wilk normality testing
- Q-Q plot generation for normal and negative hypergeometric distributions
"""
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional, Union, List
import matplotlib.pyplot as plt
from scipy import stats

from bookgap.util import LOGGER, create_unicode_safe_filename
from bookgap.plot_utils import PlotManager, FIGURE_SIZE_MEDIUM, GRID_ALPHA


# Constants for statistical analysis
MIN_GAP_BOOKS_FOR_STATISTICS = 3  # Minimum number of book gaps needed for Q-Q plots


def compute_gap_statistics(gap_size: int, book_gap_sizes: List[int]) -> tuple[float, float, float]:
    """
    Compute statistical measures for gap size relative to book gap sizes.
    
    Args:
        gap_size: Size of the current gap (number of articles)
        book_gap_sizes: List of gap sizes for books in the same topic
        
    Returns:
        Tuple of (percentile, z_score, shapiro_log10_pval)
        - percentile: Empirical percentile rank (fraction of book gaps smaller than this gap)
        - z_score: Z-score assuming normal distribution
        - shapiro_log10_pval: Log10 of p-value from Shapiro-Wilk normality test
    """
    book_gap_array = np.array(book_gap_sizes)
    
    # Empirical percentile (your original definition)
    percentile = float(np.mean(book_gap_array < gap_size))
    
    if len(book_gap_sizes) < 2:
        return percentile, float('nan'), float('nan')
    
    # Compute z-score assuming normality
    mean_gap_size = float(np.mean(book_gap_array))
    std_gap_size = float(np.std(book_gap_array, ddof=1))  # Sample standard deviation
    
    if std_gap_size > 0:
        z_score = float((gap_size - mean_gap_size) / std_gap_size)
    else:
        z_score = float('nan')
    
    # Test for normality using Shapiro-Wilk
    shapiro_log10_pval = float('nan')
    if len(book_gap_sizes) >= MIN_GAP_BOOKS_FOR_STATISTICS:
        try:
            _, pval = stats.shapiro(book_gap_sizes)
            shapiro_log10_pval = float(np.log10(pval) if pval > 0 else -np.inf)
        except Exception:
            shapiro_log10_pval = float('nan')
    
    return percentile, z_score, shapiro_log10_pval


def create_normal_qqplot(
    book_gap_sizes: Union[List[int], NDArray[np.intp]], 
    topic_name: str, 
    topic_id: str,
    plot_dir: Path,
    proposed_gap_size: Optional[int] = None
) -> Path:
    """
    Create Q-Q plot of book gap sizes against normal distribution.
    
    Args:
        book_gap_sizes: List or array of gap sizes for books in the topic
        topic_name: Display name of the topic
        topic_id: ID of the topic
        plot_dir: Directory to save the plot
        proposed_gap_size: Optional gap size to overlay on the plot
        
    Returns:
        Path to the saved plot file
    """
    fig, ax = PlotManager.create_figure(figsize=FIGURE_SIZE_MEDIUM)
    stats.probplot(book_gap_sizes, dist="norm", plot=ax)
    ax.set_title(f'Q-Q Plot: Book Gap Sizes vs Normal Distribution\nTopic: {topic_name}')
    ax.set_xlabel('Theoretical Quantiles (Normal Distribution)')
    ax.set_ylabel('Sample Quantiles (Book Gap Sizes)')
    ax.grid(True, alpha=GRID_ALPHA)
    
    # Add some statistics to the plot
    mean_gap = np.mean(book_gap_sizes)
    std_gap = np.std(book_gap_sizes, ddof=1)
    _, pval = stats.shapiro(book_gap_sizes)
    shapiro_log10_pval = float(np.log10(pval) if pval > 0 else -np.inf)

    ax.text(0.05, 0.95, f'Mean: {mean_gap:.1f}\nStd: {std_gap:.1f}\nShapiro $\\log_{{10}}$(p): {shapiro_log10_pval:.1f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overlay proposed gap size if provided
    if proposed_gap_size is not None:
        # Calculate theoretical quantile for the proposed gap size
        from scipy.stats import norm
        z_score = (proposed_gap_size - mean_gap) / std_gap if std_gap > 0 else 0
        percentile = stats.percentileofscore(book_gap_sizes, proposed_gap_size) / 100
        theoretical_quantile = norm.ppf(percentile)
        
        ax.axhline(y=proposed_gap_size, color='red', linestyle='--', alpha=0.7, 
                  label=f'Proposed Gap Size: {proposed_gap_size}')
        ax.axvline(x=float(theoretical_quantile), color='red', linestyle='--', alpha=0.7)
        ax.plot(float(theoretical_quantile), proposed_gap_size, 'ro', markersize=8, 
               label=f'Proposed Point (z={z_score:.2f})')
        ax.legend()
    
    # Save the normal Q-Q plot
    plot_file = plot_dir / f"{create_unicode_safe_filename(topic_name)}_id={topic_id}_book_gaps_qqplot_normal.png"
    PlotManager.save_and_close(fig, plot_file)
    
    LOGGER.info(f"Normal Q-Q plot saved to {plot_file}")
    return plot_file


def create_negative_hypergeometric_qqplot(
    book_gap_sizes: Union[List[int], NDArray[np.intp]],
    topic_name: str, 
    topic_id: str,
    plot_dir: Path,
    num_books: int,
    num_articles: int,
    proposed_gap_size: Optional[int] = None
) -> Optional[Path]:
    """
    Create Q-Q plot of book gap sizes against negative hypergeometric distribution.
    
    Args:
        book_gap_sizes: List or array of gap sizes for books in the topic
        topic_name: Display name of the topic
        topic_id: ID of the topic
        plot_dir: Directory to save the plot
        num_books: Total number of books in topic
        num_articles: Total number of articles in topic
        proposed_gap_size: Optional gap size to overlay on the plot
        
    Returns:
        Path to the saved plot file, or None if creation failed
    """
    if num_books <= 0 or num_articles <= 0:
        return None
        
    try:
        # Parameters for negative hypergeometric distribution:
        # M: total population size (books + articles)
        # n: number of "success" items in population (articles - ambient points)
        # r: number of "failures" to draw (1, since we stop at first landmark/book)
        M = num_books + num_articles
        n = num_articles  # ambient points (articles)
        r = 1  # stop at first landmark (book)
        
        fig, ax = PlotManager.create_figure(figsize=FIGURE_SIZE_MEDIUM)
        stats.probplot(book_gap_sizes, dist='nhypergeom', sparams=(M, n, r), plot=ax)
        ax.set_title(f'Q-Q Plot: Book Gap Sizes vs Negative Hypergeometric Distribution\nTopic: {topic_name}')
        ax.set_xlabel('Theoretical Quantiles (Negative Hypergeometric)')
        ax.set_ylabel('Sample Quantiles (Book Gap Sizes)')
        ax.grid(True, alpha=GRID_ALPHA)
        
        # Add distribution parameters to the plot
        mean_gap = np.mean(book_gap_sizes)
        nhyper_mean = stats.nhypergeom.mean(M, n, r)
        nhyper_var = stats.nhypergeom.var(M, n, r)
        
        ax.text(0.05, 0.95, 
                f'Sample Mean: {mean_gap:.1f}\nNHypergeom Mean: {nhyper_mean:.1f}\n'
                f'NHypergeom Var: {nhyper_var:.1f}\nM={M}, n={n}, r={r}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Overlay proposed gap size if provided
        if proposed_gap_size is not None:
            # Calculate theoretical quantile for the proposed gap size using negative hypergeometric
            percentile = stats.percentileofscore(book_gap_sizes, proposed_gap_size) / 100
            theoretical_quantile = stats.nhypergeom.ppf(percentile, M, n, r)
            
            ax.axhline(y=proposed_gap_size, color='red', linestyle='--', alpha=0.7,
                      label=f'Proposed Gap Size: {proposed_gap_size}')
            ax.axvline(x=float(theoretical_quantile), color='red', linestyle='--', alpha=0.7)
            ax.plot(float(theoretical_quantile), proposed_gap_size, 'ro', markersize=8,
                   label=f'Proposed Point (p={percentile:.3f})')
            ax.legend()
        
        # Save the negative hypergeometric Q-Q plot
        plot_file = plot_dir / f"{create_unicode_safe_filename(topic_name)}_id={topic_id}_book_gaps_qqplot_nhypergeom.png"
        PlotManager.save_and_close(fig, plot_file)
        
        LOGGER.info(f"Negative hypergeometric Q-Q plot saved to {plot_file}")
        return plot_file
        
    except Exception as e:
        LOGGER.warning(f"Failed to create negative hypergeometric Q-Q plot: {e}")
        return None


def create_book_gap_sizes_qqplot(
    book_gap_sizes: Union[List[int], NDArray[np.intp]], 
    topic_name: str, 
    topic_id: str,
    num_books: Optional[int] = None, 
    num_articles: Optional[int] = None, 
    proposed_gap_size: Optional[int] = None
) -> None:
    """
    Create Q-Q plots of book gap sizes against normal and negative hypergeometric distributions.
    
    This is the main entry point that consolidates both types of Q-Q plots.
    
    Args:
        book_gap_sizes: List of gap sizes for books in the topic
        topic_name: Display name of the topic
        topic_id: ID of the topic
        gap_config: Configuration object containing denoiser, drop_k, and radius_fraction
        num_books: Total number of books in topic (for negative hypergeometric)
        num_articles: Total number of articles in topic (for negative hypergeometric)
        proposed_gap_size: Optional gap size to overlay on the plots
    """
    if len(book_gap_sizes) < MIN_GAP_BOOKS_FOR_STATISTICS:
        LOGGER.info(f"Skipping Q-Q plot for topic {topic_name} - insufficient book gap data ({len(book_gap_sizes)} gaps)")
        return
    
    # Get plot directory using the configuration object
    plot_dir = Path('.')#GapDirectoryManager.get_plots_dir(analysis_config)    # Create normal distribution Q-Q plot
    create_normal_qqplot(book_gap_sizes, topic_name, topic_id, plot_dir, proposed_gap_size)
    
    # Create negative hypergeometric Q-Q plot if parameters are available
    if num_books is not None and num_articles is not None:
        create_negative_hypergeometric_qqplot(
            book_gap_sizes, topic_name, topic_id, plot_dir, 
            num_books, num_articles, proposed_gap_size
        )


# Gap statistics column definitions for pandas DataFrames
GAP_STATISTICS_COLUMNS = {
    'gap_size_percentile': 'float64',
    'gap_zscore': 'float64', 
    'shapiro_log10_pval': 'float64'
}

GAP_STATISTICS_COLUMN_NAMES = ['gap_size_percentile', 'gap_zscore', 'shapiro_log10_pval']
