#!/usr/bin/env python3
"""
Analyze the gap between books and articles across different filters.
"""

from functools import lru_cache
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Collection, Optional, Union
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import pyalex
import plotly.express as px
import plotly.graph_objects as go

from nextbook.cloud import Cloud, CloudDenoiserContext, SubCloud, WorkDenoisers, build_cloud, get_subclouds_for_topic
from nextbook.gap_finder import find_gaps_for_topic
from nextbook.cloud_index import avgref_ranks, within_cloud_nn_ranks, knn
from nextbook.parse_args import get_explicit_args, parse_args
from nextbook.util import TIMER, Filter, WorkType, LOGGER, MEMORY, FilterPair, SubField, setup_logging, get_stats
from nextbook.deduplicators import Deduplicator
from nextbook.index import HNSW, IndexType

# Define allowed denoisers using actual enum members for better static checking
ALLOWED_WORK_DENOISERS = [
    WorkDenoisers.PCA,
    WorkDenoisers.LPP,
    WorkDenoisers.ANTI_CSP,
    WorkDenoisers.JAD_EQUAL,
    WorkDenoisers.JAD_HALF_L,
    WorkDenoisers.JAD_CCP,
    WorkDenoisers.JAD_LIP,
]

# Helper to include 'NONE' as a special case (since it's not a WorkDenoiser enum member)
ALLOWED_DENOISER_NAMES = ['NONE'] + [wd.name for wd in ALLOWED_WORK_DENOISERS]

FONT_SIZE = 7

# Import plotting utilities to eliminate duplication
from nextbook.plot_utils import PlotManager, StatisticalPlotUtils, DPI
from nextbook.results_manager import BookArticleResults

def get_denoiser_display_name(denoiser: str) -> str:
    """Map denoiser names to display-friendly names with LaTeX math notation."""
    display_names = {
        'LIP': r'LIP = CCLIP($\alpha=1$)',
        # 'CCLIP': r'CCLIP($\alpha=0.5$)',
        'CCLIP_1': r'CCLIP($\alpha=0.1$)',
        'CCLIP_01': r'CCLIP($\alpha=0.01$)',
        'CCP': r'CCP = CCLIP($\alpha=0$)',
        'ANTI_LDA': 'Anti-LDA',
        # 'JAD': 'JAD (Balanced)',
        # 'JAD_BOOK_EMPHASIS': 'JAD (Book Emphasis)',
        # 'JAD_SQRT_BALANCE': 'JAD (√-Balance)',
        # 'JAD_CONSERVATIVE': 'JAD (Conservative)',
        'JAD_1_4': 'JAD (1:4 Book:Article)',
        'JAD_EQUAL': 'JAD (Equal Emphasis)',
        'JAD_PRIO_T': 'JAD (Total Scatter)',
        # 'RNOJD': 'RNOJD',
        'NONE': 'No Denoising'
    }
    return display_names.get(denoiser, denoiser)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Global BookArticleResults instance
_book_article_results = BookArticleResults(csv_path=ARTIFACTS_DIR / "book_article_analysis.csv")


@dataclass
class BookArticleAnalysis:
    """Results of analyzing books vs articles in a cloud or subcloud."""
    num_books: int
    num_articles: int
    kball_size: float
    num_books_with_avgref: int
    avgref_mean_rank: float

def analyze_book_article_gap(cloud: Cloud, topic_id: str, radius_fraction: float) -> BookArticleAnalysis:
    books, articles = get_subclouds_for_topic(cloud, topic_id)
    num_books = len(books.embeddings())
    num_articles = len(articles.embeddings())

    avgref_rank_res = avgref_rank([cloud], books)

    return BookArticleAnalysis(
        num_books=num_books,
        num_articles=num_articles,
        kball_size=kball_size(articles, books, k=1),
        num_books_with_avgref=avgref_rank_res.num_queries_with_avgref,
        avgref_mean_rank=float(avgref_rank_res.avgref_mean_rank)
    )


def _df_from_file(csv_path: Path) -> pd.DataFrame:        
    try:
        df = pd.read_csv(csv_path)
        LOGGER.info(f"Loaded existing results from {csv_path}")
    except FileNotFoundError:
        df = pd.DataFrame()
        LOGGER.info(f"No existing results found at {csv_path}, starting fresh.")
    return df


def kball_size(index: SubCloud, query: SubCloud, k: int) -> float:
    """
    Calculates the mean k-ball size.

    This metric evaluates how well the 'index' cloud separates points within the
    'query' cloud. For each point in the query cloud, it defines a hypersphere ("k-ball")
    whose radius is the distance to its k-th nearest neighbor within that same cloud.
    It then counts how many points from the 'index' cloud fall inside this ball.

    Higher is better, meaning that on average many index points are closer to a query
    point than its own k-th nearest neighbor.

    Args:
        index: The cloud of points that "separates" the query points (e.g., articles).
        query: The cloud of points for which the metric is calculated (e.g., books).
        k: The rank of the same-cloud neighbor used to define the radius of the k-ball.

    Returns:
        The mean number of index points found inside the k-balls, or NaN if no
        ranks could be computed.
    """
    ranks = within_cloud_nn_ranks(index, query, k=k).ranks    
    LOGGER.info(f"Mean k-ball size stats: {get_stats(ranks)}")
    return float(np.mean(ranks))

@dataclass(frozen=True)
class AvgrefRankResult:
    avgref_mean_rank: np.float64
    num_queries_with_avgref: int

def avgref_rank(index_clouds: Collection[Cloud | SubCloud], query: Cloud | SubCloud) -> AvgrefRankResult:
    """
    For each query point, compute its avgref (average embedding of its references in the index clouds)
    and return the mean rank of these avgrefs within the index clouds.
    """
    res = avgref_ranks(index_clouds, query, normalize_avgref_embeddings=query.normalize_embeddings)

    ranks = np.concatenate(res.per_cloud_ranks)
    assert ranks.shape == (len(res.avgref_res.corresponding_query_idxs) * len(index_clouds),)
    LOGGER.info(f"Rank stats: {get_stats(ranks)}")

    return AvgrefRankResult(
        avgref_mean_rank=np.mean(ranks),
        num_queries_with_avgref=len(res.avgref_res.corresponding_query_idxs)
    )
    
    


def hubness(index: SubCloud, query: SubCloud):
    """
    Calculate the hubness:
    1. Draw an arrow from each query point to its nearest index point.
    2. Count the number of incoming arrows to each index point.
    3. There may be more index points than query points. In this case we
    only retain the top N counts, where N is the number of query points.
    The discarded counts are guaranteed to be zero.
    4. Hubness is the Gini coefficient of the retained counts.
    """
    num_queries = len(query.embeddings())
    num_indices = len(index.embeddings())
    n = min(num_queries, num_indices)
    if n == 0:
        LOGGER.warning(f"n=min({num_queries=}, {num_indices=}) == 0, returning NaN for hubness")
        return np.nan
    
    nearest_indices = knn(index, query).I.squeeze(1)

    if np.any(nearest_indices < 0):
        assert num_indices == 0
        return np.nan

    # 1. Count how many times each of the index points is a nearest neighbor
    nn_counts = np.bincount(nearest_indices, minlength=num_indices)
    
    # 2. Sort the counts
    nn_counts.sort()
    
    assert nn_counts.shape == (num_indices,)

    # 3. Take the top N counts, where N is the number of queries
    nn_counts = nn_counts[-num_queries:]

    assert nn_counts.shape == (n,)
    
    # --- Gini coefficient calculation on the subset ---

    assert nn_counts.sum() == n

    cum_counts = nn_counts.cumsum()
    gini = (n + 1 - 2 * cum_counts.sum() / cum_counts[-1]) / n
    
    assert type(gini) is np.float64, f"Expected np.float64, got {type(gini)}"
    percentiles = np.linspace(0, 100, 11)
    percentiles = ' '.join(f"{perc}:{value:.3f}" for perc, value in zip(percentiles, np.percentile(nn_counts, percentiles)))
    LOGGER.info(f"Hubness Gini: {gini:.4f}, Percentiles: {percentiles}")
    
    return float(gini)


def plot_scatter_by_total_works(
    df: pd.DataFrame,
    artifacts_dir: Path,
    y_col: str,
    title: str,
    output_filename: str,
    hover_cols: list,
    color_col='denoiser_drop_k',
    show_labels=False,
    yaxis_title=None,
    log_y: bool = False
):
    plot_df = df.copy()
    plot_df['denoiser'] = plot_df['denoiser'].astype(str).str.strip()
    plot_df['denoiser_drop_k'] = plot_df['denoiser'] + '_k' + plot_df['drop_k'].astype(str)
    # Determine color column for plotly express
    if isinstance(color_col, (tuple, list)):
        color_col_name = '_color_col_combined'
        for i, col in enumerate(color_col):
            if i == 0:
                plot_df[color_col_name] = plot_df[col].astype(str)
            else:
                plot_df[color_col_name] += '_' + plot_df[col].astype(str)
    else:
        color_col_name = color_col
    col_values = plot_df[color_col_name]
    unique_combinations = sorted(col_values.unique())
    n_combinations = len(unique_combinations)
    if n_combinations == 0:
        raise ValueError(f"No unique combinations found for color column '{color_col_name}'")
    if n_combinations > 1:
        hsv_values = np.linspace(0, 1, n_combinations, endpoint=False)
        colors = [mcolors.hsv_to_rgb((h, 0.8, 0.9)) for h in hsv_values]
        hex_colors = [mcolors.to_hex(tuple(color)) for color in colors]
        color_map = dict(zip(unique_combinations, hex_colors))
    else:
        color_map = {unique_combinations[0]: '#1f77b4'}
    fig = px.scatter(
        plot_df,
        x='total_works',
        y=y_col,
        color=color_col_name,
        color_discrete_map=color_map,
        hover_data=hover_cols,
        hover_name='topic_name',
        title=title,
        category_orders={color_col_name: unique_combinations}
    )
    if show_labels:
        fig.add_trace(go.Scatter(
            x=plot_df['total_works'],
            y=plot_df[y_col],
            mode='text',
            text=plot_df['topic_name'],
            textposition='top center',
            showlegend=False,
            hoverinfo='skip',
            textfont=dict(size=FONT_SIZE, color='black'),
        ))
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        width=1000,
        height=700,
        font=dict(size=12),
        title_font=dict(size=16),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        ),
        xaxis_title='Total Works',
        xaxis_type='log',
        yaxis_title=yaxis_title or y_col
    )
    if log_y:
        fig.update_yaxes(type='log')
    plot_dir = artifacts_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / output_filename
    fig.write_html(output_file)

def plot_kball_size_by_total_works(df: pd.DataFrame, artifacts_dir: Path, color_col='denoiser_drop_k', show_labels=False):
    plot_scatter_by_total_works(
        df,
        artifacts_dir,
        y_col='kball_size',
        title='Book-Article Gap Analysis: K-Ball Size by Total Works',
        output_filename='kball_size_by_total_works.html',
        hover_cols=['topic_name', 'num_books', 'num_articles'],
        color_col=color_col,
        show_labels=show_labels,
        yaxis_title='K-Ball Size',
        log_y=False
    )
    
def plot_col_by_denoiser(df: pd.DataFrame, artifacts_dir: Path, y_col: str, color_col='denoiser', show_labels=False, log_y=False, weights: Optional[str] = None, hover_data: Optional[list] = None):
    """
    Plot a specified y_col for each (denoiser, drop_k) as a categorical x-axis.
    Show all points, mean, and horizontal whiskers at ±1 stddev.
    Sort denoiser_drop_k by ascending mean y_col. Draw faint lines between adjacent denoiser_drop_k groups for the same topic_id, and connect whisker means as well.
    The right y-axis shows the ratio to baseline (NONE denoiser).
    color_col: str or tuple/list of str, column(s) to use for color grouping.
    show_labels: if True, always show topic_name as text label at each point.
    log_y: if True, use log scale for y-axis.
    weights: Column name to use for weighting the mean and std calculations.
    hover_data: List of column names to include in hover information.
    """
    df = df.copy()
    df['denoiser'] = df['denoiser'].astype(str).str.strip()
    df['denoiser_drop_k'] = df['denoiser'] + '_k' + df['drop_k'].astype(str)
    
    # Drop rows where y_col is NaN
    df = df.dropna(subset=[y_col])
    
    # Handle weights parameter - create uniform weights if None
    if weights is None:
        assert 'weights_' not in df.columns, "Column 'weights_' already exists in DataFrame"
        df['weights_'] = 1.0
        weights = 'weights_'
    
    # Compute weighted means and stddevs for each group (with respect to y_col)
    def weighted_mean(x):
        return np.average(x[y_col], weights=x[weights])
    def weighted_std(x):
        values = x[y_col].to_numpy(dtype=float)
        weight_vals = x[weights].to_numpy(dtype=float)
        if np.sum(weight_vals) == 0:
            return np.nan
        mean = np.average(values, weights=weight_vals)
        variance = np.average((values - mean) ** 2, weights=weight_vals)
        return np.sqrt(variance)
    stats = df.groupby('denoiser_drop_k').apply(weighted_mean).reset_index(name='mean')
    stats['std'] = df.groupby('denoiser_drop_k').apply(weighted_std).values
    
    # # Calculate baseline for secondary y-axis
    # baseline_stats = stats[stats['denoiser_drop_k'].str.startswith('NONE_k')]
    # baseline_mean = baseline_stats['mean'].iloc[0] if not baseline_stats.empty else 1.0

    # LOGGER.info(f"Baseline mean for {y_col} (NONE denoiser): {baseline_mean:.3f}")
    
    # Sort denoiser_drop_k by ascending mean y_col
    sorted_groups = stats.sort_values('mean')['denoiser_drop_k'].tolist()
    # Set categorical order for x-axis (enforce strict Categorical)
    df['denoiser_drop_k'] = pd.Categorical(df['denoiser_drop_k'], categories=sorted_groups, ordered=True)
    # Map each denoiser_drop_k to its integer index for whisker lines
    group_to_idx = {cat: i for i, cat in enumerate(sorted_groups)}
    # Assign colors to each unique value of color_col in natural order
    col_values = df[color_col]
    if pd.api.types.is_numeric_dtype(col_values):
        unique_vals = sorted(col_values.unique(), key=lambda x: float(x))
    else:
        unique_vals = sorted(col_values.unique())
    n_vals = len(unique_vals)
    hsv_values = np.linspace(0, 1, n_vals, endpoint=False)
    colors = [mcolors.hsv_to_rgb((h, 0.8, 0.9)) for h in hsv_values]
    hex_colors = [mcolors.to_hex(tuple(color)) for color in colors]
    color_map = dict(zip(unique_vals, hex_colors))
    # Create the scatter plot (all points, colored by color_col, legend in natural order)
    fig = px.scatter(
        df,
        x='denoiser_drop_k',
        y=y_col,
        color=color_col,
        color_discrete_map=color_map,
        hover_data=hover_data,
        title=f'{y_col} by Denoiser/Drop_k',
        opacity=0.5,
        category_orders={color_col: unique_vals}
    )
    # Optionally add always-visible text labels for topic_name
    if show_labels:
        fig.add_trace(go.Scatter(
            x=df['denoiser_drop_k'],
            y=df[y_col],
            mode='text',
            text=df['topic_name'],
            textposition='top center',
            showlegend=False,
            hoverinfo='skip',
            textfont=dict(size=FONT_SIZE, color='black'),
        ))
    # Add mean marker and horizontal whisker for each group
    mean_y = []
    mean_x = []
    for i, row in stats.iterrows():
        x = row['denoiser_drop_k']
        mean = row['mean']
        std = row['std']
        mean_x.append(x)
        mean_y.append(mean)
        # Add mean marker with hover
        fig.add_trace(go.Scatter(
            x=[x], y=[mean],
            mode='markers',
            marker=dict(symbol='diamond', size=14, color='black'),
            name=f"Mean: {x}",
            showlegend=False,
            hoverinfo='x+y+text',
            text=[f"Mean {y_col}: {mean:.2f} ± {std:.2f}" if not np.isnan(std) else f"Mean {y_col}: {mean:.2f}"],
        ))
        # Add horizontal whisker at ±1 stddev (on integer axis, but show at categorical x)
        if not np.isnan(std):
            idx = group_to_idx[x]
            fig.add_trace(go.Scatter(
                x=[sorted_groups[idx]], y=[mean+std],
                mode='markers',
                marker=dict(symbol='line-ew', size=28, color='black', line_width=8),
                name=f"+1 std: {x}",
                showlegend=False,
                hoverinfo='skip',
            ))
            fig.add_trace(go.Scatter(
                x=[sorted_groups[idx]], y=[mean-std],
                mode='markers',
                marker=dict(symbol='line-ew', size=28, color='black', line_width=8),
                name=f"-1 std: {x}",
                showlegend=False,
                hoverinfo='skip',
            ))
    # Connect whisker means across adjacent denoiser_drop_k groups for the same topic_id
    for topic_id, group in df.groupby('topic_id'):
        group = group.sort_values('denoiser_drop_k')
        if len(group) < 2:
            continue
        x_vals = group['denoiser_drop_k'].astype(str).tolist()
        y_vals = group[y_col].tolist()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            line=dict(color='rgba(100,100,100,0.2)', width=2),
            showlegend=False,
            hoverinfo='skip',
        ))
        # Connect means for this topic_id (on categorical axis)
        means = stats[stats['denoiser_drop_k'].isin(x_vals)].set_index('denoiser_drop_k').loc[x_vals]['mean'].tolist()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=means,
            mode='lines',
            line=dict(color='rgba(0,0,0,0.5)', width=3, dash='dot'),
            showlegend=False,
            hoverinfo='skip',
        ))
    fig.update_layout(
        xaxis_title='Denoiser / Drop_k',
        yaxis_title=y_col,
        width=1100,
        height=700,
        font=dict(size=12),
        title_font=dict(size=16),
        xaxis={'categoryorder':'array', 'categoryarray': sorted_groups},
        showlegend=True
    )
    
    # # Add secondary y-axis showing ratio to baseline
    # if baseline_mean != 0 and not np.isnan(baseline_mean):
    #     # Get unique y values from the stats for tick positioning
    #     y_values = sorted(stats['mean'].dropna().unique())
    #     fig.update_layout(
    #         yaxis2=dict(
    #             title="Ratio to Baseline",
    #             overlaying='y',
    #             side='right',
    #             tickvals=y_values,
    #             ticktext=[f"{y/baseline_mean:.2f}" for y in y_values]
    #         )
    #     )
    # else:
    #     LOGGER.warning("Baseline mean is zero or NaN, not adding secondary y-axis.")
    
    if log_y:
        fig.update_yaxes(type='log')
    plot_dir = artifacts_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / f"{y_col}_by_denoiser_{color_col=}.html"
    fig.write_html(output_file)

def plot_col_by_denoiser_pyplot(df: pd.DataFrame, artifacts_dir: Path, y_col: str, color_col='denoiser', show_labels=False, log_y=False, weights: Optional[str] = None):
    """
    Pyplot version of plot_col_by_denoiser.
    Plot a specified y_col for each (denoiser, drop_k) as a categorical x-axis using matplotlib.
    Show all points, mean, and horizontal whiskers at ±1 stddev.
    Sort denoiser_drop_k by ascending mean y_col. Draw faint lines between adjacent denoiser_drop_k groups for the same topic_id.
    """
    df = df.copy()
    df['denoiser'] = df['denoiser'].astype(str).str.strip()
    df['denoiser_drop_k'] = df['denoiser'] + '_k' + df['drop_k'].astype(str)
    
    # Drop rows where y_col is NaN
    df = df.dropna(subset=[y_col])
    
    # Handle weights parameter - create uniform weights if None
    if weights is None:
        assert 'weights_' not in df.columns, "Column 'weights_' already exists in DataFrame"
        df['weights_'] = 1.0
        weights = 'weights_'
    
    # Compute weighted means and stddevs for each group (with respect to y_col)
    def weighted_mean(x):
        return np.average(x[y_col], weights=x[weights])
    def weighted_std(x):
        values = x[y_col].to_numpy(dtype=float)
        weight_vals = x[weights].to_numpy(dtype=float)
        if np.sum(weight_vals) == 0:
            return np.nan
        mean = np.average(values, weights=weight_vals)
        variance = np.average((values - mean) ** 2, weights=weight_vals)
        return np.sqrt(variance)
    stats = df.groupby('denoiser_drop_k').apply(weighted_mean).reset_index(name='mean')
    stats['std'] = df.groupby('denoiser_drop_k').apply(weighted_std).values
    
    # # Calculate baseline for secondary y-axis
    # baseline_stats = stats[stats['denoiser_drop_k'].str.startswith('NONE_k')]
    # baseline_mean = baseline_stats['mean'].iloc[0] if not baseline_stats.empty else 1.0

    # LOGGER.info(f"Baseline mean for {y_col} (NONE denoiser): {baseline_mean:.3f}")
    
    # Sort denoiser_drop_k by ascending mean y_col
    sorted_groups = stats.sort_values('mean')['denoiser_drop_k'].tolist()
    # Set categorical order for x-axis (enforce strict Categorical)
    df['denoiser_drop_k'] = pd.Categorical(df['denoiser_drop_k'], categories=sorted_groups, ordered=True)
    
    # Assign colors to each unique value of color_col in natural order
    col_values = df[color_col]
    if pd.api.types.is_numeric_dtype(col_values):
        unique_vals = sorted(col_values.unique(), key=lambda x: float(x))
    else:
        unique_vals = sorted(col_values.unique())
    
    # Create the plot
    fig, ax1 = PlotManager.create_figure(figsize=(14, 8))
    
    # Color map for different categories
    cmap = plt.cm.get_cmap('tab10')
    color_map = {val: cmap(i % 10) for i, val in enumerate(unique_vals)}
    
    # Plot all points
    for val in unique_vals:
        mask = df[color_col] == val
        subset = df[mask]
        x_positions = [sorted_groups.index(group) for group in subset['denoiser_drop_k']]
        ax1.scatter(x_positions, subset[y_col], alpha=0.5, color=color_map[val], label=f'{color_col}={val}', s=30)
    
    # Add means and error bars
    x_positions = list(range(len(sorted_groups)))
    means = [stats[stats['denoiser_drop_k'] == group]['mean'].iloc[0] for group in sorted_groups]
    stds = [stats[stats['denoiser_drop_k'] == group]['std'].iloc[0] for group in sorted_groups]
    
    # Plot means as diamonds
    ax1.scatter(x_positions, means, marker='D', s=80, color='black', zorder=5, label='Mean')
    
    # Add error bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        if not np.isnan(std):
            ax1.errorbar(i, mean, yerr=std, fmt='none', color='black', capsize=5, capthick=2, linewidth=2)
    
    # Connect lines for same topic_id
    for topic_id, group in df.groupby('topic_id'):
        group = group.sort_values('denoiser_drop_k')
        if len(group) < 2:
            continue
        x_vals = [sorted_groups.index(g) for g in group['denoiser_drop_k']]
        y_vals = group[y_col].tolist()
        ax1.plot(x_vals, y_vals, color='gray', alpha=0.2, linewidth=1, zorder=1)
        
        # Connect means for this topic_id
        group_means = [stats[stats['denoiser_drop_k'] == g]['mean'].iloc[0] for g in group['denoiser_drop_k']]
        ax1.plot(x_vals, group_means, color='black', alpha=0.5, linewidth=2, linestyle='--', zorder=2)
    
    # Add text labels if requested
    if show_labels:
        for _, row in df.iterrows():
            x_pos = sorted_groups.index(row['denoiser_drop_k'])
            ax1.text(x_pos, row[y_col], row['topic_name'], fontsize=FONT_SIZE, ha='center', va='bottom', alpha=0.7)
    
    # Formatting
    ax1.set_xlabel('Denoiser / Drop_k')
    ax1.set_ylabel(y_col)
    ax1.set_title(f'{y_col} by Denoiser/Drop_k')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(sorted_groups, rotation=45, ha='right')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # # Add secondary y-axis for ratio to baseline
    # ax2 = None
    # if baseline_mean != 0 and not np.isnan(baseline_mean):
    #     ax2 = ax1.twinx()
    #     ax2.set_ylabel('Ratio to Baseline')
    #     ylim = ax1.get_ylim()
    #     ax2.set_ylim((ylim[0] / baseline_mean, ylim[1] / baseline_mean))
    
    if log_y:
        ax1.set_yscale('log')
        # if ax2 is not None:
        #     ax2.set_yscale('log')
    
    # plt.tight_layout()
    plot_dir = artifacts_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / f"{y_col}_by_denoiser_{color_col=}_pyplot.png"
    PlotManager.save_and_close(fig, output_file)

def plot_col_by_drop_k(df: pd.DataFrame, artifacts_dir: Path, y_col: str, log_y: bool = False, weights: Optional[str] = None):
    """
    Plot metric vs drop_k with each denoiser as a separate series.
    For each (denoiser, drop_k) pair, average the metric across all topics.
    Treats "NONE" denoiser as drop_k = 0 baseline.
    A secondary y-axis shows the ratio to baseline (NONE denoiser).
    
    Args:
        df: DataFrame with the data
        artifacts_dir: Directory to save plots
        y_col: Column name to plot on y-axis
        log_y: if True, use log scale for y-axis
        weights: Column name to use for weighting the mean and std calculations
    """
    plot_df = df.copy()
    
    # Drop rows where the metric is NaN
    plot_df = plot_df[plot_df[y_col].notna()]
    
    if plot_df.empty:
        LOGGER.info(f"No data available for drop_k plot ({y_col}). Skipping plot.")
        return
    
    # Check that the y_col exists
    if y_col not in plot_df.columns:
        LOGGER.warning(f"Column '{y_col}' not found in data. Skipping plot.")
        return
    
    # Handle weights parameter - create uniform weights if None
    if weights is None:
        assert 'weights_' not in plot_df.columns, "Column 'weights_' already exists in DataFrame"
        plot_df['weights_'] = 1.0
        weights = 'weights_'
    
    # Group by denoiser and drop_k, then average the metric across topics
    def weighted_mean(x):
        return np.average(x[y_col], weights=x[weights])
    def weighted_std(x):
        values = x[y_col].to_numpy(dtype=float)
        weight_vals = x[weights].to_numpy(dtype=float)
        if np.sum(weight_vals) == 0:
            return np.nan
        mean = np.average(values, weights=weight_vals)
        variance = np.average((values - mean) ** 2, weights=weight_vals)
        return np.sqrt(variance)
    grouped = plot_df.groupby(['denoiser', 'drop_k']).apply(
        lambda x: pd.Series({
            'metric_mean': weighted_mean(x),
            'std': weighted_std(x),
            'count': len(x)
        })
    ).reset_index()
    
    # Sort by drop_k for proper x-axis ordering
    grouped = grouped.sort_values('drop_k')
    
    # Get baseline value and add as horizontal line
    baseline_data = grouped[grouped['denoiser'] == 'NONE']
    baseline_value = baseline_data['metric_mean'].iloc[0] if not baseline_data.empty else None

    # LOGGER.info(f"Baseline value for {y_col} (NONE denoiser): {baseline_value:.3f}")
    
    # Create the line plot
    fig = go.Figure()
    
    # Get unique denoisers and assign colors (excluding NONE since it will be a horizontal line)
    denoisers = grouped[grouped['denoiser'] != 'NONE']['denoiser'].unique()
    colors = px.colors.qualitative.Dark24[:len(denoisers)]
    
    # Use generic naming conventions
    y_axis_title = y_col.replace('_', ' ').title()
    filename = f'{y_col}_by_drop_k.html'
    
    for i, denoiser in enumerate(denoisers):
        denoiser_data = grouped[grouped['denoiser'] == denoiser].sort_values('drop_k')
        
        line_style = dict(color=colors[i % len(colors)], width=3)
        marker_style = dict(size=8, color=colors[i % len(colors)])
        denoiser_display_name = get_denoiser_display_name(denoiser)
    
        # Add main line
        fig.add_trace(go.Scatter(
            x=denoiser_data['drop_k'],
            y=denoiser_data['metric_mean'],
            mode='lines+markers',
            name=denoiser_display_name,
            line=line_style,
            marker=marker_style,
            hovertemplate=(
                f'<b>{denoiser_display_name}</b><br>' +
                'Drop K: %{x}<br>' +
                f'{y_axis_title}: %{{y:.3f}}<br>' +
                'Std: %{customdata[0]:.3f}<br>' +
                'Topic Count: %{customdata[1]}<br>' +
                '<extra></extra>'
            ),
            customdata=denoiser_data[['std', 'count']].values
        ))
    
    # Add baseline horizontal line
    if baseline_value is not None:
        x_range = grouped['drop_k']
        x_min, x_max = x_range.min(), x_range.max()
        fig.add_trace(go.Scatter(
            x=[x_min, x_max],
            y=[baseline_value, baseline_value],
            mode='lines',
            name=get_denoiser_display_name('NONE'),
            line=dict(color='black', width=4, dash='dash'),
            hovertemplate=(
                f'<b>{get_denoiser_display_name("NONE")}</b><br>' +
                f'{y_axis_title}: {baseline_value:.3f}<br>' +
                '<extra></extra>'
            )
        ))
    
    fig.update_layout(
        title=f'{y_axis_title} by Drop K (Averaged Across Topics)',
        xaxis_title='Drop K',
        yaxis_title=y_axis_title,
        width=1000,
        height=700,
        font=dict(size=12),
        title_font=dict(size=16),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        ),
        # xaxis_type='log',  # Log scale for drop_k since values span orders of magnitude
        hovermode='closest'
    )
    
    # # Add secondary y-axis showing ratio to baseline
    # if baseline_value != 0 and not np.isnan(baseline_value):
    #     # Get unique y values from the grouped data for tick positioning
    #     y_values = sorted(grouped['metric_mean'].dropna().unique())
    #     fig.update_layout(
    #         yaxis2=dict(
    #             title="Ratio to Baseline",
    #             overlaying='y',
    #             side='right',
    #             tickvals=y_values,
    #             ticktext=[f"{y/baseline_value:.2f}" for y in y_values]
    #         )
    #     )
    # else:
    #     LOGGER.warning("Baseline value is zero or NaN, not adding secondary y-axis.")
    
    if log_y:
        fig.update_yaxes(type='log')
    
    # Save the plot
    plot_dir = artifacts_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / filename
    fig.write_html(output_file)

def plot_col_by_drop_k_pyplot(df: pd.DataFrame, artifacts_dir: Path, y_col: str, log_y: bool = False, weights: Optional[str] = None):
    """
    Pyplot version of plot_col_by_drop_k.
    Plot metric vs drop_k with each denoiser as a separate series using matplotlib.
    For each (denoiser, drop_k) pair, average the metric across all topics.
    Treats "NONE" denoiser as drop_k = 0 baseline.
    """
    plot_df = df.copy()
    
    # Drop rows where the metric is NaN
    plot_df = plot_df[plot_df[y_col].notna()]
    
    if plot_df.empty:
        LOGGER.info(f"No data available for drop_k plot ({y_col}). Skipping plot.")
        return
    
    # Check that the y_col exists
    if y_col not in plot_df.columns:
        LOGGER.warning(f"Column '{y_col}' not found in data. Skipping plot.")
        return
    
    # Handle weights parameter - create uniform weights if None
    if weights is None:
        assert 'weights_' not in plot_df.columns, "Column 'weights_' already exists in DataFrame"
        plot_df['weights_'] = 1.0
        weights = 'weights_'
    
    # Group by denoiser and drop_k, then average the metric across topics
    def weighted_mean(x):
        return np.average(x[y_col], weights=x[weights])
    def weighted_std(x):
        values = x[y_col].to_numpy(dtype=float)
        weight_vals = x[weights].to_numpy(dtype=float)
        if np.sum(weight_vals) == 0:
            return np.nan
        mean = np.average(values, weights=weight_vals)
        variance = np.average((values - mean) ** 2, weights=weight_vals)
        return np.sqrt(variance)
    grouped = plot_df.groupby(['denoiser', 'drop_k']).apply(
        lambda x: pd.Series({
            'metric_mean': weighted_mean(x),
            'std': weighted_std(x),
            'count': len(x)
        })
    ).reset_index()
    
    # Sort by drop_k for proper x-axis ordering
    grouped = grouped.sort_values('drop_k')
    
    # Get baseline value for secondary y-axis
    baseline_data = grouped[grouped['denoiser'] == 'NONE']
    baseline_value = baseline_data['metric_mean'].iloc[0] if not baseline_data.empty else None

    # LOGGER.info(f"Baseline value for {y_col} (NONE denoiser): {baseline_value:.3f}")
    
    # Create the plot
    fig, ax1 = PlotManager.create_figure(figsize=(12, 8))
    
    # Get unique denoisers and assign colors/styles (excluding NONE since it will be a horizontal line)
    denoisers = grouped[grouped['denoiser'] != 'NONE']['denoiser'].unique()
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(denoisers)))
    
    for i, denoiser in enumerate(denoisers):
        denoiser_data = grouped[grouped['denoiser'] == denoiser].sort_values('drop_k')
        
        # Regular styling for other denoisers
        ax1.plot(denoiser_data['drop_k'], denoiser_data['metric_mean'], 
                color=colors[i], linewidth=2, marker='o', markersize=6, label=get_denoiser_display_name(denoiser))
    
    # Add baseline horizontal line
    if baseline_value is not None:
        x_range = grouped['drop_k']
        ax1.axhline(y=baseline_value, xmin=x_range.min(), xmax=x_range.max(),
                   color='black', linestyle='--', linewidth=3, 
                   label=get_denoiser_display_name('NONE'), zorder=10)
    
    # Formatting
    y_axis_title = y_col.replace('_', ' ').title()
    ax1.set_xlabel('Drop K')
    ax1.set_ylabel(y_axis_title)
    ax1.set_title(f'{y_axis_title} by Drop K (Averaged Across Topics)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add secondary y-axis for ratio to baseline
    ax2 = None
    # if baseline_value != 0 and not np.isnan(baseline_value):
    #     ax2 = ax1.twinx()
    #     ax2.set_ylabel('Ratio to Baseline')
    #     ylim = ax1.get_ylim()
    #     ax2.set_ylim((ylim[0] / baseline_value, ylim[1] / baseline_value))
    
    if log_y:
        ax1.set_yscale('log')
        if ax2 is not None:
            ax2.set_yscale('log')
    
    # plt.tight_layout()
    plot_dir = artifacts_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / f"{y_col}_by_drop_k_pyplot.png"
    PlotManager.save_and_close(fig, output_file)

def plot_books_vs_articles(df: pd.DataFrame, artifacts_dir: Path, show_labels=False, log_scale=True, min_articles=0):
    """
    Plot number of books vs number of articles for each topic, with an optional regression line.
    Args:
        df: DataFrame with the data
        artifacts_dir: Directory to save plots
        show_labels: If True, show topic names as text labels
        log_scale: If True, use log scale for both axes and fit regression in log-log space
        min_articles: Only include topics with at least this many articles
    """
    if df.empty:
        LOGGER.warning("No data available for books vs articles plot. Skipping.")
        return
    # Filter to just denoiser=None since num_books and num_articles are invariant wrt denoiser/drop_k
    plot_df = df.copy()
    plot_df = plot_df[plot_df['denoiser'] == 'NONE'].copy()
    if min_articles > 0:
        plot_df = plot_df[plot_df['num_articles'] >= min_articles]
    
    if plot_df.empty:
        LOGGER.warning("No data with denoiser=None found for books vs articles plot. Skipping.")
        return
    
    fig = px.scatter(
        plot_df,
        x='num_articles',
        y='num_books',
        color_discrete_sequence=['#1f77b4'],  # All points same color
        hover_data=['topic_name', 'topic_id', 'total_works'],
        hover_name='topic_name',
        title='Books vs Articles by Topic',
    )
    
    # Add regression line
    x = plot_df['num_articles'].values
    y = plot_df['num_books'].values
    assert type(x) == np.ndarray
    assert type(y) == np.ndarray
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) > 1:
        if log_scale:
            logx = np.log10(x)
            logy = np.log10(y)
            slope, intercept = np.polyfit(logx, logy, 1)
            x_fit = np.linspace(x.min(), x.max(), 200)
            y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
            label = f"log-log fit: y ~ x^{slope:.2f}"
        else:
            slope, intercept = np.polyfit(x, y, 1)
            x_fit = np.linspace(x.min(), x.max(), 200)
            y_fit = slope * x_fit + intercept
            label = f"linear fit: y = {slope:.2f}x + {intercept:.2f}"
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            name=label,
            showlegend=False,
            hoverinfo='skip',
        ))
    
    if show_labels:
        fig.add_trace(go.Scatter(
            x=plot_df['num_articles'],
            y=plot_df['num_books'],
            mode='text',
            text=plot_df['topic_name'],
            textposition='top center',
            showlegend=False,
            hoverinfo='skip',
            textfont=dict(size=FONT_SIZE, color='black'),
        ))
    
    fig.update_traces(marker=dict(size=8, opacity=0.7), showlegend=False)
    fig.update_layout(
        width=1000,
        height=700,
        font=dict(size=12),
        title_font=dict(size=16),
        showlegend=False,
        xaxis_title='Number of Articles',
        yaxis_title='Number of Books'
    )
    
    if log_scale:
        fig.update_xaxes(type='log')
        fig.update_yaxes(type='log')
    
    plot_dir = artifacts_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / ('books_vs_articles.html' if not log_scale else 'books_vs_articles_log.html')
    fig.write_html(output_file)

def plot_books_vs_articles_pyplot(df: pd.DataFrame, artifacts_dir: Path, show_labels=False, log_scale=True, min_articles=0):
    """
    Plot number of books vs number of articles for each topic using matplotlib, with an optional regression line.
    Args:
        df: DataFrame with the data
        artifacts_dir: Directory to save plots
        show_labels: If True, show topic names as text labels
        log_scale: If True, use log scale for both axes and fit regression in log-log space
        min_articles: Only include topics with at least this many articles
    """
    if df.empty:
        LOGGER.warning("No data available for books vs articles plot. Skipping.")
        return
    plot_df = df.copy()
    plot_df = plot_df[plot_df['denoiser'] == 'NONE'].copy()
    if min_articles > 0:
        plot_df = plot_df[plot_df['num_articles'] >= min_articles]
    if plot_df.empty:
        LOGGER.warning("No data with denoiser=None found for books vs articles plot. Skipping.")
        return
    x = plot_df['num_articles'].to_numpy()
    y = plot_df['num_books'].to_numpy()
    fig, ax = PlotManager.create_figure(figsize=(10, 7))
    ax.scatter(x, y, s=40, alpha=0.7)
    if show_labels:
        for i, row in plot_df.iterrows():
            ax.text(row['num_articles'], row['num_books'], row['topic_name'], fontsize=FONT_SIZE, ha='center', va='bottom', color='black')
    mask = (x > 0) & (y > 0)
    x_fit = x[mask]
    y_fit = y[mask]
    if len(x_fit) > 1:
        if log_scale:
            logx = np.log10(x_fit)
            logy = np.log10(y_fit)
            slope, intercept = np.polyfit(logx, logy, 1)
            x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
            y_line = 10 ** (slope * np.log10(x_line) + intercept)
            label = f"log-log fit: y ~ x^{slope:.2f}"
        else:
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
            y_line = slope * x_line + intercept
            label = f"linear fit: y = {slope:.2f}x + {intercept:.2f}"
        ax.plot(x_line, y_line, 'r--', linewidth=2, label=label)
    ax.set_xlabel('Number of Articles')
    ax.set_ylabel('Number of Books')
    ax.set_title('Books vs Articles by Topic')
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.legend(loc='best', fontsize=10)
    # plt.tight_layout()
    plot_dir = artifacts_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / ('books_vs_articles_pyplot.png' if not log_scale else 'books_vs_articles_log_pyplot.png')
    PlotManager.save_and_close(fig, output_file)

def do_plots(artifacts_dir: Path):

    df = _book_article_results.get_df()

    if df.empty:
        LOGGER.info("No data available for plotting. Skipping plots.")
        return

    LOGGER.info(f"Plotting {len(df)} rows of data")

    df['total_works'] = df['num_books'] + df['num_articles']

    df = df[df['denoiser'].isin(ALLOWED_DENOISER_NAMES)]

    df['kball_lift'] = df['kball_size'] * df['num_books'] / df['num_articles']
    df['kball_log_lift'] = np.log2(df['kball_lift'])
    df['avgref_mean_rank_lift'] = df['avgref_mean_rank'] * df['num_books'] / df['num_articles']
    df['avgref_mean_rank_log_lift'] = np.log2(df['avgref_mean_rank_lift'])

    plot_books_vs_articles(df, artifacts_dir, show_labels=True, log_scale=True, min_articles=0)
    plot_books_vs_articles_pyplot(df, artifacts_dir, show_labels=True, log_scale=True, min_articles=0)
    plot_books_vs_articles(df, artifacts_dir, show_labels=True, log_scale=False, min_articles=0)
    plot_books_vs_articles_pyplot(df, artifacts_dir, show_labels=True, log_scale=False, min_articles=0)

    plot_col_by_drop_k(df, artifacts_dir, 'kball_size', log_y=True, weights='total_works')
    plot_col_by_drop_k(df, artifacts_dir, 'kball_lift', log_y=True, weights='total_works')
    plot_col_by_drop_k(df, artifacts_dir, 'kball_log_lift', log_y=False, weights='total_works')
    plot_col_by_drop_k(df, artifacts_dir, 'avgref_mean_rank', log_y=True, weights='num_books_with_avgref')
    plot_col_by_drop_k(df, artifacts_dir, 'avgref_mean_rank_lift', log_y=True, weights='num_books_with_avgref')
    plot_col_by_drop_k(df, artifacts_dir, 'avgref_mean_rank_log_lift', log_y=False, weights='num_books_with_avgref')
    
    # Generate pyplot versions as well
    plot_col_by_drop_k_pyplot(df, artifacts_dir, 'kball_size', log_y=True, weights='total_works')
    plot_col_by_drop_k_pyplot(df, artifacts_dir, 'kball_lift', log_y=True, weights='total_works')
    plot_col_by_drop_k_pyplot(df, artifacts_dir, 'kball_log_lift', log_y=False, weights='total_works')
    plot_col_by_drop_k_pyplot(df, artifacts_dir, 'avgref_mean_rank', log_y=True, weights='num_books_with_avgref')
    plot_col_by_drop_k_pyplot(df, artifacts_dir, 'avgref_mean_rank_lift', log_y=True, weights='num_books_with_avgref')
    plot_col_by_drop_k_pyplot(df, artifacts_dir, 'avgref_mean_rank_log_lift', log_y=False, weights='num_books_with_avgref')
    
    for color_col in ['denoiser', 'drop_k']:
        plot_col_by_denoiser(df, artifacts_dir, y_col='kball_size', color_col=color_col, show_labels=False, log_y=True, weights='total_works', hover_data=['topic_name', 'num_books', 'num_articles'])
        plot_col_by_denoiser(df, artifacts_dir, y_col='kball_lift', color_col=color_col, show_labels=False, log_y=True, weights='total_works', hover_data=['topic_name', 'num_books', 'num_articles'])
        plot_col_by_denoiser(df, artifacts_dir, y_col='kball_log_lift', color_col=color_col, show_labels=False, log_y=False, weights='total_works', hover_data=['topic_name', 'num_books', 'num_articles'])
        plot_col_by_denoiser(df, artifacts_dir, y_col='avgref_mean_rank', color_col=color_col, show_labels=False, log_y=True, weights='num_books_with_avgref', hover_data=['topic_name', 'num_articles', 'num_books', 'num_books_with_avgref'])
        plot_col_by_denoiser(df, artifacts_dir, y_col='avgref_mean_rank_lift', color_col=color_col, show_labels=False, log_y=True, weights='num_books_with_avgref', hover_data=['topic_name', 'num_articles', 'num_books', 'num_books_with_avgref'])
        plot_col_by_denoiser(df, artifacts_dir, y_col='avgref_mean_rank_log_lift', color_col=color_col, show_labels=False, log_y=False, weights='num_books_with_avgref', hover_data=['topic_name', 'num_articles', 'num_books', 'num_books_with_avgref'])
        
        # Generate pyplot versions as well
        plot_col_by_denoiser_pyplot(df, artifacts_dir, y_col='kball_size', color_col=color_col, show_labels=False, log_y=True, weights='total_works')
        plot_col_by_denoiser_pyplot(df, artifacts_dir, y_col='kball_lift', color_col=color_col, show_labels=False, log_y=True, weights='total_works')
        plot_col_by_denoiser_pyplot(df, artifacts_dir, y_col='kball_log_lift', color_col=color_col, show_labels=False, log_y=False, weights='total_works')
        plot_col_by_denoiser_pyplot(df, artifacts_dir, y_col='avgref_mean_rank', color_col=color_col, show_labels=False, log_y=True, weights='num_books_with_avgref')
        plot_col_by_denoiser_pyplot(df, artifacts_dir, y_col='avgref_mean_rank_lift', color_col=color_col, show_labels=False, log_y=True, weights='num_books_with_avgref')
        plot_col_by_denoiser_pyplot(df, artifacts_dir, y_col='avgref_mean_rank_log_lift', color_col=color_col, show_labels=False, log_y=False, weights='num_books_with_avgref')

@MEMORY.cache
def get_sorted_topics_by_work_count(cloud: Cloud, subfield: str):
    subfields = pyalex.Subfields()
    ai_subfield: pyalex.Subfield = subfields[subfield] # type: ignore
    topics: list = ai_subfield["topics"]
    topic_work_counts = []
    for topic in topics:
        topic_id = topic["id"].split("/")[-1]
        subcloud = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id))
        num_works = len(subcloud.embeddings())
        topic_work_counts.append((num_works, topic))
    topics_sorted = [topic for _, topic in sorted(topic_work_counts, key=lambda x: -x[0])]
    return topics_sorted, topic_work_counts

def variance_retention_main():
    explicit_args = get_explicit_args()
    setup_logging(log_name=__name__.split('.')[-1], explicit_args=explicit_args)
    
    if 'drop_k' in dict(explicit_args):
        raise ValueError("many_main should not be called with a specific drop_k. Use the main function instead.")
    if 'radius_fraction' in dict(explicit_args):
        raise ValueError("many_main should not be called with a specific radius_fraction. Use the main function instead.")
    
    args = vars(parse_args())
    del args['drop_k']
    del args['radius_fraction']

    plot_variance_retention(
        filter_pair=FilterPair(Filter(subfield=SubField.AI.value), Filter()),
        **args)

def plot_variance_retention(filter_pair: FilterPair, index_type: IndexType, hnsw_m: int, hnsw_efc: int, hnsw_efs: int,
                            max_works: Optional[int], deduplicator: Optional[Deduplicator], denoiser: str) -> None:
    if index_type == IndexType.HNSW:
        index_info = HNSW(hnsw_m=hnsw_m, hnsw_efc=hnsw_efc, hnsw_efs=hnsw_efs)
    else:
        index_info = index_type.value()
    original_cloud = Cloud(
        filter_pair=filter_pair,
        index_info=index_info,
        max_works=max_works,
        deduplicator=deduplicator,
        normalize_embeddings=True,
        denoiser=None,
        drop_k=0
    )
    original_embeddings = original_cloud.embeddings()
    N, d = original_embeddings.shape
    context = CloudDenoiserContext(original_cloud.works(), original_embeddings, Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    S_T = context.S_T()
    S_L = context.S_L()
    S_B = context.S_B()

    assert S_T.shape == (d, d), f"Expected S_T shape {(d, d)}, got {S_T.shape}"
    assert S_L.shape == (d, d), f"Expected S_L shape {(d, d)}, got {S_L.shape}"
    assert S_B.shape == (d, d), f"Expected S_B shape {(d, d)}, got {S_B.shape}"

    S_T_var = np.linalg.trace(S_T)
    S_L_var = np.linalg.trace(S_L)
    S_B_var = np.linalg.trace(S_B)

    P = WorkDenoisers[denoiser].value.projector(original_cloud.works(), original_embeddings, n_components=d)

    S_T_proj = P @ S_T @ P.T
    S_L_proj = P @ S_L @ P.T
    S_B_proj = P @ S_B @ P.T

    S_T_diag = np.diag(S_T_proj)
    S_L_diag = np.diag(S_L_proj)
    S_B_diag = np.diag(S_B_proj)

    S_T_proj_vars = np.cumsum(S_T_diag) / S_T_var
    S_L_proj_vars = np.cumsum(S_L_diag) / S_L_var
    S_B_proj_vars = np.cumsum(S_B_diag) / S_B_var

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1, d + 1),
        y=S_T_proj_vars,
        mode='lines+markers',
        name='S_T Variance Retention',
        line=dict(color='blue', width=2),
        marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(1, d + 1),
        y=S_L_proj_vars,
        mode='lines+markers',
        name='S_L Variance Retention',
        line=dict(color='green', width=2),
        marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(1, d + 1),
        y=S_B_proj_vars,
        mode='lines+markers',
        name='S_B Variance Retention',
        line=dict(color='red', width=2),
        marker=dict(size=5)
    ))
    fig.update_layout(
        title='Variance Retention by Projection',
        xaxis_title='Number of Components',
        yaxis_title='Cumulative Variance Retention',
        width=1000,
        height=700,
    )
    plot_dir = Path("artifacts") / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / f"variance_retention_{denoiser}.html"
    fig.write_html(output_file)
    LOGGER.info(f"Variance retention plot saved to {output_file}")

                
    


def main(drop_k: int, max_works: Optional[int], deduplicator: Optional[Deduplicator], radius_fraction: float, denoiser: str,
         index_type: IndexType, hnsw_m: int, hnsw_efc: int, hnsw_efs: int):

    # Standard cloud configuration
    if index_type == IndexType.HNSW:
        index_info = HNSW(hnsw_m=hnsw_m, hnsw_efc=hnsw_efc, hnsw_efs=hnsw_efs)
    else:
        index_info = index_type.value()
    # Use passed-in values for max_works, deduplicator, normalize_embeddings, denoiser, drop_k
    # Create the AI cloud (subfield=AI, topic=None)
    ai_filter_pair = FilterPair(Filter(subfield=SubField.AI.value), Filter())
    ai_cloud = Cloud(
        filter_pair=ai_filter_pair,
        index_info=index_info,
        max_works=max_works,
        deduplicator=deduplicator,
        normalize_embeddings=True,
        denoiser=denoiser if denoiser != "NONE" else None,
        drop_k=drop_k
    )
    
    

    topics, _topic_work_counts = get_sorted_topics_by_work_count(
        # Use the AI cloud with denoiser=None and drop_k=0 for caching
        replace(ai_cloud, denoiser=None, drop_k=0),
        subfield=SubField.AI.value)

    for i, topic in enumerate(topics):
        topic_name = topic["display_name"]
        if _book_article_results.has_row(topic_name=topic_name, denoiser=denoiser, drop_k=drop_k):
            continue
        topic_id = topic["id"]
        LOGGER.info("-+-"*10 + f"\nAnalyzing topic {i+1}/{len(topics)}: {topic_name} ({topic_id})\nDenoiser: {denoiser}, Drop_k: {drop_k}\n" + "-+-"*10)
        topic_id_suf = topic_id.split("/")[-1]
        analysis = analyze_book_article_gap(ai_cloud, topic_id_suf, radius_fraction=radius_fraction)
        result_row = {
            'topic_name': topic_name,
            'topic_id': topic_id,
            'denoiser': denoiser,
            'drop_k': drop_k,
            **vars(analysis)
        }
        _book_article_results.add_row(**result_row)
        if (i+1) % 20 == 0 or i == len(topics) - 1:
            do_plots(ARTIFACTS_DIR)
# python -c "from nextbook.analyze_book_article_gap import many_main; many_main()"
def many_main():
    explicit_args = get_explicit_args()
    setup_logging(log_name=__name__.split('.')[-1], explicit_args=explicit_args)
    
    if 'denoiser' in dict(explicit_args):
        raise ValueError("many_main should not be called with a specific denoiser. Use the main function instead.")
    if 'drop_k' in dict(explicit_args):
        raise ValueError("many_main should not be called with a specific drop_k. Use the main function instead.")
    
    work_denoisers = [denoiser.name for denoiser in WorkDenoisers]
    allowed_denoisers = [wd.name for wd in ALLOWED_WORK_DENOISERS]
    LOGGER.info(f"All denoisers: {ALLOWED_DENOISER_NAMES}")
    drop_ks = [1, 2, 3, 4, 5, 6, 7, 8]
    LOGGER.info(f"All drop_k values: {drop_ks}")
    
    args = vars(parse_args())
    del args['denoiser']
    del args['drop_k']

    do_plots(ARTIFACTS_DIR)

    if "NONE" in ALLOWED_DENOISER_NAMES:
        # Run with denoiser=None first
        LOGGER.info("Running analysis with denoiser=None (baseline)")
        main(**args, denoiser="NONE", drop_k=0)
    
    for drop_k in drop_ks:
        for denoiser in allowed_denoisers:
            LOGGER.info(f"Running analysis with denoiser={denoiser}, drop_k={drop_k}")
            main(**args, denoiser=denoiser, drop_k=drop_k)

if __name__ == "__main__":
    from nextbook.script_utils import run_script_main
    run_script_main(main, __file__)
