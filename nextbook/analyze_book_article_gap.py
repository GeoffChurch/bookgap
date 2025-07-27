#!/usr/bin/env python3
"""
Analyze the gap between books and articles across different filters.
"""

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Union
import matplotlib.colors as mcolors

import pyalex
import plotly.express as px
import plotly.graph_objects as go

from nextbook.cloud import Cloud, SubCloud, get_ball_coverage, within_cloud_nn_ranks, faiss_search_outside_epsilon, WorkDenoisers
from nextbook.parse_args import get_explicit_args, parse_args
from nextbook.util import Filter, WorkType, LOGGER, FilterPair, SubField, setup_logging, get_stats, MEMORY
from nextbook.deduplicators import Deduplicator
from nextbook.index import HNSW, IndexType

FONT_SIZE = 7

@dataclass
class BookArticleAnalysis:
    """Results of analyzing books vs articles in a cloud or subcloud."""
    num_books: int
    num_articles: int
    largest_gap: float
    total_works: int
    kball_size: float

def analyze_book_article_gap(cloud: Cloud, topic_id: str, radius_fraction: float) -> BookArticleAnalysis:
    books = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_type=WorkType.BOOK.value))
    articles = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_type=WorkType.ARTICLE.value))
    num_books = len(books.embeddings())
    num_articles = len(articles.embeddings())
    total_works = num_books + num_articles
    coverage_result = get_ball_coverage(articles, books, radius_fraction=radius_fraction)
    largest_gap = float(np.max(coverage_result.ball_counts)) if len(coverage_result.ball_counts) > 0 else 0.0

    return BookArticleAnalysis(
        num_books=num_books,
        num_articles=num_articles,
        largest_gap=largest_gap,
        total_works=total_works,
        kball_size=kball_size(articles, books, k=1)
    )

def write_results_to_csv(df, results, csv_path):
    if results:
        new_df = pd.DataFrame(results)
        if not df.empty:
            # Ensure column compatibility
            assert list(df.columns) == list(new_df.columns), f"Column mismatch: {list(df.columns)} != {list(new_df.columns)}"
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = new_df
        # Sort by largest gap (descending) and then by filter name
        # df = df.sort_values(by=['largest_gap', 'topic_name'], ascending=[False, True], na_position='last')
        # Sort by descending gap significance
        # df = df.sort_values(by=['gap_significance'], ascending=[False])
        df = df.sort_values(by=['total_works', 'drop_k', 'denoiser'])
        # Save to CSV
        df.to_csv(csv_path, index=False, float_format='%.5f')
    return df

def kball_size(index: SubCloud, query: SubCloud, k: int) -> float:
    """
    Calculates the mean k-ball size as a fraction of the index cloud.

    This metric evaluates how well the 'index' cloud separates points within the
    'query' cloud. For each point in the query cloud, it defines a hypersphere ("k-ball")
    whose radius is the distance to its k-th nearest neighbor within that same cloud.
    It then counts how many points from the 'index' cloud fall inside this ball,
    and divides by the total number of points in the index cloud.

    Higher is better, meaning that on average a large fraction of index points
    are closer to a query point than its own k-th nearest neighbor.

    Args:
        index: The cloud of points that "separates" the query points (e.g., articles).
        query: The cloud of points for which the metric is calculated (e.g., books).
        k: The rank of the same-cloud neighbor used to define the radius of the k-ball.

    Returns:
        The mean fraction of index points found inside the k-balls, or NaN if no
        ranks could be computed.
    """
    ranks = within_cloud_nn_ranks(index, query, k=k).ranks

    if ranks is None or len(ranks) == 0:
        LOGGER.warning(f"No ranks found for k={k}, returning NaN")
        return np.nan
    
    num_index_points = len(index.embeddings())
    if num_index_points == 0:
        LOGGER.warning(f"Index cloud has no points, returning NaN")
        return np.nan
    
    LOGGER.info(f"Mean k-ball size stats: {get_stats(ranks)}")
    
    # Divide by number of index points to get fraction
    return float(np.mean(ranks) / num_index_points)

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
    
    nearest_indices = faiss_search_outside_epsilon(index, query,is_intra_cloud_search=False, k=1, epsilon=0).I_outside.squeeze(1)

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
        hover_cols=['topic_name', 'largest_gap', 'gap_significance', 'num_books', 'num_articles'],
        color_col=color_col,
        show_labels=show_labels,
        yaxis_title='K-Ball Size',
        log_y=False
    )
def plot_gap_significance_by_total_works(df: pd.DataFrame, artifacts_dir: Path, color_col='denoiser_drop_k', show_labels=False, denoisers=None):
    plot_df = df.copy()
    if denoisers is not None:
        plot_df = plot_df[plot_df['denoiser'].isin(denoisers)]
    plot_df['log_gap_significance'] = np.log2(plot_df['gap_significance'].replace(0, np.nan))
    plot_scatter_by_total_works(
        plot_df,
        artifacts_dir,
        y_col='log_gap_significance',
        title='Gap Significance by Total Works',
        output_filename='gap_significance_by_total_works.html',
        hover_cols=['topic_name', 'largest_gap', 'num_books', 'num_articles', 'log_gap_significance'],
        color_col=color_col,
        show_labels=show_labels,
        yaxis_title='Gap Significance',
        log_y=False
    )
    
def plot_col_by_denoiser(df: pd.DataFrame, artifacts_dir: Path, y_col: str, color_col='denoiser', show_labels=False):
    """
    Plot a specified y_col for each (denoiser, drop_k) as a categorical x-axis.
    Show all points, mean, and horizontal whiskers at ±1 stddev.
    Sort denoiser_drop_k by ascending mean y_col. Draw faint lines between adjacent denoiser_drop_k groups for the same topic_id, and connect whisker means as well.
    color_col: str or tuple/list of str, column(s) to use for color grouping.
    show_labels: if True, always show topic_name as text label at each point.
    """
    df = df.copy()
    df['denoiser'] = df['denoiser'].astype(str).str.strip()
    df['denoiser_drop_k'] = df['denoiser'] + '_k' + df['drop_k'].astype(str)
    # Compute weighted means and stddevs for each group (with respect to y_col)
    def weighted_mean(x):
        return np.average(x[y_col], weights=x['total_works'])
    def weighted_std(x):
        values = x[y_col].to_numpy(dtype=float)
        weights = x['total_works'].to_numpy(dtype=float)
        if np.sum(weights) == 0:
            return np.nan
        mean = np.average(values, weights=weights)
        variance = np.average((values - mean) ** 2, weights=weights)
        return np.sqrt(variance)
    stats = df.groupby('denoiser_drop_k').apply(weighted_mean).reset_index(name='mean')
    stats['std'] = df.groupby('denoiser_drop_k').apply(weighted_std).values
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
        hover_data=['topic_name', 'largest_gap', 'gap_significance', 'num_books', 'num_articles'],
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
    plot_dir = artifacts_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / f"{y_col}_by_denoiser_{color_col=}.html"
    fig.write_html(output_file)

def plot_metric_by_drop_k(df: pd.DataFrame, artifacts_dir: Path, y_col: str):
    """
    Plot metric vs drop_k with each denoiser as a separate series.
    For each (denoiser, drop_k) pair, average the metric across all topics.
    Treats "NONE" denoiser as drop_k = 0 baseline.
    
    Args:
        df: DataFrame with the data
        artifacts_dir: Directory to save plots
        y_col: Column name to plot on y-axis
    """
    plot_df = df.copy()
    
    # For NONE denoiser, set drop_k to 0 regardless of actual drop_k value
    plot_df.loc[plot_df['denoiser'] == 'NONE', 'drop_k'] = 0

    # Drop rows where the metric is NaN
    plot_df = plot_df[plot_df[y_col].notna()]
    
    if plot_df.empty:
        LOGGER.info(f"No data available for drop_k plot ({y_col}). Skipping plot.")
        return
    
    # Check that the y_col exists
    if y_col not in plot_df.columns:
        LOGGER.warning(f"Column '{y_col}' not found in data. Skipping plot.")
        return
    
    # Group by denoiser and drop_k, then average the metric across topics
    grouped = plot_df.groupby(['denoiser', 'drop_k'])[y_col].agg(['mean', 'std', 'count']).reset_index()
    grouped.columns = ['denoiser', 'drop_k', 'metric_mean', 'std', 'count']
    
    # Sort by drop_k for proper x-axis ordering
    grouped = grouped.sort_values('drop_k')
    
    # Create the line plot
    fig = go.Figure()
    
    # Get unique denoisers and assign colors
    denoisers = grouped['denoiser'].unique()
    colors = px.colors.qualitative.Dark24[:len(denoisers)]
    
    # Use generic naming conventions
    y_axis_title = y_col.replace('_', ' ').title()
    filename = f'{y_col}_by_drop_k.html'
    
    for i, denoiser in enumerate(denoisers):
        denoiser_data = grouped[grouped['denoiser'] == denoiser].sort_values('drop_k')
        
        # Special styling for NONE denoiser (baseline)
        if denoiser == 'NONE':
            line_style = dict(color='black', width=4, dash='dash')
            marker_style = dict(size=10, color='black', symbol='diamond')
            denoiser_display_name = 'No Denoising (Baseline)'

            # Add baseline line and markers
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
        else:
            line_style = dict(color=colors[i % len(colors)], width=3)
            marker_style = dict(size=8, color=colors[i % len(colors)])
            denoiser_display_name = denoiser
        
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
            
            # # Add error bars (±s std)
            # s = 0.01
            # fig.add_trace(go.Scatter(
            #     x=denoiser_data['drop_k'].tolist() + denoiser_data['drop_k'].tolist()[::-1],
            #     y=(denoiser_data['log_nearest_book_rank_mean'] + s*denoiser_data['std']).tolist() + 
            #         (denoiser_data['log_nearest_book_rank_mean'] - s*denoiser_data['std']).tolist()[::-1],
            #     fill='toself',
            #     fillcolor=f'rgba{tuple(list(mcolors.to_rgb(colors[i % len(colors)])) + [0.2])}',
            #     line=dict(color='rgba(255,255,255,0)'),
            #     hoverinfo="skip",
            #     showlegend=False,
            #     name=f'{denoiser} ±{s}σ'
            # ))
    
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
    
    # Save the plot
    plot_dir = artifacts_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / filename
    fig.write_html(output_file)
    LOGGER.info(f"Saved drop_k plot ({y_col}) to {output_file}")

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
        hover_data=['topic_name', 'topic_id', 'largest_gap', 'gap_significance', 'total_works'],
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

def do_plots(df: pd.DataFrame, artifacts_dir: Path, show_labels=True):
    if df.empty:
        LOGGER.info("No data available for plotting. Skipping plots.")
        return
    
    df = df.copy()
    
    plot_books_vs_articles(df, artifacts_dir, show_labels=show_labels, log_scale=True, min_articles=10_000)
    plot_books_vs_articles(df, artifacts_dir, show_labels=show_labels, log_scale=False, min_articles=10_000)

    plot_metric_by_drop_k(df, artifacts_dir, 'kball_size')

    for color_col in ['denoiser', 'drop_k']:
        plot_gap_significance_by_total_works(df, artifacts_dir, color_col=color_col, show_labels=show_labels, denoisers=['NONE'])
        plot_col_by_denoiser(df, artifacts_dir, y_col='kball_size', color_col=color_col, show_labels=show_labels)

@MEMORY.cache
def get_sorted_topics_by_work_count(cloud: Cloud, subfield: str):
    subfields = pyalex.Subfields()
    ai_subfield: pyalex.Subfield = subfields[subfield] # type: ignore
    topics: list = ai_subfield["topics"]
    topic_work_counts = []
    for topic in topics:
        topic_id = topic["id"].split("/")[-1]
        subcloud = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id))
        num_works = len(subcloud.works())
        topic_work_counts.append((num_works, topic))
    topics_sorted = [topic for _, topic in sorted(topic_work_counts, key=lambda x: -x[0])]
    return topics_sorted, topic_work_counts

def main(drop_k: int, max_works: Optional[int], deduplicator: Optional[Deduplicator], radius_fraction: float, denoiser: str,
         index_type: IndexType, hnsw_m: int, hnsw_efc: int, hnsw_efs: int):
    # Set up output directory and CSV file
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_path = artifacts_dir / "book_article_analysis.csv"
    
    # Load existing results if they exist
    try:
        df = pd.read_csv(csv_path)
        # Ensure drop_k is always an integer after reading from CSV
        if 'drop_k' in df.columns:
            LOGGER.info(f"drop_k dtype before conversion: {df['drop_k'].dtype}")
            df['drop_k'] = pd.to_numeric(df['drop_k'], errors='coerce').astype('Int64')
            LOGGER.info(f"drop_k dtype after conversion: {df['drop_k'].dtype}")
        LOGGER.info(f"Loaded existing results from {csv_path}")
    except FileNotFoundError:
        df = pd.DataFrame()
        LOGGER.info(f"Starting fresh analysis - {csv_path} not found")
    
    key_columns = ['topic_name', 'denoiser', 'drop_k']
    existing_keys = [tuple(row) for row in df[key_columns].to_numpy()] if not df.empty else set()
    results = []

    do_plots(df, artifacts_dir)
    
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
    
    

    topics_sorted, _topic_work_counts = get_sorted_topics_by_work_count(
        # Use the AI cloud with denoiser=None and drop_k=0 for caching
        replace(ai_cloud, denoiser=None, drop_k=0),
        subfield=SubField.AI.value)
    topics = topics_sorted[:10][::-1]

    for i, topic in enumerate(topics):
        topic_name = topic["display_name"]
        key_tuple = (topic_name, denoiser, drop_k)
        if key_tuple in existing_keys:
            LOGGER.info(f"Skipping topic '{topic_name}' - already analyzed")
            continue
        topic_id = topic["id"]
        LOGGER.info("-+-"*10 + f"\nAnalyzing topic {i+1}/{len(topics)}: {topic_name} ({topic_id})")
        topic_id_suf = topic_id.split("/")[-1]
        analysis = analyze_book_article_gap(ai_cloud, topic_id_suf, radius_fraction=radius_fraction)
        result_row = {
            'topic_name': topic_name,
            'topic_id': topic_id,
            'denoiser': denoiser,
            'drop_k': drop_k,
            'gap_significance': analysis.largest_gap / (analysis.num_articles / analysis.num_books) if analysis.num_books > 0 and analysis.num_articles > 0 else 0.0,
            **vars(analysis)
        }
        results.append(result_row)
        LOGGER.info(f"Completed analysis for topic {topic_id}: {result_row}")
        # Write to CSV after each iteration
        df = write_results_to_csv(df, [result_row], csv_path)
        if i % 2 == 0 or i == len(topics) - 1:
            do_plots(df, artifacts_dir)

# python -c "from nextbook.analyze_book_article_gap import many_main; many_main()"
def many_main():
    explicit_args = get_explicit_args()
    setup_logging(log_name=__name__.split('.')[-1], explicit_args=explicit_args)
    
    if 'denoiser' in dict(explicit_args):
        raise ValueError("many_main should not be called with a specific denoiser. Use the main function instead.")
    if 'drop_k' in dict(explicit_args):
        raise ValueError("many_main should not be called with a specific drop_k. Use the main function instead.")
    
    denoisers = [denoiser.name for denoiser in WorkDenoisers]
    LOGGER.info(f"All denoisers: {denoisers}")
    drop_ks = [1, 5, 10, 20, 50, 100, 200]
    LOGGER.info(f"All drop_k values: {drop_ks}")
    
    args = vars(parse_args())
    del args['denoiser']
    del args['drop_k']

    # Run with denoiser=None first
    main(**args, denoiser="NONE", drop_k=0)
    
    for drop_k in drop_ks:
        for denoiser in denoisers:
            LOGGER.info(f"Running analysis with denoiser={denoiser}, drop_k={drop_k}")
            main(**args, denoiser=denoiser, drop_k=drop_k)

if __name__ == "__main__":
    setup_logging(log_name=Path(__file__).stem, explicit_args=get_explicit_args())
    main(**vars(parse_args()))
