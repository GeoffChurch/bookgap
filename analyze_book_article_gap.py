#!/usr/bin/env python3
"""
Analyze the gap between books and articles across different filters.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import matplotlib.colors as mcolors

import pyalex
import plotly.express as px
import plotly.graph_objects as go

from nextbook.cloud import Cloud, SubCloud, get_ball_coverage, cross_cloud_nn_ranks
from nextbook.parse_args import parse_args
from nextbook.util import Filters, Filter, WorkType, LOGGER, TIMER, FilterPair, SubField, Topic, setup_logging, get_stats
from nextbook.index import HNSW, IndexType

@dataclass
class BookArticleAnalysis:
    """Results of analyzing books vs articles in a cloud or subcloud."""
    num_books: int
    num_articles: int
    largest_gap: float
    total_works: int
    nearest_article_rank_mean: float
    nearest_article_rank_median: float
    nearest_book_rank_mean: float
    nearest_book_rank_median: float

@TIMER.time(formatters=dict(cloud=lambda c: f"<omitted>"))
def analyze_book_article_gap(cloud: Cloud, topic_id: str) -> BookArticleAnalysis:
    books = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_type=WorkType.BOOK.value))
    articles = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_type=WorkType.ARTICLE.value))
    book_works = books.works()
    article_works = articles.works()
    num_books = len(book_works)
    num_articles = len(article_works)
    total_works = num_books + num_articles
    LOGGER.info(f"{topic_id}: {num_books=}, {num_articles=}, {total_works=}")
    coverage_result = get_ball_coverage(articles, books, radius_fraction=0.75)
    largest_gap = float(np.max(coverage_result.ball_counts)) if len(coverage_result.ball_counts) > 0 else 0.0
    LOGGER.info(f"{topic_id}: Largest gap (max ball count): {largest_gap}")
    ranks_of_nearest_article = cross_cloud_nn_ranks(articles, books).ranks
    assert ranks_of_nearest_article is not None
    nearest_article_stats = get_stats(ranks_of_nearest_article)
    ranks_of_nearest_book = cross_cloud_nn_ranks(books, articles).ranks
    assert ranks_of_nearest_book is not None
    nearest_book_stats = get_stats(ranks_of_nearest_book)

    
    return BookArticleAnalysis(
        num_books=num_books,
        num_articles=num_articles,
        largest_gap=largest_gap,
        total_works=total_works,
        nearest_article_rank_mean=nearest_article_stats['mean'],
        nearest_article_rank_median=nearest_article_stats['median'],
        nearest_book_rank_mean=nearest_book_stats['mean'],
        nearest_book_rank_median=nearest_book_stats['median']
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
        df = df.sort_values(by=['gap_significance'], ascending=[False])
        # Save to CSV
        df.to_csv(csv_path, index=False, float_format='%.3f')
        LOGGER.info(f"Saved results to {csv_path}")
    return df

@TIMER.time()
def plot_nn_book_rank_by_total_works(df: pd.DataFrame, artifacts_dir: Path):
    """
    Generate an HTML Plotly plot of the CSV results.
    X-axis: total_works (log scale)
    Y-axis: nearest_book_rank_mean
    Color: different (denoiser, drop_k) combinations using HSV colormap
    """
    valid_df = df.copy()
    # Fill missing denoiser values with 'None' for plotting
    if 'denoiser' in valid_df.columns:
        valid_df['denoiser'] = valid_df['denoiser'].fillna('None')
    # Create a combined denoiser-drop_k column for coloring
    valid_df = valid_df.copy()
    valid_df['denoiser_drop_k'] = valid_df['denoiser'].astype(str) + '_k' + valid_df['drop_k'].astype(str)
    # Get unique denoiser-drop_k combinations
    unique_combinations = valid_df['denoiser_drop_k'].unique()
    n_combinations = len(unique_combinations)
    # Generate colors using HSV colormap interpolation
    if n_combinations > 1:
        hsv_values = np.linspace(0, 1, n_combinations, endpoint=False)
        colors = [mcolors.hsv_to_rgb((h, 0.8, 0.9)) for h in hsv_values]
        # Convert to hex colors for Plotly
        hex_colors = [mcolors.to_hex(tuple(color)) for color in colors]
        color_map = dict(zip(unique_combinations, hex_colors))
    else:
        color_map = {unique_combinations[0]: '#1f77b4'}  # Default blue
    # Create the scatter plot
    fig = px.scatter(
        valid_df,
        x='total_works',
        y='nearest_book_rank_mean',
        color='denoiser_drop_k',
        color_discrete_map=color_map,
        hover_data=['topic_name', 'largest_gap', 'gap_significance', 'num_books', 'num_articles'],
        hover_name='topic_name',
        title='Book-Article Gap Analysis: Nearest Neighbor Ranks by Total Works'
    )
    # Update marker size and opacity for better visibility
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    # Update layout for better appearance
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
        yaxis_title='Nearest Book Rank Mean'
    )
    # Save the plot
    plot_dir = artifacts_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / "nn_book_rank_by_total_works.html"
    fig.write_html(output_file)
    LOGGER.info(f"Saved interactive plot to {output_file}")

@TIMER.time()
def plot_book_rank_by_denoiser(df: pd.DataFrame, artifacts_dir: Path):
    """
    Plot lift (nearest_book_rank_mean / (num_articles / num_books)) for each (denoiser, drop_k) as a categorical x-axis.
    Show all points, mean, and whiskers at ±1 stddev.
    Sort denoiser_drop_k by ascending mean lift. Draw faint lines between adjacent denoiser_drop_k groups for the same topic_id.
    """
    df = df.copy()
    # Fill missing denoiser values with 'None' for plotting
    if 'denoiser' in df.columns:
        df['denoiser'] = df['denoiser'].fillna('None')
    # Ensure drop_k is string and strip whitespace from denoiser
    df['denoiser'] = df['denoiser'].astype(str).str.strip()
    df['drop_k'] = df['drop_k'].astype(str).str.strip()
    df['denoiser_drop_k'] = df['denoiser'] + '_k' + df['drop_k']
    # Compute lift metric: set lift=1.0 where num_books==0, else normal value, then log2
    raw_lift = np.where(
        df['num_books'].astype(float) == 0,
        1.0,
        df['nearest_book_rank_mean'] / (df['num_articles'] / df['num_books'])
    )
    df['lift'] = np.log2(raw_lift)
    # Compute means and stddevs for each group (with respect to log-lift)
    stats = df.groupby('denoiser_drop_k')['lift'].agg(['mean', 'std']).reset_index()
    # Sort denoiser_drop_k by ascending mean log-lift
    sorted_groups = stats.sort_values('mean')['denoiser_drop_k'].tolist()
    # Set categorical order for x-axis (enforce strict Categorical)
    df['denoiser_drop_k'] = pd.Categorical(df['denoiser_drop_k'], categories=sorted_groups, ordered=True)
    # Create the strip plot (all points)
    fig = px.strip(
        df,
        x='denoiser_drop_k',
        y='lift',
        hover_data=['topic_name', 'largest_gap', 'gap_significance', 'num_books', 'num_articles', 'nearest_book_rank_mean'],
        title='Lift of Nearest Book Rank Mean by Denoiser/Drop_k',
        stripmode='overlay',
        color='denoiser_drop_k',
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    # Add mean and stddev whiskers for each group
    for i, row in stats.iterrows():
        x = row['denoiser_drop_k']
        mean = row['mean']
        std = row['std']
        # Add mean marker with hover
        fig.add_trace(go.Scatter(
            x=[x], y=[mean],
            mode='markers',
            marker=dict(symbol='diamond', size=14, color='black'),
            name=f"Mean: {x}",
            showlegend=False,
            hoverinfo='x+y+text',
            text=[f"Mean lift: {mean:.2f} ± {std:.2f}" if not np.isnan(std) else f"Mean lift: {mean:.2f}"],
        ))
        # Add whiskers at ±1 stddev
        if not np.isnan(std):
            fig.add_trace(go.Scatter(
                x=[x, x], y=[mean-std, mean+std],
                mode='lines',
                line=dict(color='black', width=3),
                name=f"±1 std: {x}",
                showlegend=False,
                hoverinfo='skip',
            ))
    # Draw faint lines between adjacent denoiser_drop_k groups for the same topic_id
    for topic_id, group in df.groupby('topic_id'):
        group = group.sort_values('denoiser_drop_k')
        if len(group) < 2:
            continue
        x_vals = group['denoiser_drop_k'].astype(str).tolist()
        y_vals = group['lift'].tolist()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            line=dict(color='rgba(100,100,100,0.2)', width=2),
            showlegend=False,
            hoverinfo='skip',
        ))
    fig.update_layout(
        xaxis_title='Denoiser / Drop_k',
        yaxis_title='Lift: log(Mean Rank of Nearest Book / Expected Mean Rank)',
        width=1100,
        height=700,
        font=dict(size=12),
        title_font=dict(size=16),
        xaxis={'categoryorder':'array', 'categoryarray': sorted_groups}
    )
    plot_dir = artifacts_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / "book_rank_by_denoiser.html"
    fig.write_html(output_file)
    LOGGER.info(f"Saved book rank by denoiser plot to {output_file}")

def plot_gap_significance_by_total_works(df: pd.DataFrame, artifacts_dir: Path):
    """
    Plot total_works (x-axis) vs log2(gap_significance) (y-axis), colored by denoiser_drop_k.
    Output: gap_significance_by_total_works.html
    """
    plot_df = df.copy()
    if 'denoiser' in plot_df.columns:
        plot_df['denoiser'] = plot_df['denoiser'].fillna('None')
    plot_df['denoiser'] = plot_df['denoiser'].astype(str).str.strip()
    plot_df['drop_k'] = plot_df['drop_k'].astype(str).str.strip()
    plot_df['denoiser_drop_k'] = plot_df['denoiser'] + '_k' + plot_df['drop_k']
    # Take log2 of gap_significance, handle nonpositive values gracefully
    plot_df['log_gap_significance'] = np.log2(plot_df['gap_significance'].replace(0, np.nan))
    unique_combinations = plot_df['denoiser_drop_k'].unique()
    n_combinations = len(unique_combinations)
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
        y='log_gap_significance',
        color='denoiser_drop_k',
        color_discrete_map=color_map,
        hover_data=['topic_name', 'largest_gap', 'num_books', 'num_articles', 'log_gap_significance'],
        hover_name='topic_name',
        title='Gap Significance by Total Works'
    )
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
        yaxis_title='Gap Significance',
    )
    plot_dir = artifacts_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / "gap_significance_by_total_works.html"
    fig.write_html(output_file)
    LOGGER.info(f"Saved gap significance plot to {output_file}")

def do_plots(df: pd.DataFrame, artifacts_dir: Path):
    plot_nn_book_rank_by_total_works(df, artifacts_dir)
    plot_book_rank_by_denoiser(df, artifacts_dir)
    plot_gap_significance_by_total_works(df, artifacts_dir)

def main(drop_k: int, log_level: str, max_works: Optional[int], duplicate_expectation: float, radius_fraction: float, denoiser: Optional[str],
         index_type: IndexType, hnsw_m: int, hnsw_efc: int, hnsw_efs: int):
    setup_logging(__name__, log_level=log_level)
    # Set up output directory and CSV file
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_path = artifacts_dir / "book_article_analysis.csv"
    
    # Load existing results if they exist
    try:
        df = pd.read_csv(csv_path)
        LOGGER.info(f"Loaded existing results from {csv_path}")
    except FileNotFoundError:
        df = pd.DataFrame()
        LOGGER.info(f"Starting fresh analysis - {csv_path} not found")
    
    key_columns = ['topic_name', 'denoiser', 'drop_k']
    existing_keys = set(tuple(row) for row in df[key_columns].to_numpy()) if not df.empty else set()
    results = []

    do_plots(df, artifacts_dir)
    
    # Standard cloud configuration
    if index_type == IndexType.HNSW:
        index_info = HNSW(hnsw_m=hnsw_m, hnsw_efc=hnsw_efc, hnsw_efs=hnsw_efs)
    else:
        index_info = index_type.value()
    # Use passed-in values for max_works, duplicate_expectation, normalize_embeddings, denoiser, drop_k
    # Create the AI cloud (subfield=AI, topic=None)
    ai_filter_pair = FilterPair(Filter(subfield=SubField.AI.value), Filter())
    ai_cloud = Cloud(
        filter_pair=ai_filter_pair,
        index_info=index_info,
        max_works=max_works,
        duplicate_expectation=duplicate_expectation,
        normalize_embeddings=True,
        denoiser=denoiser,
        drop_k=drop_k
    )
    
    subfields = pyalex.Subfields()
    ai_subfield: pyalex.Subfield = subfields["1702"]
    topics: list = ai_subfield["topics"]
    LOGGER.info(f"{type(subfields)=} {type(ai_subfield)=} {type(topics)=}")
    for topic in topics:
        LOGGER.info(f"{type(topic)=} {topic=}")
        topic_name = topic["display_name"]
        key_tuple = (topic_name, str(denoiser), drop_k)
        if key_tuple in existing_keys:
            LOGGER.info(f"Skipping topic '{topic_name}' - already analyzed")
            continue
        topic_id = topic["id"]
        LOGGER.info(f"Analyzing topic: {topic_name} ({topic_id})")
        # Create SubClouds for this topic: one for books, one for articles

        topic_id_suf = topic_id.split("/")[-1]
        # books_subcloud = SubCloud(parent=ai_cloud, post_filter=Filter(topic=topic_id_suf, work_type=WorkType.BOOK.value))
        # articles_subcloud = SubCloud(parent=ai_cloud, post_filter=Filter(topic=topic_id_suf, work_type=WorkType.ARTICLE.value))
        # analysis = analyze_book_article_gap(books_subcloud, articles_subcloud)
        analysis = analyze_book_article_gap(ai_cloud, topic_id_suf)
        result_row = {
            'topic_name': topic_name,
            'topic_id': topic_id,
            'denoiser': denoiser,
            'drop_k': drop_k,
            'gap_significance': analysis.largest_gap / (analysis.num_articles / analysis.num_books) if analysis.num_books > 0 else 0.0,
            **vars(analysis)
        }
        results.append(result_row)
        LOGGER.info(f"Completed analysis for topic {topic_id}: {result_row}")
        # Write to CSV after each iteration
        df = write_results_to_csv(df, [result_row], csv_path)
        
        do_plots(df, artifacts_dir)

    # Print summary at the end
    print("\n" + "="*80)
    print("BOOK/ARTICLE ANALYSIS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    # Print some key insights
    valid_gaps = df[df['largest_gap'].notna()]
    if not valid_gaps.empty:
        max_gap_row = valid_gaps.loc[valid_gaps['largest_gap'].idxmax()]
        print(f"\nLargest gap found: {max_gap_row['largest_gap']:.3f} in filter '{max_gap_row['topic_name']}'")
        total_books = df['num_books'].sum()
        total_articles = df['num_articles'].sum()
        print(f"Total across all filters: {total_books} books, {total_articles} articles")


if __name__ == "__main__":
    main(**vars(parse_args(log_prefix=__file__)))
