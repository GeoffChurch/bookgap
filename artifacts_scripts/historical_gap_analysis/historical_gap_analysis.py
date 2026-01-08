#!/usr/bin/env python3
"""Historical gap analysis - validate book gap definitions against historical publication patterns.

This script analyzes whether books historically tend to fill large gaps in the embedding space,
which would validate the p-ball gap definition used in the main analysis.

For a given year Y:
- Build a cloud using books published up to year Y (landmark set)
- Identify books and articles published in year Y+1 (test sets)
- Compute gap sizes for test books and articles with respect to the landmark books
- Output statistics to CSV for analysis
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from numpy.typing import NDArray

from bookgap.cloud import Cloud
from bookgap.configs.config_store import register_configs
from bookgap.subcloud import SubCloud
from bookgap.types import CellType
from bookgap.util import LOGGER, MEMORY, TIMER, WorkType, Filter, get_stats, create_unicode_safe_filename, setup_logging

from .config import HistoricalAnalysisConfig

# Initialize logging via shared utility to ensure DEBUG output
setup_logging(log_name="historical_gap_analysis", log_level="DEBUG", include_time=True)

# Register configs
register_configs()

# Register historical analysis config
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name="historical_analysis", node=HistoricalAnalysisConfig(year=2010))


@dataclass(frozen=True)
class TemporalFilter(Filter):
    """Filter works by publication year and type."""
    
    max_year: Optional[int] = None
    exact_year: Optional[int] = None
    work_type: Optional[WorkType] = None
    
    def check_work(self, work: dict[str, Any]) -> bool:
        # Check parent filter first
        if not super().check_work(work):
            LOGGER.info(f"[TF] Excluded by parent filter: {work.get('id', '')}")
            return False

        pub_year = _extract_year(work.get("publication_date"))
        if pub_year is None:
            LOGGER.info(f"[TF] No pub_year: {work.get('id', '')} pub_date={work.get('publication_date')}")
            return False

        if self.max_year is not None and pub_year > self.max_year:
            # LOGGER.info(f"[TF] Excluded by max_year: {work.get('id', '')} pub_year={pub_year} max_year={self.max_year}")
            return False

        if self.exact_year is not None and pub_year != self.exact_year:
            # LOGGER.info(f"[TF] Excluded by exact_year: {work.get('id', '')} pub_year={pub_year} exact_year={self.exact_year}")
            return False

        if self.work_type is not None:
            work_type_str = work.get("type", "").lower()
            # Handle both WorkType enum and string values
            target_work_type = self.work_type.value if isinstance(self.work_type, WorkType) else self.work_type
            if target_work_type == "book" and work_type_str != "book":
                # LOGGER.info(f"[TF] Excluded by work_type=BOOK: {work.get('id', '')} type={work_type_str}")
                return False
            elif target_work_type in ["article", "preprint"] and work_type_str not in ["article", "preprint"]:
                # LOGGER.info(f"[TF] Excluded by work_type=ARTICLE: {work.get('id', '')} type={work_type_str}")
                return False

        # Uncomment to see all passing works:
        # LOGGER.info(f"[TF] Included: {work.get('id', '')} pub_year={pub_year} type={work.get('type', '')}")
        return True


@TIMER.time()
@MEMORY.cache
def _build_temporal_subcloud(
    cloud: Cloud,
    max_year: int,
    work_type: WorkType,
) -> SubCloud:
    """Build a SubCloud containing only works up to and including max_year.
    
    Args:
        cloud: Full cloud to filter
        max_year: Maximum publication year (inclusive)
        work_type: Type of works to include (BOOK or ARTICLE)
        
    Returns:
        SubCloud with filtered works and embeddings
    """
    parent_post_filter = cloud.filter_pair.post
    post_filter = TemporalFilter(
        max_year=max_year,
        work_type=work_type,
        subfield=parent_post_filter.subfield,
        topic=parent_post_filter.topic,
    )
    subcloud = SubCloud(parent=cloud, post_filter=post_filter)
    
    num_works = len(subcloud.embeddings())
    LOGGER.info(
        f"Built temporal subcloud for {work_type.value} up to year {max_year}: "
        f"{num_works} works"
    )
    
    return subcloud


@TIMER.time()
@MEMORY.cache
def _build_temporal_test_set(
    cloud: Cloud,
    target_year: int,
    work_type: WorkType,
) -> SubCloud:
    """Build a SubCloud containing only works from target_year.
    
    Args:
        cloud: Full cloud to filter
        target_year: Year to select works from
        work_type: Type of works to include (BOOK or ARTICLE)
        
    Returns:
        SubCloud with filtered works and embeddings
    """
    parent_post_filter = cloud.filter_pair.post
    post_filter = TemporalFilter(
        exact_year=target_year,
        work_type=work_type,
        subfield=parent_post_filter.subfield,
        topic=parent_post_filter.topic,
    )
    subcloud = SubCloud(parent=cloud, post_filter=post_filter)
    
    num_works = len(subcloud.embeddings())
    LOGGER.info(
        f"Built test set for {work_type.value} from year {target_year}: "
        f"{num_works} works"
    )
    
    return subcloud


def _extract_year(date_str: Optional[str]) -> Optional[int]:
    """Extract year from publication_date string.
    
    Args:
        date_str: Date string in format "YYYY-MM-DD" or "YYYY"
        
    Returns:
        Year as integer, or None if invalid/missing
    """
    if not date_str:
        return None
    
    try:
        # Handle "YYYY-MM-DD" or just "YYYY"
        return int(date_str.split("-")[0])
    except (ValueError, IndexError):
        raise ValueError(f"Invalid date string: {date_str}")


@TIMER.time()
def _compute_gap_sizes(
    landmark: SubCloud,
    query: SubCloud,
    ambient: SubCloud,
    radius_fraction: float,
    cell_type: CellType,
    is_intra_cloud: bool,
) -> NDArray[np.int64]:
    """Compute gap sizes for query points with respect to landmark set.
    
    Args:
        landmark: Landmark SubCloud (e.g., books up to year Y)
        query: Query SubCloud (e.g., books from year Y+1)
        radius_fraction: Radius fraction for p-ball definition
        cell_type: Type of cell to use for gap computation
        
    Returns:
        Array of gap sizes (number of ambient points in each gap)
    """
    LOGGER.info(
        f"Computing gap sizes for {len(query.embeddings())} query points "
        f"with respect to {len(landmark.embeddings())} landmark books "
        f"and {len(ambient.embeddings())} ambient articles (p={radius_fraction})"
    )
    gap_sizes = cell_type.value.compute_gap_sizes_for_embeddings(
        ambient=ambient,
        landmark=landmark,
        embeddings=query.embeddings(),
        radius_fraction=radius_fraction,
        k=1,
        is_intra_cloud=is_intra_cloud,
        show_progress=True,
    )
    gap_sizes = np.array(gap_sizes, dtype=np.int64)
    LOGGER.info(f"Gap size statistics: {get_stats(gap_sizes)}")
    return gap_sizes


def _plot_gap_distributions(
    book_gaps: NDArray[np.int64],
    article_gaps: NDArray[np.int64],
    landmark_gaps: NDArray[np.int64],
    output_path: Path,
    year: int,
) -> None:
    """Plot overlaid histograms of gap size distributions."""
    plt.figure(figsize=(10, 6))
    
    # Determine binning range based on all data
    all_data = np.concatenate([book_gaps, article_gaps, landmark_gaps])
    if len(all_data) == 0:
        LOGGER.warning("No data to plot histograms.")
        return
        
    max_val = np.max(all_data)
    tick_locs = None
    tick_labels = None

    if np.any(all_data == 0):
        max_orig = np.max(all_data)
        landmark_gaps = np.log1p(landmark_gaps)
        article_gaps = np.log1p(article_gaps)
        book_gaps = np.log1p(book_gaps)
        all_data = np.log1p(all_data)
        
        xlabel = 'Gap Size (Number of Ambient Articles)'
        max_val = np.max(all_data)
        bins = np.linspace(0, max_val, 50)
        
        # Custom ticks for log1p scale
        tick_vals = [0]
        if max_orig >= 1:
            tick_vals.append(1)
        curr = 10
        while curr <= max_orig:
            tick_vals.append(curr)
            curr *= 10
        
        tick_locs = np.log1p(tick_vals)
        tick_labels = [str(v) for v in tick_vals]
    else:
        max_val = np.max(all_data)
        if max_val > 0:
            # Use log-spaced bins for log-scale x-axis
            # Start from 1 (10^0) to max_val
            bins = np.logspace(0, np.log10(max_val), 50)
            plt.xscale('log')
        else:
            bins = np.linspace(0, max_val, 50)
        xlabel = 'Gap Size (Number of Ambient Articles)'
    
    # Convert bins to list to satisfy type checker
    bins_list = bins.tolist() if isinstance(bins, np.ndarray) else bins

    plt.hist(landmark_gaps, bins=bins_list, alpha=0.5, label='Prior Books (Landmarks)', density=True, color='gray')
    plt.hist(article_gaps, bins=bins_list, alpha=0.5, label='New Articles', density=True, color='blue')
    plt.hist(book_gaps, bins=bins_list, alpha=0.5, label='New Books', density=True, color='red')
    
    if tick_locs is not None:
        plt.xticks(tick_locs, tick_labels)
    
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.title(f'Gap Size Distributions (Year {year})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path)
    plt.close()
    LOGGER.info(f"Saved histogram plot to {output_path}")


def _run_analysis(cfg) -> None:
    """Execute the historical gap analysis for a single year.
    
    Args:
        cfg: HistoricalAnalysisConfig object
    """
    output_dir = Path(cfg.output_dir) if cfg.output_dir else Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log startup
    LOGGER.info("=" * 60)
    LOGGER.info("Historical Gap Analysis")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Year: {cfg.year}")
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info("")
    
    # Build the cloud
    cloud: Cloud = instantiate(cfg.experiment.cloud)
    LOGGER.info(f"Cloud: {cloud.name()}")
    LOGGER.info(f"Embedder: {cloud.embedder_model.value}")
    LOGGER.info(f"Denoiser: {cloud.denoiser}")
    LOGGER.info(f"Drop K: {cloud.drop_k}")
    
    # Build temporal subclouds
    landmark_year = cfg.year
    test_year = cfg.year + 1
    
    LOGGER.info(f"Building landmark set (books up to {landmark_year})...")
    landmark_books = _build_temporal_subcloud(cloud, landmark_year, WorkType.BOOK)
    LOGGER.info(
        f"[Diagnostics] Landmark filter='{landmark_books.post_filter}' type={type(landmark_books.post_filter).__name__} "
        f"parent_works={len(cloud.works())} filtered={len(landmark_books.embeddings())}"
    )
    # Log filter attributes to verify construction
    lf = landmark_books.post_filter
    LOGGER.info(
        f"[FilterAttrs] Landmark: max_year={getattr(lf, 'max_year', 'N/A')} "
        f"exact_year={getattr(lf, 'exact_year', 'N/A')} work_type={getattr(lf, 'work_type', 'N/A')} "
        f"subfield={getattr(lf, 'subfield', 'N/A')} topic={getattr(lf, 'topic', 'N/A')}"
    )
    # Log sample parent works to verify data fields
    sample_parent_works = cloud.works()[:3]
    LOGGER.info(f"[Sample] First 3 parent works:")
    for i, w in enumerate(sample_parent_works):
        LOGGER.info(
            f"  [{i}] id={w.get('id', 'N/A')} pub_date={w.get('publication_date', 'N/A')} "
            f"type={w.get('type', 'N/A')} primary_topic={w.get('primary_topic', {}).get('id', 'N/A')}"
        )
    
    LOGGER.info(f"Building test sets (year {test_year})...")
    test_books = _build_temporal_test_set(cloud, test_year, WorkType.BOOK)
    LOGGER.info(
        f"[Diagnostics] TestBooks filter='{test_books.post_filter}' type={type(test_books.post_filter).__name__} "
        f"parent_works={len(cloud.works())} filtered={len(test_books.embeddings())}"
    )
    tbf = test_books.post_filter
    LOGGER.info(
        f"[FilterAttrs] TestBooks: max_year={getattr(tbf, 'max_year', 'N/A')} "
        f"exact_year={getattr(tbf, 'exact_year', 'N/A')} work_type={getattr(tbf, 'work_type', 'N/A')} "
        f"subfield={getattr(tbf, 'subfield', 'N/A')} topic={getattr(tbf, 'topic', 'N/A')}"
    )
    test_articles = _build_temporal_test_set(cloud, test_year, WorkType.ARTICLE)
    LOGGER.info(
        f"[Diagnostics] TestArticles filter='{test_articles.post_filter}' type={type(test_articles.post_filter).__name__} "
        f"parent_works={len(cloud.works())} filtered={len(test_articles.embeddings())}"
    )
    taf = test_articles.post_filter
    LOGGER.info(
        f"[FilterAttrs] TestArticles: max_year={getattr(taf, 'max_year', 'N/A')} "
        f"exact_year={getattr(taf, 'exact_year', 'N/A')} work_type={getattr(taf, 'work_type', 'N/A')} "
        f"subfield={getattr(taf, 'subfield', 'N/A')} topic={getattr(taf, 'topic', 'N/A')}"
    )
    
    num_prior_books = len(landmark_books.embeddings())
    cloud_for_ambient = landmark_books.parent
    if cloud_for_ambient.filter_pair.post:
        raise ValueError("Ambient cloud's parent should not have a post filter applied")
    article_filter = TemporalFilter(
        max_year=landmark_year,
        work_type=WorkType.ARTICLE,
    )
    prior_articles_subcloud = SubCloud(parent=cloud_for_ambient, post_filter=article_filter)
    num_prior_articles = len(prior_articles_subcloud.embeddings())
    num_test_books = len(test_books.embeddings())
    num_test_articles = len(test_articles.embeddings())
    
    LOGGER.info(f"Prior books (up to {landmark_year}): {num_prior_books}")
    LOGGER.info(f"Prior articles (any year): {num_prior_articles}")
    LOGGER.info(f"Test books ({test_year}): {num_test_books}")
    LOGGER.info(f"Test articles ({test_year}): {num_test_articles}")
    
    # Check minimum requirements
    if num_test_books < cfg.min_books_per_year:
        LOGGER.warning(
            f"Skipping analysis: only {num_test_books} books in {test_year} "
            f"(minimum {cfg.min_books_per_year} required)"
        )
        return
    
    if num_test_articles < cfg.min_articles_per_year:
        LOGGER.warning(
            f"Skipping analysis: only {num_test_articles} articles in {test_year} "
            f"(minimum {cfg.min_articles_per_year} required)"
        )
        return
    
    if num_prior_books == 0:
        LOGGER.warning(f"Skipping analysis: no prior books found up to {landmark_year}")
        return
    
    # Compute gap sizes
    LOGGER.info(f"Computing gap sizes for test books...")
    book_gap_sizes = _compute_gap_sizes(
        landmark=landmark_books,
        query=test_books,
        ambient=prior_articles_subcloud,
        radius_fraction=cfg.experiment.gap.radius_fraction,
        cell_type=cfg.experiment.gap.cell_type,
        is_intra_cloud=False,
    )
    LOGGER.info(f"Computing gap sizes for test articles...")
    article_gap_sizes = _compute_gap_sizes(
        landmark=landmark_books,
        query=test_articles,
        ambient=prior_articles_subcloud,
        radius_fraction=cfg.experiment.gap.radius_fraction,
        cell_type=cfg.experiment.gap.cell_type,
        is_intra_cloud=False,
    )
    # Compute gap sizes for landmark books (for percentile/zscore calculation)
    LOGGER.info(f"Computing gap sizes for landmark books (for statistics)...")
    landmark_gap_sizes = _compute_gap_sizes(
        landmark=landmark_books,
        query=landmark_books,
        ambient=prior_articles_subcloud,
        radius_fraction=cfg.experiment.gap.radius_fraction,
        cell_type=cfg.experiment.gap.cell_type,
        is_intra_cloud=True,
    )
    
    # Compute statistics
    book_stats = get_stats(book_gap_sizes)
    article_stats = get_stats(article_gap_sizes)
    landmark_stats = get_stats(landmark_gap_sizes)
    
    LOGGER.info(f"Book gap sizes: {book_stats}")
    LOGGER.info(f"Article gap sizes: {article_stats}")
    LOGGER.info(f"Landmark book gap sizes: {landmark_stats}")
    
    # Prepare CSV output
    # Sanitize names for filesystem safety using shared utility
    safe_embedder = create_unicode_safe_filename(cloud.embedder_model.value)
    safe_denoiser = create_unicode_safe_filename(cloud.denoiser)
    
    # Also write a summary CSV
    summary_filename = (
        f"historical_gap_analysis_summary_"
        f"year{cfg.year}_"
        f"p{cfg.experiment.gap.radius_fraction:.2f}_"
        f"{safe_embedder}_"
        f"{safe_denoiser}_"
        f"drop{cloud.drop_k}.csv"
    )
    summary_path = output_dir / summary_filename
    
    LOGGER.info(f"Writing summary to {summary_path}")
    
    # Calculate quantiles
    quantiles = list(range(0, 101, 10))
    
    def _get_qs(data):
        if len(data) == 0:
            return [float('nan')] * len(quantiles)
        return np.percentile(data, quantiles).tolist()

    book_qs = _get_qs(book_gap_sizes)
    article_qs = _get_qs(article_gap_sizes)
    landmark_qs = _get_qs(landmark_gap_sizes)

    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        
        header = [
            "year",
            "p",
            "embedder_model",
            "denoiser",
            "drop_k",
            "prior_books",
            "prior_articles",
            "num_test_books",
            "num_test_articles",
            "book_gap_size_mean",
            "book_gap_size_std",
        ] + [f"book_gap_size_p{q}" for q in quantiles] + [
            "article_gap_size_mean",
            "article_gap_size_std",
        ] + [f"article_gap_size_p{q}" for q in quantiles] + [
            "landmark_gap_size_mean",
            "landmark_gap_size_std",
        ] + [f"landmark_gap_size_p{q}" for q in quantiles]
        
        writer.writerow(header)
        
        row = [
            cfg.year,
            cfg.experiment.gap.radius_fraction,
            cloud.embedder_model.value,
            cloud.denoiser,
            cloud.drop_k,
            num_prior_books,
            num_prior_articles,
            num_test_books,
            num_test_articles,
            book_stats['mean'],
            book_stats['std'],
        ] + book_qs + [
            article_stats['mean'],
            article_stats['std'],
        ] + article_qs + [
            landmark_stats['mean'],
            landmark_stats['std'],
        ] + landmark_qs
        
        writer.writerow(row)
    
    # Generate plot
    plot_filename = (
        f"historical_gap_analysis_plot_"
        f"year{cfg.year}_"
        f"p{cfg.experiment.gap.radius_fraction:.2f}_"
        f"{safe_embedder}_"
        f"{safe_denoiser}_"
        f"drop{cloud.drop_k}.png"
    )
    plot_path = output_dir / plot_filename
    _plot_gap_distributions(
        book_gaps=book_gap_sizes,
        article_gaps=article_gap_sizes,
        landmark_gaps=landmark_gap_sizes,
        output_path=plot_path,
        year=cfg.year,
    )

    LOGGER.info("=" * 60)
    LOGGER.info("ANALYSIS COMPLETED SUCCESSFULLY")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Summary written to: {summary_path}")
    LOGGER.info(f"Plot written to: {plot_path}")


@hydra.main(version_base=None, config_path=None, config_name="historical_analysis")
def main(cfg: HistoricalAnalysisConfig) -> None:
    """Main entry point for historical gap analysis."""
    _run_analysis(cfg)


if __name__ == "__main__":
    main()
