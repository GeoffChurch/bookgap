"""Context export functionality for BookGap analysis.

This module generates context files for gap analysis:
- gap_N_context.json: Structured context data for each gap
- gap_N_analysis.txt: Detailed human-readable gap analysis

Ports key functionality from write_single_kball_gap() in gap_finder.py
with standardized formatting and Hydra output directory structure.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from bookgap.llm.data_structures import GapWithContext
from bookgap.gap_statistics import compute_gap_statistics
from bookgap.gap_finder import group_gap_articles_by_covering_books, get_book_to_articles_mapping, TOP_ARTICLES_TO_DISPLAY
from bookgap.util import create_unicode_safe_filename, LOGGER


def save_gap_context(gap_with_context: GapWithContext, output_dir: Path) -> Dict[str, Path]:
    """Save gap context as both JSON and text analysis files.
    
    Args:
        gap_with_context: Gap with associated context data
        output_dir: Base output directory (Hydra managed)
        
    Returns:
        Dict with 'json' and 'txt' keys mapping to saved file paths
        
    Raises:
        ValueError: If gap data is insufficient for analysis
    """
    gap_idx = gap_with_context.gap_idx
    topic_name = gap_with_context.topic_name
    topic_id = gap_with_context.topic_id
    
    LOGGER.info(f"Generating context files for gap {gap_idx}: {topic_name}")
    
    # Create contexts directory
    contexts_dir = output_dir / "contexts"
    contexts_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate safe filename base
    safe_topic = create_unicode_safe_filename(topic_name)
    base_name = f"gap_{gap_idx}_{safe_topic}_id={topic_id}"
    
    json_path = contexts_dir / f"gap_{gap_idx}_context.json"
    txt_path = contexts_dir / f"{base_name}_analysis.txt"
    
    # Extract gap data
    gap = gap_with_context.gap
    articles = gap_with_context.articles
    books = gap_with_context.books
    
    gap_dists_idxs = gap.get_dists_idxs()
    gap_size = len(gap_dists_idxs.idxs)
    
    if gap_size == 0:
        raise ValueError(f"Gap {gap_idx} has no articles")
    
    # Get work metadata
    article_works = articles.works()
    book_works = books.works()
    
    # Compute gap statistics (need book gap sizes - simplified for now)
    # TODO: In full implementation, this would come from the broader analysis context
    book_gap_sizes = [len(book_works)]  # Placeholder - in real implementation this would be computed
    gap_size_percentile, gap_zscore, shapiro_log10_pval = compute_gap_statistics(gap_size, book_gap_sizes)
    
    # Get book coverage information
    book_coverage = group_gap_articles_by_covering_books(gap)
    book_to_articles_mapping = get_book_to_articles_mapping(gap)
    
    # 1. Generate JSON context file
    context_data = _build_json_context(
        gap_with_context, gap_size, gap_size_percentile, gap_zscore, shapiro_log10_pval,
        article_works, book_works, book_coverage, gap_dists_idxs
    )
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(context_data, f, indent=2, ensure_ascii=False)
    
    LOGGER.info(f"Gap context JSON written to {json_path}")
    
    # 2. Generate text analysis file (ported from write_single_kball_gap)
    _write_gap_analysis_text(
        txt_path, gap_with_context, gap_size, gap_size_percentile, gap_zscore, shapiro_log10_pval,
        article_works, book_works, book_coverage, book_to_articles_mapping, gap_dists_idxs
    )
    
    LOGGER.info(f"Gap analysis text written to {txt_path}")
    
    return {"json": json_path, "txt": txt_path}


def _build_json_context(
    gap_with_context: GapWithContext,
    gap_size: int,
    gap_size_percentile: float,
    gap_zscore: float,
    shapiro_log10_pval: float,
    article_works: List[Dict[str, Any]],
    book_works: List[Dict[str, Any]],
    book_coverage: Dict[int, List[int]],
    gap_dists_idxs: Any
) -> Dict[str, Any]:
    """Build the structured JSON context data."""
    
    # Extract gap articles with their distances
    gap_articles = []
    for i, article_idx in enumerate(gap_dists_idxs.idxs):
        article_work = article_works[article_idx]
        gap_articles.append({
            "index": int(article_idx),
            "distance_from_center": float(gap_dists_idxs.sqdists[i]),
            "id": article_work.get("id", "N/A"),
            "title": article_work.get("display_name", "N/A"),
            "abstract": article_work.get("abstract", "N/A"),
            "is_center": i == 0  # First article is the center
        })
    
    # Extract covering books with coverage info
    covering_books = []
    for book_idx, covered_article_idxs in book_coverage.items():
        book_work = book_works[book_idx]
        covering_books.append({
            "index": int(book_idx),
            "id": book_work.get("id", "N/A"),
            "title": book_work.get("display_name", "N/A"),
            "abstract": book_work.get("abstract", "N/A"),
            "articles_covered": len(covered_article_idxs),
            "coverage_percentage": len(covered_article_idxs) / gap_size if gap_size > 0 else 0.0,
            "covered_article_indices": [int(idx) for idx in covered_article_idxs]
        })
    
    # Sort books by coverage (most covering first)
    covering_books.sort(key=lambda x: x["articles_covered"], reverse=True)
    
    context_data = {
        "metadata": {
            "topic_name": gap_with_context.topic_name,
            "topic_id": gap_with_context.topic_id,
            "gap_idx": gap_with_context.gap_idx,
            "generation_timestamp": str(np.datetime64('now')),
        },
        "gap_statistics": {
            "gap_size": gap_size,
            "gap_size_percentile": float(gap_size_percentile),
            "gap_zscore": float(gap_zscore) if not np.isnan(gap_zscore) else None,
            "shapiro_log10_pval": float(shapiro_log10_pval) if not np.isnan(shapiro_log10_pval) else None,
            "total_articles": len(article_works),
            "total_books": len(book_works),
            "covering_books_count": len(book_coverage)
        },
        "gap_articles": gap_articles,
        "covering_books": covering_books
    }
    
    return context_data


def _write_gap_analysis_text(
    txt_path: Path,
    gap_with_context: GapWithContext,
    gap_size: int,
    gap_size_percentile: float,
    gap_zscore: float,
    shapiro_log10_pval: float,
    article_works: List[Dict[str, Any]],
    book_works: List[Dict[str, Any]],
    book_coverage: Dict[int, List[int]],
    book_to_articles_mapping: Dict[int, Any],
    gap_dists_idxs: Any
) -> None:
    """Write detailed gap analysis text file.
    
    Ports the text generation logic from write_single_kball_gap() in gap_finder.py.
    """
    topic_name = gap_with_context.topic_name
    topic_id = gap_with_context.topic_id
    gap_idx = gap_with_context.gap_idx
    
    num_articles_total = len(article_works)
    num_books_total = len(book_works)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        # Header section
        f.write(f"Gap Analysis for Topic: {topic_name}\n")
        f.write(f"Topic ID: {topic_id}\n")
        f.write(f"Gap Index: {gap_idx} (0-indexed)\n")
        f.write(f"Number of articles in gap: {gap_size}\n")
        f.write(f"Total books in topic: {num_books_total}\n")
        f.write(f"Total articles in topic: {num_articles_total}\n")
        f.write("\n" + "=" * 80 + "\n")
        
        # Statistical analysis section
        f.write("STATISTICAL ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Gap size percentile rank: {gap_size_percentile:.3f}\n")
        f.write(f"  (Fraction of book gaps smaller than this gap)\n")
        if not np.isnan(gap_zscore):
            f.write(f"Gap z-score (assuming normality): {gap_zscore:.3f}\n")
        else:
            f.write("Gap z-score: N/A (insufficient data or zero variance)\n")
        f.write("=" * 80 + "\n\n")
        
        # Gap articles section - most central articles
        closest_n = min(TOP_ARTICLES_TO_DISPLAY, gap_size)
        f.write(f"{closest_n} most central articles in the gap:\n\n")
        for i, article_idx in enumerate(gap_dists_idxs.idxs[:closest_n]):
            article_work = article_works[article_idx]
            f.write(
                f"Article {i + 1} - {article_work.get('id', 'N/A')} - {article_work.get('display_name', 'N/A')} - "
                f"squared distance: {gap_dists_idxs.sqdists[i]:.4f}\n"
            )
            f.write(f"  Abstract: {article_work.get('abstract', 'N/A')}\n\n")
        
        # Furthest article
        if gap_size > 1:
            f.write("Furthest article from the center of the gap:\n\n")
            furthest_idx = int(np.argmax(gap_dists_idxs.sqdists))
            furthest_article_idx = gap_dists_idxs.idxs[furthest_idx]
            furthest_article_work = article_works[furthest_article_idx]
            f.write(
                f"Furthest article - Id: {furthest_article_work.get('id', 'N/A')} - Title: {furthest_article_work.get('display_name', 'N/A')} - "
                f"Squared distance from center: {gap_dists_idxs.sqdists[furthest_idx]:.4f}\n"
            )
            f.write(f"  Abstract: {furthest_article_work.get('abstract', 'N/A')}\n\n")
        
        # Books section
        if len(book_coverage) == 0:
            f.write("No books found in this topic.\n")
        else:
            sorted_books = sorted(book_coverage.items(), key=lambda x: len(x[1]), reverse=True)
            f.write(f"{len(sorted_books)} books covering articles in the gap, in order of decreasing coverage:\n\n")
            for i, (book_idx, covered_article_idxs) in enumerate(sorted_books):
                book_work = book_works[book_idx]
                num_covered = len(covered_article_idxs)
                f.write(
                    f"Book {i + 1}/{len(sorted_books)} - Id: {book_work.get('id', 'N/A')} - Title: {book_work.get('display_name', 'N/A')}\n"
                )
                f.write(f"  Abstract: {book_work.get('abstract', 'N/A')}\n")
                f.write(f"  Number of articles covered: {num_covered} ({(num_covered / gap_size) if gap_size > 0 else 0:.2%})\n")
                
                # Show closest article in gap to this book
                if book_idx in book_to_articles_mapping:
                    dists_idxs = book_to_articles_mapping[book_idx]
                    if len(dists_idxs.idxs) > 0:
                        closest_article_idx = dists_idxs.idxs[0]
                        closest_distance = float(dists_idxs.sqdists[0])
                        closest_article_work = article_works[closest_article_idx]
                        f.write("  Closest article in gap:\n")
                        f.write(f"    Title: {closest_article_work.get('display_name', 'N/A')}\n")
                        f.write(f"    ID: {closest_article_work.get('id', 'N/A')}\n")
                        f.write(f"    Abstract: {closest_article_work.get('abstract', 'N/A')}\n")
                        f.write(f"    Squared distance: {closest_distance:.4f}\n")
                f.write("\n" + "-" * 60 + "\n\n")


def generate_all_contexts(gaps: List[GapWithContext], output_dir: Path) -> List[Path]:
    """Generate context files for all gaps.
    
    Args:
        gaps: List of gaps with context
        output_dir: Base output directory (Hydra managed)
        
    Returns:
        List of paths to generated context files (both JSON and TXT)
    """
    LOGGER.info(f"Generating context files for {len(gaps)} gaps")
    
    all_paths = []
    successful_contexts = 0
    
    for gap_with_context in gaps:
        try:
            paths = save_gap_context(gap_with_context, output_dir)
            all_paths.extend([paths["json"], paths["txt"]])
            successful_contexts += 1
            
        except Exception as e:
            LOGGER.error(f"Failed to generate context for gap {gap_with_context.gap_idx}: {e}")
            continue
    
    LOGGER.info(f"Generated context files for {successful_contexts}/{len(gaps)} gaps")
    LOGGER.info(f"Total files created: {len(all_paths)}")
    
    return all_paths
