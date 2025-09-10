from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Tuple, Protocol, Union
import abc
import numpy as np
import pandas as pd
from numpy.typing import NDArray
import pyalex
import random
import matplotlib.axes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from nextbook.denoisers import LPPDenoiser
from nextbook.cloud import Cloud, SubCloud, build_cloud, get_subclouds_for_topic
from nextbook.cloud_index import knn
from nextbook.deduplicators import Deduplicator
from nextbook.index import IndexType
from nextbook.util import LOGGER, SubField, create_unicode_safe_filename
from nextbook.gap_directory_manager import GapDirectoryManager
from nextbook.config_manager import AnalysisConfiguration, GapConfiguration, LLMConfiguration
from nextbook.llm.gap_algorithms.kball_algorithm import KballGap, DistsAndIdxs
from nextbook.llm.gap_algorithms.voronoi_algorithm import VoronoiGap
from nextbook.llm.llm_implementation import Implementation, get_generator
from nextbook.types import CellType
from nextbook.llm.protocols import ProposalResult
from nextbook.llm.context_builder import build_proposal_context
from nextbook.plot_utils import (
    PlotManager, FIGURE_SIZE_LARGE, GRID_ALPHA,
    BOOK_COLOR, GAP_COLOR
)
from nextbook.results_manager import GapResults
from nextbook.gap_statistics import compute_gap_statistics, create_book_gap_sizes_qqplot
import abc

# Machine learning and algorithmic constants
NEAREST_NEIGHBOR_K = 5  # Default k for neighbor searches in visualization
RANDOM_STATE = 42       # Random state for reproducibility

# Analysis and reporting constants
TOP_ARTICLES_TO_DISPLAY = 10  # Number of closest articles to show in gap analysis

# Validation and error constants
MIN_REQUIRED_DIMENSION = 1  # Minimum required dimension for distance arrays
CENTER_ARTICLE_DISTANCE = 0  # Expected squared distance for center article (self-distance)
INVALID_LANDMARK_INDEX = -1   # Indicator for no nearest landmark found



class GapProtocol(Protocol):
    """Protocol defining the common interface for gap analysis objects.
    
    This eliminates the need for isinstance checks by defining the methods
    that both KballGap and VoronoiGap must implement.
    """
    
    @property
    def dists_idxs(self) -> DistsAndIdxs:
        """Return the distances and indices of articles in the gap."""
        ...
    
    @property 
    def ambient(self) -> 'SubCloud':
        """Return the ambient space (articles)."""
        ...
        
    @property
    def landmark(self) -> 'SubCloud':  
        """Return the landmark space (books)."""
        ...
    
    def get_covering_book_indices(self) -> list[np.intp]:
        """Get the indices of books that cover articles in the gap."""
        ...
        
    def get_book_to_articles_mapping(self) -> dict[np.intp, DistsAndIdxs]:
        """Get mapping from book indices to their covered articles with distances."""
        ...


@dataclass(frozen=True)
class GapVisualizationData:
    """
    Container for gap visualization data including both flattened neighbor sets
    and detailed FAISS search results for efficient lookups.
    """
    # Flattened neighbor sets for collecting visualization items
    gap_to_article_neighbors: set[int]
    gap_to_book_neighbors: set[int] 
    book_to_article_neighbors: set[int]
    book_to_book_neighbors: set[int]
    
    # Detailed FAISS search results for efficient article->book lookups
    gap_article_to_book_search: Optional[NDArray[np.intp]]  # [gap_article_idx] -> book neighbors
    book_to_article_search: Optional[NDArray[np.intp]]      # [book_idx] -> article neighbors
    
    # Mappings for quick lookups
    gap_article_indices: NDArray[np.intp]
    covering_book_indices: list[np.intp]
    
    def get_nearest_book_for_article(self, gap_article_idx: int) -> Optional[int]:
        """
        Efficiently get the nearest book for a gap article.
        
        Args:
            gap_article_idx: Index of the gap article
            
        Returns:
            Index of the nearest book, or None if not found
        """
        if self.gap_article_to_book_search is None:
            return None
            
        # Find position of this article in gap_article_indices
        gap_positions = np.where(self.gap_article_indices == gap_article_idx)[0]
        if len(gap_positions) == 0:
            return None
            
        gap_position = gap_positions[0]
        if gap_position >= len(self.gap_article_to_book_search):
            return None
            
        # Return the nearest book (first in the search results)
        nearest_books = self.gap_article_to_book_search[gap_position]
        return nearest_books[0] if len(nearest_books) > 0 else None





def get_covering_book_indices(gap: GapProtocol) -> list[np.intp]:
    """Get the indices of books that cover articles in the gap."""
    return gap.get_covering_book_indices()


def get_book_to_articles_mapping(gap: GapProtocol) -> dict[np.intp, DistsAndIdxs]:
    """Get mapping from book indices to their covered articles with distances."""
    return gap.get_book_to_articles_mapping()


def group_gap_articles_by_covering_books(gap: GapProtocol) -> dict[np.intp, list[np.intp]]:
    """Group gap articles by their covering books (simplified version without distances)."""
    book_to_articles = get_book_to_articles_mapping(gap)
    return {book_idx: dists_idxs.idxs.tolist() for book_idx, dists_idxs in book_to_articles.items()}


ARTIFACTS_DIR = GapDirectoryManager.get_base_artifacts_dir()

def _df_from_file(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, dtype={
            'topic_name': 'string',
            'topic_id': 'string', 
            'denoiser': 'string',
            'drop_k': 'int64',
            'gap_idx': 'int64',
            'num_books': 'int64',
            'num_articles': 'int64',
            'gap_size': 'int64',
            'gap_size_percentile': 'float64',
            'gap_zscore': 'float64',
            'shapiro_log10_pval': 'float64'
        })
        LOGGER.info(f"Loaded {len(df)} rows from {csv_path}")
    except FileNotFoundError:
        LOGGER.info(f"File {csv_path} not found, creating empty DataFrame")
        df = pd.DataFrame(columns=['topic_name', 'topic_id', 'denoiser', 'drop_k', 'gap_idx', 'num_books', 'num_articles', 'gap_size', 'gap_size_percentile', 'gap_zscore', 'shapiro_log10_pval'])
        df = df.astype({
            'topic_name': 'string',
            'topic_id': 'string', 
            'denoiser': 'string',
            'drop_k': 'int64',
            'gap_idx': 'int64',
            'num_books': 'int64',
            'num_articles': 'int64',
            'gap_size': 'int64',
            'gap_size_percentile': 'float64',
            'gap_zscore': 'float64',
            'shapiro_log10_pval': 'float64'
        })
    return df


def write_single_kball_gap(
    gap: GapProtocol,
    gap_idx: int,
    topic_name: str,
    topic_id: str,
    analysis_config: AnalysisConfiguration,
    books: SubCloud,
    articles: SubCloud,
    book_gap_sizes: list[int],
) -> None:
    """Write detailed analysis, prompt, and CSV row for a single k-ball gap.

    Requires explicit requested prompt counts in analysis_config; computes actual prompt counts
    and topic totals, computes statistics, writes artifacts, and records a row in results CSV.
    """
    # Compute gap size and statistics
    gap_size = len(gap.dists_idxs.idxs)
    gap_size_percentile, gap_zscore, shapiro_log10_pval = compute_gap_statistics(gap_size, book_gap_sizes)
    LOGGER.info(f"Found gap {gap_idx} for topic {topic_name} with {gap_size} articles "
                f"(percentile: {gap_size_percentile:.3f}, z-score: {gap_zscore:.3f})")

    # Generate LLM proposal and capture actual prompt counts used
    proposal_result = generate_book_proposal(
        gap, gap_idx, topic_name, topic_id, analysis_config, books, articles
    )
    actual_num_prompt_articles = proposal_result.counts.num_articles_for_llm
    actual_num_prompt_books = proposal_result.counts.num_books_for_llm

    # Results CSV
    results = GapResults(analysis_config=analysis_config)
    num_articles_total = len(articles.embeddings())
    num_books_total = len(books.embeddings())
    results.add_row(
        topic_name=topic_name,
        topic_id=topic_id,
        gap_idx=gap_idx,
        gap_size=gap_size,
        gap_size_percentile=gap_size_percentile,
        gap_zscore=gap_zscore,
        shapiro_log10_pval=shapiro_log10_pval,
        num_articles=num_articles_total,
        num_books=num_books_total,
        actual_num_prompt_articles=actual_num_prompt_articles,
        actual_num_prompt_books=actual_num_prompt_books,
    )

    # Detailed gap analysis text file
    analysis_dir = GapDirectoryManager.get_analysis_dir(analysis_config, gap_idx)
    filename = f"{create_unicode_safe_filename(topic_name)}_id={topic_id}_analysis.txt"
    gap_file = analysis_dir / filename

    article_works = articles.works()
    book_works = books.works()
    book_coverage = group_gap_articles_by_covering_books(gap)

    with open(gap_file, 'w', encoding='utf-8') as f:
        f.write(f"Gap Analysis for Topic: {topic_name}\n")
        f.write(f"Topic ID: {topic_id}\n")
        f.write(f"Gap Index: {gap_idx} (0-indexed)\n")
        f.write(f"Denoiser: {analysis_config.gap_config.denoiser}\n")
        f.write(f"Drop K: {analysis_config.gap_config.drop_k}\n")
        f.write(f"Radius Fraction: {analysis_config.gap_config.radius_fraction}\n")
        f.write(f"Number of articles in gap: {gap_size}\n")
        f.write(f"Total books in topic: {num_books_total}\n")
        f.write(f"Total articles in topic: {num_articles_total}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("STATISTICAL ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Gap size percentile rank: {gap_size_percentile:.3f}\n")
        f.write(f"  (Fraction of book gaps smaller than this gap)\n")
        if not np.isnan(gap_zscore):
            f.write(f"Gap z-score (assuming normality): {gap_zscore:.3f}\n")
        else:
            f.write("Gap z-score: N/A (insufficient data or zero variance)\n")
        f.write("=" * 80 + "\n\n")

        # Center article
        assert np.argmin(gap.dists_idxs.sqdists) == 0, (
            f"Expected the closest article to be the first one in the gap, got {np.argmin(gap.dists_idxs.sqdists)}"
        )
        # center_article_idx = gap.dists_idxs.idxs[0]  # Center is the first article
        # if len(book_works) > 0:
        #     # Find the book that covers the center article
        #     closest_book_idx = None
        #     closest_book_distance = None
            
        #     # # Find which book covers the center article
        #     # for book_idx, articles_in_book in book_coverage.items():
        #     #     if center_article_idx in articles_in_book:
        #     #         closest_book_idx = book_idx
        #     #         # For VoronoiGap, we don't have distances readily available, so use placeholder
        #     #         closest_book_distance = 0.0
        #     #         break
            
        #     # if closest_book_idx is not None:
        #     #     closest_book_work = book_works[closest_book_idx]
        #     #     f.write(f"Closest book to gap center (distance: {closest_book_distance:.4f}):\n")
        #     #     f.write(f"{closest_book_work.get('id', 'N/A')}: {closest_book_work.get('display_name', 'N/A')}\n")
        #     #     f.write(f"  {closest_book_work.get('abstract', 'N/A')}\n")
        #     #     f.write("\n" + "=" * 80 + "\n\n")

        # List the top articles closest to the center of the gap
        closest_n = min(TOP_ARTICLES_TO_DISPLAY, gap_size)
        f.write(f"{closest_n} most central articles in the gap:\n\n")
        for i, article_idx in enumerate(gap.dists_idxs.idxs[:closest_n]):
            article_work = article_works[article_idx]
            f.write(
                f"Article {i + 1} - {article_work.get('id', 'N/A')} - {article_work.get('display_name', 'N/A')} - "
                f"squared distance: {gap.dists_idxs.sqdists[i]:.4f}\n"
            )
            f.write(f"  Abstract: {article_work.get('abstract', 'N/A')}\n\n")

        f.write("Furthest article from the center of the gap:\n\n")
        furthest_idx = int(np.argmax(gap.dists_idxs.sqdists))
        furthest_article_idx = gap.dists_idxs.idxs[furthest_idx]
        furthest_article_work = article_works[furthest_article_idx]
        f.write(
            f"Furthest article - Id: {furthest_article_work.get('id', 'N/A')} - Title: {furthest_article_work.get('display_name', 'N/A')} - "
            f"Squared distance from center: {gap.dists_idxs.sqdists[furthest_idx]:.4f}\n"
        )
        f.write(f"  Abstract: {furthest_article_work.get('abstract', 'N/A')}\n\n")

        if len(book_coverage) == 0:
            f.write("No books found in this topic.\n")
        else:
            sorted_books = sorted(book_coverage.items(), key=lambda x: len(x[1]), reverse=True)
            book_to_articles_mapping = get_book_to_articles_mapping(gap)
            f.write(f"{len(sorted_books)} books covering articles in the gap, in order of decreasing coverage:\n\n")
            for i, (book_idx, covered_article_idxs) in enumerate(sorted_books):
                book_work = book_works[book_idx]
                num_covered = len(covered_article_idxs)
                f.write(
                    f"Book {i + 1}/{len(sorted_books)} - Id: {book_work.get('id', 'N/A')} - Title: {book_work.get('display_name', 'N/A')}\n"
                )
                f.write(f"  Abstract: {book_work.get('abstract', 'N/A')}\n")
                f.write(f"  Number of articles covered: {num_covered} ({(num_covered / gap_size) if gap_size > 0 else 0:.2%})\n")
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

    LOGGER.info(f"Detailed gap analysis written to {gap_file}")


def write_llm_prompt(gap: GapProtocol, gap_idx: int, topic_name: str, topic_id: str, 
                     analysis_config: AnalysisConfiguration, 
                     books: SubCloud, articles: SubCloud) -> Tuple[int, int]:
    """Write an LLM prompt for generating a book proposal based on the gap analysis.
    
    Returns:
        Tuple[int, int]: (num_articles_for_llm, num_books_for_llm)
    """
    gap_size = len(gap.dists_idxs.idxs)
    
    # Get work metadata
    article_works = articles.works()
    book_works = books.works()
    
    # Sample random articles from the gap (up to configured limit)
    sample_size = min(analysis_config.llm_config.requested_num_prompt_articles, gap_size)
    sampled_indices = random.sample(range(gap_size), sample_size)
    sampled_articles = [article_works[gap.dists_idxs.idxs[i]] for i in sampled_indices]
    
    # Analyze book coverage using the configured minimum coverage threshold
    covering_books = []
    # Get book coverage using helper function
    book_coverage = group_gap_articles_by_covering_books(gap)
    
    # # Find books that cover at least the minimum required coverage of the gap articles
    sorted_books = sorted(book_coverage.items(), key=lambda x: len(x[1]), reverse=True)
    covered_articles = set()
    # target_coverage = gap_size * analysis_config.prompt_book_min_coverage
    
    for book_idx, covered_article_idxs in sorted_books[:analysis_config.llm_config.requested_num_prompt_books]:
        covering_books.append(book_works[book_idx])
        covered_articles.update(covered_article_idxs)
        # if len(covered_articles) >= target_coverage:
        #     break
    
    prompts_llm_dir = GapDirectoryManager.get_prompts_llm_dir(analysis_config, gap_idx)
    
    prompt_filename = f"{create_unicode_safe_filename(topic_name)}_id={topic_id}_prompt.txt"
    prompt_file = prompts_llm_dir / prompt_filename
    
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write("You are a publishing expert and academic researcher. Based on the research articles ")
        if covering_books:
            f.write(f"and related books ")
        f.write("provided below, please propose a title, abstract, and table of contents for a textbook that would ")
        f.write("prepare students to understand research in this area.\n\n")
        
        f.write("## Sample Articles\n\n")
        
        for i, article in enumerate(sampled_articles, 1):
            f.write(f"**Article {i}**: {article.get('display_name', 'N/A')}\n")
            f.write(f"\t*Abstract*: {article.get('abstract', 'N/A')}\n\n")
        
        if covering_books:
            f.write("## Related Existing Books\n\n")
            for i, book in enumerate(covering_books, 1):
                f.write(f"**Book {i}**: {book.get('display_name', 'N/A')}\n")
                f.write(f"\t*Abstract*: {book.get('abstract', 'N/A')}\n\n")
        
        f.write("## Required Output Format\n\n")
        f.write("Format your response as valid JSON with EXACTLY the following structure. ")
        f.write("Do not add any text before or after this JSON:\n\n")
        f.write("{\n")
        f.write('  "title": "<Proposed Title>",\n')
        f.write('  "abstract": "<Abstract>",\n')
        f.write('  "chapters": [\n')
        f.write('    {\n')
        f.write('      "chapter_title": "<Chapter Name>",\n')
        f.write('      "chapter_description": "<Description of concepts covered (1-2 sentences)>"\n')
        f.write('    },\n')
        f.write('    {\n')
        f.write('      "chapter_title": "<Chapter Name>",\n')
        f.write('      "chapter_description": "<Description of concepts covered (1-2 sentences)>"\n')
        f.write('    }\n')
        f.write('    // ... continue for 10-15 chapters\n')
        f.write('  ]\n')
        f.write('}\n\n')
        f.write("## Guidelines\n\n")
        f.write("- Use the EXACT JSON structure shown above.\n")
        f.write("- Replace ALL angle-bracketed placeholders (e.g., <Proposed Title>) with your content.\n")
        f.write("- Include 10-15 chapters in the chapters array.\n")
        f.write("- Build up systematically from basic to advanced concepts.\n")
    LOGGER.info(f"LLM prompt written to {prompt_file}")
    
    # Return the counts for CSV tracking
    num_articles_for_llm = len(sampled_articles)
    num_books_for_llm = len(covering_books)
    return num_articles_for_llm, num_books_for_llm


def generate_book_proposal(
    gap: GapProtocol, 
    gap_idx: int, 
    topic_name: str, 
    topic_id: str, 
    analysis_config: AnalysisConfiguration, 
    books: SubCloud, 
    articles: SubCloud
) -> ProposalResult:
    """Generate a textbook proposal using the configured LLM generator.
    
    This function replaces write_llm_prompt with a clean architecture that uses
    the configured generator implementation (structured or prompt-based).
    
    Args:
        gap: The gap analysis result
        gap_idx: Index of this gap (0-based)
        topic_name: Human-readable topic name
        topic_id: Unique topic identifier
        analysis_config: Analysis configuration including LLM settings
        books: SubCloud of books for this topic
        articles: SubCloud of articles for this topic
        
    Returns:
        ProposalResult with generated proposal and metadata
        
    Raises:
        ProposalGenerationError: If generation fails for any reason
    """
    # Build the proposal context using deterministic sampling
    context = build_proposal_context(
        gap=gap,
        gap_idx=gap_idx,
        topic_name=topic_name,
        topic_id=topic_id,
        analysis_config=analysis_config,
        books=books,
        articles=articles
    )
    
    # Get the appropriate generator for the configured implementation
    generator = get_generator(analysis_config.llm_config.implementation, analysis_config)
    
    # Generate the proposal
    result = generator.generate(context)
    
    return result


def _perform_gap_neighbor_searches(gap: GapProtocol, books: SubCloud, articles: SubCloud, k: int = 5) -> GapVisualizationData:
    """Perform the four FAISS neighbor searches for gap visualization."""
    gap_article_indices = gap.dists_idxs.idxs
    covering_book_indices = get_covering_book_indices(gap)
    
    # 1. Nearest articles to in-gap articles
    gap_to_article_neighbors = set()
    # if len(gap_article_indices) > 0:
    #     res = knn(articles, articles, k=k+1) # +1 to exclude self-neighbor
    #     gap_to_article_neighbors.update(res.I.flatten())
    
    # 2. Nearest books to in-gap articles - KEEP DETAILED RESULTS
    gap_to_book_neighbors = set()
    gap_article_to_book_search = None
    if len(gap_article_indices) > 0 and len(books.embeddings()) > 0:
        res = knn(books, articles, k=k)
        # Only get neighbors for gap articles
        gap_book_neighbors = res.I[gap_article_indices]
        gap_to_book_neighbors.update(gap_book_neighbors.flatten())
        # Keep detailed search results for efficient lookups
        gap_article_to_book_search = gap_book_neighbors
    
    # 3. Nearest articles to covering books - KEEP DETAILED RESULTS
    book_to_article_neighbors = set()
    book_to_article_search = None
    # if len(covering_book_indices) > 0:
    #     res = knn(articles, books, k=k)
    #     # Only get neighbors for covering books
    #     book_article_neighbors = res.I[covering_book_indices]
    #     book_to_article_neighbors.update(book_article_neighbors.flatten())
    #     # Keep detailed search results for efficient lookups
    #     book_to_article_search = book_article_neighbors
    
    # 4. Nearest books to covering books
    book_to_book_neighbors = set()
    if len(covering_book_indices) > 0:
        res = knn(books, books, k=k+1) # +1 to exclude self-neighbor
        # Only get neighbors for covering books
        book_book_neighbors = res.I[covering_book_indices]
        book_to_book_neighbors.update(book_book_neighbors.flatten())
    
    return GapVisualizationData(
        gap_to_article_neighbors=gap_to_article_neighbors,
        gap_to_book_neighbors=gap_to_book_neighbors,
        book_to_article_neighbors=book_to_article_neighbors,
        book_to_book_neighbors=book_to_book_neighbors,
        gap_article_to_book_search=gap_article_to_book_search,
        book_to_article_search=book_to_article_search,
        gap_article_indices=gap_article_indices,
        covering_book_indices=covering_book_indices
    )


def _collect_visualization_data(gap: GapProtocol, gap_idx: int, books: SubCloud, articles: SubCloud, proposal_embeddings: Optional[list] = None):
    """
    Helper function to collect all embeddings and metadata for gap visualization.

    Returns:
        tuple: (base_embeddings, item_types, all_items, gap_article_positions_map, all_book_positions_map, neighbor_data, proposal_embeddings_array)
    """
    gap_size = len(gap.dists_idxs.idxs)
    
    if gap_size == 0:
        return None, None, None, None, None, None, None, None
    
    # Get all embeddings and work data
    all_article_embeddings = articles.embeddings()
    all_book_embeddings = books.embeddings()
    
    # Get gap articles
    gap_article_indices = gap.dists_idxs.idxs
    
    # Get covered points analysis to find books covering the gap
    covering_book_indices = get_covering_book_indices(gap)
    
    # Perform the four neighbor searches
    k = NEAREST_NEIGHBOR_K
    neighbor_data = _perform_gap_neighbor_searches(gap, books, articles, k)
    
    # Collect all unique items - simplified to just distinguish articles vs books
    item_registry = {}  # (work_type, idx) -> item_type
    
    # Add all articles (gap articles + neighbor articles)
    all_article_indices = [int(idx) for idx in gap_article_indices]
    all_article_indices.extend([int(idx) for idx in neighbor_data.gap_to_article_neighbors.union(neighbor_data.book_to_article_neighbors)])
    for idx in set(all_article_indices):  # Remove duplicates
        item_registry[('article', idx)] = 'article'
    
    # Add all books (covering books + neighbor books)
    all_book_indices = [int(idx) for idx in covering_book_indices]
    all_book_indices.extend([int(idx) for idx in neighbor_data.gap_to_book_neighbors.union(neighbor_data.book_to_book_neighbors)])
    for idx in set(all_book_indices):  # Remove duplicates
        item_registry[('book', idx)] = 'book'
    
    # Build final lists with unique items
    all_items = []
    all_embeddings_list = []
    item_types = []
    
    for (work_type, idx), item_type in item_registry.items():
        all_items.append((work_type, idx))
        if work_type == 'article':
            all_embeddings_list.append(all_article_embeddings[idx])
        else:  # work_type == 'book'
            all_embeddings_list.append(all_book_embeddings[idx])
        item_types.append(item_type)
    
    if len(all_embeddings_list) < 3:
        return None, None, None, None, None, None, None, None
    
    # Base embeddings (do not include proposals here)
    base_embeddings = np.vstack(all_embeddings_list)
    
    # Prepare proposal embeddings array (kept separate)
    proposal_embeddings_arr = None
    if proposal_embeddings:
        prop_list = [np.asarray(pe) for pe in proposal_embeddings]
        if len(prop_list) > 0:
            proposal_embeddings_arr = np.vstack(prop_list)
    
    # Create position mappings for connection lines  
    gap_article_positions_map = {}
    all_book_positions_map = {}  # Now includes all books, not just covering books
    
    for i, (item_type, (work_type, work_idx)) in enumerate(zip(item_types, all_items)):
        if work_type == 'article' and work_idx in gap_article_indices:
            gap_article_positions_map[work_idx] = i
        elif work_type == 'book':
            all_book_positions_map[work_idx] = i
    
    return (base_embeddings, item_types, all_items, gap_article_positions_map, 
            all_book_positions_map, neighbor_data, proposal_embeddings_arr)


def _create_gap_plot(
    embeddings_2d: NDArray,
    item_types: list[str],
    all_items: list[tuple[str, int]],
    gap_article_positions_map: dict,
    all_book_positions_map: dict,
    covered_points_analysis,
    neighbor_data,
    gap: GapProtocol,
    topic_name: str,
    gap_idx: int,
    method_name: str,
    pc_variances: tuple[float, float] | None,
    ax: matplotlib.axes.Axes,
    proposals_2d: Optional[NDArray] = None,
) -> None:
    """Minimal gap plotting helper used by plotters and overlays.

    Plots base articles and books, highlights articles in the gap, and labels the plot.
    This intentionally keeps the visualization simple.
    """
    n = len(embeddings_2d)
    if n == 0:
        return

    # Determine masks for books and articles
    is_book = [work_type == 'book' for (work_type, _idx) in all_items[:n]]
    is_article = [work_type == 'article' for (work_type, _idx) in all_items[:n]]

    emb = embeddings_2d[:n]

    # Compute positions of gap articles within the local plotting array
    gap_positions_all = list(gap_article_positions_map.values())
    gap_positions = [p for p in gap_positions_all if p < n]
    gap_pos_set = set(gap_positions)

    # Plot non-gap articles only (so 'Articles' isn't shown if we only drew gap articles)
    article_idx = [i for i, v in enumerate(is_article) if v]
    non_gap_article_idx = [i for i in article_idx if i not in gap_pos_set]
    if len(non_gap_article_idx) > 0:
        ax.scatter(emb[non_gap_article_idx, 0], emb[non_gap_article_idx, 1], c=GAP_COLOR, s=30, alpha=0.6, label='Articles', zorder=1)

    # Highlight gap articles (center and others)
    if len(gap_positions) > 0:
        ax.scatter(emb[gap_positions, 0], emb[gap_positions, 1], c=GAP_COLOR, s=90, alpha=0.8, marker='o', label='Gap Articles', zorder=2)
        # Mark center article specially (first in gap.dists_idxs.idxs)
        try:
            center_global_idx = gap.dists_idxs.idxs[0]
            center_pos = gap_article_positions_map.get(int(center_global_idx), None)
            if center_pos is not None and center_pos < n:
                ax.scatter(emb[center_pos, 0], emb[center_pos, 1], marker='*', c=GAP_COLOR, s=120, edgecolors='k', linewidths=1.2, label='Gap Center', zorder=4)
        except Exception:
            pass

    # Plot books
    book_idx = [i for i, v in enumerate(is_book) if v]
    if len(book_idx) > 0:
        ax.scatter(emb[book_idx, 0], emb[book_idx, 1], c=BOOK_COLOR, s=80, marker='s', alpha=0.8, label='Books', zorder=3)

    # Plot proposed book(s) as large blue circles with black border
    if proposals_2d is not None and len(proposals_2d) > 0:
        ax.scatter(proposals_2d[:, 0], proposals_2d[:, 1],
                   c=BOOK_COLOR, marker='*', s=120, edgecolors='black', linewidths=1.2,
                   label='Proposed Book', zorder=4)

    # Title without variance
    ax.set_title(f"Gap {gap_idx} - {topic_name} ({method_name})")
    # Axis labels: include per-PC variance if PCA
    if method_name.upper() == 'PCA' and pc_variances is not None:
        ax.set_xlabel(f"PC1 ({pc_variances[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pc_variances[1]:.1%})")
    else:
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend()

def prepare_gap_visualization_data(gap: GapProtocol, gap_idx: int, num_books: int, books: 'SubCloud', articles: 'SubCloud', proposal_embeddings: Optional[list] = None):
    """
    Public wrapper to collect embeddings and metadata for gap visualization.
    Returns the same tuple as the internal collector, and includes proposal_embeddings_arr as final element.
    """
    return _collect_visualization_data(gap, gap_idx, books, articles, proposal_embeddings=proposal_embeddings)


class GapPlotter(abc.ABC):
    def __init__(self, gap: GapProtocol, gap_idx: int, topic_name: str, topic_id: str, analysis_config: AnalysisConfiguration, books: 'SubCloud', articles: 'SubCloud') -> None:
        self.gap = gap
        self.gap_idx = gap_idx
        self.topic_name = topic_name
        self.topic_id = topic_id
        self.analysis_config = analysis_config
        self.books = books
        self.articles = articles
        
        # Properties for existing code
        self.denoiser = analysis_config.gap_config.denoiser
        self.drop_k = analysis_config.gap_config.drop_k
        self.radius_fraction = analysis_config.gap_config.radius_fraction
        self.num_articles = analysis_config.llm_config.requested_num_prompt_articles
        self.num_books = analysis_config.llm_config.requested_num_prompt_books
    @abc.abstractmethod
    def get_embeddings_2d(self,
                         all_embeddings: NDArray,
                         item_types: list[str],
                         all_items: list,
                         gap_article_positions_map: dict,
                         all_book_positions_map: dict,
                         neighbor_data
    ) -> tuple[NDArray, tuple[float, float] | None]:
        """Return 2D embeddings and method-specific info (e.g., variance_info)."""
        pass

    @abc.abstractmethod
    def method_name(self) -> str:
        pass

    def plot_and_save(self) -> None:
        # Collect data (no proposals for standard plotter)
        data = _collect_visualization_data(self.gap, self.gap_idx, self.books, self.articles, proposal_embeddings=None)
        if data[0] is None:
            LOGGER.warning(f"Skipping {self.method_name()} visualization - insufficient data")
            return
        base_embeddings, item_types, all_items, gap_article_positions_map, all_book_positions_map, neighbor_data, _proposal_embeddings = data
        embeddings_2d, pc_variances = self.get_embeddings_2d(base_embeddings, item_types, all_items, gap_article_positions_map, all_book_positions_map, neighbor_data)
        fig, ax = PlotManager.create_figure(figsize=FIGURE_SIZE_LARGE)
        _create_gap_plot(embeddings_2d, item_types, all_items, gap_article_positions_map,
                        all_book_positions_map, None, neighbor_data,
                        self.gap, self.topic_name, self.gap_idx, self.method_name(), pc_variances, ax=ax)
        
        # Use the analysis configuration for centralized directory management
        plots_dir = GapDirectoryManager.get_plots_dir(self.analysis_config, self.gap_idx)
        plot_file = plots_dir / f"{create_unicode_safe_filename(self.topic_name)}_gap_{self.gap_idx}_{self.method_name().lower()}_visualization.png"
        PlotManager.save_and_close(fig, plot_file)
        LOGGER.info(f"{self.method_name()} gap visualization saved to {plot_file}")


# Shared projection helper used by plotters and overlays

def _compute_base_projection(
    method: str,
    base_embeddings: NDArray,
    item_types: list[str],
    all_items: list[tuple[str, int]],
    book_to_articles_mapping,
    neighbor_data,
    gap: GapProtocol,
    books: 'SubCloud',
    articles: 'SubCloud',
    proposal_embeddings: Optional[NDArray] = None
) -> Tuple[NDArray, tuple[float, float] | None, Callable[[NDArray], NDArray]]:
    """Compute 2D projection. Fit PCA on base + proposals when proposals are provided.

    This function intentionally supports only PCA to keep the projection simple.
    """
    # Stack embeddings for fitting if proposals provided; keep base_embeddings for indexing
    if proposal_embeddings is not None and len(proposal_embeddings) > 0:
        all_embeddings = np.vstack([base_embeddings, proposal_embeddings])
    else:
        all_embeddings = base_embeddings

    method_upper = method.upper()
    if method_upper == 'PCA':
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        embeddings_2d = pca.fit_transform(all_embeddings)
        # Return per-axis variances (PC1, PC2) for labeling axes
        evr = pca.explained_variance_ratio_
        pc_variances = (float(evr[0]), float(evr[1])) if evr.size >= 2 else (float(evr[0]), 0.0)
        return embeddings_2d, pc_variances, pca.transform

    if method_upper == 'LPP':
        raise NotImplementedError("LPP projection is not yet implemented in this context.")
        gap_article_indices = gap.dists_idxs.idxs
        covering_book_indices = list(book_to_articles_mapping.keys())
        k = 5
        item_neighbors = defaultdict(set)
        global_to_local = { (work_type, global_idx): local_idx for local_idx, (work_type, global_idx) in enumerate(all_items) }
        # 1. Article-article neighbors
        if len(gap_article_indices) > 0:
            res = knn(articles, articles, k=k+1)
            for gap_article_idx in gap_article_indices:
                if ('article', gap_article_idx) in global_to_local:
                    neighbors = res.I[gap_article_idx]
                    for neighbor_idx in neighbors:
                        if ('article', neighbor_idx) in global_to_local and neighbor_idx != gap_article_idx:
                            item_neighbors[('article', gap_article_idx)].add(global_to_local[('article', neighbor_idx)])
        # 2. Article-book neighbors
        if len(gap_article_indices) > 0 and len(books.embeddings()) > 0:
            res = knn(books, articles, k=k)
            for gap_article_idx in gap_article_indices:
                if ('article', gap_article_idx) in global_to_local:
                    neighbors = res.I[gap_article_idx]
                    for neighbor_idx in neighbors:
                        if ('book', neighbor_idx) in global_to_local:
                            item_neighbors[('article', gap_article_idx)].add(global_to_local[('book', neighbor_idx)])
        # 3. Book-article neighbors
        if len(covering_book_indices) > 0:
            res = knn(articles, books, k=k)
            for covering_book_idx in covering_book_indices:
                if ('book', covering_book_idx) in global_to_local:
                    neighbors = res.I[covering_book_idx]
                    for neighbor_idx in neighbors:
                        if ('article', neighbor_idx) in global_to_local:
                            item_neighbors[('book', covering_book_idx)].add(global_to_local[('article', neighbor_idx)])
        # 4. Book-book neighbors
        if len(covering_book_indices) > 0:
            res = knn(books, books, k=k+1)
            for covering_book_idx in covering_book_indices:
                if ('book', covering_book_idx) in global_to_local:
                    neighbors = res.I[covering_book_idx]
                    for neighbor_idx in neighbors:
                        if ('book', neighbor_idx) in global_to_local and neighbor_idx != covering_book_idx:
                            item_neighbors[('book', covering_book_idx)].add(global_to_local[('book', neighbor_idx)])
        # Build neighbor indices
        neighbor_indices_list = []
        for work_type, global_idx in all_items:
            neighbors = list(item_neighbors[(work_type, global_idx)])
            neighbors = neighbors[:k]
            local_idx = global_to_local[(work_type, global_idx)]
            while len(neighbors) < k:
                neighbors.append(neighbors[0] if neighbors else local_idx)
            neighbor_indices_list.append(neighbors)
        neighbor_indices = np.array(neighbor_indices_list)
        # LPP projector
        context = _GapVisualizationContext(all_embeddings, neighbor_indices, item_types, k)
        lpp_denoiser = LPPDenoiser()
        projector, _eigs = lpp_denoiser.projector(context, n_components=2)
        embeddings_2d = all_embeddings @ projector
        transform_fn = lambda X: X @ projector
        return embeddings_2d, "", transform_fn

    raise NotImplementedError(f"Projection method not supported: {method}")



class PCAGapPlotter(GapPlotter):
    def method_name(self) -> str:
        return "PCA"
    def get_embeddings_2d(self,
                         all_embeddings: NDArray,
                         item_types: list[str],
                         all_items: list,
                         gap_article_positions_map: dict,
                         all_book_positions_map: dict,
                         neighbor_data
    ) -> tuple[NDArray, tuple[float, float] | None]:
        # Get book coverage data for _compute_base_projection 
        book_to_articles_mapping = get_book_to_articles_mapping(self.gap)
        embeddings_2d, pc_variances, _ = _compute_base_projection(
            'PCA', all_embeddings, item_types, all_items,
            book_to_articles_mapping, neighbor_data, self.gap, self.books, self.articles
        )
        return embeddings_2d, pc_variances

class TSNEGapPlotter(GapPlotter):
    def method_name(self) -> str:
        return "t-SNE"
    def get_embeddings_2d(self,
                         all_embeddings: NDArray,
                         item_types: list[str],
                         all_items: list,
                         gap_article_positions_map: dict,
                         all_book_positions_map: dict,
                         neighbor_data
    ) -> tuple[NDArray, tuple[float, float] | None]:
        n_samples = len(all_embeddings)
        perplexity = min(30, max(5, n_samples // 4))
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(all_embeddings)
        return embeddings_2d, None

class LPPGapPlotter(GapPlotter):
    def method_name(self) -> str:
        return "LPP"
    def get_embeddings_2d(self,
                         all_embeddings: NDArray,
                         item_types: list[str],
                         all_items: list,
                         gap_article_positions_map: dict,
                         all_book_positions_map: dict,
                         neighbor_data
    ) -> tuple[NDArray, tuple[float, float] | None]:
        book_to_articles_mapping = get_book_to_articles_mapping(self.gap)
        embeddings_2d, _pc_variances, _ = _compute_base_projection(
            'LPP', all_embeddings, item_types, all_items,
            book_to_articles_mapping, neighbor_data, self.gap, self.books, self.articles
        )
        return embeddings_2d, None


def find_gaps_for_topic(
    books: SubCloud,
    articles: SubCloud,
    radius_fraction: float,
    k: int = 1,
    is_intra_cloud: bool = False,
    cell_type: CellType = CellType.KBALL # TODO this is leading to incorrect usage in batch_operations.py
):
    """Return gaps for the given books/articles subclouds and parameters."""
    return cell_type.value.find_gaps(
        ambient=articles,
        landmark=books,
        query=articles,
        radius_fraction=radius_fraction,
        k=k,
        is_intra_cloud=is_intra_cloud
    )

def write_gaps(
    cloud: Cloud,
    topic_name: str,
    topic_id: str,
    radius_fraction: float,
    num_gaps: int,
    requested_num_prompt_articles: int,
    requested_num_prompt_books: int,
    cell_type: str,
    generate_plots: bool = True,
):
    """Analyze the top gaps for a topic, write artifacts, and record results.

    Requires explicit requested_num_prompt_* values. No defaults.
    """
    # Convert string cell_type to enum
    if cell_type == 'kball':
        cell_type_enum = CellType.KBALL
    elif cell_type == 'voronoi':
        cell_type_enum = CellType.VORONOI
    else:
        raise ValueError(f"Unsupported cell_type: {cell_type}")
        
    # Build subclouds and topic stats
    books, articles = get_subclouds_for_topic(cloud, topic_id)
    num_books = len(books.embeddings())
    num_articles = len(articles.embeddings())
    LOGGER.info(f"Topic {topic_name}: {num_books} books, {num_articles} articles")

    # Centralized analysis configuration
    gap_config = GapConfiguration(
        denoiser=cloud.denoiser,
        drop_k=cloud.drop_k,
        radius_fraction=radius_fraction,
        cell_type=cell_type_enum,
    )
    
    llm_config = LLMConfiguration(
        implementation=Implementation.PROMPT,  # Default to prompt implementation for now
        requested_num_prompt_articles=requested_num_prompt_articles,
        requested_num_prompt_books=requested_num_prompt_books,
    )
    
    analysis_config = AnalysisConfiguration.from_components(gap_config, llm_config)

    if num_articles == 0:
        LOGGER.info(f"No articles found for topic {topic_id}, recording empty gaps")
        results = GapResults(analysis_config=analysis_config)
        for gap_idx in range(num_gaps):
            results.add_row(
                topic_name=topic_name,
                topic_id=topic_id,
                gap_idx=gap_idx,
                gap_size=0,
                gap_size_percentile=float("nan"),
                gap_zscore=float("nan"),
                shapiro_log10_pval=float("nan"),
                num_articles=num_articles,
                num_books=num_books,
                actual_num_prompt_articles=0,
                actual_num_prompt_books=0,
            )
        return

    # Book gap sizes for statistics
    book_gap_sizes = [
        len(g.dists_idxs.idxs)
        for g in cell_type_enum.value.find_gaps(
            ambient=articles,
            landmark=books,
            query=books,
            radius_fraction=radius_fraction,
            k=1,
            is_intra_cloud=True,
        )
    ]
    # TODO: cell_type='voronoi' would use something like this (maybe this is already working - need to check):
    # book_gap_sizes = actual_voronoi_coverage(ambient=articles.embeddings(), landmark=books.index(), ambient_is_landmark=False)

    # Q-Q plots
    create_book_gap_sizes_qqplot(
        book_gap_sizes, topic_name, topic_id, analysis_config, num_books=num_books, num_articles=num_articles
    )

    # Top gaps over articles
    gap_generator = cell_type_enum.value.find_gaps(
        ambient=articles, landmark=books, query=articles, radius_fraction=radius_fraction, k=1, is_intra_cloud=False
    )
    # TODO: cell_type='voronoi' would use something like this (maybe this is already working - need to check):
    # gap_generator = voronoi_gaps(ambient=articles, landmark=books, query=articles)

    for gap_idx in range(num_gaps):
        try:
            gap = next(gap_generator)
        except StopIteration:
            LOGGER.warning(
                f"No gap {gap_idx} found for topic {topic_name} ({topic_id}) - generator exhausted"
            )
            break

        # Write artifacts and row
        write_single_kball_gap(
            gap, gap_idx, topic_name, topic_id, analysis_config, books, articles, book_gap_sizes
        )

        # Plot the gap visualization (if enabled)
        if generate_plots:
            PCAGapPlotter(gap, gap_idx, topic_name, topic_id, analysis_config, books, articles).plot_and_save()
        else:
            LOGGER.info(f"Skipping plot generation for gap {gap_idx} of topic {topic_name}")




def save_gap_plot_with_proposals(
    gap: KballGap | VoronoiGap,
    gap_idx: int,
    topic_name: str,
    topic_id: str,
    analysis_config: AnalysisConfiguration,
    books: 'SubCloud',
    articles: 'SubCloud',
    method: str,
    proposal_embeddings: list,
    proposal_labels: Optional[list[str]] = None,
) -> Path:
    # Collect base visualization data (append proposals into the data collection)
    num_books = len(books.embeddings())
    data = prepare_gap_visualization_data(gap, gap_idx, num_books, books, articles, proposal_embeddings=proposal_embeddings)
    if not data or data[0] is None:
        LOGGER.warning(f"Skipping {method} overlay - insufficient data")
        raise ValueError("Insufficient data for plotting")
    (
        base_embeddings,
        item_types,
        all_items,
        gap_article_positions_map,
        all_book_positions_map,
        neighbor_data,
        proposal_embeddings_arr,
    ) = data

    # Compute projection using base + proposals (PCA fits on both if proposals provided)
    book_to_articles_mapping = get_book_to_articles_mapping(gap)
    embeddings_2d_all, pc_variances, transform_fn = _compute_base_projection(
        method, base_embeddings, item_types, all_items,
        book_to_articles_mapping, neighbor_data, gap, books, articles,
        proposal_embeddings=proposal_embeddings_arr
    )

    # Split projections into base items and proposals using the length of base_embeddings
    base_n = len(base_embeddings)
    if embeddings_2d_all.shape[0] < base_n:
        # Fallback: nothing to plot
        embeddings_2d = embeddings_2d_all
        proposals_2d = None
    else:
        embeddings_2d = embeddings_2d_all[:base_n]
        proposals_2d = embeddings_2d_all[base_n:] if embeddings_2d_all.shape[0] > base_n else None

    # Create plot with base items and proposals
    fig, ax = PlotManager.create_figure(figsize=FIGURE_SIZE_LARGE)
    _create_gap_plot(
        embeddings_2d,
        item_types[:len(embeddings_2d)],
        all_items[:len(embeddings_2d)],
        gap_article_positions_map,
        all_book_positions_map,
        None,  # No longer need covered_points_analysis
        neighbor_data,
        gap,
        topic_name,
        gap_idx,
        method.upper(),
        pc_variances,
        ax=ax,
        proposals_2d=proposals_2d,
    )

    # Save alongside existing plots
    # Use LLM-specific plots directory since this plot includes proposals
    plots_dir = GapDirectoryManager.get_plots_llm_dir(analysis_config, gap_idx)
    plot_file = plots_dir / f"{create_unicode_safe_filename(topic_name)}_gap_{gap_idx}_{method.lower()}_visualization_with_proposals.png"
    PlotManager.save_and_close(fig, plot_file)
    LOGGER.info(f"{method.upper()} gap visualization with proposals saved to {plot_file}")
    return plot_file


def save_gap_books_vs_proposals_plot(
    gap: KballGap | VoronoiGap,
    gap_idx: int,
    topic_name: str,
    topic_id: str,
    analysis_config: AnalysisConfiguration,
    books: 'SubCloud',
    articles: 'SubCloud',
    method: str,
    proposal_embeddings: list,
    proposal_labels: Optional[list[str]] = None,
) -> Path:
    """Save a plot comparing gap articles' proximity to neighboring books vs proposals.

    Requirements:
    - Proposals must be provided (non-empty).
    - Plots gap articles, neighboring books (that cover gap articles), and proposals.
    - Articles colored by nearest neighbor class in the ORIGINAL embedding space:
        * If closer to a neighboring book than to the nearest proposal -> color BOOK_COLOR.
        * Otherwise -> color GAP_COLOR.
    - Books are squares (marker='s') colored BOOK_COLOR.
    - Proposals are squares (marker='s') colored GAP_COLOR with black border.
    - Articles are circles (marker='o').
    """
    if proposal_embeddings is None or len(proposal_embeddings) == 0:
        raise ValueError("proposal_embeddings must be provided and non-empty")

    # Collect base data: gap articles and covering (neighboring) books
    all_article_embeddings = articles.embeddings()
    all_book_embeddings = books.embeddings()

    gap_article_indices = gap.dists_idxs.idxs
    covering_book_indices = get_covering_book_indices(gap)

    # Build base items and embeddings (articles first, then books)
    article_vecs = [all_article_embeddings[int(idx)] for idx in gap_article_indices]
    book_vecs = [all_book_embeddings[int(idx)] for idx in covering_book_indices] if len(covering_book_indices) > 0 else []

    base_items: list[tuple[str, int]] = [(
        'article', int(idx)
    ) for idx in gap_article_indices] + ([
        ('book', int(idx)) for idx in covering_book_indices
    ] if len(covering_book_indices) > 0 else [])

    base_embeddings = np.vstack(article_vecs + (book_vecs if len(book_vecs) > 0 else [])) if len(article_vecs) + len(book_vecs) > 0 else None
    if base_embeddings is None or base_embeddings.shape[0] < 3:
        raise ValueError("Insufficient base items to plot")

    # Prepare proposals array (required)
    proposals_arr = np.vstack([np.asarray(pe) for pe in proposal_embeddings])

    # Compute projection using base + proposals (PCA fits on both if proposals provided)
    # Build minimal placeholders for parameters not used by PCA path
    item_types = ['article'] * len(gap_article_indices) + ['book'] * len(covering_book_indices)
    neighbor_data = _perform_gap_neighbor_searches(gap, books, articles, k=5)
    book_to_articles_mapping = get_book_to_articles_mapping(gap)
    embeddings_2d_all, pc_variances, _ = _compute_base_projection(
        method, base_embeddings, item_types, base_items,
        book_to_articles_mapping, neighbor_data, gap, books, articles,
        proposal_embeddings=proposals_arr
    )

    # Split projections into base items and proposals using the length of base_embeddings
    base_n = len(base_embeddings)
    embeddings_2d = embeddings_2d_all[:base_n]
    proposals_2d = embeddings_2d_all[base_n:]

    # Determine counts and slices
    num_articles = len(gap_article_indices)
    num_books = len(covering_book_indices)
    article_pos = np.arange(num_articles)
    book_pos = np.arange(num_articles, num_articles + num_books) if num_books > 0 else np.array([], dtype=int)

    # Classify each article by nearest in ORIGINAL space: book vs proposal
    A = np.vstack(article_vecs)  # (A, d)
    B = np.vstack(book_vecs) if len(book_vecs) > 0 else None  # (B, d) or None
    P = proposals_arr  # (P, d)

    # Distances to books
    if B is not None and B.shape[0] > 0:
        # Compute squared distances (A,B)
        ab_diffs = A[:, None, :] - B[None, :, :]
        ab_sq_dists = np.einsum('ijk,ijk->ij', ab_diffs, ab_diffs)
        min_book_d = np.min(ab_sq_dists, axis=1)
    else:
        # No books; set +inf so proposals always closer
        min_book_d = np.full(A.shape[0], np.inf, dtype=float)

    # Distances to proposals (always present)
    ap_diffs = A[:, None, :] - P[None, :, :]
    ap_sq_dists = np.einsum('ijk,ijk->ij', ap_diffs, ap_diffs)
    min_prop_d = np.min(ap_sq_dists, axis=1)

    closer_to_book_mask = min_book_d < min_prop_d
    closer_to_prop_mask = ~closer_to_book_mask

    # Create plot
    fig, ax = PlotManager.create_figure(figsize=FIGURE_SIZE_LARGE)

    handles = []
    labels = []

    # Plot books (neighboring)
    if num_books > 0:
        h_books = ax.scatter(
            embeddings_2d[book_pos, 0], embeddings_2d[book_pos, 1],
            c=BOOK_COLOR, s=80, marker='s', edgecolors='black', linewidths=1.2,
            alpha=0.8, label='Book', zorder=2
        )
        handles.append(h_books)
        labels.append('Book')

    # Plot proposals (squares, GAP_COLOR, black border)
    if proposals_2d is not None and len(proposals_2d) > 0:
        h_props = ax.scatter(
            proposals_2d[:, 0], proposals_2d[:, 1],
            c=GAP_COLOR, s=80, marker='s', edgecolors='black', linewidths=1.2,
            alpha=0.8, label='Proposal', zorder=2
        )
        handles.append(h_props)
        labels.append('Proposal')

    # Plot articles classified by nearest
    if np.any(closer_to_book_mask):
        idx = article_pos[closer_to_book_mask]
        h_ab = ax.scatter(
            embeddings_2d[idx, 0], embeddings_2d[idx, 1],
            c=BOOK_COLOR, s=70, marker='o', alpha=0.85,
            label='Gap Article (closer to existing book)', zorder=1
        )
        handles.append(h_ab)
        labels.append('Gap Article (closer to existing book)')

    if np.any(closer_to_prop_mask):
        idx = article_pos[closer_to_prop_mask]
        h_ap = ax.scatter(
            embeddings_2d[idx, 0], embeddings_2d[idx, 1],
            c=GAP_COLOR, s=70, marker='o', alpha=0.85,
            label='Gap Articles (closer to proposal)', zorder=1
        )
        handles.append(h_ap)
        labels.append('Gap Articles (closer to proposal)')

    # Compute reassignment stats for title
    num_reassigned = int(np.sum(closer_to_prop_mask))
    reassigned_pct = (100.0 * num_reassigned / num_articles) if num_articles > 0 else 0.0

    # Title without variance; axes with PCA variances
    title_core = f"Gap {gap_idx} - {topic_name} ({method.upper()}) | Books vs Proposals | Reassigned: {num_reassigned}/{num_articles} ({reassigned_pct:.1f}%)"
    ax.set_title(title_core)
    if method.upper() == 'PCA' and pc_variances is not None:
        ax.set_xlabel(f"PC1 ({pc_variances[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pc_variances[1]:.1%})")
    else:
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
    ax.grid(True, alpha=GRID_ALPHA)

    if len(handles) > 0:
        ax.legend()

    # Save alongside existing plots
    # Use LLM-specific plots directory since this plot includes proposals  
    plots_dir = GapDirectoryManager.get_plots_llm_dir(analysis_config, gap_idx)
    plot_file = plots_dir / f"{create_unicode_safe_filename(topic_name)}_gap_{gap_idx}_{method.lower()}_books_vs_proposals.png"
    PlotManager.save_and_close(fig, plot_file)
    LOGGER.info(f"{method.upper()} gap 'books vs proposals' plot saved to {plot_file}")
    return plot_file


def main(drop_k: int, max_works: Optional[int], deduplicator: Optional[Deduplicator], radius_fraction: float, denoiser: str,
         index_type: IndexType, hnsw_m: int, hnsw_efc: int, hnsw_efs: int, generate_plots: bool,
         requested_num_prompt_articles: int, requested_num_prompt_books: int, cell_type: str):
    # Build the AI cloud using the helper
    ai_cloud = build_cloud(
        drop_k=drop_k,
        max_works=max_works,
        deduplicator=deduplicator,
        denoiser=denoiser,
        index_type=index_type,
        hnsw_m=hnsw_m,
        hnsw_efc=hnsw_efc,
        hnsw_efs=hnsw_efs
    )
    subfields = pyalex.Subfields()
    ai_subfield: pyalex.Subfield = subfields[SubField.AI.value] # type: ignore
    topics: list = ai_subfield["topics"]
    for i, topic in enumerate(topics):
        topic_name = topic["display_name"]
        LOGGER.info(f"Analyzing topic {i + 1}/{len(topics)}: {topic_name} ({topic['id']})")
        topic_id = topic["id"].split("/")[-1]  # Extract the topic ID from the URL
        
        write_gaps(
            cloud=ai_cloud,
            topic_name=topic_name,
            topic_id=topic_id,
            radius_fraction=radius_fraction,
            num_gaps=1,
            requested_num_prompt_articles=requested_num_prompt_articles,
            requested_num_prompt_books=requested_num_prompt_books,
            cell_type=cell_type,
            generate_plots=generate_plots
        )


# if __name__ == "__main__":
#     from nextbook.script_utils import run_script_main
#     run_script_main(main, __file__, generate_plots=True)