from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from numpy.typing import NDArray
import pyalex
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.sparse as sp
from nextbook.denoisers import LPPDenoiser, DenoiserContext
from nextbook.cloud import Cloud, SubCloud
from nextbook.cloud_index import faiss_search_outside_epsilon, get_ball_coverage
from nextbook.deduplicators import Deduplicator
from nextbook.index import HNSW, IndexType
from nextbook.parse_args import get_explicit_args, parse_args
from nextbook.util import LOGGER, Filter, FilterPair, SubField, WorkType, create_unicode_safe_filename, setup_logging

@dataclass(frozen=True)
class DistsAndIdxs:
    dists: NDArray[np.float32]
    idxs: NDArray[np.intp]

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

@dataclass(frozen=True)
class Gap:
    index: SubCloud
    query: SubCloud
    radius_fraction: float
    dists_idxs: DistsAndIdxs  # Distances and indices of the index points that are in the gap.

    @dataclass(frozen=True)
    class CoveredPoints:
        index: SubCloud
        query: SubCloud
        radius_fraction: float
        qidx_to_covered_index_points: dict[np.intp, DistsAndIdxs]  # List of distances and idxs for each index point covered by the corresponding query point.
    def closest_query_points(self):
        qidx_to_covered_iidxs = defaultdict(list)
        qidx_to_covered_dists = defaultdict(list)
        res = faiss_search_outside_epsilon(index=self.query, query=self.index, is_intra_cloud_search=False, k=1, epsilon=0)
        closest_q_to_each_i = res.I_outside[self.dists_idxs.idxs].squeeze(1)
        if np.any(closest_q_to_each_i < 0):
            assert np.all(closest_q_to_each_i < 0)
            return Gap.CoveredPoints(
                index=self.index,
                query=self.query,
                radius_fraction=self.radius_fraction,
                qidx_to_covered_index_points={}
            )

        closest_q_dists = res.D_outside[self.dists_idxs.idxs].squeeze(1)

        for dist, iidx, qidx in zip(closest_q_dists, self.dists_idxs.idxs, closest_q_to_each_i):
            qidx_to_covered_iidxs[qidx].append(iidx)
            qidx_to_covered_dists[qidx].append(dist)
        qidxs = np.unique(closest_q_to_each_i).tolist()  # Ensure qidxs are unique
        assert set(qidxs) == set(closest_q_to_each_i)
        assert set(qidxs) == set(qidx_to_covered_iidxs.keys())
        assert set(qidxs) == set(qidx_to_covered_dists.keys())
        qidx_to_covered_index_points = {qidx: DistsAndIdxs(dists=np.array(qidx_to_covered_dists[qidx], dtype=np.float32),
                                             idxs=np.array(qidx_to_covered_iidxs[qidx], dtype=np.intp))
                            for qidx in qidxs}  # Exclude -1 indices which indicate no nearest query point found
        return Gap.CoveredPoints(
            index=self.index,
            query=self.query,
            radius_fraction=self.radius_fraction,
            qidx_to_covered_index_points=qidx_to_covered_index_points
        )
        



    def __post_init__(self):
        if not np.all(np.diff(self.dists_idxs.dists) >= 0):
            raise ValueError("Distances must be sorted in non-decreasing order")
        if not self.dists_idxs.dists.ndim == 1:
            raise ValueError("Distances must be a 1D array")
        if not self.dists_idxs.dists.shape == self.dists_idxs.idxs.shape:
            raise ValueError("Distances and indices must have the same shape")
        if not self.dists_idxs.dists[0] == 0:
            raise ValueError("The closest point in the gap must have distance 0 (it is the center point)")

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Color constants for visualization
BOOK_COLOR = 'blue'
GAP_COLOR = 'orange'

def get_gap_dir(denoiser: str, drop_k: int, radius_fraction: float, gap_idx: int) -> Path:
    """
    Get the directory path for a specific gap and create it if it doesn't exist.
    
    Args:
        denoiser: Name of the denoiser used
        drop_k: Drop-k parameter used
        radius_fraction: Radius fraction for gap analysis
        gap_idx: Index of the gap (0-indexed)
        
    Returns:
        Path to the gap directory
    """
    gap_dir = ARTIFACTS_DIR / "gaps" / f"{denoiser}_k={drop_k}_radiusfrac={radius_fraction:.2f}" / f"gap_{gap_idx}"
    gap_dir.mkdir(parents=True, exist_ok=True)
    return gap_dir

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
            'gap_size': 'int64'
        })
        LOGGER.info(f"Loaded {len(df)} rows from {csv_path}")
    except FileNotFoundError:
        LOGGER.info(f"File {csv_path} not found, creating empty DataFrame")
        df = pd.DataFrame(columns=['topic_name', 'topic_id', 'denoiser', 'drop_k', 'gap_idx', 'num_books', 'num_articles', 'gap_size'])
        df = df.astype({
            'topic_name': 'string',
            'topic_id': 'string', 
            'denoiser': 'string',
            'drop_k': 'int64',
            'gap_idx': 'int64',
            'num_books': 'int64',
            'num_articles': 'int64',
            'gap_size': 'int64'
        })
    return df

class Results:
    _csv_path = ARTIFACTS_DIR / "gaps" / "gap_analysis.csv"
    _key_columns = ['topic_name', 'topic_id', 'denoiser', 'drop_k', 'gap_idx']
    _df = _df_from_file(_csv_path)
    
    @classmethod
    def add_row(cls, topic_name: str, topic_id: str, denoiser: str, drop_k: int, gap_idx: int, num_books: int, num_articles: int, gap_size: int):
        cls._csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        new_row = pd.DataFrame([{
            'topic_name': topic_name,
            'topic_id': topic_id,
            'denoiser': denoiser,
            'drop_k': drop_k,
            'gap_idx': gap_idx,
            'num_books': num_books,
            'num_articles': num_articles,
            'gap_size': gap_size
        }])

        LOGGER.info(f"Adding row:\n{new_row}")

        # Check if row already exists
        if len(cls._df) > 0:
            mask = cls._df['topic_name'] == topic_name
            mask &= cls._df['topic_id'] == topic_id
            mask &= cls._df['denoiser'] == denoiser
            mask &= cls._df['drop_k'] == drop_k
            mask &= cls._df['gap_idx'] == gap_idx
            
            if mask.any():
                LOGGER.warning(f"Row already exists. Updating existing row. Old row:\n{cls._df[mask]}")
                cls._df.loc[mask, 'num_books'] = num_books
                cls._df.loc[mask, 'num_articles'] = num_articles
                cls._df.loc[mask, 'gap_size'] = gap_size
            else:
                cls._df = pd.concat([cls._df, new_row], ignore_index=True)
        else:
            cls._df = pd.concat([cls._df, new_row], ignore_index=True)
        
        cls._df.to_csv(cls._csv_path, index=False)
        LOGGER.info(f"Saved results to {cls._csv_path}")
    
    @classmethod
    def get_df(cls) -> pd.DataFrame:
        return cls._df.copy()


def gaps(index: SubCloud, query: SubCloud, radius_fraction):
    """
    Find gaps where there are many index points (e.g. articles) but no query points (e.g. books).
    """
    covered_points = get_ball_coverage(index, query, radius_fraction=radius_fraction).covered_points
    while True:
        highest_coverage_idx = np.argmax([len(idxs) for _dists, idxs in covered_points])
        dists, idxs = covered_points[highest_coverage_idx]
        if len(idxs) == 0:
            LOGGER.info("No more gaps found")
            break
        order = np.argsort(dists)
        dists = dists[order]
        idxs = idxs[order]

        # assert highest_coverage_idx in idxs
        assert idxs[0] == highest_coverage_idx, "The first index should be the one with the highest coverage"
        assert dists[0] == 0, "The distance for the highest coverage index should be 0"
        
        yield Gap(index=index, query=query, radius_fraction=radius_fraction, 
                 dists_idxs=DistsAndIdxs(dists=dists, idxs=idxs))
        for idx in idxs:
            # Prevent all covered points from being topic centers
            covered_points[idx] = (np.array([], dtype=dists.dtype), np.array([], dtype=idxs.dtype))
        for idx, (dists_, idxs_) in enumerate(covered_points):
            # Remove all points in idxs from all other covered points
            mask = np.isin(idxs_, idxs, invert=True)
            covered_points[idx] = (dists_[mask], idxs_[mask])


def write_single_gap(gap: Gap, gap_idx: int, topic_name: str, topic_id: str, denoiser: str, drop_k: int, 
                     radius_fraction: float, num_books: int, num_articles: int, 
                     books: SubCloud, articles: SubCloud) -> None:
    """
    Write detailed analysis for a single gap to a file and add entry to Results CSV.
    
    Args:
        gap: The Gap object to analyze
        gap_idx: Index of this gap (0-indexed)
        topic_name: Display name of the topic
        topic_id: ID of the topic
        denoiser: Name of the denoiser used
        drop_k: Drop-k parameter used
        radius_fraction: Radius fraction for gap analysis
        num_books: Total number of books in topic
        num_articles: Total number of articles in topic
        books: SubCloud for books
        articles: SubCloud for articles
    """
    gap_size = len(gap.dists_idxs.idxs)
    LOGGER.info(f"Found gap {gap_idx} for topic {topic_name} with {gap_size} articles")
    
    # Add to Results CSV
    Results.add_row(topic_name, topic_id, denoiser, drop_k, gap_idx, num_books, num_articles, gap_size)
    
    # Create detailed gap analysis file
    gap_dir = get_gap_dir(denoiser, drop_k, radius_fraction, gap_idx)
    analysis_dir = gap_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize filename and add gap index
    filename = f"{create_unicode_safe_filename(topic_name)}_id={topic_id}_analysis.txt"
    gap_file = analysis_dir / filename
    
    # Get work metadata
    article_works = articles.works()
    book_works = books.works() if num_books > 0 else []
    
    # Get covered points analysis
    covered_points_analysis = gap.closest_query_points()
    
    with open(gap_file, 'w', encoding='utf-8') as f:
        f.write(f"Gap Analysis for Topic: {topic_name}\n")
        f.write(f"Topic ID: {topic_id}\n")
        f.write(f"Gap Index: {gap_idx} (0-indexed)\n")
        f.write(f"Denoiser: {denoiser}\n")
        f.write(f"Drop K: {drop_k}\n")
        f.write(f"Radius Fraction: {radius_fraction}\n")
        f.write(f"Number of articles in gap: {gap_size}\n")
        f.write(f"Total books in topic: {num_books}\n")
        f.write(f"Total articles in topic: {num_articles}\n")
        f.write("=" * 80 + "\n\n")

        # Find the closest book to the center article
        assert np.argmin(gap.dists_idxs.dists) == 0, f"Expected the closest article to be the first one in the gap, got {np.argmin(gap.dists_idxs.dists)}"
        center_article_idx = gap.dists_idxs.idxs[0]  # Center is the first article
        if num_books > 0:
            # Find the book that covers the center article
            closest_book_idx = None
            closest_book_distance = None
            
            for book_idx, covered_points in covered_points_analysis.qidx_to_covered_index_points.items():
                # Check if this book covers the center article
                center_positions = np.where(covered_points.idxs == center_article_idx)[0]
                if len(center_positions) > 0:
                    closest_book_idx = book_idx
                    closest_book_distance = covered_points.dists[center_positions[0]]
                    break
            
            assert closest_book_idx is not None
            closest_book_work = book_works[closest_book_idx]
            f.write(f"Closest book to gap center (distance: {closest_book_distance:.4f}):\n")
            f.write(f"{closest_book_work.get('id', 'N/A')}: {closest_book_work.get('display_name', 'N/A')}\n")
            f.write(f"  {closest_book_work.get('abstract', 'N/A')}\n")
            f.write("\n" + "=" * 80 + "\n\n")

        # List the top articles closest to the center of the gap
        closest_n = min(10, gap_size)
        f.write(f"{closest_n} most central articles in the gap:\n\n")
        for i, article_idx in enumerate(gap.dists_idxs.idxs[:closest_n]):
            article_work = article_works[article_idx]
            f.write(f"Article {i + 1} - {article_work.get('id', 'N/A')} - {article_work.get('display_name', 'N/A')} - distance: {gap.dists_idxs.dists[i]:.4f}\n")
            f.write(f"  Abstract: {article_work.get('abstract', 'N/A')}\n")
            f.write("\n")

        f.write("Furthest article from the center of the gap:\n\n")
        furthest_idx = np.argmax(gap.dists_idxs.dists)
        furthest_article_idx = gap.dists_idxs.idxs[furthest_idx]
        furthest_article_work = article_works[furthest_article_idx]
        f.write(f"Furthest article - Id: {furthest_article_work.get('id', 'N/A')} - Title: {furthest_article_work.get('display_name', 'N/A')} - Distance from center: {gap.dists_idxs.dists[furthest_idx]:.4f}\n")
        f.write(f"  Abstract: {furthest_article_work.get('abstract', 'N/A')}\n")
        
        f.write("=" * 80 + "\n\n")

        if num_books == 0:
            f.write("No books found in this topic.\n\n")
        else:
            # Group covered articles by covering book and sort by number of articles covered
            book_coverage = defaultdict(list)
            book_distances = defaultdict(list)
            
            for book_idx, covered_points in covered_points_analysis.qidx_to_covered_index_points.items():
                if len(covered_points.idxs) > 0:
                    for article_idx, dist in zip(covered_points.idxs, covered_points.dists):
                        # article_idx here is the index in the original articles subcloud
                        # We need to find which position in the gap this article corresponds to
                        gap_position = np.where(gap.dists_idxs.idxs == article_idx)[0]
                        book_coverage[book_idx].append(gap_position[0])
                        book_distances[book_idx].append(dist)
            
            # Sort books by number of articles covered (descending)
            sorted_books = sorted(book_coverage.items(), key=lambda x: len(x[1]), reverse=True)
            
            f.write(f"{len(sorted_books)} books covering articles in the gap, in order of decreasing coverage:\n\n")
            
            for i, (book_idx, covered_article_idxs) in enumerate(sorted_books):
                book_work = book_works[book_idx]
                num_covered = len(covered_article_idxs)
                
                f.write(f"Book {i + 1}/{len(sorted_books)} - Id: {book_work.get('id', 'N/A')} - Title: {book_work.get('display_name', 'N/A')}\n")
                f.write(f"  Abstract: {book_work.get('abstract', 'N/A')}\n")
                f.write(f"  Number of articles covered: {num_covered} ({num_covered / gap_size:.2%})\n")
                
                # Find the closest article (minimum distance)
                distances = book_distances[book_idx]
                min_dist_idx = np.argmin(distances)
                gap_position = covered_article_idxs[min_dist_idx]
                closest_distance = distances[min_dist_idx]
                
                # Get the article from the gap using the gap position
                gap_article_idx = gap.dists_idxs.idxs[gap_position]
                closest_article_work = article_works[gap_article_idx]
                
                f.write(f"  Closest article in gap:\n")
                f.write(f"    Title: {closest_article_work.get('display_name', 'N/A')}\n")
                f.write(f"    ID: {closest_article_work.get('id', 'N/A')}\n")
                f.write(f"    Abstract: {closest_article_work.get('abstract', 'N/A')}\n")
                f.write(f"    Distance: {closest_distance:.4f}\n")
                
                f.write("\n" + "-" * 60 + "\n\n")
    
    LOGGER.info(f"Detailed gap analysis written to {gap_file}")


def write_llm_prompt(gap: Gap, gap_idx: int, topic_name: str, topic_id: str, denoiser: str, drop_k: int, 
                     radius_fraction: float, num_books: int, num_articles: int, 
                     books: SubCloud, articles: SubCloud) -> None:
    """
    Write an LLM prompt for generating a book proposal based on the gap analysis.
    
    Args:
        gap: The Gap object to analyze
        gap_idx: Index of this gap (0-indexed)
        topic_name: Display name of the topic
        topic_id: ID of the topic
        denoiser: Name of the denoiser used
        drop_k: Drop-k parameter used
        radius_fraction: Radius fraction for gap analysis
        num_books: Total number of books in topic
        num_articles: Total number of articles in topic
        books: SubCloud for books
        articles: SubCloud for articles
    """
    gap_size = len(gap.dists_idxs.idxs)
    
    # Create prompt file
    gap_dir = get_gap_dir(denoiser, drop_k, radius_fraction, gap_idx)
    prompts_dir = gap_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
    prompt_filename = f"{create_unicode_safe_filename(topic_name)}_id={topic_id}_prompt.txt"
    prompt_file = prompts_dir / prompt_filename
    
    # Get work metadata
    article_works = articles.works()
    book_works = books.works()
    
    # Sample 20 random articles from the gap
    sample_size = min(20, gap_size)
    sampled_indices = random.sample(range(gap_size), sample_size)
    sampled_articles = [article_works[gap.dists_idxs.idxs[i]] for i in sampled_indices]
    
    # Only analyze book coverage if books exist
    covering_books = []
    if num_books > 0:
        # Get covered points analysis to find books covering the gap
        covered_points_analysis = gap.closest_query_points()
        
        # Find books that cover at least 50% of the gap articles
        book_coverage = defaultdict(list)
        for book_idx, covered_points in covered_points_analysis.qidx_to_covered_index_points.items():
            if len(covered_points.idxs) > 0:
                book_coverage[book_idx] = covered_points.idxs.tolist()
        
        # Sort books by coverage and select until we cover at least 50% of gap
        sorted_books = sorted(book_coverage.items(), key=lambda x: len(x[1]), reverse=True)
        covered_articles = set()
        target_coverage = gap_size * 0.5
        
        for book_idx, covered_article_idxs in sorted_books:
            covering_books.append(book_works[book_idx])
            covered_articles.update(covered_article_idxs)
            if len(covered_articles) >= target_coverage:
                break
    
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write("# Textbook Proposal Prompt\n\n")
        f.write("You are a publishing expert and academic researcher. Based on the research articles and related books ")
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
        f.write("Format your response as a complete LaTeX subsection using EXACTLY the following structure. ")
        f.write("Do not add any text before or after this LaTeX code:\n\n")
        f.write("```\n")
        f.write("\\subsection{[Proposed Title]}\n\n")
        f.write("\\textbf{Abstract:} [Abstract]\n\n")
        f.write("\\textbf{Table of Contents:}\n")
        f.write("\\begin{enumerate}\n")
        f.write("    \\item [Chapter Name] - [Description of concepts covered (1-2 sentences)]\n")
        f.write("    \\item [Chapter Name] - [Description of concepts covered (1-2 sentences)]\n")
        f.write("    % ...\n")
        f.write("\\end{enumerate}\n\n")
        f.write("```\n\n")
        f.write("## Guidelines\n\n")
        f.write("- Use the EXACT LaTeX structure shown above, so that it can be directly copied into a LaTeX document.\n")
        f.write("- Replace ALL bracketed placeholders with your content.\n")
        f.write("- Include 10-15 chapters in the enumerated list.\n")
        f.write("- Build up systematically from basic to advanced concepts.\n")
    
    LOGGER.info(f"LLM prompt written to {prompt_file}")


def _perform_gap_neighbor_searches(gap: Gap, books: SubCloud, articles: SubCloud, covered_points_analysis: Gap.CoveredPoints, k: int = 5) -> GapVisualizationData:
    """
    Perform the four FAISS neighbor searches for gap visualization.
    
    Returns:
        GapVisualizationData: Contains both flattened neighbor sets and detailed search results
    """
    gap_article_indices = gap.dists_idxs.idxs
    covering_book_indices = list(covered_points_analysis.qidx_to_covered_index_points.keys())
    
    # 1. Nearest articles to in-gap articles
    gap_to_article_neighbors = set()
    if len(gap_article_indices) > 0:
        res = faiss_search_outside_epsilon(articles, articles, is_intra_cloud_search=True, k=k, epsilon=0)
        gap_to_article_neighbors.update(res.I_outside.flatten())
    
    # 2. Nearest books to in-gap articles - KEEP DETAILED RESULTS
    gap_to_book_neighbors = set()
    gap_article_to_book_search = None
    if len(gap_article_indices) > 0 and len(books.embeddings()) > 0:
        res = faiss_search_outside_epsilon(books, articles, is_intra_cloud_search=False, k=k, epsilon=0)
        # Only get neighbors for gap articles
        gap_book_neighbors = res.I_outside[gap_article_indices]
        gap_to_book_neighbors.update(gap_book_neighbors.flatten())
        # Keep detailed search results for efficient lookups
        gap_article_to_book_search = gap_book_neighbors
    
    # 3. Nearest articles to covering books - KEEP DETAILED RESULTS
    book_to_article_neighbors = set()
    book_to_article_search = None
    if len(covering_book_indices) > 0:
        res = faiss_search_outside_epsilon(articles, books, is_intra_cloud_search=False, k=k, epsilon=0)
        # Only get neighbors for covering books
        book_article_neighbors = res.I_outside[covering_book_indices]
        book_to_article_neighbors.update(book_article_neighbors.flatten())
        # Keep detailed search results for efficient lookups
        book_to_article_search = book_article_neighbors
    
    # 4. Nearest books to covering books
    book_to_book_neighbors = set()
    if len(covering_book_indices) > 0:
        res = faiss_search_outside_epsilon(books, books, is_intra_cloud_search=True, k=k, epsilon=0)
        # Only get neighbors for covering books
        book_book_neighbors = res.I_outside[covering_book_indices]
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


def _collect_visualization_data(gap: Gap, gap_idx: int, num_books: int, books: SubCloud, articles: SubCloud):
    """
    Helper function to collect all embeddings and metadata for gap visualization.
    
    Returns:
        tuple: (all_embeddings, item_types, all_items, gap_article_positions_map, all_book_positions_map, covered_points_analysis, neighbor_data)
    """
    gap_size = len(gap.dists_idxs.idxs)
    
    if gap_size == 0:
        return None, None, None, None, None, None, None
    
    # Get all embeddings and work data
    all_article_embeddings = articles.embeddings()
    all_book_embeddings = books.embeddings()
    
    # Get gap articles
    gap_article_indices = gap.dists_idxs.idxs
    
    # Get covered points analysis to find books covering the gap
    covered_points_analysis = gap.closest_query_points()
    covering_book_indices = list(covered_points_analysis.qidx_to_covered_index_points.keys())
    
    # Perform the four neighbor searches
    k = 5
    neighbor_data = _perform_gap_neighbor_searches(gap, books, articles, covered_points_analysis, k)
    
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
        return None, None, None, None, None, None, None
    
    # Concatenate embeddings
    all_embeddings = np.vstack(all_embeddings_list)
    
    # Create position mappings for connection lines  
    gap_article_positions_map = {}
    all_book_positions_map = {}  # Now includes all books, not just covering books
    
    for i, (item_type, (work_type, work_idx)) in enumerate(zip(item_types, all_items)):
        if work_type == 'article' and work_idx in gap_article_indices:
            gap_article_positions_map[work_idx] = i
        elif work_type == 'book':
            all_book_positions_map[work_idx] = i
    
    return (all_embeddings, item_types, all_items, gap_article_positions_map, 
            all_book_positions_map, covered_points_analysis, neighbor_data)


def _create_gap_plot(embeddings_2d, item_types, all_items, gap_article_positions_map, 
                    all_book_positions_map, covered_points_analysis, neighbor_data: GapVisualizationData, 
                    gap: Gap, topic_name: str, gap_idx: int, method_name: str, variance_info: str = ""):
    """
    Helper function to create the actual plot from 2D embeddings.
    
    Args:
        embeddings_2d: 2D coordinates for plotting
        item_types: List of item type strings
        all_items: List of (work_type, idx) tuples
        gap_article_positions_map: Mapping from gap article indices to position in embeddings_2d
        all_book_positions_map: Mapping from all book indices to position in embeddings_2d
        covered_points_analysis: Gap coverage analysis
        neighbor_data: Neighbor search results
        gap: The Gap object
        topic_name: Name of the topic
        gap_idx: Gap index
        method_name: Name of dimensionality reduction method (e.g., "PCA", "t-SNE")
        variance_info: Additional info about variance explained (for PCA)
    """
    gap_article_indices = gap.dists_idxs.idxs
    covering_book_indices = list(covered_points_analysis.qidx_to_covered_index_points.keys())
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Determine which articles will be re-assigned to center point
    center_article_idx = gap_article_indices[0]  # Center is the first article in the gap
    center_pos = None
    reassigned_articles = set()  # Articles that will be re-assigned to center
    
    if center_article_idx in gap_article_positions_map:
        center_pos_idx = gap_article_positions_map[center_article_idx]
        center_pos = embeddings_2d[center_pos_idx]
        
        # Check each article to see if it's closer to center than to its nearest book
        # This applies to ALL articles, not just gap articles
        for i, (work_type, work_idx) in enumerate(all_items):
            if work_type == 'article':
                article_pos = embeddings_2d[i]
                
                # Use efficient lookup to find the nearest book
                nearest_book_idx = neighbor_data.get_nearest_book_for_article(work_idx)
                
                if nearest_book_idx is not None and nearest_book_idx in all_book_positions_map:
                    book_pos_idx = all_book_positions_map[nearest_book_idx]
                    book_pos = embeddings_2d[book_pos_idx]
                    
                    # Calculate distances in 2D visualization space
                    distance_to_book = np.linalg.norm(article_pos - book_pos)
                    distance_to_center = np.linalg.norm(article_pos - center_pos)
                    
                    # If closer to center, this article will be re-assigned
                    if distance_to_center < distance_to_book:
                        reassigned_articles.add(work_idx)
    
    # Plot all books
    book_positions = []
    article_positions_reassigned = []
    article_positions_normal = []
    
    for i, (work_type, work_idx) in enumerate(all_items):
        if work_type == 'book':
            book_positions.append(i)
        elif work_type == 'article':
            if work_idx in reassigned_articles:
                article_positions_reassigned.append(i)
            else:
                article_positions_normal.append(i)
    
    # # Plot articles
    # article_positions = article_positions_reassigned + article_positions_normal
    # article_coords = embeddings_2d[article_positions]
    # plt.scatter(article_coords[:, 0], article_coords[:, 1], 
    #             c='gray', marker='o', s=50, alpha=0.5, label='Articles', zorder=1)

    # Plot unreassigned articles
    unreassigned_article_coords = embeddings_2d[article_positions_normal]
    plt.scatter(unreassigned_article_coords[:, 0], unreassigned_article_coords[:, 1], 
                c='gray', marker='.', s=50, alpha=0.2, label='Unreassigned Articles', zorder=1)
    
    # Plot reassigned articles
    reassigned_article_coords = embeddings_2d[article_positions_reassigned]
    plt.scatter(reassigned_article_coords[:, 0], reassigned_article_coords[:, 1], 
                c=GAP_COLOR, marker='.', s=50, alpha=0.5, label='Reassigned Articles', zorder=2)
    
    # Plot books
    book_coords = embeddings_2d[book_positions]
    plt.scatter(book_coords[:, 0], book_coords[:, 1], 
                c=BOOK_COLOR, marker='^', s=100, alpha=0.7, label='Books', zorder=3)
    
    # Draw lines from each article to its nearest book or gap center
    for i, (work_type, work_idx) in enumerate(all_items):
        if work_type == 'article':
            article_pos = embeddings_2d[i]
            
            if work_idx in reassigned_articles and center_pos is not None:
                # Reassigned articles: yellow line to gap center
                plt.plot([article_pos[0], center_pos[0]], [article_pos[1], center_pos[1]], 
                        color=GAP_COLOR, alpha=0.3, linewidth=1, linestyle='-', zorder=4)
            else:
                # Un-reassigned articles: blue line to nearest book
                nearest_book_idx = neighbor_data.get_nearest_book_for_article(work_idx)
                
                if nearest_book_idx is not None and nearest_book_idx in all_book_positions_map:
                    book_pos_idx = all_book_positions_map[nearest_book_idx]
                    book_pos = embeddings_2d[book_pos_idx]
                    
                    plt.plot([article_pos[0], book_pos[0]], [article_pos[1], book_pos[1]], 
                            color=BOOK_COLOR, alpha=0.3, linewidth=1, linestyle='-', zorder=4)
    
    # Plot center article
    if center_pos is not None:
        plt.scatter(center_pos[0], center_pos[1], 
                   c=GAP_COLOR, marker='^', s=200, alpha=1.0, 
                   edgecolors='black', linewidths=1, 
                   label='Gap Center', zorder=5)
    
    plt.xlabel(f'{method_name} 1{variance_info}')
    plt.ylabel(f'{method_name} 2{variance_info}')
    plt.title(f'Gap Visualization ({method_name}): {topic_name} (Gap {gap_idx})\n'
             f'{len(gap_article_indices)} articles in gap, {len(reassigned_articles)} will be reassigned')
    plt.legend()
    plt.grid(True, alpha=0.3)


def plot_gap_visualization_pca(gap: Gap, gap_idx: int, topic_name: str, topic_id: str, denoiser: str, drop_k: int, 
                              radius_fraction: float, num_books: int, num_articles: int, 
                              books: SubCloud, articles: SubCloud) -> None:
    """
    Create a 2D PCA visualization of the gap showing articles, books, and their relationships.
    """
    LOGGER.info(f"Creating PCA gap visualization for topic {topic_name}, gap {gap_idx}")
    
    # Collect visualization data
    data = _collect_visualization_data(gap, gap_idx, num_books, books, articles)
    if data[0] is None:
        LOGGER.warning(f"Skipping PCA visualization - insufficient data")
        return
    
    all_embeddings, item_types, all_items, gap_article_positions_map, all_book_positions_map, covered_points_analysis, neighbor_data = data
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Create variance info string
    variance_info = f' ({pca.explained_variance_ratio_[0]:.1%} variance)'
    
    # Create the plot
    _create_gap_plot(embeddings_2d, item_types, all_items, gap_article_positions_map,
                    all_book_positions_map, covered_points_analysis, neighbor_data,
                    gap, topic_name, gap_idx, "PCA", variance_info)
    
    # Save the plot
    gap_dir = get_gap_dir(denoiser, drop_k, radius_fraction, gap_idx)
    plots_dir = gap_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_file = plots_dir / f"{create_unicode_safe_filename(topic_name)}_gap_{gap_idx}_pca_visualization.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    LOGGER.info(f"PCA gap visualization saved to {plot_file}")


def plot_gap_visualization_tsne(gap: Gap, gap_idx: int, topic_name: str, topic_id: str, denoiser: str, drop_k: int, 
                               radius_fraction: float, num_books: int, num_articles: int, 
                               books: SubCloud, articles: SubCloud) -> None:
    """
    Create a 2D t-SNE visualization of the gap showing articles, books, and their relationships.
    """
    LOGGER.info(f"Creating t-SNE gap visualization for topic {topic_name}, gap {gap_idx}")
    
    # Collect visualization data
    data = _collect_visualization_data(gap, gap_idx, num_books, books, articles)
    if data[0] is None:
        LOGGER.warning(f"Skipping t-SNE visualization - insufficient data")
        return
    
    all_embeddings, item_types, all_items, gap_article_positions_map, all_book_positions_map, covered_points_analysis, neighbor_data = data
    
    # Apply t-SNE to reduce to 2D
    # Use lower perplexity for small datasets
    n_samples = len(all_embeddings)
    perplexity = min(30, max(5, n_samples // 4))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Create the plot
    _create_gap_plot(embeddings_2d, item_types, all_items, gap_article_positions_map,
                    all_book_positions_map, covered_points_analysis, neighbor_data,
                    gap, topic_name, gap_idx, "t-SNE")
    
    # Save the plot
    gap_dir = get_gap_dir(denoiser, drop_k, radius_fraction, gap_idx)
    plots_dir = gap_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_file = plots_dir / f"{create_unicode_safe_filename(topic_name)}_gap_{gap_idx}_tsne_visualization.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    LOGGER.info(f"t-SNE gap visualization saved to {plot_file}")


class _GapVisualizationContext(DenoiserContext):
    """
    Custom DenoiserContext for gap visualization that uses the k-NN graph
    from FAISS searches as the affinity matrix.
    """
    def __init__(self, embeddings: NDArray, neighbor_indices: NDArray, k: int = 5):
        """
        Args:
            embeddings: All embeddings for visualization items
            neighbor_indices: Array of shape (n_items, k) containing neighbor indices
            k: Number of neighbors per item
        """
        self._embeddings = embeddings
        self._neighbor_indices = neighbor_indices
        self._k = k
        self._affinity_matrix = None
        
    def _build_affinity_matrix(self) -> sp.csr_matrix:
        """Build sparse affinity matrix from neighbor indices."""
        if self._affinity_matrix is not None:
            return self._affinity_matrix
            
        n_items = len(self._embeddings)
        row_indices = []
        col_indices = []
        data = []
        
        for i, neighbors in enumerate(self._neighbor_indices):
            for neighbor_idx in neighbors:
                if 0 <= neighbor_idx < n_items:  # Valid neighbor
                    row_indices.append(i)
                    col_indices.append(neighbor_idx)
                    data.append(1.0)  # Unweighted graph
        
        self._affinity_matrix = sp.csr_matrix(
            (data, (row_indices, col_indices)), 
            shape=(n_items, n_items)
        )
        return self._affinity_matrix
    
    def A(self) -> sp.csr_matrix:
        return self._build_affinity_matrix()
    
    def X(self) -> NDArray:
        return self._embeddings
    
    def X1(self) -> NDArray:
        # Not used for LPP, but required by interface
        return self._embeddings[:len(self._embeddings)//2]
    
    def X2(self) -> NDArray:
        # Not used for LPP, but required by interface  
        return self._embeddings[len(self._embeddings)//2:]


def plot_gap_visualization_lpp(gap: Gap, gap_idx: int, topic_name: str, topic_id: str, denoiser: str, drop_k: int, 
                              radius_fraction: float, num_books: int, num_articles: int, 
                              books: SubCloud, articles: SubCloud) -> None:
    """
    Create a 2D LPP visualization of the gap using the k-NN graph from FAISS searches.
    """
    LOGGER.info(f"Creating LPP gap visualization for topic {topic_name}, gap {gap_idx}")
    
    # Collect visualization data
    data = _collect_visualization_data(gap, gap_idx, num_books, books, articles)
    if data[0] is None:
        LOGGER.warning(f"Skipping LPP visualization - insufficient data")
        return
    
    all_embeddings, item_types, all_items, gap_article_positions_map, all_book_positions_map, covered_points_analysis, neighbor_data = data
    
    # Get the neighbor searches
    gap_article_indices = gap.dists_idxs.idxs
    covering_book_indices = list(covered_points_analysis.qidx_to_covered_index_points.keys())
    k = 5
    
    # Get the actual FAISS search results to build proper neighbor relationships
    # We need to re-run the searches to get the detailed neighbor mappings
    item_neighbors = defaultdict(set)  # Maps (work_type, global_idx) -> set of local neighbor indices
    
    # Create a mapping from global indices to local positions in all_embeddings
    global_to_local = {}
    for local_idx, (work_type, global_idx) in enumerate(all_items):
        global_to_local[(work_type, global_idx)] = local_idx
    
    gap_article_indices_set = set(gap_article_indices)
    covering_book_indices_set = set(covering_book_indices)
    
    # 1. Get neighbors for gap articles from article-to-article search
    if len(gap_article_indices) > 0:
        res = faiss_search_outside_epsilon(articles, articles, is_intra_cloud_search=True, k=k, epsilon=0)
        for i, gap_article_idx in enumerate(gap_article_indices):
            if ('article', gap_article_idx) in global_to_local:
                neighbors = res.I_outside[gap_article_idx]
                for neighbor_idx in neighbors:
                    if ('article', neighbor_idx) in global_to_local and neighbor_idx != gap_article_idx:
                        item_neighbors[('article', gap_article_idx)].add(global_to_local[('article', neighbor_idx)])
    
    # 2. Get book neighbors for gap articles from article-to-book search  
    if len(gap_article_indices) > 0 and len(books.embeddings()) > 0:
        res = faiss_search_outside_epsilon(books, articles, is_intra_cloud_search=False, k=k, epsilon=0)
        for gap_article_idx in gap_article_indices:
            if ('article', gap_article_idx) in global_to_local:
                neighbors = res.I_outside[gap_article_idx]
                for neighbor_idx in neighbors:
                    if ('book', neighbor_idx) in global_to_local:
                        item_neighbors[('article', gap_article_idx)].add(global_to_local[('book', neighbor_idx)])
    
    # 3. Get article neighbors for covering books from book-to-article search
    if len(covering_book_indices) > 0:
        res = faiss_search_outside_epsilon(articles, books, is_intra_cloud_search=False, k=k, epsilon=0)
        for covering_book_idx in covering_book_indices:
            if ('book', covering_book_idx) in global_to_local:
                neighbors = res.I_outside[covering_book_idx]
                for neighbor_idx in neighbors:
                    if ('article', neighbor_idx) in global_to_local:
                        item_neighbors[('book', covering_book_idx)].add(global_to_local[('article', neighbor_idx)])
    
    # 4. Get book neighbors for covering books from book-to-book search
    if len(covering_book_indices) > 0:
        res = faiss_search_outside_epsilon(books, books, is_intra_cloud_search=True, k=k, epsilon=0)
        for covering_book_idx in covering_book_indices:
            if ('book', covering_book_idx) in global_to_local:
                neighbors = res.I_outside[covering_book_idx]
                for neighbor_idx in neighbors:
                    if ('book', neighbor_idx) in global_to_local and neighbor_idx != covering_book_idx:
                        item_neighbors[('book', covering_book_idx)].add(global_to_local[('book', neighbor_idx)])
    
    # Build neighbor indices list efficiently
    neighbor_indices_list = []
    for work_type, global_idx in all_items:
        neighbors = list(item_neighbors[(work_type, global_idx)])
        
        # Pad or truncate to exactly k neighbors
        neighbors = neighbors[:k]  # Take first k
        local_idx = global_to_local[(work_type, global_idx)]
        while len(neighbors) < k:
            neighbors.append(neighbors[0] if neighbors else local_idx)  # Self-loop if no neighbors
        
        neighbor_indices_list.append(neighbors)
    
    neighbor_indices = np.array(neighbor_indices_list)
    
    # Create LPP context and apply LPP
    context = _GapVisualizationContext(all_embeddings, neighbor_indices, k)
    lpp_denoiser = LPPDenoiser()
    
    try:
        projector, eigenvals = lpp_denoiser.projector(context, n_components=2)
        embeddings_2d = all_embeddings @ projector
        
        # Create the plot
        _create_gap_plot(embeddings_2d, item_types, all_items, gap_article_positions_map,
                        all_book_positions_map, covered_points_analysis, neighbor_data,
                        gap, topic_name, gap_idx, "LPP")
        
        # Save the plot
        gap_dir = get_gap_dir(denoiser, drop_k, radius_fraction, gap_idx)
        plots_dir = gap_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_file = plots_dir / f"{create_unicode_safe_filename(topic_name)}_gap_{gap_idx}_lpp_visualization.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        LOGGER.info(f"LPP gap visualization saved to {plot_file}")
        
    except Exception as e:
        LOGGER.warning(f"LPP visualization failed: {e}")


def plot_gap_visualization(gap: Gap, gap_idx: int, topic_name: str, topic_id: str, denoiser: str, drop_k: int, 
                          radius_fraction: float, num_books: int, num_articles: int, 
                          books: SubCloud, articles: SubCloud) -> None:
    """
    Create PCA, t-SNE, and LPP visualizations of the gap.
    """
    plot_gap_visualization_pca(gap, gap_idx, topic_name, topic_id, denoiser, drop_k,
                              radius_fraction, num_books, num_articles, books, articles)
    # plot_gap_visualization_lpp(gap, gap_idx, topic_name, topic_id, denoiser, drop_k,
    #                           radius_fraction, num_books, num_articles, books, articles)
    # plot_gap_visualization_tsne(gap, gap_idx, topic_name, topic_id, denoiser, drop_k,
    #                            radius_fraction, num_books, num_articles, books, articles)

def write_gaps(cloud: Cloud, topic_name: str, topic_id: str, radius_fraction: float, num_gaps: int):
    """
    Analyze the top num_gaps gaps for a topic and write detailed information to files.
    
    Args:
        cloud: The main cloud object
        topic_name: Display name of the topic
        topic_id: ID of the topic
        radius_fraction: Radius fraction for gap analysis
        num_gaps: Number of top gaps to analyze and write files for
    """
    # Create subclouds for books and articles
    books = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_type=WorkType.BOOK.value))
    articles = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_type=WorkType.ARTICLE.value))
    
    denoiser = cloud.denoiser if cloud.denoiser is not None else "NONE"
    drop_k = cloud.drop_k
    num_books = len(books.embeddings())
    num_articles = len(articles.embeddings())
    
    LOGGER.info(f"Topic {topic_name}: {num_books} books, {num_articles} articles")
    
    if num_articles == 0:
        LOGGER.info(f"No articles found for topic {topic_id}, skipping gap analysis")
        for gap_idx in range(num_gaps):
            Results.add_row(topic_name, topic_id, denoiser, drop_k, gap_idx, num_books, num_articles, 0)
        return
    
    # Find the top gaps
    gap_generator = gaps(articles, books, radius_fraction=radius_fraction)
    for gap_idx in range(num_gaps):
        try:
            gap = next(gap_generator)
        except StopIteration:
            LOGGER.warning(f"No gap {gap_idx} found for topic {topic_name} ({topic_id}) - generator exhausted")
            # Add empty entries for remaining gaps
            for remaining_gap_idx in range(gap_idx, num_gaps):
                Results.add_row(topic_name, topic_id, denoiser, drop_k, remaining_gap_idx, num_books, num_articles, 0)
            break
        
        # Process this gap
        write_single_gap(gap, gap_idx, topic_name, topic_id, denoiser, drop_k, 
                        radius_fraction, num_books, num_articles, books, articles)

        # Generate LLM prompt for book proposal
        write_llm_prompt(gap, gap_idx, topic_name, topic_id, denoiser, drop_k, 
                        radius_fraction, num_books, num_articles, books, articles)

        # Plot the gap visualization
        plot_gap_visualization(gap, gap_idx, topic_name, topic_id, denoiser, drop_k, 
                              radius_fraction, num_books, num_articles, books, articles)





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
            num_gaps=1
        )


if __name__ == "__main__":
    setup_logging(log_name=Path(__file__).stem, explicit_args=get_explicit_args())
    main(**vars(parse_args()))
