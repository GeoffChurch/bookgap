"""Gap visualization plots for BookGap analysis.

This module provides functions for generating gap plots with proposals,
porting functionality from the original gap_finder.py with hardcoded settings.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict, Any
import numpy as np
from numpy.typing import NDArray
from hydra.core.hydra_config import HydraConfig
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.axes

from bookgap.llm.data_structures import GapWithContext
from bookgap.llm.protocols import ParsedProposal
from bookgap.gap_types import DistsAndIdxs
from bookgap.gap_finder import GapProtocol
from bookgap.cloud import SubCloud, get_embedding_model
from bookgap.cloud_index import knn
from bookgap.plot_utils import (
    PlotManager, FIGURE_SIZE_LARGE, GRID_ALPHA,
    BOOK_COLOR, GAP_COLOR
)
from bookgap.util import LOGGER, create_unicode_safe_filename

# Hardcoded plotting settings
FIGURE_SIZE = (12, 10)
DPI = 300
FORMAT = "png"
ALPHA_GRID = 0.3
NEAREST_NEIGHBOR_K = 5


def _compute_proposal_embeddings(proposals: List[ParsedProposal]) -> List[np.ndarray]:
    """Compute embeddings for proposals by encoding title + abstract.
    
    Args:
        proposals: List of parsed proposals
        
    Returns:
        List of embedding vectors for the proposals
    """
    if not proposals:
        return []
    
    embedding_model = get_embedding_model()
    embeddings = []
    
    for proposal in proposals:
        try:
            # Combine title and abstract as done in batch_operations.py
            title = proposal.proposal.title if proposal.proposal.title else ""
            abstract = proposal.proposal.abstract if proposal.proposal.abstract else ""
            text = f"{title} {abstract}".strip()
            
            if text:
                vec = embedding_model.encode(text)
                embeddings.append(vec)
                LOGGER.debug(f"Computed embedding for proposal: {title[:50]}...")
            else:
                LOGGER.warning(f"Empty proposal text for {proposal.custom_id}")
        except Exception as e:
            LOGGER.error(f"Failed to compute embedding for proposal {proposal.custom_id}: {e}")
            continue
    
    LOGGER.info(f"Computed embeddings for {len(embeddings)}/{len(proposals)} proposals")
    return [np.array(emb) for emb in embeddings]  # Convert to numpy arrays


def save_gap_plot_with_proposals(
    gap_with_context: GapWithContext,
    proposals: List[ParsedProposal],
    method: str = "PCA"
) -> Path:
    """Save gap visualization plot with proposals overlay.
    
    Args:
        gap_with_context: Gap with associated context data
        proposals: List of parsed proposals to overlay on plot
        method: Projection method ("PCA" or "TSNE")
        
    Returns:
        Path to saved plot file
        
    Raises:
        ValueError: If insufficient data for plotting
    """
    # Extract data from gap_with_context
    gap = gap_with_context.gap
    gap_idx = gap_with_context.gap_idx
    topic_name = gap_with_context.topic_name
    topic_id = gap_with_context.topic_id
    books = gap_with_context.books
    articles = gap_with_context.articles
    
    # Compute proposal embeddings from title + abstract
    LOGGER.info(f"Computing embeddings for {len(proposals)} proposals for gap {gap_idx}")
    proposal_embeddings = _compute_proposal_embeddings(proposals)
    
    # Prepare visualization data with computed proposal embeddings
    num_books = len(books.embeddings())
    data = _prepare_gap_visualization_data(
        gap, gap_idx, num_books, books, articles, 
        proposal_embeddings=proposal_embeddings
    )
    
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
    
    # Ensure base_embeddings is not None
    if base_embeddings is None:
        raise ValueError("Base embeddings cannot be None")
    
    # Compute projection
    book_to_articles_mapping = _get_book_to_articles_mapping(gap)
    embeddings_2d_all, pc_variances, transform_fn = _compute_base_projection(
        method, base_embeddings, item_types, all_items,
        book_to_articles_mapping, neighbor_data, gap, books, articles,
        proposal_embeddings=proposal_embeddings_arr
    )
    
    # Split projections into base items and proposals
    base_n = len(base_embeddings)
    if embeddings_2d_all.shape[0] < base_n:
        embeddings_2d = embeddings_2d_all
        proposals_2d = None
    else:
        embeddings_2d = embeddings_2d_all[:base_n]
        proposals_2d = embeddings_2d_all[base_n:] if embeddings_2d_all.shape[0] > base_n else None
    
    # Create plot
    fig, ax = PlotManager.create_figure(figsize=FIGURE_SIZE)
    _create_gap_plot(
        embeddings_2d,
        item_types[:len(embeddings_2d)],
        all_items[:len(embeddings_2d)],
        gap_article_positions_map,
        all_book_positions_map,
        None,  # No covered_points_analysis needed
        neighbor_data,
        gap,
        topic_name,
        gap_idx,
        method.upper(),
        pc_variances,
        ax=ax,
        proposals_2d=proposals_2d,
    )
    
    # Save to Hydra output directory
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_file = plots_dir / f"{create_unicode_safe_filename(topic_name)}_gap_{gap_idx}_{method.lower()}_visualization_with_proposals.png"
    PlotManager.save_and_close(fig, plot_file)
    LOGGER.info(f"{method.upper()} gap visualization with proposals saved to {plot_file}")
    
    return plot_file


def _prepare_gap_visualization_data(
    gap: GapProtocol, 
    gap_idx: int, 
    num_books: int, 
    books: SubCloud, 
    articles: SubCloud, 
    proposal_embeddings: Optional[list] = None
):
    """Prepare data for gap visualization (ported from gap_finder.py)."""
    # Port the logic from prepare_gap_visualization_data
    # This is a complex function that prepares embeddings, mappings, etc.
    
    # Get gap articles and neighbor searches
    gap_article_indices = gap.get_dists_idxs().idxs
    neighbor_data = _perform_gap_neighbor_searches(gap, books, articles, k=NEAREST_NEIGHBOR_K)
    
    # Sample articles and books for visualization
    all_article_embeddings = articles.embeddings()
    all_book_embeddings = books.embeddings()
    
    # Build base embeddings list
    base_embeddings = []
    item_types = []
    all_items = []
    gap_article_positions_map = {}
    all_book_positions_map = {}
    
    # Add gap articles first
    for i, article_idx in enumerate(gap_article_indices):
        base_embeddings.append(all_article_embeddings[int(article_idx)])
        item_types.append('article')
        all_items.append(('article', int(article_idx)))
        gap_article_positions_map[int(article_idx)] = i
    
    # Add some neighboring books (simplified sampling)
    # For now, just add first few books
    max_books_to_show = min(20, num_books)
    for book_idx in range(max_books_to_show):
        if book_idx < len(all_book_embeddings):
            base_embeddings.append(all_book_embeddings[book_idx])
            item_types.append('book')
            all_items.append(('book', book_idx))
            all_book_positions_map[book_idx] = len(base_embeddings) - 1
    
    # Add some neighboring articles (simplified sampling)
    # Sample some articles around the gap
    max_articles_to_show = min(50, len(all_article_embeddings))
    article_sample = np.random.choice(
        len(all_article_embeddings), 
        size=min(max_articles_to_show, len(all_article_embeddings)), 
        replace=False
    )
    
    for article_idx in article_sample:
        if article_idx not in gap_article_positions_map:  # Don't duplicate gap articles
            base_embeddings.append(all_article_embeddings[article_idx])
            item_types.append('article')
            all_items.append(('article', int(article_idx)))
    
    # Handle proposal embeddings
    proposal_embeddings_arr = None
    if proposal_embeddings and len(proposal_embeddings) > 0:
        proposal_embeddings_arr = np.vstack(proposal_embeddings)
    
    if not base_embeddings:
        LOGGER.error("No base embeddings available for plotting")
        return None
        
    base_embeddings_array = np.vstack(base_embeddings)
    
    return (
        base_embeddings_array,
        item_types,
        all_items,
        gap_article_positions_map,
        all_book_positions_map,
        neighbor_data,
        proposal_embeddings_arr,
    )


def _perform_gap_neighbor_searches(gap: GapProtocol, books: SubCloud, articles: SubCloud, k: int = 5):
    """Perform neighbor searches for gap visualization."""
    # Simplified version - just return empty dict for now
    # In full implementation, this would do KNN searches
    return {}


def _get_book_to_articles_mapping(gap: GapProtocol):
    """Get mapping of books to articles for the gap."""
    # Simplified version - return empty dict for now
    return {}


def _compute_base_projection(
    method: str,
    base_embeddings: NDArray,
    item_types: list,
    all_items: list,
    book_to_articles_mapping: dict,
    neighbor_data: dict,
    gap: GapProtocol,
    books: SubCloud,
    articles: SubCloud,
    proposal_embeddings: Optional[NDArray] = None
) -> Tuple[NDArray, Optional[Tuple[float, float]], Optional[Callable]]:
    """Compute 2D projection of embeddings using PCA or t-SNE."""
    
    # Combine base embeddings with proposal embeddings if available
    if proposal_embeddings is not None:
        all_embeddings = np.vstack([base_embeddings, proposal_embeddings])
    else:
        all_embeddings = base_embeddings
    
    # Apply dimensionality reduction
    if method.upper() == "PCA":
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(all_embeddings)
        
        # Calculate explained variance
        pc_variances = (
            reducer.explained_variance_ratio_[0],
            reducer.explained_variance_ratio_[1]
        )
        transform_fn = reducer.transform
        
    elif method.upper() == "TSNE":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings) - 1))
        embeddings_2d = reducer.fit_transform(all_embeddings)
        pc_variances = None  # t-SNE doesn't have explained variance
        transform_fn = None  # t-SNE doesn't have a transform method
        
    else:
        raise ValueError(f"Unknown projection method: {method}")
    
    return embeddings_2d, pc_variances, transform_fn


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
    pc_variances: Optional[Tuple[float, float]],
    ax: matplotlib.axes.Axes,
    proposals_2d: Optional[NDArray] = None,
) -> None:
    """Create gap plot with articles, books, and proposals."""
    
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
    
    # Plot non-gap articles
    article_idx = [i for i, v in enumerate(is_article) if v]
    non_gap_article_idx = [i for i in article_idx if i not in gap_pos_set]
    if len(non_gap_article_idx) > 0:
        ax.scatter(emb[non_gap_article_idx, 0], emb[non_gap_article_idx, 1], 
                  c=GAP_COLOR, s=30, alpha=0.6, label='Articles', zorder=1)
    
    # Highlight gap articles
    if len(gap_positions) > 0:
        ax.scatter(emb[gap_positions, 0], emb[gap_positions, 1], 
                  c='red', s=60, alpha=0.8, label='Gap Articles', zorder=3)
    
    # Plot books
    book_idx = [i for i, v in enumerate(is_book) if v]
    if len(book_idx) > 0:
        ax.scatter(emb[book_idx, 0], emb[book_idx, 1], 
                  c=BOOK_COLOR, s=80, marker='s', alpha=0.7, label='Books', zorder=2)
    
    # Plot proposals if available
    if proposals_2d is not None and len(proposals_2d) > 0:
        ax.scatter(proposals_2d[:, 0], proposals_2d[:, 1], 
                  c='purple', s=100, marker='^', alpha=0.9, 
                  label='Proposals', zorder=4, edgecolors='black', linewidths=1)
    
    # Set title and labels
    title = f"Gap {gap_idx} - {topic_name} ({method_name})"
    if proposals_2d is not None and len(proposals_2d) > 0:
        title += f" with {len(proposals_2d)} Proposals"
    ax.set_title(title)
    
    # Set axis labels with variance if available
    if method_name == "PCA" and pc_variances is not None:
        ax.set_xlabel(f"PC1 ({pc_variances[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pc_variances[1]:.1%})")
    else:
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
    
    ax.grid(True, alpha=ALPHA_GRID)
    ax.legend()


def generate_gap_plots_for_results(
    gaps: List[GapWithContext], 
    results: List[ParsedProposal],
    output_dir: Path
) -> List[Path]:
    """Generate gap plots for all gaps with their corresponding proposals.
    
    Args:
        gaps: List of gaps with context
        results: List of all parsed proposals
        output_dir: Base output directory (Hydra managed)
        
    Returns:
        List of paths to generated plot files
    """
    plot_paths = []
    
    # Group proposals by gap index (extract from custom_id)
    proposals_by_gap: Dict[int, List[ParsedProposal]] = {}
    for proposal in results:
        # Extract gap index from custom_id (assuming format like "gap_0", "gap_1", etc.)
        try:
            if "_" in proposal.custom_id:
                gap_idx = int(proposal.custom_id.split("_")[1])
                if gap_idx not in proposals_by_gap:
                    proposals_by_gap[gap_idx] = []
                proposals_by_gap[gap_idx].append(proposal)
        except (ValueError, IndexError) as e:
            LOGGER.warning(f"Could not parse gap index from custom_id {proposal.custom_id}: {e}")
            continue
    
    # Generate plots for each gap
    for gap_with_context in gaps:
        gap_idx = gap_with_context.gap_idx
        gap_proposals = proposals_by_gap.get(gap_idx, [])
        
        try:
            plot_path = save_gap_plot_with_proposals(
                gap_with_context, 
                gap_proposals,
                method="PCA"  # Hardcoded to PCA for now
            )
            plot_paths.append(plot_path)
            
        except Exception as e:
            LOGGER.error(f"Failed to generate plot for gap {gap_idx}: {e}")
            continue
    
    LOGGER.info(f"Generated {len(plot_paths)} gap plots with proposals")
    return plot_paths
