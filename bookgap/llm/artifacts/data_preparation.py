"""
Unified data preparation pipeline for gap visualizations.

This module consolidates the complex data preparation logic that's currently
duplicated across gap_plots.py and gap_finder.py into a single, testable,
well-documented pipeline.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from bookgap.cloud_index import knn
from bookgap.llm.data_structures import GapProtocol, GapWithContext
from bookgap.llm.protocols import ParsedProposal
from bookgap.subcloud import SubCloud
from bookgap.util import LOGGER

from bookgap.visualization import VisualizationData, ProjectionMethod, PositionMap


class GapVisualizationDataPreparer:
    """
    Unified data preparation pipeline for all gap visualizations.
    
    This class replaces the complex, duplicated logic in multiple plotting functions
    with a single, well-tested, configurable pipeline.
    """
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the data preparer.
        
        Args:
            embedding_model: OpenAI embedding model to use for proposals
        """
        self.embedding_model = embedding_model
    
    def prepare_standard_gap_data(
        self,
        gap_with_context: GapWithContext,
        proposals: Optional[List[ParsedProposal]] = None,
        method: ProjectionMethod = ProjectionMethod.pca()
    ) -> VisualizationData:
        """
        Prepare data for standard gap visualization with optional proposal overlay.
        
        This replaces the complex _prepare_gap_visualization_data function
        with a clean, well-documented interface.
        
        Args:
            gap_with_context: Gap with associated context data
            proposals: Optional list of proposals to overlay
            method: Dimensionality reduction method to use
            
        Returns:
            VisualizationData ready for plotting
            
        Raises:
            ValueError: If insufficient data for visualization
        """
        # Compute proposal embeddings if provided
        proposal_embeddings = None
        proposal_labels = None
        if proposals:
            proposal_embeddings = [p.embedding for p in proposals]
            proposal_labels = [f"{p.proposal.title[:50]}..." if len(p.proposal.title) > 50 else p.proposal.title 
                             for p in proposals]
        
        # Prepare base embeddings and metadata
        base_data = self._prepare_base_data(gap_with_context.gap, gap_with_context.books, gap_with_context.articles)
        if not base_data:
            raise ValueError("Insufficient data for visualization")
            
        base_embeddings, item_types, all_items, gap_positions, book_positions = base_data
        
        # Perform dimensionality reduction
        embeddings_2d, method_info = self._project_to_2d(
            base_embeddings, method, proposal_embeddings
        )
        
        # Project proposals if provided
        proposal_embeddings_2d = None
        if proposal_embeddings is not None:
            # Proposals are projected with the same transformation
            proposal_embeddings_2d = self._project_proposals(
                proposal_embeddings, method, base_embeddings
            )
        
        # Find gap center position
        center_position = self._find_gap_center_position(gap_with_context.gap, gap_positions, all_items)
        
        return VisualizationData(
            embeddings_2d=embeddings_2d,
            item_types=item_types,
            all_items=all_items,
            gap_article_positions=gap_positions,
            book_positions=book_positions,
            proposal_embeddings_2d=proposal_embeddings_2d,
            proposal_labels=proposal_labels,
            center_position=center_position,
            method_info=method_info
        )
    
    def prepare_books_vs_proposals_data(
        self,
        gap: GapProtocol,
        books: SubCloud,
        articles: SubCloud,
        proposals: List[ParsedProposal],
        method: ProjectionMethod = ProjectionMethod.pca()
    ) -> VisualizationData:
        """
        Prepare data for books vs proposals comparison visualization.
        
        This implements the logic from save_gap_books_vs_proposals_plot
        with a clean, testable interface.
        
        Args:
            gap: Gap to visualize
            books: Book cloud
            articles: Article cloud  
            proposals: List of proposals (required)
            method: Dimensionality reduction method
            
        Returns:
            VisualizationData for books vs proposals plot
            
        Raises:
            ValueError: If proposals are empty or insufficient data
        """
        if not proposals:
            raise ValueError("Proposals are required for books vs proposals plot")
            
        LOGGER.info(f"Preparing books vs proposals data for {len(proposals)} proposals")
        
        # Get gap and covering book indices
        gap_article_indices = gap.get_dists_idxs().idxs
        covering_book_indices = self._get_covering_book_indices(gap)
        
        if len(gap_article_indices) == 0:
            raise ValueError("No gap articles found")
            
        # Build embeddings and metadata
        all_article_embeddings = articles.embeddings()
        all_book_embeddings = books.embeddings()
        
        # Extract gap articles and covering books
        article_vecs = [all_article_embeddings[int(idx)] for idx in gap_article_indices]
        book_vecs = ([all_book_embeddings[int(idx)] for idx in covering_book_indices] 
                    if covering_book_indices else [])
        
        # Build combined embeddings
        base_embeddings = np.vstack(article_vecs + book_vecs) if book_vecs else np.vstack(article_vecs)
        
        # Build metadata
        item_types = (['article'] * len(gap_article_indices) + 
                     ['book'] * len(covering_book_indices))
        all_items = ([(('article', int(idx))) for idx in gap_article_indices] +
                    [('book', int(idx)) for idx in covering_book_indices])
        
        # Compute proposal embeddings
        proposal_embeddings = [p.embedding for p in proposals]
        proposal_labels = [p.proposal.title for p in proposals]
        
        # Project to 2D
        embeddings_2d, method_info = self._project_to_2d(
            base_embeddings, method, proposal_embeddings
        )
        
        proposal_embeddings_2d = self._project_proposals(
            proposal_embeddings, method, base_embeddings
        )
        
        # Position mappings
        gap_positions = list(range(len(gap_article_indices)))
        book_positions = list(range(len(gap_article_indices), len(all_items)))
        
        return VisualizationData(
            embeddings_2d=embeddings_2d,
            item_types=item_types,
            all_items=all_items,
            gap_article_positions=gap_positions,
            book_positions=book_positions,
            proposal_embeddings_2d=proposal_embeddings_2d,
            proposal_labels=proposal_labels,
            method_info=method_info
        )
    
    def _prepare_base_data(
        self, 
        gap: GapProtocol, 
        books: SubCloud, 
        articles: SubCloud
    ) -> Optional[Tuple[np.ndarray, List[str], List[Tuple[str, int]], List[int], List[int]]]:
        """
        Prepare base embeddings and metadata for visualization.
        
        This consolidates the complex data preparation logic from the original functions.
        """
        # Get gap article indices (ordered by increasing distance from center)
        gap_article_indices = gap.get_dists_idxs().idxs
        if len(gap_article_indices) == 0:
            LOGGER.warning("No gap articles found")
            return None
            
        # Get neighbor data (4-way FAISS searches) and covering books
        neighbor_data = self._perform_gap_neighbor_searches(gap, books, articles)
        covering_book_indices = self._get_covering_book_indices(gap)
            
        # Extract embeddings
        all_article_embeddings = articles.embeddings()
        all_book_embeddings = books.embeddings()
        
        # Build the combined embedding matrix and metadata
        embeddings_list = []
        item_types = []
        all_items = []
        gap_positions = []
        book_positions = []
        current_pos = 0
        
        # Add gap articles first (preserve order so center is first)
        for idx in gap_article_indices:
            embeddings_list.append(all_article_embeddings[int(idx)])
            item_types.append('article')
            all_items.append(('article', int(idx)))
            gap_positions.append(current_pos)
            current_pos += 1
            
        # Determine article neighbors (non-gap)
        neighbor_article_indices: List[int] = []
        for key in ('gap_to_article_neighbors', 'book_to_article_neighbors'):
            if key in neighbor_data and neighbor_data[key] is not None:
                neighbor_article_indices.extend(int(i) for i in neighbor_data[key])
        # Unique and exclude gap articles
        gap_set = set(int(i) for i in gap_article_indices.tolist())
        neighbor_article_indices = [i for i in sorted(set(neighbor_article_indices)) if i not in gap_set]
        
        # Add neighbor articles (classified implicitly as non-gap articles)
        for idx in neighbor_article_indices:
            embeddings_list.append(all_article_embeddings[int(idx)])
            item_types.append('article')
            all_items.append(('article', int(idx)))
            current_pos += 1
        
        # Determine book indices to include: covering books + neighbor books
        neighbor_book_indices: List[int] = []
        for key in ('gap_to_book_neighbors', 'book_to_book_neighbors'):
            if key in neighbor_data and neighbor_data[key] is not None:
                neighbor_book_indices.extend(int(i) for i in neighbor_data[key])
        base_book_indices = sorted(set(int(i) for i in covering_book_indices))
        neighbor_book_indices = [i for i in sorted(set(neighbor_book_indices)) if i not in set(base_book_indices)]
        
        # Add covering books first
        for book_idx in base_book_indices:
            if 0 <= int(book_idx) < len(all_book_embeddings):
                embeddings_list.append(all_book_embeddings[int(book_idx)])
                item_types.append('book')
                all_items.append(('book', int(book_idx)))
                book_positions.append(current_pos)
                current_pos += 1
        # Add neighbor books
        for book_idx in neighbor_book_indices:
            if 0 <= int(book_idx) < len(all_book_embeddings):
                embeddings_list.append(all_book_embeddings[int(book_idx)])
                item_types.append('book')
                all_items.append(('book', int(book_idx)))
                book_positions.append(current_pos)
                current_pos += 1
        
        if len(embeddings_list) < 2:
            LOGGER.warning("Insufficient items for visualization")
            return None
            
        base_embeddings = np.vstack(embeddings_list)
        
        return base_embeddings, item_types, all_items, gap_positions, book_positions
    
    def _project_to_2d(
        self,
        embeddings: np.ndarray,
        method: ProjectionMethod,
        proposal_embeddings: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Project high-dimensional embeddings to 2D using specified method.
        
        Args:
            embeddings: Base embeddings to project
            method: Projection method configuration
            proposal_embeddings: Optional proposal embeddings to include in fit
            
        Returns:
            Tuple of (projected_embeddings, method_info_dict)
        """
        # Combine base and proposal embeddings for fitting if proposals provided
        if proposal_embeddings is not None:
            proposal_array = np.vstack([np.array(pe) for pe in proposal_embeddings])
            fit_embeddings = np.vstack([embeddings, proposal_array])
        else:
            fit_embeddings = embeddings
            
        if method.name.upper() == "PCA":
            projector = PCA(**method.params)
            projector.fit(fit_embeddings)
            embeddings_2d = projector.transform(embeddings)
            
            # Return variance information
            method_info = {
                "explained_variance_ratio": projector.explained_variance_ratio_.tolist(),
                "total_variance": float(np.sum(projector.explained_variance_ratio_))
            }
            
        elif method.name.upper() == "T-SNE":
            # Adjust perplexity for small datasets - it must be less than n_samples
            params = method.params.copy()
            n_samples = embeddings.shape[0]
            original_perplexity = params.get("perplexity", 30.0)
            
            if original_perplexity >= n_samples:
                # Use a reasonable fraction of the sample size, minimum 1
                adjusted_perplexity = max(1.0, min(n_samples - 1, n_samples // 3))
                params["perplexity"] = adjusted_perplexity
                LOGGER.info(f"Adjusted t-SNE perplexity from {original_perplexity} to {adjusted_perplexity} for {n_samples} samples")
                
            projector = TSNE(**params)
            # t-SNE requires fitting on the data we want to transform
            embeddings_2d = projector.fit_transform(embeddings)
            method_info = {
                "perplexity": params.get("perplexity", 30.0),
                "original_perplexity": original_perplexity,
                "n_samples": n_samples
            }
            
        else:
            raise ValueError(f"Unsupported projection method: {method.name}")
            
        return embeddings_2d, method_info
    
    def _project_proposals(
        self,
        proposal_embeddings: List[np.ndarray],
        method: ProjectionMethod,
        base_embeddings: np.ndarray
    ) -> np.ndarray:
        """Project proposal embeddings using the same transformation as base data."""
        if method.name.upper() == "PCA":
            # For PCA, we can transform new data using the fitted model
            proposal_array = np.vstack([np.array(pe) for pe in proposal_embeddings])
            combined = np.vstack([base_embeddings, proposal_array])
            
            projector = PCA(**method.params)
            all_projected = projector.fit_transform(combined)
            
            # Return only the proposal projections
            return all_projected[len(base_embeddings):]
            
        elif method.name.upper() == "T-SNE":
            # For t-SNE, we need to include proposals in the original fit
            proposal_array = np.vstack([np.array(pe) for pe in proposal_embeddings])
            combined = np.vstack([base_embeddings, proposal_array])
            
            # Adjust perplexity for small datasets - it must be less than n_samples
            params = method.params.copy()
            n_samples = combined.shape[0]
            original_perplexity = params.get("perplexity", 30.0)
            
            if original_perplexity >= n_samples:
                # Use a reasonable fraction of the sample size, minimum 1
                adjusted_perplexity = max(1.0, min(n_samples - 1, n_samples // 3))
                params["perplexity"] = adjusted_perplexity
                LOGGER.info(f"Adjusted t-SNE perplexity from {original_perplexity} to {adjusted_perplexity} for {n_samples} samples (in proposal projection)")
                
            projector = TSNE(**params)
            all_projected = projector.fit_transform(combined)
            
            # Return only the proposal projections  
            return all_projected[len(base_embeddings):]
            
        else:
            raise ValueError(f"Unsupported projection method: {method.name}")
    
    def _perform_gap_neighbor_searches(
        self, 
        gap: GapProtocol, 
        books: SubCloud, 
        articles: SubCloud, 
        k: int = 5
    ) -> Dict[str, Any]:
        """Perform the four FAISS neighbor searches for gap visualization.

        Returns a dict with sets of neighbor indices and detailed search arrays
        for efficient lookups, mirroring gap_finder._perform_gap_neighbor_searches.
        """
        gap_article_indices = gap.get_dists_idxs().idxs
        result: Dict[str, Any] = {}
        # Match original gap_finder defaults: skip these two searches for now, but preserve API.
        SKIP_GAP_TO_ARTICLE_NEIGHBORS = True
        SKIP_BOOK_TO_ARTICLE_NEIGHBORS = True

        # 1) Nearest articles to in-gap articles (exclude self-neighbor)
        gap_to_article_neighbors: set[int] = set()
        if not SKIP_GAP_TO_ARTICLE_NEIGHBORS:
            if len(gap_article_indices) > 0 and len(articles.embeddings()) > 0:
                res_articles = knn(articles, articles, k=k+1)  # +1 to include self
                # Take neighbors for the specific gap articles, excluding the first column (self)
                gap_article_neighbors = res_articles.I[gap_article_indices]
                if gap_article_neighbors.ndim == 2 and gap_article_neighbors.shape[1] >= 2:
                    gap_to_article_neighbors.update(gap_article_neighbors[:, 1:].flatten().tolist())
                else:
                    gap_to_article_neighbors.update(gap_article_neighbors.flatten().tolist())
        result["gap_to_article_neighbors"] = gap_to_article_neighbors

        # 2) Nearest books to in-gap articles (keep detailed per-article results)
        gap_to_book_neighbors: set[int] = set()
        gap_article_to_book_search = None
        if len(gap_article_indices) > 0 and len(books.embeddings()) > 0:
            res_books_for_articles = knn(books, articles, k=k)
            gap_book_neighbors = res_books_for_articles.I[gap_article_indices]
            gap_to_book_neighbors.update(gap_book_neighbors.flatten().tolist())
            gap_article_to_book_search = gap_book_neighbors
        result["gap_to_book_neighbors"] = gap_to_book_neighbors
        result["gap_article_to_book_search"] = gap_article_to_book_search

        # 3) Nearest articles to covering books (keep detailed per-book results)
        book_to_article_neighbors: set[int] = set()
        book_to_article_search = None
        covering_book_indices = self._get_covering_book_indices(gap)
        if not SKIP_BOOK_TO_ARTICLE_NEIGHBORS:
            if len(covering_book_indices) > 0 and len(articles.embeddings()) > 0:
                res_articles_for_books = knn(articles, books, k=k)
                book_article_neighbors = res_articles_for_books.I[covering_book_indices]
                book_to_article_neighbors.update(book_article_neighbors.flatten().tolist())
                book_to_article_search = book_article_neighbors
        result["book_to_article_neighbors"] = book_to_article_neighbors
        result["book_to_article_search"] = book_to_article_search
        result["covering_book_indices"] = covering_book_indices

        # 4) Nearest books to covering books (exclude self-neighbor)
        book_to_book_neighbors: set[int] = set()
        if len(covering_book_indices) > 0 and len(books.embeddings()) > 0:
            res_books_for_books = knn(books, books, k=k+1)  # +1 to include self
            neighbors = res_books_for_books.I[covering_book_indices]
            if neighbors.ndim == 2 and neighbors.shape[1] >= 2:
                book_to_book_neighbors.update(neighbors[:, 1:].flatten().tolist())
            else:
                book_to_book_neighbors.update(neighbors.flatten().tolist())
        result["book_to_book_neighbors"] = book_to_book_neighbors
        result["k"] = k
        return result
    
    def _get_covering_book_indices(self, gap: GapProtocol) -> List[int]:
        """Get indices of books that cover the gap articles using gap API."""
        try:
            return [int(i) for i in list(gap.get_covering_book_indices())]
        except Exception:
            return []
    
    def _find_gap_center_position(
        self,
        gap: GapProtocol,
        gap_positions: List[int],
        all_items: List[Tuple[str, int]]
    ) -> Optional[int]:
        """Find the position of the gap center article in the visualization."""
        # By construction we add gap articles first in the same order as
        # gap.get_dists_idxs().idxs, which has the center first.
        return gap_positions[0] if gap_positions else None
