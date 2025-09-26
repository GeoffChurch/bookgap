"""
Visualization data structures for plotting components.

This module provides standardized data containers for visualization
that can be safely imported anywhere without circular dependencies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class VisualizationData:
    """
    Standardized data structure for ALL gap visualizations.
    
    Replaces the complex 7-tuple returns from _prepare_gap_visualization_data
    with a clean, typed, documented structure that's trivial to mock and test.
    """
    # Core embedding data
    embeddings_2d: NDArray                    # Shape: (n_items, 2) - 2D projected embeddings
    item_types: List[str]                     # Length: n_items - 'article' or 'book' for each item
    all_items: List[Tuple[str, int]]         # Length: n_items - (work_type, idx) for each item
    
    # Position mappings (indices into embeddings_2d array)
    gap_article_positions: List[int]          # Positions of gap articles
    book_positions: List[int]                 # Positions of books
    
    # Optional proposal data
    proposal_embeddings_2d: Optional[NDArray] = None  # Shape: (n_proposals, 2)
    proposal_labels: Optional[List[str]] = None       # Human-readable proposal names
    
    # Optional special positions
    center_position: Optional[int] = None     # Position of gap center article
    neighbor_positions: List[int] = field(default_factory=list)  # Neighboring book positions
    
    # Method-specific metadata
    method_info: Dict[str, Any] = field(default_factory=dict)  # PCA variance, t-SNE perplexity, etc.
    
    def __post_init__(self):
        """Validate data consistency."""
        n_items = len(self.embeddings_2d)
        
        # Validate lengths match
        if len(self.item_types) != n_items:
            raise ValueError(f"item_types length {len(self.item_types)} != embeddings length {n_items}")
        if len(self.all_items) != n_items:
            raise ValueError(f"all_items length {len(self.all_items)} != embeddings length {n_items}")
            
        # Validate position indices are in bounds
        all_positions = (self.gap_article_positions + self.book_positions + 
                        self.neighbor_positions + 
                        ([self.center_position] if self.center_position is not None else []))
        
        for pos in all_positions:
            if not (0 <= pos < n_items):
                raise ValueError(f"Position {pos} out of bounds for {n_items} items")
                
        # Validate proposal data consistency
        if self.proposal_embeddings_2d is not None:
            if self.proposal_labels is not None:
                if len(self.proposal_embeddings_2d) != len(self.proposal_labels):
                    raise ValueError("proposal_embeddings_2d and proposal_labels must have same length")

    @property
    def n_items(self) -> int:
        """Number of base items (articles + books)."""
        return len(self.embeddings_2d)
    
    @property
    def n_proposals(self) -> int:
        """Number of proposals."""
        return len(self.proposal_embeddings_2d) if self.proposal_embeddings_2d is not None else 0
    
    @property
    def article_positions(self) -> List[int]:
        """Positions of all articles (gap + non-gap)."""
        return [i for i, item_type in enumerate(self.item_types) if item_type == 'article']
    
    @property
    def non_gap_article_positions(self) -> List[int]:
        """Positions of non-gap articles."""
        return [pos for pos in self.article_positions if pos not in self.gap_article_positions]


@dataclass(frozen=True)
class PlotConfig:
    """
    Configuration for plot appearance and behavior.
    
    Centralizes all the magic constants and styling choices that are
    currently scattered across multiple plotting functions.
    """
    # Figure properties
    figsize: Tuple[float, float] = (12, 10)
    dpi: int = 300
    
    # Plot identification
    title: str = ""
    method_name: str = "PCA"  # "PCA", "t-SNE", etc.
    topic_name: str = ""
    gap_idx: Optional[int] = None
    
    # File output
    save_path: Optional[Path] = None
    file_format: str = "png"
    
    # Visual styling
    show_legend: bool = True
    show_grid: bool = True
    grid_alpha: float = 0.3
    
    # Colors (using existing plot_utils constants)
    article_color: str = "orange"
    gap_article_color: str = "orange" 
    book_color: str = "lightblue"
    proposal_color: str = "green"
    center_color: str = "red"
    
    # Marker properties
    article_marker: str = "o"
    book_marker: str = "s"
    proposal_marker: str = "*"
    center_marker: str = "*"
    
    # Sizes
    article_size: int = 30
    gap_article_size: int = 90
    book_size: int = 80
    proposal_size: int = 120
    center_size: int = 120
    
    # Transparency
    article_alpha: float = 0.6
    gap_article_alpha: float = 0.8
    book_alpha: float = 0.8
    proposal_alpha: float = 1.0
    center_alpha: float = 1.0
    
    def build_save_path(self, base_dir: Path) -> Path:
        """Build standardized save path from configuration."""
        if self.save_path is not None:
            return self.save_path
            
        # Build filename from components
        filename_parts = []
        if self.topic_name:
            filename_parts.append(self.topic_name.replace(" ", "_"))
        if self.gap_idx is not None:
            filename_parts.append(f"gap_{self.gap_idx}")
        filename_parts.append(self.method_name.lower())
        
        filename = "_".join(filename_parts) + f".{self.file_format}"
        return base_dir / filename


@dataclass(frozen=True)
class ProjectionMethod:
    """
    Configuration for dimensionality reduction methods.
    
    Encapsulates the method-specific parameters that are currently
    hardcoded in various places throughout the plotting code.
    """
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def pca(cls, n_components: int = 2) -> 'ProjectionMethod':
        """Standard PCA configuration."""
        return cls(
            name="PCA",
            params={"n_components": n_components}
        )
    
    @classmethod 
    def tsne(cls, n_components: int = 2, perplexity: float = 30.0, 
             random_state: int = 42) -> 'ProjectionMethod':
        """Standard t-SNE configuration."""
        return cls(
            name="t-SNE", 
            params={
                "n_components": n_components,
                "perplexity": perplexity,
                "random_state": random_state
            }
        )


# Type aliases for clarity
PositionMap = Dict[int, int]  # Maps original indices to projected array positions
NeighborData = Dict[str, Any]  # Flexible container for neighbor search results
