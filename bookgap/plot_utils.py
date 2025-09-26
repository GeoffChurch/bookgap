"""
Centralized plotting utilities to eliminate matplotlib duplication patterns.

This module consolidates the repetitive matplotlib plotting operations found across:
- gap_finder.py: Gap visualizations, Q-Q plots  
- analyze_book_article_gap.py: Statistical plots, scatter plots
- data_sources.py: Distribution plots, histograms
- cloud.py: Singular value plots
"""

import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

from bookgap.visualization import VisualizationData, PlotConfig

# Centralized plotting constants (eliminates duplication across files)
DPI = 300
FIGURE_SIZE_LARGE = (12, 10)  # For main gap visualizations
FIGURE_SIZE_MEDIUM = (8, 6)   # For Q-Q plots and statistics  
FIGURE_SIZE_SMALL = (6, 4)    # For small plots
BBOX_INCHES_SETTING = 'tight'
GRID_ALPHA = 0.3

# Color constants for visualization
BOOK_COLOR = 'lightblue'
GAP_COLOR = 'orange'
PROPOSAL_COLOR = 'green'


class PlotManager:
    """
    Centralized matplotlib plot manager that eliminates duplicate patterns:
    - plt.subplots() + figsize configuration
    - plt.savefig() + dpi + bbox_inches configuration
    - plt.close() cleanup
    - Common plot styling and formatting
    """
    
    @staticmethod
    def create_figure(figsize: Tuple[float, float] = FIGURE_SIZE_MEDIUM) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """
        Create a matplotlib figure with standardized configuration.
        
        Args:
            figsize: Figure size tuple (width, height)
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    
    @staticmethod
    def create_subplots(nrows: int = 1, ncols: int = 1, 
                       figsize: Tuple[float, float] = FIGURE_SIZE_MEDIUM,
                       **subplot_kwargs) -> Tuple[matplotlib.figure.Figure, Union[matplotlib.axes.Axes, NDArray]]:
        """
        Create matplotlib subplots with standardized configuration.
        
        Args:
            nrows: Number of subplot rows
            ncols: Number of subplot columns  
            figsize: Figure size tuple (width, height)
            **subplot_kwargs: Additional arguments passed to plt.subplots()
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **subplot_kwargs)
        return fig, axes
    
    @staticmethod
    def save_and_close(fig: matplotlib.figure.Figure, file_path: Path, 
                      dpi: int = DPI, 
                      bbox_inches: str = BBOX_INCHES_SETTING,
                      **savefig_kwargs) -> None:
        """
        Save and close a matplotlib figure with standardized configuration.
        
        Args:
            fig: The matplotlib figure to save
            file_path: Path where to save the figure
            dpi: Dots per inch for the saved figure
            bbox_inches: Bounding box inches setting
            **savefig_kwargs: Additional arguments passed to fig.savefig()
            
        Raises:
            RuntimeError: If the figure has already been closed
        """
        # Check if figure is already closed
        if not plt.fignum_exists(fig.number):
            raise RuntimeError(f"Figure {fig.number} has already been closed and cannot be saved")
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with standardized settings
        fig.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches, **savefig_kwargs)
        plt.close(fig)
    
    @staticmethod
    def configure_axes(ax: matplotlib.axes.Axes, 
                      title: Optional[str] = None,
                      xlabel: Optional[str] = None, 
                      ylabel: Optional[str] = None,
                      grid: bool = True,
                      grid_alpha: float = GRID_ALPHA,
                      **style_kwargs) -> None:
        """
        Configure matplotlib axes with common styling patterns.
        
        Args:
            ax: The matplotlib axes to configure
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            grid: Whether to show grid
            grid_alpha: Grid transparency
            **style_kwargs: Additional styling arguments
        """
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if grid:
            ax.grid(True, alpha=grid_alpha)
    
    @staticmethod
    def create_and_save_plot(plot_function, file_path: Path, 
                           figsize: Tuple[float, float] = FIGURE_SIZE_MEDIUM,
                           dpi: int = DPI,
                           bbox_inches: str = BBOX_INCHES_SETTING,
                           **plot_kwargs) -> None:
        """
        Create, configure, save and close a plot in one call.
        
        Args:
            plot_function: Function that takes (fig, ax) and creates the plot
            file_path: Path where to save the figure
            figsize: Figure size tuple
            dpi: Dots per inch for saved figure
            bbox_inches: Bounding box setting
            **plot_kwargs: Arguments passed to plot_function
        """
        fig, ax = PlotManager.create_figure(figsize=figsize)
        plot_function(fig, ax, **plot_kwargs)
        PlotManager.save_and_close(fig, file_path, dpi=dpi, bbox_inches=bbox_inches)


class GapPlotUtils:
    """
    Specialized plotting utilities for gap analysis visualizations.
    Consolidates the gap plotting patterns from gap_finder.py.
    """
    
    @staticmethod
    def create_gap_scatter(ax: matplotlib.axes.Axes,
                          embeddings_2d: NDArray,
                          item_types: list[str],
                          all_items: list,
                          gap_positions: list[int],
                          book_positions: list[int],
                          center_position: Optional[int] = None,
                          proposals_2d: Optional[NDArray] = None) -> None:
        """
        Create standardized gap scatter plot on given axes.
        
        Args:
            ax: Matplotlib axes to plot on
            embeddings_2d: 2D embeddings array
            item_types: List of item types ('article' or 'book')
            all_items: List of (work_type, idx) tuples
            gap_positions: Positions of gap articles in embeddings array
            book_positions: Positions of books in embeddings array  
            center_position: Position of gap center article
            proposals_2d: Optional 2D embeddings for proposal books
        """
        n = len(embeddings_2d)
        if n == 0:
            return
        
        # Plot articles (non-gap)
        article_positions = [i for i, item_type in enumerate(item_types) if item_type == 'article']
        non_gap_articles = [p for p in article_positions if p not in gap_positions]
        if non_gap_articles:
            ax.scatter(embeddings_2d[non_gap_articles, 0], embeddings_2d[non_gap_articles, 1], 
                      c=GAP_COLOR, s=30, alpha=0.6, label='Articles', zorder=1)
        
        # Plot gap articles
        if gap_positions:
            ax.scatter(embeddings_2d[gap_positions, 0], embeddings_2d[gap_positions, 1],
                      c=GAP_COLOR, s=90, alpha=0.8, marker='o', label='Gap Articles', zorder=2)
        
        # Plot gap center
        if center_position is not None and center_position < n:
            ax.scatter(embeddings_2d[center_position, 0], embeddings_2d[center_position, 1],
                      marker='*', c=GAP_COLOR, s=120, edgecolors='k', linewidths=1.2, 
                      label='Gap Center', zorder=4)
        
        # Plot books
        if book_positions:
            ax.scatter(embeddings_2d[book_positions, 0], embeddings_2d[book_positions, 1],
                      c=BOOK_COLOR, s=80, marker='s', alpha=0.8, label='Books', zorder=3)
        
        # Plot proposals
        if proposals_2d is not None and len(proposals_2d) > 0:
            ax.scatter(proposals_2d[:, 0], proposals_2d[:, 1],
                      c=BOOK_COLOR, marker='*', s=120, edgecolors='black', linewidths=1.2,
                      label='Proposed Book', zorder=4)
    
    @staticmethod
    def configure_gap_plot_axes(ax: matplotlib.axes.Axes,
                               gap_idx: int,
                               topic_name: str,
                               method_name: str,
                               pc_variances: Optional[Tuple[float, float]] = None) -> None:
        """
        Configure axes for gap plots with standardized labels and formatting.
        
        Args:
            ax: Matplotlib axes to configure
            gap_idx: Gap index for title
            topic_name: Topic name for title
            method_name: Visualization method name (PCA, t-SNE, etc.)
            pc_variances: Optional PC variance percentages for PCA plots
        """
        ax.set_title(f"Gap {gap_idx} - {topic_name} ({method_name})")
        
        if method_name.upper() == 'PCA' and pc_variances is not None:
            ax.set_xlabel(f"PC1 ({pc_variances[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pc_variances[1]:.1%})")
        else:
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
        
        ax.grid(True, alpha=GRID_ALPHA)
        
        # Add legend if there are labeled elements
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()


class StatisticalPlotUtils:
    """
    Utilities for statistical plots like Q-Q plots, histograms, etc.
    Consolidates patterns from analyze_book_article_gap.py and data_sources.py.
    """
    
    @staticmethod
    def create_histogram(ax: matplotlib.axes.Axes,
                        data: NDArray,
                        bins: Union[int, str, np.ndarray] = 'auto',
                        title: Optional[str] = None,
                        xlabel: Optional[str] = None,
                        ylabel: str = 'Frequency',
                        log_scale: bool = False,
                        **hist_kwargs) -> None:
        """
        Create standardized histogram with common formatting.
        
        Args:
            ax: Matplotlib axes to plot on
            data: Data to histogram
            bins: Histogram bins specification
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            log_scale: Whether to use log scale for y-axis
            **hist_kwargs: Additional arguments passed to ax.hist()
        """
        ax.hist(data, bins=bins, **hist_kwargs)  # type: ignore
        
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.grid(True, alpha=GRID_ALPHA)
    
    @staticmethod
    def create_scatter_with_fit(ax: matplotlib.axes.Axes,
                               x: NDArray, y: NDArray,
                               fit_line: bool = True,
                               log_scale: bool = False,
                               title: Optional[str] = None,
                               xlabel: Optional[str] = None,
                               ylabel: Optional[str] = None,
                               **scatter_kwargs) -> None:
        """
        Create scatter plot with optional regression line.
        
        Args:
            ax: Matplotlib axes to plot on
            x: X-axis data
            y: Y-axis data
            fit_line: Whether to add regression line
            log_scale: Whether to use log scale for both axes
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            **scatter_kwargs: Additional arguments passed to ax.scatter()
        """
        ax.scatter(x, y, **scatter_kwargs)
        
        if fit_line and len(x) > 1:
            # Filter valid data for fitting
            mask = (x > 0) & (y > 0) if log_scale else np.isfinite(x) & np.isfinite(y)
            x_fit, y_fit = x[mask], y[mask]
            
            if len(x_fit) > 1:
                if log_scale:
                    # Log-log fit
                    log_x, log_y = np.log10(x_fit), np.log10(y_fit)
                    slope, intercept = np.polyfit(log_x, log_y, 1)
                    x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
                    y_line = 10 ** (slope * np.log10(x_line) + intercept)
                    label = f"log-log fit: y ~ x^{slope:.2f}"
                else:
                    # Linear fit
                    slope, intercept = np.polyfit(x_fit, y_fit, 1)
                    x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
                    y_line = slope * x_line + intercept
                    label = f"linear fit: y = {slope:.2f}x + {intercept:.2f}"
                
                ax.plot(x_line, y_line, 'r--', linewidth=2, label=label)
                ax.legend()
        
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        ax.grid(True, alpha=GRID_ALPHA)


# Modern unified plotting engine
class UnifiedGapPlotter:
    """
    Modern, unified plotting engine for all gap visualizations.
    
    This class replaces the scattered plotting logic across multiple files
    with a single, well-tested, configurable plotting system.
    """
    
    def __init__(self):
        """Initialize the plotter."""
        pass
    
    def plot_gap_visualization(
        self,
        data: VisualizationData,
        config: PlotConfig
    ) -> matplotlib.figure.Figure:
        """
        Create a gap visualization plot using standardized data structures.
        
        This replaces all the complex plotting logic in gap_plots.py and gap_finder.py
        with a single, clean, well-tested function.
        
        Args:
            data: Standardized visualization data
            config: Plot configuration
            
        Returns:
            Matplotlib figure ready for saving
        """
        fig, ax = PlotManager.create_figure(config.figsize)
        
        # Plot non-gap articles
        if data.non_gap_article_positions:
            ax.scatter(
                data.embeddings_2d[data.non_gap_article_positions, 0],
                data.embeddings_2d[data.non_gap_article_positions, 1],
                c=config.article_color,
                s=config.article_size,
                alpha=config.article_alpha,
                marker=config.article_marker,
                label='Articles',
                zorder=1
            )
        
        # Plot gap articles
        if data.gap_article_positions:
            ax.scatter(
                data.embeddings_2d[data.gap_article_positions, 0],
                data.embeddings_2d[data.gap_article_positions, 1],
                c=config.gap_article_color,
                s=config.gap_article_size,
                alpha=config.gap_article_alpha,
                marker=config.article_marker,
                label='Gap Articles',
                zorder=2
            )
        
        # Plot books
        if data.book_positions:
            ax.scatter(
                data.embeddings_2d[data.book_positions, 0],
                data.embeddings_2d[data.book_positions, 1],
                c=config.book_color,
                s=config.book_size,
                alpha=config.book_alpha,
                marker=config.book_marker,
                label='Books',
                zorder=3
            )
        
        # Plot gap center
        if data.center_position is not None:
            ax.scatter(
                data.embeddings_2d[data.center_position, 0],
                data.embeddings_2d[data.center_position, 1],
                c=config.center_color,
                s=config.center_size,
                alpha=config.center_alpha,
                marker=config.center_marker,
                edgecolors='black',
                linewidths=1.2,
                label='Gap Center',
                zorder=4
            )
        
        # Plot proposals
        if data.proposal_embeddings_2d is not None and len(data.proposal_embeddings_2d) > 0:
            ax.scatter(
                data.proposal_embeddings_2d[:, 0],
                data.proposal_embeddings_2d[:, 1],
                c=config.proposal_color,
                s=config.proposal_size,
                alpha=config.proposal_alpha,
                marker=config.proposal_marker,
                edgecolors='black',
                linewidths=1.2,
                label='Proposals',
                zorder=5
            )
        
        # Configure axes
        self._configure_axes(ax, data, config)
        
        return fig
    
    def _configure_axes(
        self,
        ax: matplotlib.axes.Axes,
        data: VisualizationData,
        config: PlotConfig
    ) -> None:
        """Configure axes with standardized styling."""
        # Set title
        if config.title:
            ax.set_title(config.title)
        elif config.topic_name and config.gap_idx is not None:
            ax.set_title(f"Gap {config.gap_idx} - {config.topic_name} ({config.method_name})")
        
        # Set axis labels based on method
        if config.method_name.upper() == 'PCA' and 'explained_variance_ratio' in data.method_info:
            var_ratios = data.method_info['explained_variance_ratio']
            if len(var_ratios) >= 2:
                ax.set_xlabel(f"PC1 ({var_ratios[0]:.1%})")
                ax.set_ylabel(f"PC2 ({var_ratios[1]:.1%})")
            else:
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
        else:
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
        
        # Configure grid
        if config.show_grid:
            ax.grid(True, alpha=config.grid_alpha)
        
        # Configure legend
        if config.show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
    
    def save_plot(
        self,
        fig: matplotlib.figure.Figure,
        config: PlotConfig,
        base_dir: Path
    ) -> Path:
        """
        Save plot with standardized configuration.
        
        Args:
            fig: Figure to save
            config: Plot configuration
            base_dir: Base directory for saving
            
        Returns:
            Path where plot was saved
        """
        save_path = config.build_save_path(base_dir)
        PlotManager.save_and_close(fig, save_path, dpi=config.dpi)
        return save_path
    
    def create_and_save_gap_plot(
        self,
        data: VisualizationData,
        config: PlotConfig,
        base_dir: Path
    ) -> Path:
        """
        Create and save a gap visualization plot in one call.
        
        This is the main entry point that replaces all the individual
        plotting functions with a single, unified interface.
        
        Args:
            data: Standardized visualization data
            config: Plot configuration
            base_dir: Base directory for saving
            
        Returns:
            Path where plot was saved
        """
        fig = self.plot_gap_visualization(data, config)
        return self.save_plot(fig, config, base_dir)


# Convenience functions that maintain backward compatibility
def create_figure(figsize: Tuple[float, float] = FIGURE_SIZE_MEDIUM) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Convenience function for creating standardized figures."""
    return PlotManager.create_figure(figsize)

def save_and_close_figure(fig: matplotlib.figure.Figure, file_path: Path) -> None:
    """Convenience function for saving and closing figures with standard settings."""
    PlotManager.save_and_close(fig, file_path)
