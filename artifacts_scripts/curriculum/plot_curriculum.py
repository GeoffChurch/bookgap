#!/usr/bin/env python3
"""Visualize curriculum by projecting topic centroids to 2D and drawing the sequence.

This module provides functions to generate 2D visualizations showing:
- Topic centroids projected via PCA
- Arrows connecting topics in linearization order
- Optional labels with topic keywords

Use via the main.py unified entry point or import functions directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

from bookgap.util import LOGGER

from .dimensionality_reduction import DimensionalityReductionMethod, project_to_2d
from .summarize_topics import load_curriculum_outputs
from .sampling import sample_by_relevance


def compute_topic_centroids(
    embeddings: NDArray[np.floating],
    frontier_nodes: NDArray[np.int64],
    leaf_to_cluster: NDArray[np.int64],
) -> NDArray[np.floating]:
    """Compute centroids for each frontier topic.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim).
        frontier_nodes: Array of frontier node indices.
        leaf_to_cluster: Array mapping leaf indices to cluster IDs.
        
    Returns:
        Array of shape (n_topics, embedding_dim) with topic centroids.
    """
    n_topics = len(frontier_nodes)
    embedding_dim = embeddings.shape[1]
    
    centroids = np.zeros((n_topics, embedding_dim), dtype=np.float64)
    
    for topic_idx in range(n_topics):
        topic_mask = leaf_to_cluster == topic_idx
        if topic_mask.sum() > 0:
            centroids[topic_idx] = embeddings[topic_mask].mean(axis=0)
    
    return centroids


def load_curriculum_data(
    curriculum_dir: Path,
    embeddings: NDArray[np.floating],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.floating], NDArray[np.int64]]:
    """Load curriculum outputs and prepare data for visualization.
    
    Args:
        curriculum_dir: Path to directory containing curriculum outputs.
        embeddings: Array of shape (n_samples, embedding_dim).
        
    Returns:
        Tuple of (frontier_nodes, leaf_to_cluster, centroids, linearization_order).
    """
    LOGGER.info(f"Loading curriculum outputs from {curriculum_dir}...")
    outputs = load_curriculum_outputs(curriculum_dir)
    
    n_topics = len(outputs.frontier_nodes)
    LOGGER.info(f"Loaded curriculum with {n_topics} topics")
    
    LOGGER.info("Computing topic centroids...")
    centroids = compute_topic_centroids(embeddings, outputs.frontier_nodes, outputs.point_to_cluster)
    
    if outputs.metadata is None:
        raise ValueError("metadata not found in curriculum outputs")
    linearization_order = np.arange(len(outputs.frontier_nodes))
    
    return outputs.frontier_nodes, outputs.point_to_cluster, centroids, linearization_order


def draw_curriculum_arrows(
    ax: Axes,
    centroids_2d: NDArray[np.floating],
    linearization_order: NDArray[np.int64],
    reference_range: NDArray[np.floating],
    arrow_color: str = "black",
    arrow_lw: float = 1.5,
    arrow_rad: float = 0.1,
    margin_fraction: float = 0.03,
    zorder: int = 10,
) -> None:
    """Draw arrows connecting topics in linearization order.
    
    Args:
        ax: Matplotlib axes to draw on.
        centroids_2d: 2D projected centroids.
        linearization_order: Order in which to connect topics.
        reference_range: Data range for margin calculation.
        arrow_color: Color of arrows.
        arrow_lw: Line width of arrows.
        arrow_rad: Arc radius for curved arrows.
        margin_fraction: Fraction of range to use for arrow margins.
        zorder: Z-order for arrow elements.
    """
    n_topics = len(linearization_order)
    sorted_centroids = centroids_2d[linearization_order]
    
    for i in range(n_topics - 1):
        start = sorted_centroids[i]
        end = sorted_centroids[i + 1]
        
        direction = end - start
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
            margin = margin_fraction * (reference_range.max() - reference_range.min())
            start_adj = start + direction * margin
            end_adj = end - direction * margin
            
            ax.annotate(
                "",
                xy=end_adj,
                xytext=start_adj,
                arrowprops=dict(
                    arrowstyle="->",
                    color=arrow_color,
                    lw=arrow_lw,
                    connectionstyle=f"arc3,rad={arrow_rad}",
                ),
                zorder=zorder,
            )


def plot_centroids_with_labels(
    ax: Axes,
    centroids_2d: NDArray[np.floating],
    linearization_order: NDArray[np.int64],
    colors: Optional[List | NDArray] = None,
    cmap: Optional[Colormap | ScalarMappable] = None,
    point_size: int = 200,
    label_fontsize: int = 10,
    label_color: str = "white",
    edgecolors: str = "black",
    linewidths: float = 1.5,
    zorder_scatter: int = 10,
    zorder_label: int = 11,
) -> None:
    """Plot centroids with rank labels.
    
    Args:
        ax: Matplotlib axes to draw on.
        centroids_2d: 2D projected centroids.
        linearization_order: Order in which topics are ranked.
        colors: Precomputed colors for each topic (used if cmap is None).
        cmap: Colormap or ScalarMappable to use instead of colors array.
        point_size: Size of scatter points.
        label_fontsize: Font size for rank labels.
        label_color: Color for rank labels.
        edgecolors: Edge color for scatter points.
        linewidths: Edge line width for scatter points.
        zorder_scatter: Z-order for scatter points.
        zorder_label: Z-order for label annotations.
    """
    n_topics = len(linearization_order)
    sorted_centroids = centroids_2d[linearization_order]
    
    for i in range(n_topics):
        topic_idx = linearization_order[i]
        x, y = sorted_centroids[i]
        
        # Determine color
        if colors is not None:
            point_color = [colors[topic_idx]]
        elif cmap is not None:
            if isinstance(cmap, ScalarMappable):
                point_color = [cmap.to_rgba(topic_idx % 10)]
            else:
                # cmap is a Colormap
                point_color = [cmap(topic_idx % 10)]
        else:
            point_color = "C0"
        
        ax.scatter(
            x, y,
            s=point_size,
            c=point_color,
            edgecolors=edgecolors,
            linewidths=linewidths,
            zorder=zorder_scatter,
        )
        
        # Add rank label
        ax.annotate(
            str(i + 1),
            (x, y),
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            color=label_color,
            zorder=zorder_label,
        )


def style_plot(
    ax: Axes,
    reducer: Any,
    method: DimensionalityReductionMethod,
    n_topics: int,
    n_samples: Optional[int] = None,
) -> None:
    """Apply consistent styling to curriculum plot.
    
    Args:
        ax: Matplotlib axes to style.
        reducer: Fitted reducer object (PCA, LDA, or UMAP).
        method: The reduction method used.
        n_topics: Number of topics.
        n_samples: Optional number of samples (for title).
    """
    title = f"Curriculum Sequence ({n_topics} topics"
    if n_samples is not None:
        title += f", {n_samples} works"
    title += ")"
    
    # Set axis labels based on method
    if method == DimensionalityReductionMethod.PCA and hasattr(reducer, 'explained_variance_ratio_'):
        ax.set_xlabel(f"PC1 ({reducer.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({reducer.explained_variance_ratio_[1]:.1%})")
    elif method == DimensionalityReductionMethod.LDA:
        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")
    elif method == DimensionalityReductionMethod.UMAP:
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    else:
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
    
    ax.set_title(title)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_curriculum(
    curriculum_dir: Path,
    embeddings: NDArray[np.floating],
    output_path: Optional[Path] = None,
    topic_keywords: Optional[Dict[Union[int, str], List[tuple[str, float]]]] = None,
    figsize: tuple[float, float] = (12, 10),
    dpi: int = 150,
    show_labels: bool = True,
    arrow_color: str = "black",
    point_size: int = 200,
    method: DimensionalityReductionMethod = DimensionalityReductionMethod.LDA,
    umap_config: Optional[Any] = None,
    lpp_config: Optional[Any] = None,
    citation_graph: Optional[sp.csr_matrix] = None,
    background: bool = False,
    background_alpha: float = 0.4,
    background_size: int = 10,
    max_points_per_topic: Optional[int] = None,
) -> None:
    """Generate 2D visualization of curriculum.
    
    Args:
        curriculum_dir: Path to directory containing curriculum outputs.
        embeddings: Array of shape (n_samples, embedding_dim).
        output_path: Path to save figure (default: curriculum_dir/curriculum_plot.png).
        topic_keywords: Optional dict of topic keywords for labels.
        figsize: Figure size in inches.
        dpi: Resolution for saved figure.
        show_labels: Whether to show topic labels.
        arrow_color: Color for sequence arrows.
        point_size: Size of topic markers.
        method: Dimensionality reduction method.
        umap_config: UMAPConfig instance (required if method is UMAP).
        citation_graph: Optional citation adjacency matrix for UMAP.
        background: If True, include background embeddings in the plot and compute centroids
            visually from a sampled subset of embeddings instead of projecting centroids.
        background_alpha: Alpha for background points (only used when background=True).
        background_size: Size of background points (only used when background=True).
        max_points_per_topic: Max points to plot per topic for background sampling.
    """
    # Load curriculum data
    frontier_nodes, leaf_to_cluster, centroids, linearization_order = load_curriculum_data(
        curriculum_dir, embeddings
    )
    n_topics = len(frontier_nodes)

    if n_topics < 2:
        LOGGER.error("Need at least 2 topics to plot")
        return

    # Hardcoded flag: if False and method=LPP, fit LPP on weighted topic centroids
    # rather than directly on the underlying works.
    APPLY_LPP_TO_WORKS = True

    # If background plotting is requested, create a subset of points and project them.
    if background:
        # Load point_masses for relevance-based sampling
        point_masses = None
        point_masses_path = curriculum_dir / "point_masses.npy"
        if point_masses_path.exists():
            point_masses = np.load(point_masses_path)

        LOGGER.info(f"Loaded curriculum with {n_topics} topics, {len(embeddings)} works")

        indices_to_plot: List[int] = []
        per_topic_indices: Dict[int, NDArray[np.int64]] = {}

        # Gather per-topic sampled indices
        for topic_idx in range(n_topics):
            indices = np.where(leaf_to_cluster == topic_idx)[0]
            if max_points_per_topic is not None and len(indices) > max_points_per_topic:
                if point_masses is not None:
                    indices = sample_by_relevance(indices, point_masses, max_points_per_topic)
                else:
                    indices = np.random.choice(indices, size=max_points_per_topic, replace=False)
            per_topic_indices[topic_idx] = indices
            indices_to_plot.extend(indices.tolist())

        # Sample unassigned points as well
        unassigned_indices = np.where(leaf_to_cluster == -1)[0]
        if len(unassigned_indices) > 0:
            if max_points_per_topic is not None and len(unassigned_indices) > max_points_per_topic:
                if point_masses is not None:
                    unassigned_indices = sample_by_relevance(unassigned_indices, point_masses, max_points_per_topic)
                else:
                    unassigned_indices = np.random.choice(unassigned_indices, size=max_points_per_topic, replace=False)
            per_topic_indices[-1] = unassigned_indices
            indices_to_plot.extend(unassigned_indices.tolist())

        indices_to_plot = sorted(set(indices_to_plot))
        indices_to_plot_arr = np.array(indices_to_plot, dtype=np.int64)
        index_pos_map = {orig_idx: pos for pos, orig_idx in enumerate(indices_to_plot_arr.tolist())}
        embeddings_subset = embeddings[indices_to_plot_arr]
        labels_subset = leaf_to_cluster[indices_to_plot_arr]

        weights_subset_raw = None
        weights_subset_norm = None
        if point_masses is not None:
            weights_subset_raw = point_masses[indices_to_plot_arr]
            weights_subset_raw = np.asarray(weights_subset_raw, dtype=np.float64)
            weights_subset_raw = np.clip(weights_subset_raw, 0.0, None)
            if weights_subset_raw.max() > 0:
                weights_subset_norm = weights_subset_raw / weights_subset_raw.max()

        # Slice citation graph to subset if provided
        citation_graph_subset = None
        if citation_graph is not None:
            citation_graph_subset = citation_graph[indices_to_plot_arr][:, indices_to_plot_arr]

        use_topic_lpp = (
            method == DimensionalityReductionMethod.LPP and not APPLY_LPP_TO_WORKS
        )

        if use_topic_lpp:
            embeddings_2d = embeddings_subset[:, :2].copy()
            if citation_graph_subset is None:
                raise ValueError("LPP requires citation_graph")

            # Compute weighted centroids in embedding space
            centroids_weighted = np.zeros((n_topics, embeddings_subset.shape[1]), dtype=np.float64)
            for topic_idx in range(n_topics):
                mask = labels_subset == topic_idx
                if not mask.any():
                    continue
                topic_emb = embeddings_subset[mask]
                if weights_subset_raw is not None:
                    topic_weights = weights_subset_raw[mask]
                    w_sum = topic_weights.sum()
                    if w_sum > 0:
                        centroids_weighted[topic_idx] = np.average(topic_emb, axis=0, weights=topic_weights)
                    else:
                        centroids_weighted[topic_idx] = topic_emb.mean(axis=0)
                else:
                    centroids_weighted[topic_idx] = topic_emb.mean(axis=0)

            # Build topic-level citation graph
            rows, cols = citation_graph_subset.nonzero()
            data = citation_graph_subset.data
            src_topics = labels_subset[rows]
            dst_topics = labels_subset[cols]
            topic_mask = (src_topics >= 0) & (dst_topics >= 0)
            topic_graph = sp.csr_matrix(
                (
                    data[topic_mask],
                    (src_topics[topic_mask], dst_topics[topic_mask]),
                ),
                shape=(n_topics, n_topics),
            ).tocsr()

            centroids_2d, reducer = project_to_2d(
                centroids_weighted,
                labels=None,
                method=method,
                umap_config=umap_config,
                lpp_config=lpp_config,
                citation_graph=topic_graph,
            )

            spread = np.ptp(centroids_2d, axis=0)
            if np.all(spread < 1e-6):
                LOGGER.warning(
                    "LPP-on-centroids collapsed (near-zero spread). Falling back to PCA for plotting."
                )
                centroids_2d, reducer = project_to_2d(
                    centroids_weighted,
                    labels=None,
                    method=DimensionalityReductionMethod.PCA,
                    umap_config=None,
                    lpp_config=None,
                    citation_graph=None,
                )
                if hasattr(reducer, "transform"):
                    embeddings_2d = reducer.transform(embeddings_subset)
                else:
                    embeddings_2d = embeddings_subset[:, :2].copy()

            # Apply LPP projection to works using the centroid-fitted components
            if isinstance(reducer, dict) and reducer.get("name") == "lpp":
                A = reducer.get("components")
                if A is None:
                    raise ValueError("LPP reducer missing components")
                if lpp_config is not None and getattr(lpp_config, "center", False):
                    center = centroids_weighted.mean(axis=0)
                    embeddings_centered = embeddings_subset - center
                else:
                    embeddings_centered = embeddings_subset
                embeddings_2d = embeddings_centered.dot(A)
        else:
            # Project subset to 2D
            embeddings_2d, reducer = project_to_2d(
                embeddings_subset,
                labels=labels_subset,
                method=method,
                umap_config=umap_config,
                lpp_config=lpp_config,
                citation_graph=citation_graph_subset,
            )

            # Compute centroids in 2D for visual centering (weighted by relevance if available)
            centroids_2d = np.zeros((n_topics, 2), dtype=np.float64)
            for topic_idx in range(n_topics):
                topic_orig_indices = per_topic_indices.get(topic_idx, np.array([], dtype=np.int64))
                if len(topic_orig_indices) > 0:
                    subset_positions = [index_pos_map[i] for i in topic_orig_indices.tolist() if i in index_pos_map]
                    if len(subset_positions) > 0:
                        if point_masses is not None:
                            weights = point_masses[topic_orig_indices]
                            weights = np.asarray(weights, dtype=np.float64)
                            weights = np.clip(weights, 0.0, None)
                            w_sum = weights.sum()
                            if w_sum > 0:
                                centroids_2d[topic_idx] = np.average(
                                    embeddings_2d[subset_positions],
                                    axis=0,
                                    weights=weights,
                                )
                            else:
                                centroids_2d[topic_idx] = embeddings_2d[subset_positions].mean(axis=0)
                        else:
                            centroids_2d[topic_idx] = embeddings_2d[subset_positions].mean(axis=0)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Color map for topics
        cmap = plt.get_cmap("tab10")

        # Precompute relevance weights for sizing/opacity if available
        weights_subset = weights_subset_norm

        LOGGER.info(f"Plotting background points with max_points_per_topic={max_points_per_topic}")
        MIN_SCALE = 1.0
        MAX_SCALE = 50.0
        for topic_idx in range(n_topics):
            topic_orig_indices = per_topic_indices.get(topic_idx, np.array([], dtype=np.int64))
            subset_positions = [index_pos_map[i] for i in topic_orig_indices.tolist() if i in index_pos_map]
            if len(subset_positions) == 0:
                continue
            # Scale size and alpha by relevance if available
            if weights_subset is not None:
                topic_weights = weights_subset[subset_positions]
                size_scale = MIN_SCALE + (MAX_SCALE - MIN_SCALE) * (topic_weights ** 2)
                sizes = background_size * size_scale
                point_colors = np.tile(cmap(topic_idx % 10), (len(subset_positions), 1))
                point_colors[:, 3] = background_alpha
            else:
                sizes = background_size
                point_colors = [cmap(topic_idx % 10)]
            ax.scatter(
                embeddings_2d[subset_positions, 0],
                embeddings_2d[subset_positions, 1],
                s=sizes,
                c=point_colors,
                label=None,
                zorder=5,
            )

        # Plot unassigned points in gray
        if -1 in per_topic_indices:
            unassigned_orig = per_topic_indices[-1]
            subset_positions = [index_pos_map[i] for i in unassigned_orig.tolist() if i in index_pos_map]
            if len(subset_positions) > 0:
                if weights_subset is not None:
                    unassigned_weights = weights_subset[subset_positions]
                    size_scale = 0.05 + (MAX_SCALE - 0.05) * (unassigned_weights ** 2)
                    sizes = background_size * size_scale
                    point_colors = np.tile(np.array([0.5, 0.5, 0.5, 1.0]), (len(subset_positions), 1))
                    point_colors[:, 3] = background_alpha / 2
                else:
                    sizes = background_size
                    point_colors = "gray"
                ax.scatter(
                    embeddings_2d[subset_positions, 0],
                    embeddings_2d[subset_positions, 1],
                    s=sizes,
                    c=point_colors,
                    zorder=5,
                )

        # Draw arrows
        draw_curriculum_arrows(
            ax, centroids_2d, linearization_order, embeddings_2d,
            arrow_color=arrow_color,
            arrow_lw=2,
            arrow_rad=0.05,
            margin_fraction=0.02,
            zorder=15,
        )

        # Plot centroids with labels
        plot_centroids_with_labels(
            ax, centroids_2d, linearization_order,
            cmap=cmap,
            point_size=300,
            label_fontsize=11,
            zorder_scatter=2,
            zorder_label=16,
        )

        # Add keyword labels if available
        if show_labels and topic_keywords:
            sorted_centroids = centroids_2d[linearization_order]
            for i in range(n_topics):
                topic_idx = linearization_order[i]
                x, y = sorted_centroids[i]
                keywords = topic_keywords.get(topic_idx) or topic_keywords.get(str(topic_idx))
                if keywords:
                    if isinstance(keywords[0], (tuple, list)):
                        label = ", ".join([str(kw) for kw, _ in keywords[:3]])
                    else:
                        label = ", ".join([str(kw) for kw in keywords[:3]])
                    ax.annotate(
                        label,
                        (x, y),
                        xytext=(5, 10),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.8,
                    )

        # Style plot
        style_plot(ax, reducer, method, n_topics, len(embeddings))

        plt.tight_layout()

        # Save
        if output_path is None:
            output_path = curriculum_dir / "curriculum_plot_with_background.png"

        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        LOGGER.info(f"Saved plot to {output_path}")
        plt.close(fig)

        return

    # Non-background plotting: project centroids to 2D and plot them
    centroids_2d, reducer = project_to_2d(
        centroids,
        labels=None,  # Centroids don't have individual labels
        method=method,
        umap_config=umap_config,
        lpp_config=lpp_config,
        citation_graph=None,  # Citation graph not applicable for centroids only
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color map for topics (by position in sequence)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / (n_topics - 1)) for i in range(n_topics)]

    # Draw arrows
    draw_curriculum_arrows(
        ax, centroids_2d, linearization_order, centroids_2d,
        arrow_color=arrow_color,
        arrow_lw=1.5,
        arrow_rad=0.1,
        margin_fraction=0.03,
    )

    # Plot centroids with labels
    plot_centroids_with_labels(
        ax, centroids_2d, linearization_order,
        colors=colors,
        point_size=point_size,
        label_fontsize=10,
    )

    # Add keyword labels if available
    if show_labels and topic_keywords:
        sorted_centroids = centroids_2d[linearization_order]
        for i in range(n_topics):
            topic_idx = linearization_order[i]
            x, y = sorted_centroids[i]
            keywords = topic_keywords.get(topic_idx) or topic_keywords.get(str(topic_idx))
            if keywords:
                if isinstance(keywords[0], (tuple, list)):
                    label = ", ".join([str(kw) for kw, _ in keywords[:3]])
                else:
                    label = ", ".join([str(kw) for kw in keywords[:3]])
                ax.annotate(
                    label,
                    (x, y),
                    xytext=(5, 10),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.8,
                )

    # Style plot
    style_plot(ax, reducer, method, n_topics)

    # Add legend for sequence direction
    sorted_centroids = centroids_2d[linearization_order]
    ax.annotate(
        "Start",
        sorted_centroids[0],
        xytext=(-20, -20),
        textcoords="offset points",
        fontsize=10,
        color="green",
        fontweight="bold",
    )
    ax.annotate(
        "End",
        sorted_centroids[-1],
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        color="red",
        fontweight="bold",
    )

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = curriculum_dir / "curriculum_plot.png"

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    LOGGER.info(f"Saved plot to {output_path}")

    plt.close(fig)


def plot_curriculum_with_background(
    curriculum_dir: Path,
    embeddings: NDArray[np.floating],
    output_path: Optional[Path] = None,
    topic_keywords: Optional[Dict[Union[int, str], List[tuple[str, float]]]] = None,
    figsize: tuple[float, float] = (12, 10),
    dpi: int = 300,
    background_alpha: float = 0.4,
    background_size: int = 10,
    max_points_per_topic: Optional[int] = None,
    show_labels: bool = True,
    method: DimensionalityReductionMethod = DimensionalityReductionMethod.LDA,
    umap_config: Optional[Any] = None,
    lpp_config: Optional[Any] = None,
    citation_graph: Optional[sp.csr_matrix] = None,
) -> None:
    """Plot curriculum with background embeddings.
    
    Similar to plot_curriculum but includes all embeddings projected to 2D
    as a background scatter plot.
    
    Args:
        curriculum_dir: Path to directory containing curriculum outputs.
        embeddings: Array of shape (n_samples, embedding_dim).
        output_path: Path to save figure.
        topic_keywords: Optional dict of topic keywords for labels.
        figsize: Figure size in inches.
        dpi: Resolution for saved figure.
        background_alpha: Alpha for background points.
        background_size: Size of background points.
        max_points_per_topic: Max points to plot per topic (None = all). If set, samples
            from each topic's points.
        show_labels: Whether to show topic keyword labels.
        method: Dimensionality reduction method.
        umap_config: UMAPConfig instance (required if method is UMAP).
        citation_graph: Optional citation adjacency matrix for UMAP.
    """
    # Thin wrapper kept for compatibility: forward to unified `plot_curriculum`.
    LOGGER.info("plot_curriculum_with_background is deprecated; forwarding to plot_curriculum(background=True)")
    plot_curriculum(
        curriculum_dir,
        embeddings,
        output_path=output_path,
        topic_keywords=topic_keywords,
        figsize=figsize,
        dpi=dpi,
        background=True,
        background_alpha=background_alpha,
        background_size=background_size,
        max_points_per_topic=max_points_per_topic,
        show_labels=show_labels,
        method=method,
        umap_config=umap_config,
        lpp_config=lpp_config,
        citation_graph=citation_graph,
    )
