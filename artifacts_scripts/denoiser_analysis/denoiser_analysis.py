#!/usr/bin/env python3
"""Denoiser analysis - comprehensive comparison of denoising strategies.

This script analyzes and compares different denoising strategies by computing
metrics on the AI cloud without requiring pre-cached denoised clouds.

Key metrics:
- SNR Score: Signal-to-Noise Ratio where Signal = Within-Class Variance (S_W)
             and Noise = Between-Class Gap (S_B) + Graph Inconsistency (S_L)
- Class Separability: How much the projection separates books from articles
- Variance Retention: Total variance preserved by the projection
- Locality Preservation: How well local structure is preserved

Usage:
    python -m artifacts_scripts.denoiser_analysis.denoiser_analysis
    python -m artifacts_scripts.denoiser_analysis.denoiser_analysis denoisers='[PCA,CCLIP,WC_CCLIP]'
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from numpy.typing import NDArray
from sklearn.preprocessing import normalize

from bookgap.cloud import load_works
from bookgap.cloud_denoiser import (
    CloudDenoiserContext,
    ProjectiveDenoiser,
    WorkDenoisers,
)
from bookgap.configs.config_store import register_configs
from bookgap.embedder import Embedder
from bookgap.util import (
    LOGGER,
    TIMER,
    Filter,
    SubField,
    WorkType,
    setup_logging,
)

from .config import DenoiserAnalysisConfig

# Initialize logging
setup_logging(log_name="denoiser_analysis", log_level="DEBUG", include_time=True)

# Register configs
register_configs()

# Register denoiser analysis config
cs = ConfigStore.instance()
cs.store(name="denoiser_analysis", node=DenoiserAnalysisConfig())


@dataclass
class DenoiserMetrics:
    """Computed metrics for a single denoiser."""
    
    name: str
    snr_score: float
    variance_retention: float
    between_class_scatter: float
    within_class_scatter: float
    laplacian_scatter: float
    total_scatter: float
    book_scatter: float
    article_scatter: float
    eigenvalues: Optional[NDArray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "denoiser": self.name,
            "snr_score": self.snr_score,
            "variance_retention": self.variance_retention,
            "between_class_scatter": self.between_class_scatter,
            "within_class_scatter": self.within_class_scatter,
            "laplacian_scatter": self.laplacian_scatter,
            "total_scatter": self.total_scatter,
            "book_scatter": self.book_scatter,
            "article_scatter": self.article_scatter,
        }


def calculate_snr_score(
    context: CloudDenoiserContext,
    P: NDArray,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """
    Calculates the Signal-to-Noise Ratio of the projection P.
    
    Signal = Within-Class Variance (S_W)
    Noise = Between-Class Gap (S_B) + Graph Inconsistency (S_L)
    
    Args:
        context: Denoiser context with data
        P: Projection matrix (d x k)
        alpha: Weight for between-class scatter in noise term
        beta: Weight for Laplacian scatter in noise term
        
    Returns:
        SNR score (higher is better)
    """
    # Get S_W (Within-Class Scatter)
    S_W = sum(context.S_subset(i) for i in range(context.num_subsets()))
    S_B = context.S_B_multi(list(range(context.num_subsets())))
    S_L = context.S_L()
    
    # Project the matrices: Trace(P.T @ S @ P)
    signal = np.trace(P.T @ S_W @ P)
    
    noise_matrix = alpha * S_B + beta * S_L
    noise = np.trace(P.T @ noise_matrix @ P)
    
    return float(signal / noise) if noise > 1e-10 else 0.0


def compute_projected_scatter(S: NDArray, P: NDArray) -> float:
    """Compute trace of projected scatter matrix: Tr(P.T @ S @ P)."""
    return float(np.trace(P.T @ S @ P))


@TIMER.time()
def compute_denoiser_metrics(
    name: str,
    P: NDArray,
    context: CloudDenoiserContext,
    snr_alpha: float,
    snr_beta: float,
    original_dim: int,
    eigvals: Optional[NDArray] = None,
) -> DenoiserMetrics:
    """Compute all metrics for a single denoiser projection."""
    
    # Get scatter matrices
    S_T = context.S_T()
    S_B = context.S_B_multi(list(range(context.num_subsets())))
    S_L = context.S_L()
    S_book = context.S_subset(0)
    S_article = context.S_subset(1)
    S_W = S_book + S_article
    
    # Compute projected scatters
    proj_S_T = compute_projected_scatter(S_T, P)
    proj_S_B = compute_projected_scatter(S_B, P)
    proj_S_L = compute_projected_scatter(S_L, P)
    proj_S_W = compute_projected_scatter(S_W, P)
    proj_S_book = compute_projected_scatter(S_book, P)
    proj_S_article = compute_projected_scatter(S_article, P)
    
    # Compute SNR
    snr = calculate_snr_score(context, P, alpha=snr_alpha, beta=snr_beta)
    
    # Variance retention: ratio of projected total scatter to original
    original_total_scatter = np.trace(S_T)
    variance_retention = proj_S_T / original_total_scatter if original_total_scatter > 0 else 0.0
    
    return DenoiserMetrics(
        name=name,
        snr_score=snr,
        variance_retention=variance_retention,
        between_class_scatter=proj_S_B,
        within_class_scatter=proj_S_W,
        laplacian_scatter=proj_S_L,
        total_scatter=proj_S_T,
        book_scatter=proj_S_book,
        article_scatter=proj_S_article,
        eigenvalues=eigvals,
    )


@TIMER.time(formatters=dict(works=lambda w: f"<{len(w)} works>", embeddings=lambda e: f"<shape={e.shape}>"))
def train_denoiser(
    denoiser_name: str,
    works: List[Dict[str, Any]],
    embeddings: NDArray,
    drop_k: int,
    book_filter: Filter,
    article_filter: Filter,
) -> Tuple[NDArray, Optional[NDArray]]:
    """Train a denoiser and return its projection matrix.
    
    Returns:
        Tuple of (projection_matrix, eigenvalues)
    """
    denoiser_enum = WorkDenoisers[denoiser_name]
    denoiser_impl = denoiser_enum.value
    
    # Handle NoneDenoiser specially
    if denoiser_name == "NONE":
        if drop_k != 0:
            raise ValueError(f"NONE denoiser requires drop_k=0, got {drop_k}")
        P = np.eye(embeddings.shape[1], dtype=embeddings.dtype)
        return P, None
    
    # For ProjectiveDenoiser, we need to access the internal denoiser
    if isinstance(denoiser_impl, ProjectiveDenoiser):
        inner_denoiser = denoiser_impl.denoiser_impl
        n_components = embeddings.shape[1] - drop_k
        
        # Build context
        context = CloudDenoiserContext(works, embeddings, book_filter, article_filter)
        
        # Get projector
        P, eigvals = inner_denoiser.projector(context, n_components)
        
        # Orthonormalize
        Q, R = np.linalg.qr(P, mode="reduced")
        
        return Q, eigvals
    
    raise ValueError(f"Unsupported denoiser type: {type(denoiser_impl)}")


def plot_snr_comparison(
    metrics_list: List[DenoiserMetrics],
    output_path: Path,
) -> None:
    """Bar chart comparing SNR scores across denoisers."""
    plt.figure(figsize=(14, 6))
    
    names = [m.name for m in metrics_list]
    snr_scores = [m.snr_score for m in metrics_list]
    
    # Sort by SNR score
    sorted_idx = np.argsort(snr_scores)[::-1]
    names = [names[i] for i in sorted_idx]
    snr_scores = [snr_scores[i] for i in sorted_idx]
    
    cmap = plt.colormaps.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(names)))
    
    bars = plt.bar(range(len(names)), snr_scores, color=colors)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.ylabel("SNR Score (Signal / Noise)")
    plt.xlabel("Denoiser")
    plt.title("Signal-to-Noise Ratio by Denoiser\n(Higher = Better)")
    plt.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, snr_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    LOGGER.info(f"Saved SNR comparison plot to {output_path}")


def plot_scatter_decomposition(
    metrics_list: List[DenoiserMetrics],
    output_path: Path,
) -> None:
    """Stacked bar chart showing scatter matrix decomposition."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    names = [m.name for m in metrics_list]
    s_b = [m.between_class_scatter for m in metrics_list]
    s_l = [m.laplacian_scatter for m in metrics_list]
    s_w = [m.within_class_scatter for m in metrics_list]
    
    # Sort by total (S_B + S_L) ascending (less noise is better)
    noise = [b + l for b, l in zip(s_b, s_l)]
    sorted_idx = np.argsort(noise)
    names = [names[i] for i in sorted_idx]
    s_b = [s_b[i] for i in sorted_idx]
    s_l = [s_l[i] for i in sorted_idx]
    s_w = [s_w[i] for i in sorted_idx]
    
    x = np.arange(len(names))
    width = 0.35
    
    # Noise bars (S_B + S_L)
    bars1 = ax.bar(x - width/2, s_b, width, label='Between-Class (S_B)', color='red', alpha=0.7)
    bars2 = ax.bar(x - width/2, s_l, width, bottom=s_b, label='Laplacian (S_L)', color='orange', alpha=0.7)
    
    # Signal bars (S_W)
    bars3 = ax.bar(x + width/2, s_w, width, label='Within-Class (S_W)', color='green', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Projected Scatter (Trace)")
    ax.set_xlabel("Denoiser")
    ax.set_title("Scatter Matrix Decomposition by Denoiser\n(Green = Signal, Red+Orange = Noise)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    LOGGER.info(f"Saved scatter decomposition plot to {output_path}")


def plot_variance_vs_snr(
    metrics_list: List[DenoiserMetrics],
    output_path: Path,
) -> None:
    """Scatter plot of variance retention vs SNR score."""
    plt.figure(figsize=(10, 8))
    
    names = [m.name for m in metrics_list]
    snr_scores = [m.snr_score for m in metrics_list]
    var_retention = [m.variance_retention for m in metrics_list]
    
    plt.scatter(var_retention, snr_scores, s=100, alpha=0.7)
    
    for i, name in enumerate(names):
        plt.annotate(name, (var_retention[i], snr_scores[i]), 
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    plt.xlabel("Variance Retention (Fraction)")
    plt.ylabel("SNR Score")
    plt.title("Variance Retention vs Signal-to-Noise Ratio\n(Upper-right corner is ideal)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    LOGGER.info(f"Saved variance vs SNR plot to {output_path}")


def plot_class_scatter_balance(
    metrics_list: List[DenoiserMetrics],
    output_path: Path,
) -> None:
    """Plot showing balance between book and article scatter."""
    plt.figure(figsize=(12, 6))
    
    names = [m.name for m in metrics_list]
    book_scatter = [m.book_scatter for m in metrics_list]
    article_scatter = [m.article_scatter for m in metrics_list]
    
    # Compute ratio (book / article)
    ratios = [b / a if a > 1e-10 else 0 for b, a in zip(book_scatter, article_scatter)]
    
    # Sort by ratio
    sorted_idx = np.argsort(ratios)
    names = [names[i] for i in sorted_idx]
    ratios = [ratios[i] for i in sorted_idx]
    
    colors = ['red' if r < 0.5 else 'orange' if r < 0.8 else 'green' if r < 1.2 else 'orange' if r < 2 else 'red' 
              for r in ratios]
    
    plt.barh(range(len(names)), ratios, color=colors, alpha=0.7)
    plt.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Equal balance')
    
    plt.yticks(range(len(names)), names)
    plt.xlabel("Book Scatter / Article Scatter Ratio")
    plt.ylabel("Denoiser")
    plt.title("Class Scatter Balance by Denoiser\n(1.0 = Equal treatment of both classes)")
    plt.legend()
    plt.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    LOGGER.info(f"Saved class scatter balance plot to {output_path}")


def plot_metric_heatmap(
    metrics_list: List[DenoiserMetrics],
    output_path: Path,
) -> None:
    """Heatmap of normalized metrics across all denoisers."""
    # Prepare data
    names = [m.name for m in metrics_list]
    
    # Select metrics to display
    metric_names = [
        "SNR Score",
        "Variance Retention", 
        "Between-Class (↓)",
        "Within-Class (↑)",
        "Laplacian (↓)",
        "Book Scatter",
        "Article Scatter",
    ]
    
    data = np.array([
        [m.snr_score for m in metrics_list],
        [m.variance_retention for m in metrics_list],
        [m.between_class_scatter for m in metrics_list],
        [m.within_class_scatter for m in metrics_list],
        [m.laplacian_scatter for m in metrics_list],
        [m.book_scatter for m in metrics_list],
        [m.article_scatter for m in metrics_list],
    ])
    
    # Normalize each row to [0, 1] for visualization
    data_normalized = np.zeros_like(data)
    for i in range(data.shape[0]):
        row = data[i]
        row_min, row_max = row.min(), row.max()
        if row_max - row_min > 1e-10:
            data_normalized[i] = (row - row_min) / (row_max - row_min)
        else:
            data_normalized[i] = 0.5
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    im = ax.imshow(data_normalized, cmap="RdYlGn", aspect="auto")
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels(metric_names)
    
    # Add text annotations with actual values
    for i in range(len(metric_names)):
        for j in range(len(names)):
            value = data[i, j]
            text = f"{value:.2f}" if abs(value) < 100 else f"{value:.1e}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                   color="white" if data_normalized[i, j] < 0.3 or data_normalized[i, j] > 0.7 else "black")
    
    plt.colorbar(im, ax=ax, label="Normalized Value (row-wise)")
    plt.title("Denoiser Metrics Heatmap\n(↑ = higher is better, ↓ = lower is better)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    LOGGER.info(f"Saved metrics heatmap to {output_path}")


def plot_cclip_family_comparison(
    metrics_list: List[DenoiserMetrics],
    output_path: Path,
) -> None:
    """Compare CCLIP vs WC_CCLIP family of denoisers."""
    # Filter to CCLIP family
    cclip_family = ["CCP", "CCLIP_01", "CCLIP_1", "CCLIP", "LIP",
                    "WC_CCP", "WC_CCLIP_01", "WC_CCLIP_1", "WC_CCLIP", "WC_LIP"]
    
    family_metrics = [m for m in metrics_list if m.name in cclip_family]
    if len(family_metrics) < 2:
        LOGGER.warning("Not enough CCLIP family denoisers to plot comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare data
    names = [m.name for m in family_metrics]
    is_wc = ["WC_" in n for n in names]
    colors = ["blue" if wc else "red" for wc in is_wc]
    
    # SNR scores
    ax = axes[0, 0]
    snr = [m.snr_score for m in family_metrics]
    ax.bar(range(len(names)), snr, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("SNR Score")
    ax.set_title("SNR Score (Higher = Better)")
    ax.grid(axis="y", alpha=0.3)
    
    # Variance retention
    ax = axes[0, 1]
    var_ret = [m.variance_retention for m in family_metrics]
    ax.bar(range(len(names)), var_ret, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Variance Retention")
    ax.set_title("Variance Retention (Higher = Better)")
    ax.grid(axis="y", alpha=0.3)
    
    # Between-class scatter
    ax = axes[1, 0]
    s_b = [m.between_class_scatter for m in family_metrics]
    ax.bar(range(len(names)), s_b, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Between-Class Scatter")
    ax.set_title("Between-Class Scatter (Lower = Better)")
    ax.grid(axis="y", alpha=0.3)
    
    # Within-class scatter
    ax = axes[1, 1]
    s_w = [m.within_class_scatter for m in family_metrics]
    ax.bar(range(len(names)), s_w, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Within-Class Scatter")
    ax.set_title("Within-Class Scatter (Higher = Better)")
    ax.grid(axis="y", alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='CCLIP (S_T constraint)'),
        Patch(facecolor='blue', alpha=0.7, label='WC_CCLIP (S_W constraint)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_path, dpi=150)
    plt.close()
    LOGGER.info(f"Saved CCLIP family comparison to {output_path}")


def analyze_scatter_correlations(
    context: CloudDenoiserContext,
    output_path: Path,
) -> None:
    """Analyze correlations between scatter matrices S_B, S_L, S_W, S_T.
    
    This helps understand if minimizing S_B and S_L are conflicting or aligned objectives.
    """
    LOGGER.info("Analyzing scatter matrix correlations...")
    
    # Get scatter matrices
    S_T = context.S_T()
    S_B = context.S_B_multi(list(range(context.num_subsets())))
    S_L = context.S_L()
    S_book = context.S_subset(0)
    S_article = context.S_subset(1)
    S_W = S_book + S_article
    
    # Flatten matrices for correlation analysis
    matrices = {
        "S_T (Total)": S_T.flatten(),
        "S_B (Between)": S_B.flatten(),
        "S_L (Laplacian)": S_L.flatten(),
        "S_W (Within)": S_W.flatten(),
        "S_book": S_book.flatten(),
        "S_article": S_article.flatten(),
    }
    
    # Compute correlation matrix
    names = list(matrices.keys())
    n = len(names)
    corr_matrix = np.zeros((n, n))
    
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            corr = np.corrcoef(matrices[name_i], matrices[name_j])[0, 1]
            corr_matrix[i, j] = corr
    
    # Compute some key statistics
    LOGGER.info("\n" + "=" * 50)
    LOGGER.info("SCATTER MATRIX CORRELATION ANALYSIS")
    LOGGER.info("=" * 50)
    
    # Key correlations
    s_b_flat = matrices["S_B (Between)"]
    s_l_flat = matrices["S_L (Laplacian)"]
    s_w_flat = matrices["S_W (Within)"]
    s_t_flat = matrices["S_T (Total)"]
    
    corr_b_l = np.corrcoef(s_b_flat, s_l_flat)[0, 1]
    corr_b_w = np.corrcoef(s_b_flat, s_w_flat)[0, 1]
    corr_l_w = np.corrcoef(s_l_flat, s_w_flat)[0, 1]
    corr_b_t = np.corrcoef(s_b_flat, s_t_flat)[0, 1]
    corr_l_t = np.corrcoef(s_l_flat, s_t_flat)[0, 1]
    
    LOGGER.info(f"Correlation(S_B, S_L) = {corr_b_l:.4f}")
    LOGGER.info(f"  -> {'ALIGNED' if corr_b_l > 0.5 else 'ORTHOGONAL' if abs(corr_b_l) < 0.3 else 'WEAKLY RELATED'}: "
                f"Minimizing class separation {'helps' if corr_b_l > 0 else 'hurts'} locality")
    
    LOGGER.info(f"Correlation(S_B, S_W) = {corr_b_w:.4f}")
    LOGGER.info(f"Correlation(S_L, S_W) = {corr_l_w:.4f}")
    LOGGER.info(f"Correlation(S_B, S_T) = {corr_b_t:.4f}")
    LOGGER.info(f"Correlation(S_L, S_T) = {corr_l_t:.4f}")
    
    # Trace statistics
    LOGGER.info(f"\nTrace statistics:")
    LOGGER.info(f"  Tr(S_T) = {np.trace(S_T):.4f}")
    LOGGER.info(f"  Tr(S_B) = {np.trace(S_B):.6f}")
    LOGGER.info(f"  Tr(S_L) = {np.trace(S_L):.2f}")
    LOGGER.info(f"  Tr(S_W) = {np.trace(S_W):.4f}")
    LOGGER.info(f"  Tr(S_B) / Tr(S_T) = {np.trace(S_B) / np.trace(S_T):.6f} (fraction of variance due to class separation)")
    LOGGER.info(f"  Tr(S_W) / Tr(S_T) = {np.trace(S_W) / np.trace(S_T):.4f} (fraction of variance within classes)")
    
    # Check S_T = S_B + S_W decomposition
    decomp_error = np.linalg.norm(S_T - S_B - S_W) / np.linalg.norm(S_T)
    LOGGER.info(f"  ||S_T - S_B - S_W|| / ||S_T|| = {decomp_error:.6f} (should be ~0 if decomposition holds)")
    
    # Plot correlation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Correlation heatmap
    ax = axes[0]
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    
    # Add correlation values
    for i in range(n):
        for j in range(n):
            text = f"{corr_matrix[i, j]:.2f}"
            color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Scatter Matrix Correlations\n(element-wise correlation of flattened matrices)")
    
    # Eigenvalue spectrum comparison
    ax = axes[1]
    
    # Get top eigenvalues of each matrix (normalized)
    n_eigs = min(50, S_T.shape[0])
    
    eigs_T = np.linalg.eigvalsh(S_T)[::-1][:n_eigs]
    eigs_B = np.linalg.eigvalsh(S_B)[::-1][:n_eigs]
    eigs_L = np.linalg.eigvalsh(S_L)[::-1][:n_eigs]
    eigs_W = np.linalg.eigvalsh(S_W)[::-1][:n_eigs]
    
    # Normalize by trace for comparison
    eigs_T_norm = eigs_T / np.sum(eigs_T) if np.sum(eigs_T) > 0 else eigs_T
    eigs_B_norm = eigs_B / np.sum(np.abs(eigs_B)) if np.sum(np.abs(eigs_B)) > 0 else eigs_B
    eigs_L_norm = eigs_L / np.sum(np.abs(eigs_L)) if np.sum(np.abs(eigs_L)) > 0 else eigs_L
    eigs_W_norm = eigs_W / np.sum(eigs_W) if np.sum(eigs_W) > 0 else eigs_W
    
    ax.semilogy(range(n_eigs), eigs_T_norm, 'k-', label='S_T (Total)', linewidth=2)
    ax.semilogy(range(n_eigs), np.maximum(eigs_B_norm, 1e-10), 'r--', label='S_B (Between)', linewidth=2)
    ax.semilogy(range(n_eigs), np.maximum(eigs_L_norm, 1e-10), 'orange', linestyle='--', label='S_L (Laplacian)', linewidth=2)
    ax.semilogy(range(n_eigs), eigs_W_norm, 'g-', label='S_W (Within)', linewidth=2)
    
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Normalized Eigenvalue (log scale)")
    ax.set_title("Eigenvalue Spectra of Scatter Matrices\n(fraction of total trace)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    LOGGER.info(f"Saved scatter correlations plot to {output_path}")
    
    # Additional analysis: Principal angles between subspaces
    LOGGER.info("\nPrincipal subspace analysis:")
    
    # Get top-k eigenvectors of S_B and S_L
    k = 10
    _, V_B = np.linalg.eigh(S_B)
    _, V_L = np.linalg.eigh(S_L)
    
    V_B_top = V_B[:, -k:]  # Top k eigenvectors
    V_L_top = V_L[:, -k:]
    
    # Compute principal angles via SVD of V_B.T @ V_L
    _, singular_values, _ = np.linalg.svd(V_B_top.T @ V_L_top)
    principal_angles = np.arccos(np.clip(singular_values, -1, 1)) * 180 / np.pi
    
    LOGGER.info(f"  Principal angles between top-{k} S_B and S_L eigenvectors:")
    LOGGER.info(f"  {principal_angles}")
    LOGGER.info(f"  Mean angle: {np.mean(principal_angles):.1f}° (90° = orthogonal, 0° = aligned)")


def _run_analysis(cfg: DenoiserAnalysisConfig) -> None:
    """Execute the denoiser analysis."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log startup
    LOGGER.info("=" * 60)
    LOGGER.info("Denoiser Analysis")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info(f"Drop K: {cfg.drop_k}")
    LOGGER.info(f"SNR weights: alpha={cfg.snr_alpha}, beta={cfg.snr_beta}")
    LOGGER.info("")
    
    # Determine which denoisers to analyze
    if cfg.denoisers is None:
        # Analyze all projective denoisers (exclude NONE unless drop_k=0)
        denoiser_names = [d.name for d in WorkDenoisers if d.name != "NONE" or cfg.drop_k == 0]
    else:
        denoiser_names = list(cfg.denoisers)
    
    LOGGER.info(f"Analyzing {len(denoiser_names)} denoisers: {denoiser_names}")
    
    # Get embedder model from config (already an enum)
    embedder_model = cfg.experiment.cloud.embedder_model
    if isinstance(embedder_model, str):
        embedder_model = Embedder.ModelName[embedder_model]
    LOGGER.info(f"Embedder model: {embedder_model.value}")
    
    # Load the AI cloud data (without denoising)
    ai_pre_filter = Filter(
        subfield=SubField.AI.value,
        work_types=(WorkType.BOOK.value, WorkType.ARTICLE.value),
    )
    
    LOGGER.info("Loading works and embeddings...")
    works, embeddings = load_works(ai_pre_filter, cfg.max_works, embedder_model)
    LOGGER.info(f"Loaded {len(works)} works with embeddings shape {embeddings.shape}")
    
    # Normalize embeddings
    embeddings = normalize(embeddings, norm="l2", axis=1)
    LOGGER.info("Normalized embeddings to unit length")
    
    # Define filters for books and articles
    book_filter = Filter(work_types=(WorkType.BOOK.value,))
    article_filter = Filter(work_types=(WorkType.ARTICLE.value,))
    
    # Count works by type
    num_books = sum(1 for w in works if w.get("type") == "book")
    num_articles = sum(1 for w in works if w.get("type") in ("article", "preprint"))
    LOGGER.info(f"Books: {num_books}, Articles: {num_articles}")
    
    # Build context for metric computation (shared across all denoisers)
    LOGGER.info("Building denoiser context...")
    context = CloudDenoiserContext(works, embeddings, book_filter, article_filter)
    
    # Pre-compute scatter matrices (they're cached, but let's log timing)
    LOGGER.info("Pre-computing scatter matrices...")
    _ = context.S_T()
    _ = context.S_B()
    _ = context.S_L()
    _ = context.S_subset(0)
    _ = context.S_subset(1)
    LOGGER.info("Scatter matrices computed")
    
    # Analyze each denoiser
    metrics_list: List[DenoiserMetrics] = []
    original_dim = embeddings.shape[1]
    
    for denoiser_name in denoiser_names:
        LOGGER.info(f"\n{'='*40}")
        LOGGER.info(f"Analyzing denoiser: {denoiser_name}")
        LOGGER.info(f"{'='*40}")
        
        try:
            # Train denoiser
            P, eigvals = train_denoiser(
                denoiser_name=denoiser_name,
                works=works,
                embeddings=embeddings,
                drop_k=cfg.drop_k,
                book_filter=book_filter,
                article_filter=article_filter,
            )
            
            LOGGER.info(f"Projection matrix shape: {P.shape}")
            
            # Compute metrics
            metrics = compute_denoiser_metrics(
                name=denoiser_name,
                P=P,
                context=context,
                snr_alpha=cfg.snr_alpha,
                snr_beta=cfg.snr_beta,
                original_dim=original_dim,
                eigvals=eigvals,
            )
            
            metrics_list.append(metrics)
            
            LOGGER.info(f"SNR Score: {metrics.snr_score:.4f}")
            LOGGER.info(f"Variance Retention: {metrics.variance_retention:.4f}")
            LOGGER.info(f"Between-Class Scatter: {metrics.between_class_scatter:.4f}")
            LOGGER.info(f"Within-Class Scatter: {metrics.within_class_scatter:.4f}")
            LOGGER.info(f"Laplacian Scatter: {metrics.laplacian_scatter:.4f}")
            
        except Exception as e:
            LOGGER.error(f"Failed to analyze {denoiser_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not metrics_list:
        LOGGER.error("No denoisers were successfully analyzed!")
        return
    
    # Write results to CSV
    csv_filename = f"denoiser_analysis_drop{cfg.drop_k}_{embedder_model.name}.csv"
    csv_path = output_dir / csv_filename
    
    LOGGER.info(f"\nWriting results to {csv_path}")
    
    df = pd.DataFrame([m.to_dict() for m in metrics_list])
    df = df.sort_values("snr_score", ascending=False)
    df.to_csv(csv_path, index=False)
    
    # Generate visualizations
    LOGGER.info("\nGenerating visualizations...")
    
    plot_snr_comparison(
        metrics_list,
        output_dir / f"snr_comparison_drop{cfg.drop_k}.png",
    )
    
    plot_scatter_decomposition(
        metrics_list,
        output_dir / f"scatter_decomposition_drop{cfg.drop_k}.png",
    )
    
    plot_variance_vs_snr(
        metrics_list,
        output_dir / f"variance_vs_snr_drop{cfg.drop_k}.png",
    )
    
    plot_class_scatter_balance(
        metrics_list,
        output_dir / f"class_scatter_balance_drop{cfg.drop_k}.png",
    )
    
    plot_metric_heatmap(
        metrics_list,
        output_dir / f"metrics_heatmap_drop{cfg.drop_k}.png",
    )
    
    plot_cclip_family_comparison(
        metrics_list,
        output_dir / f"cclip_family_comparison_drop{cfg.drop_k}.png",
    )
    
    # Analyze scatter matrix correlations
    analyze_scatter_correlations(
        context,
        output_dir / f"scatter_correlations_drop{cfg.drop_k}.png",
    )
    
    # Print summary table
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("ANALYSIS SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info(f"\nTop 5 denoisers by SNR Score:")
    for i, m in enumerate(sorted(metrics_list, key=lambda x: x.snr_score, reverse=True)[:5]):
        LOGGER.info(f"  {i+1}. {m.name}: SNR={m.snr_score:.4f}, VarRet={m.variance_retention:.4f}")
    
    LOGGER.info(f"\nResults written to: {csv_path}")
    LOGGER.info(f"Plots written to: {output_dir}")


@hydra.main(version_base=None, config_path=None, config_name="denoiser_analysis")
def main(cfg: DenoiserAnalysisConfig) -> None:
    """Main entry point for denoiser analysis."""
    _run_analysis(cfg)


if __name__ == "__main__":
    main()
