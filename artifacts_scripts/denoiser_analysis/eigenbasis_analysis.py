#!/usr/bin/env python3
"""Eigenbasis analysis - analyze how denoiser eigenbases capture scatter matrix variance.

This script analyzes the full eigenbasis of each denoiser (no truncation) to understand
how each denoiser's principal directions relate to the key scatter matrices:
- S_T: Total scatter (overall variance)
- S_B: Between-class scatter (class separation)
- S_W: Within-class scatter (semantic variance within classes)
- S_L: Laplacian scatter (graph inconsistency)

For each denoiser, we compute the % variance explained by each eigenvector for all 4 matrices.
This reveals:
- Whether a denoiser successfully pushes "noise" (S_B, S_L) into low-variance directions
- How correlated the variance structures of different matrices are
- Which denoisers best separate "signal" (S_W) from "noise" (S_B, S_L)

Usage:
    python -m artifacts_scripts.denoiser_analysis.eigenbasis_analysis
    python -m artifacts_scripts.denoiser_analysis.eigenbasis_analysis denoisers='[PCA,CCLIP,WC_CCLIP]'
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
    Filter,
    SubField,
    WorkType,
    setup_logging,
)

from .config import DenoiserAnalysisConfig

# Initialize logging
setup_logging(log_name="eigenbasis_analysis", log_level="DEBUG", include_time=True)

# Register configs
register_configs()

# Register config (reuse DenoiserAnalysisConfig, drop_k is ignored)
cs = ConfigStore.instance()
cs.store(name="eigenbasis_analysis", node=DenoiserAnalysisConfig())


@dataclass
class EigenbasisResult:
    """Results of eigenbasis analysis for a single denoiser."""
    
    name: str
    eigenvectors: NDArray  # (d, d) - full eigenbasis
    eigenvalues: Optional[NDArray]  # eigenvalues if available
    variance_explained: Dict[str, NDArray]  # matrix_name -> (d,) array of % variance per eigenvector


def get_full_eigenbasis(
    denoiser_name: str,
    context: CloudDenoiserContext,
) -> Tuple[NDArray, Optional[NDArray]]:
    """Get the full eigenbasis from a denoiser (all components, no truncation).
    
    Args:
        denoiser_name: Name of the denoiser from WorkDenoisers enum
        context: Pre-built context with citation graph already computed
    
    Returns:
        Tuple of (eigenvectors, eigenvalues) where eigenvectors is (d, d)
    """
    d = context.X().shape[1]
    
    denoiser_enum = WorkDenoisers[denoiser_name]
    denoiser_impl = denoiser_enum.value
    
    # Handle NoneDenoiser specially - use identity
    if denoiser_name == "NONE":
        return np.eye(d, dtype=context.X().dtype), None
    
    # For ProjectiveDenoiser, get full eigenbasis
    if isinstance(denoiser_impl, ProjectiveDenoiser):
        inner_denoiser = denoiser_impl.denoiser_impl
        
        # Get full projector (all components) - reuse the passed context
        P, eigvals = inner_denoiser.projector(context, n_components=d)
        
        # Ensure we have the full basis
        if P.shape[1] < d:
            LOGGER.warning(f"{denoiser_name}: Only got {P.shape[1]} eigenvectors, padding with zeros")
            P_full = np.zeros((d, d), dtype=P.dtype)
            P_full[:, :P.shape[1]] = P
            P = P_full
        
        return P, eigvals
    
    raise ValueError(f"Unsupported denoiser type: {type(denoiser_impl)}")


def compute_variance_explained(
    eigenvectors: NDArray,
    scatter_matrices: Dict[str, NDArray],
) -> Dict[str, NDArray]:
    """Compute % variance explained by each eigenvector for each scatter matrix.
    
    Args:
        eigenvectors: (d, d) matrix where columns are eigenvectors
        scatter_matrices: Dict mapping matrix name to (d, d) scatter matrix
        
    Returns:
        Dict mapping matrix name to (d,) array of variance fractions per eigenvector
        
    Note:
        Eigenvectors from GEP solvers are normalized s.t. v.T @ B @ v = 1, not v.T @ v = 1.
        We normalize to unit length here so variance values are comparable across denoisers
        and represent "variance along this direction" rather than solver-specific scaling.
    """
    # Normalize eigenvectors to unit length for fair comparison
    norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)  # Avoid division by zero
    eigenvectors_normalized = eigenvectors / norms
    
    result = {}
    
    for name, S in scatter_matrices.items():
        trace_S = np.trace(S)
        if trace_S < 1e-10:
            # Matrix is essentially zero
            result[name] = np.zeros(eigenvectors_normalized.shape[1])
            continue
        
        # Compute v_i.T @ S @ v_i for each eigenvector
        # Efficient: diag(V.T @ S @ V)
        projected = eigenvectors_normalized.T @ S @ eigenvectors_normalized
        variances = np.diag(projected)
        
        # Convert to percentage
        result[name] = variances / trace_S
    
    return result


def plot_variance_by_matrix(
    results: List[EigenbasisResult],
    output_path: Path,
    max_components: Optional[int] = None,
) -> None:
    """Create 2x2 subplot: one panel per scatter matrix, showing all denoisers."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    matrix_names = ["S_T", "S_B", "S_W", "S_L"]
    matrix_titles = [
        "S_T (Total Scatter)",
        "S_B (Between-Class)",
        "S_W (Within-Class)", 
        "S_L (Laplacian)",
    ]
    
    # Color map for denoisers
    n_denoisers = len(results)
    cmap = plt.colormaps.get_cmap('tab20')
    colors = [cmap(i / n_denoisers) for i in range(n_denoisers)]
    
    # Determine max components from data if not specified
    if max_components is None and results:
        max_components = len(next(iter(results[0].variance_explained.values())))
    
    for ax_idx, (matrix_name, title) in enumerate(zip(matrix_names, matrix_titles)):
        ax = axes[ax_idx]
        
        for i, res in enumerate(results):
            var_exp = res.variance_explained.get(matrix_name)
            if var_exp is None:
                continue
            
            # Cumulative variance explained
            n_comp = max_components if max_components else len(var_exp)
            cumvar = np.cumsum(var_exp[:n_comp])
            
            ax.plot(range(1, len(cumvar) + 1), cumvar, 
                   label=res.name, color=colors[i], alpha=0.8, linewidth=1.5)
        
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Variance Explained")
        ax.set_title(title)
        ax.set_xlim(1, max_components or 100)
        # Auto-scale y-axis to fit data
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    
    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    
    plt.suptitle("Cumulative Variance Explained by Denoiser Eigenbasis\n(per Scatter Matrix)", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(right=0.82)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    LOGGER.info(f"Saved variance-by-matrix plot to {output_path}")


def plot_single_denoiser(
    result: EigenbasisResult,
    output_dir: Path,
    max_components: Optional[int] = None,
) -> None:
    """Create a plot for a single denoiser showing all 4 scatter matrices."""
    
    matrix_styles = {
        "S_T": ("black", "-", "S_T (Total)"),
        "S_B": ("red", "--", "S_B (Between)"),
        "S_W": ("green", "-", "S_W (Within)"),
        "S_L": ("orange", "--", "S_L (Laplacian)"),
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine max components from data if not specified
    if max_components is None:
        max_components = len(next(iter(result.variance_explained.values())))
    
    for matrix_name, (color, linestyle, label) in matrix_styles.items():
        var_exp = result.variance_explained.get(matrix_name)
        if var_exp is None:
            continue
        
        n_comp = max_components if max_components else len(var_exp)
        cumvar = np.cumsum(var_exp[:n_comp])
        ax.plot(range(1, len(cumvar) + 1), cumvar,
               color=color, linestyle=linestyle, label=label, linewidth=2)
    
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title(f"Variance Explained by {result.name} Eigenbasis")
    ax.set_xlim(1, max_components or 100)
    # Auto-scale y-axis to fit data
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    output_path = output_dir / f"eigenbasis_{result.name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    LOGGER.info(f"Saved {result.name} eigenbasis plot to {output_path}")


def plot_components_to_threshold(
    results: List[EigenbasisResult],
    output_path: Path,
    thresholds: List[float] = [0.5, 0.9, 0.95, 0.99],
) -> None:
    """Bar chart: how many components needed to reach X% variance for each matrix."""
    
    matrix_names = ["S_T", "S_B", "S_W", "S_L"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax_idx, threshold in enumerate(thresholds):
        ax = axes[ax_idx]
        
        denoiser_names = [r.name for r in results]
        x = np.arange(len(denoiser_names))
        width = 0.2
        
        for i, matrix_name in enumerate(matrix_names):
            components_needed = []
            for res in results:
                var_exp = res.variance_explained.get(matrix_name)
                if var_exp is None:
                    components_needed.append(0)
                    continue
                cumvar = np.cumsum(var_exp)
                idx = np.searchsorted(cumvar, threshold)
                components_needed.append(min(idx + 1, len(var_exp)))
            
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, components_needed, width, label=matrix_name, alpha=0.8)
        
        ax.set_xlabel("Denoiser")
        ax.set_ylabel("Components Needed")
        ax.set_title(f"Components to Reach {threshold*100:.0f}% Variance")
        ax.set_xticks(x)
        ax.set_xticklabels(denoiser_names, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle("Components Needed to Capture Variance Threshold", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    LOGGER.info(f"Saved components-to-threshold plot to {output_path}")


def plot_variance_separation(
    results: List[EigenbasisResult],
    output_path: Path,
    k_values: List[int] = [10, 20, 50, 100],
) -> None:
    """Plot showing how well each denoiser separates signal (S_W) from noise (S_B, S_L).
    
    For each k, compute:
    - Signal captured in top-k: sum of S_W variance in first k components
    - Noise captured in top-k: sum of (S_B + S_L) variance in first k components
    - Ideal: high signal, low noise in top-k
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax_idx, k in enumerate(k_values):
        ax = axes[ax_idx]
        
        signal_captured = []
        noise_b_captured = []
        noise_l_captured = []
        denoiser_names = []
        
        for res in results:
            var_w = res.variance_explained.get("S_W")
            var_b = res.variance_explained.get("S_B")
            var_l = res.variance_explained.get("S_L")
            
            if var_w is None or var_b is None or var_l is None:
                continue
            
            denoiser_names.append(res.name)
            signal_captured.append(np.sum(var_w[:k]))
            noise_b_captured.append(np.sum(var_b[:k]))
            noise_l_captured.append(np.sum(var_l[:k]))
        
        x = np.arange(len(denoiser_names))
        width = 0.25
        
        ax.bar(x - width, signal_captured, width, label='S_W (Signal)', color='green', alpha=0.7)
        ax.bar(x, noise_b_captured, width, label='S_B (Class Noise)', color='red', alpha=0.7)
        ax.bar(x + width, noise_l_captured, width, label='S_L (Graph Noise)', color='orange', alpha=0.7)
        
        ax.set_xlabel("Denoiser")
        ax.set_ylabel("Cumulative Variance in Top-k")
        ax.set_title(f"Signal vs Noise in Top {k} Components")
        ax.set_xticks(x)
        ax.set_xticklabels(denoiser_names, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    plt.suptitle("Signal (S_W) vs Noise (S_B, S_L) Separation by Denoiser", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    LOGGER.info(f"Saved variance separation plot to {output_path}")


def _result_to_rows(
    result: EigenbasisResult,
    max_components: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convert a single EigenbasisResult to CSV rows."""
    rows = []
    num_components = len(next(iter(result.variance_explained.values())))
    n_comp = max_components if max_components else num_components
    for component_idx in range(min(n_comp, num_components)):
        row = {
            "denoiser": result.name,
            "component": component_idx + 1,
        }
        for matrix_name, var_exp in result.variance_explained.items():
            row[f"{matrix_name}_var_explained"] = var_exp[component_idx] if component_idx < len(var_exp) else 0
            row[f"{matrix_name}_cumvar"] = np.sum(var_exp[:component_idx+1]) if component_idx < len(var_exp) else np.sum(var_exp)
        rows.append(row)
    return rows


def append_result_to_csv(
    result: EigenbasisResult,
    output_path: Path,
    max_components: Optional[int] = None,
) -> None:
    """Append a single denoiser's results to CSV (incremental saving)."""
    rows = _result_to_rows(result, max_components)
    df = pd.DataFrame(rows)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Append mode: write header only if file doesn't exist
    write_header = not output_path.exists()
    df.to_csv(output_path, mode='a', index=False, header=write_header)
    LOGGER.info(f"Appended {result.name} results to {output_path}")


def load_completed_denoisers(output_path: Path) -> set:
    """Load set of denoiser names already in CSV (for resume support)."""
    if not output_path.exists():
        return set()
    try:
        df = pd.read_csv(output_path, usecols=['denoiser'])
        completed = set(df['denoiser'].unique())
        return completed
    except Exception as e:
        LOGGER.warning(f"Could not read existing CSV for resume: {e}")
        return set()


def load_results_from_csv(output_path: Path) -> List[EigenbasisResult]:
    """Load EigenbasisResults from CSV (for regenerating summary plots after resume).
    
    Note: This reconstructs variance_explained from the CSV but not eigenvectors/eigenvalues
    (which aren't stored). This is sufficient for generating summary plots.
    """
    if not output_path.exists():
        return []
    
    try:
        df = pd.read_csv(output_path)
        results = []
        
        for denoiser_name in df['denoiser'].unique():
            denoiser_df = df[df['denoiser'] == denoiser_name].sort_values('component')
            
            # Reconstruct variance_explained dict
            variance_explained = {}
            for col in denoiser_df.columns:
                if col.endswith('_var_explained'):
                    matrix_name = col.replace('_var_explained', '')
                    variance_explained[matrix_name] = denoiser_df[col].values
            
            results.append(EigenbasisResult(
                name=denoiser_name,
                eigenvectors=np.array([]),  # Not stored in CSV
                eigenvalues=None,
                variance_explained=variance_explained,
            ))
        
        LOGGER.info(f"Loaded {len(results)} results from {output_path}")
        return results
    except Exception as e:
        LOGGER.warning(f"Could not load results from CSV: {e}")
        return []


def save_results_csv(
    results: List[EigenbasisResult],
    output_path: Path,
    max_components: Optional[int] = None,
) -> None:
    """Save variance explained data to CSV (full rewrite)."""
    
    rows = []
    for res in results:
        rows.extend(_result_to_rows(res, max_components))
    
    df = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info(f"Saved variance data to {output_path}")


def _run_analysis(cfg: DenoiserAnalysisConfig) -> None:
    """Execute the eigenbasis analysis."""
    output_dir = Path(cfg.output_dir) / "eigenbasis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("=" * 60)
    LOGGER.info("Eigenbasis Analysis")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Output directory: {output_dir}")
    
    # Determine which denoisers to analyze
    if cfg.denoisers is None:
        denoiser_names = [d.name for d in WorkDenoisers if d.name != "NONE"]
    else:
        denoiser_names = list(cfg.denoisers)
    
    # Apply blacklist
    skip_set = set(cfg.skip_denoisers)
    if skip_set:
        original_count = len(denoiser_names)
        denoiser_names = [d for d in denoiser_names if d not in skip_set]
        skipped = original_count - len(denoiser_names)
        if skipped > 0:
            LOGGER.info(f"Skipping {skipped} blacklisted denoisers: {list(skip_set)}")
    
    LOGGER.info(f"Analyzing {len(denoiser_names)} denoisers: {denoiser_names}")
    
    # Get embedder model
    embedder_model = cfg.experiment.cloud.embedder_model
    if isinstance(embedder_model, str):
        embedder_model = Embedder.ModelName[embedder_model]
    LOGGER.info(f"Embedder model: {embedder_model.value}")
    
    # Load data
    ai_pre_filter = Filter(
        subfield=SubField.AI.value,
        work_types=(WorkType.BOOK.value, WorkType.ARTICLE.value),
    )
    
    LOGGER.info("Loading works and embeddings...")
    works, embeddings = load_works(ai_pre_filter, cfg.max_works, embedder_model)
    LOGGER.info(f"Loaded {len(works)} works with embeddings shape {embeddings.shape}")
    
    embeddings = normalize(embeddings, norm="l2", axis=1)
    LOGGER.info("Normalized embeddings")
    
    # Define filters
    book_filter = Filter(work_types=(WorkType.BOOK.value,))
    article_filter = Filter(work_types=(WorkType.ARTICLE.value,))
    
    # Build context and get scatter matrices
    LOGGER.info("Building context and computing scatter matrices...")
    context = CloudDenoiserContext(works, embeddings, book_filter, article_filter)
    
    # Compute properly weighted S_W: S_W = (n_0/n)*S_0 + (n_1/n)*S_1
    # This ensures S_T = S_B + S_W relationship holds
    n_0 = context.X_n(0).shape[0]
    n_1 = context.X_n(1).shape[0]
    n_total = n_0 + n_1
    LOGGER.info(f"Class sizes: n_0={n_0}, n_1={n_1}, total={n_total}")
    
    S_W = (n_0 / n_total) * context.S_subset(0) + (n_1 / n_total) * context.S_subset(1)
    
    scatter_matrices = {
        "S_T": context.S_T(),
        "S_B": context.S_B_multi(list(range(context.num_subsets()))),
        "S_W": S_W,
        "S_L": context.S_L(),
    }
    
    LOGGER.info("Scatter matrix traces:")
    for name, S in scatter_matrices.items():
        LOGGER.info(f"  Tr({name}) = {np.trace(S):.4f}")
    
    # Check for existing results (resume support)
    csv_path = output_dir / f"eigenbasis_variance_{embedder_model.name}.csv"
    completed_denoisers = load_completed_denoisers(csv_path)
    if completed_denoisers:
        LOGGER.info(f"Found {len(completed_denoisers)} completed denoisers in existing CSV (resume mode)")
        LOGGER.info(f"  Completed: {sorted(completed_denoisers)}")
    
    # Analyze each denoiser
    results: List[EigenbasisResult] = []
    
    for denoiser_name in denoiser_names:
        # Skip if already completed (resume support)
        if denoiser_name in completed_denoisers:
            LOGGER.info(f"\nSkipping {denoiser_name} (already in CSV)")
            continue
        
        LOGGER.info(f"\nAnalyzing {denoiser_name}...")
        
        try:
            eigenvectors, eigenvalues = get_full_eigenbasis(
                denoiser_name=denoiser_name,
                context=context,
            )
            
            LOGGER.info(f"  Got eigenbasis with shape {eigenvectors.shape}")
            
            # Compute variance explained
            var_explained = compute_variance_explained(eigenvectors, scatter_matrices)
            
            # Log summary
            for matrix_name, var_exp in var_explained.items():
                cumvar_10 = np.sum(var_exp[:10])
                cumvar_50 = np.sum(var_exp[:50])
                LOGGER.info(f"  {matrix_name}: top-10={cumvar_10:.3f}, top-50={cumvar_50:.3f}")
            
            result = EigenbasisResult(
                name=denoiser_name,
                eigenvectors=eigenvectors,
                eigenvalues=eigenvalues,
                variance_explained=var_explained,
            )
            results.append(result)
            
            # Save outputs incrementally (crash-safe)
            plot_single_denoiser(result, output_dir)
            append_result_to_csv(result, csv_path)
            
        except Exception as e:
            LOGGER.error(f"Failed to analyze {denoiser_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        LOGGER.error("No denoisers were successfully analyzed!")
        return
    
    # Generate outputs
    LOGGER.info("\nGenerating outputs...")
    
    # Note: CSV was saved incrementally during analysis.
    # If we have new results, the CSV already contains them.
    # We only do a full rewrite if we want to consolidate (e.g., after resume).
    if results and completed_denoisers:
        # Load previously completed results from CSV for the summary plots
        LOGGER.info("Loading previously completed results for summary plots...")
        # The incremental CSV already has all data, just log it
        LOGGER.info(f"CSV at {csv_path} contains all results (incremental + resumed)")
    
    # For summary plots, we need all results including any from previous runs
    # If we resumed, `results` only has new ones, so reload from CSV
    if completed_denoisers and csv_path.exists():
        LOGGER.info("Reloading all results from CSV for summary plots...")
        all_results = load_results_from_csv(csv_path)
    else:
        all_results = results
    
    if not all_results:
        LOGGER.warning("No results available for summary plots")
    else:
        # 2x2 plot: one panel per matrix
        plot_variance_by_matrix(
            all_results,
            output_dir / "variance_by_matrix.png",
        )
        
        # Note: Individual per-denoiser plots are saved incrementally during analysis
        
        # Components to threshold
        plot_components_to_threshold(
            all_results,
            output_dir / "components_to_threshold.png",
        )
        
        # Signal vs noise separation
        plot_variance_separation(
            all_results,
            output_dir / "signal_noise_separation.png",
        )
    
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("EIGENBASIS ANALYSIS COMPLETE")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Results written to: {output_dir}")


def regenerate_plots(csv_path: Path, output_dir: Optional[Path] = None) -> None:
    """Regenerate all plots from an existing CSV file.
    
    Args:
        csv_path: Path to eigenbasis_variance_*.csv file
        output_dir: Output directory for plots (defaults to csv_path.parent)
    """
    if output_dir is None:
        output_dir = csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(f"Regenerating plots from {csv_path}")
    LOGGER.info(f"Output directory: {output_dir}")
    
    results = load_results_from_csv(csv_path)
    if not results:
        LOGGER.error("No results loaded from CSV!")
        return
    
    LOGGER.info(f"Loaded {len(results)} denoisers")
    
    # Generate individual plots
    for result in results:
        plot_single_denoiser(result, output_dir)
    
    # Generate summary plots
    plot_variance_by_matrix(results, output_dir / "variance_by_matrix.png")
    plot_components_to_threshold(results, output_dir / "components_to_threshold.png")
    plot_variance_separation(results, output_dir / "signal_noise_separation.png")
    
    LOGGER.info("Plot regeneration complete!")


@hydra.main(version_base=None, config_path=None, config_name="eigenbasis_analysis")
def main(cfg: DenoiserAnalysisConfig) -> None:
    """Main entry point for eigenbasis analysis."""
    _run_analysis(cfg)


if __name__ == "__main__":
    import sys
    
    # Check for --regenerate-plots flag
    if len(sys.argv) >= 2 and sys.argv[1] == "--regenerate-plots":
        if len(sys.argv) < 3:
            print("Usage: python -m artifacts_scripts.denoiser_analysis.eigenbasis_analysis --regenerate-plots <csv_path> [output_dir]")
            sys.exit(1)
        csv_path = Path(sys.argv[2])
        output_dir = Path(sys.argv[3]) if len(sys.argv) >= 4 else None
        regenerate_plots(csv_path, output_dir)
    else:
        main()
