"""
Modern gap visualization with unified plotting architecture.

This module provides clean, well-tested gap plotting functionality using
standardized data structures and a unified plotting engine.
"""

from pathlib import Path
from typing import List, Dict

from bookgap.llm.data_structures import GapWithContext
from bookgap.llm.protocols import ParsedProposal
from bookgap.plot_utils import UnifiedGapPlotter
from bookgap.util import LOGGER
from bookgap.visualization import PlotConfig, ProjectionMethod

from .data_preparation import GapVisualizationDataPreparer


def save_gap_plot_with_proposals(
    gap_with_context: GapWithContext,
    proposals: List[ParsedProposal],
    method: str = "PCA"
) -> Path:
    """
    Save gap visualization plot with proposals overlay.
    
    COMPLETELY REWRITTEN to use modern, testable architecture.
    
    Args:
        gap_with_context: Gap with associated context data
        proposals: List of parsed proposals to overlay on plot
        method: Projection method ("PCA" or "TSNE")
        
    Returns:
        Path to saved plot file
        
    Raises:
        ValueError: If insufficient data for plotting
    """
    LOGGER.info(f"Creating gap plot for gap {gap_with_context.gap_idx} with {len(proposals)} proposals")
    
    # Step 1: Prepare visualization data using unified pipeline
    preparer = GapVisualizationDataPreparer()
    projection_method = (ProjectionMethod.pca() if method.upper() == "PCA" 
                        else ProjectionMethod.tsne())
    
    try:
        data = preparer.prepare_standard_gap_data(
            gap_with_context, proposals, projection_method
        )
    except ValueError as e:
        LOGGER.error(f"Failed to prepare visualization data: {e}")
        raise
    
    # Step 2: Configure plot appearance
    config = PlotConfig(
        method_name=method,
        topic_name=gap_with_context.topic_name,
        gap_idx=gap_with_context.gap_idx,
        figsize=(12, 10),
        show_legend=True
    )
    
    # Step 3: Create and save plot using unified engine
    plotter = UnifiedGapPlotter()
    
    # Determine save directory - using a simple approach for now
    from hydra.core.hydra_config import HydraConfig
    try:
        hydra_cfg = HydraConfig.get()
        base_dir = Path(hydra_cfg.runtime.output_dir) / "plots" / f"gap_{gap_with_context.gap_idx}"
    except:
        # Fallback to temp directory if Hydra not available
        base_dir = Path("/tmp/gap_plots") / f"gap_{gap_with_context.gap_idx}"
    
    try:
        save_path = plotter.create_and_save_gap_plot(data, config, base_dir)
        LOGGER.info(f"Saved gap plot to {save_path}")
        return save_path
        
    except Exception as e:
        LOGGER.error(f"Failed to create gap plot: {e}")
        raise ValueError(f"Failed to create gap plot: {e}")


def generate_gap_plots_for_results(
    gaps: List[GapWithContext], 
    results: List[ParsedProposal],
    output_dir: Path
) -> List[Path]:
    """
    Generate gap plots for all gaps with their corresponding proposals.
    
    REWRITTEN to use modern architecture and better error handling.
    
    Args:
        gaps: List of gaps with context
        results: List of all parsed proposals
        output_dir: Base output directory (Hydra managed)
        
    Returns:
        List of paths to generated plot files
    """
    LOGGER.info(f"Generating plots for {len(gaps)} gaps with {len(results)} total proposals")
    
    # Group proposals by gap index (extract from custom_id)
    proposals_by_gap: Dict[int, List[ParsedProposal]] = {}
    for proposal in results:
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
    plot_paths = []
    successful_plots = 0
    
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
            successful_plots += 1
            
        except Exception as e:
            LOGGER.error(f"Failed to generate plot for gap {gap_idx}: {e}")
            continue
    
    LOGGER.info(f"Generated {successful_plots}/{len(gaps)} gap plots successfully")
    return plot_paths
