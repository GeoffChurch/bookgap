"""Workflow artifacts orchestration for BookGap analysis.

This module coordinates the generation of all analysis artifacts,
including plots, context files, results tables, and other outputs.
"""

from pathlib import Path
from typing import List
from hydra.core.hydra_config import HydraConfig

from ..data_structures import GapWithContext, BatchResults, ArtifactPaths
from bookgap.llm.artifacts.gap_plots import generate_gap_plots_for_results
from bookgap.util import LOGGER


def generate_phase1_artifacts(gaps: List[GapWithContext], results: BatchResults) -> ArtifactPaths:
    """Generate Phase 1 artifacts: gap plots with proposals.
    
    Args:
        gaps: List of gaps with context
        results: Batch results containing proposals
        
    Returns:
        ArtifactPaths with plot paths populated
    """
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    LOGGER.info(f"Generating Phase 1 artifacts in {output_dir}")
    
    # Generate gap plots with proposals
    plot_paths = generate_gap_plots_for_results(gaps, results.proposals, output_dir)
    
    LOGGER.info(f"Phase 1 complete: Generated {len(plot_paths)} gap plots")
    
    return ArtifactPaths(plots=plot_paths)


def generate_all_artifacts(gaps: List[GapWithContext], results: BatchResults) -> ArtifactPaths:
    """Generate all artifacts (currently only Phase 1 implemented).
    
    This function will be expanded as more phases are implemented.
    
    Args:
        gaps: List of gaps with context
        results: Batch results containing proposals
        
    Returns:
        ArtifactPaths with all available artifact paths
    """
    # For now, only Phase 1 is implemented
    return generate_phase1_artifacts(gaps, results)
