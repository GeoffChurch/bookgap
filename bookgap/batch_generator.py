#!/usr/bin/env python3
"""
Batch Proposal Generator - Clean entry point for LLM proposal generation.

This is the main entry point for the new batch-first architecture. It uses Hydra
for configuration management and provides a simple CLI interface.

Usage:
    python batch_generator.py                                     # Default preset (standard)
    python batch_generator.py --config-name anti_lda_drop1        # Switch to a registered preset
    python batch_generator.py --config-name standard batch.num_gaps=5  # Override specific values
    python batch_generator.py gap.cell_type=VORONOI llm.temperature=0.4  # Override nested fields
"""

import hydra
from hydra.core.hydra_config import HydraConfig
import os

from bookgap.configs.config import ExperimentConfig
from bookgap.configs.config_store import register_configs
from bookgap.llm.generators.workflow_functions import run_batch_workflow
from bookgap.util import LOGGER


# Ensure configs are registered
register_configs()


@hydra.main(version_base=None, config_path="configs", config_name="standard")
def main(cfg: ExperimentConfig) -> None:
    """Main entry point for batch proposal generation.
    
    Args:
        cfg: Structured experiment configuration (automatically converted by Hydra)
    """
    # Log startup information
    LOGGER.info("=" * 60)
    LOGGER.info("BookGap Batch Proposal Generator")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Working directory: {os.getcwd()}")
    LOGGER.info(f"Output directory: {HydraConfig.get().runtime.output_dir}")
    LOGGER.info("")
    
    # Validate environment
    if not os.getenv("OPENAI_API_KEY"):
        LOGGER.error("OPENAI_API_KEY environment variable is required")
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # Run the complete batch workflow
    results = run_batch_workflow(cfg)
    
    # Log completion
    LOGGER.info("=" * 60)
    LOGGER.info("WORKFLOW COMPLETED SUCCESSFULLY")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Generated proposals: {len(results.proposals)}")
    LOGGER.info(f"Output directory: {HydraConfig.get().runtime.output_dir}")


if __name__ == "__main__":
    main()
