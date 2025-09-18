#!/usr/bin/env python3
"""
Batch Proposal Generator - Clean entry point for LLM proposal generation.

This is the main entry point for the new batch-first architecture. It uses Hydra
for configuration management and provides a simple CLI interface.

Usage:
    python batch_generator.py                                     # Default (production) configuration
    python batch_generator.py config=debug                        # Debug preset  
    python batch_generator.py config=production                   # Production preset (explicit)
    python batch_generator.py config=high_quality                 # High quality preset
    python batch_generator.py batch.num_gaps=50                   # Override specific values
    python batch_generator.py gap=debug llm=creative              # Mix component presets
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


@hydra.main(version_base=None, config_path=None, config_name="production")
def main(cfg: ExperimentConfig) -> None:
    """Main entry point for batch proposal generation.
    
    Args:
        cfg: Structured experiment configuration (automatically converted by Hydra)
    """
    # Log startup information
    LOGGER.info("=" * 60)
    LOGGER.info("BookGap Batch Proposal Generator")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Experiment: {cfg.experiment_name}")
    LOGGER.info(f"Description: {cfg.description}")
    LOGGER.info(f"Working directory: {os.getcwd()}")
    LOGGER.info(f"Output directory: {HydraConfig.get().runtime.output_dir}")
    LOGGER.info("")
    
    # Validate environment
    if not os.getenv("OPENAI_API_KEY"):
        LOGGER.error("OPENAI_API_KEY environment variable is required")
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    try:
        # Run the complete batch workflow
        LOGGER.info(f"Starting batch workflow with {cfg.batch.num_gaps} gaps...")
        results = run_batch_workflow(cfg)
        
        # Log completion
        LOGGER.info("=" * 60)
        LOGGER.info("WORKFLOW COMPLETED SUCCESSFULLY")
        LOGGER.info("=" * 60)
        LOGGER.info(f"Batch ID: {results['batch_id']}")
        LOGGER.info(f"Generated proposals: {results['num_results']}")
        LOGGER.info(f"Request counts: {results['request_counts']}")
        LOGGER.info(f"Output directory: {HydraConfig.get().runtime.output_dir}")
        
    except Exception as e:
        LOGGER.error("=" * 60)
        LOGGER.error("WORKFLOW FAILED")
        LOGGER.error("=" * 60)
        LOGGER.error(f"Error: {str(e)}")
        LOGGER.error(f"Error type: {type(e).__name__}")
        if hasattr(e, 'context'):
            LOGGER.error(f"Error context: {e.context}")  # type: ignore
        raise


if __name__ == "__main__":
    main()
