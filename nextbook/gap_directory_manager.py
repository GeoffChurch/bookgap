"""
Centralized directory management for gap analysis workflows.

This module consolidates all directory creation and path management logic
used across gap_finder.py and batch_operations.py, eliminating duplicate
mkdir() patterns and ensuring consistent directory structure.
"""

from pathlib import Path
from typing import Optional, Union
from bookgap.config_manager import AnalysisConfiguration


class GapDirectoryManager:
    """Centralized directory management for gap analysis workflows."""
    
    # Base directories - shared constants
    ARTIFACTS_DIR = Path("artifacts")
    GAPS_SUBDIR = "gaps"
    
    # Standard subdirectories within gap structures
    ANALYSIS_SUBDIR = "analysis"
    PLOTS_SUBDIR = "plots" 
    PROMPTS_SUBDIR = "prompts"
    RESPONSES_SUBDIR = "responses"
    BATCH_REQUESTS_SUBDIR = "batch_requests"
    BATCH_RESULTS_SUBDIR = "batch_results"
    
    @classmethod
    def ensure_directory(cls, path: Path) -> Path:
        """Ensure directory exists using standard mkdir pattern.
        
        Args:
            path: Directory path to create
            
        Returns:
            The same path after ensuring it exists
        """
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @classmethod
    def get_base_artifacts_dir(cls) -> Path:
        """Get the base artifacts directory, creating if needed."""
        return cls.ensure_directory(cls.ARTIFACTS_DIR)
    
    @classmethod
    def get_gaps_base_dir(cls) -> Path:
        """Get the base gaps directory, creating if needed."""
        return cls.ensure_directory(cls.ARTIFACTS_DIR / cls.GAPS_SUBDIR)
    
    @classmethod
    def get_config_dir(cls, config: AnalysisConfiguration) -> Path:
        """Get configuration directory for gap analysis parameters.
        
        Args:
            config: AnalysisConfiguration object
            
        Returns:
            Path to the configuration directory
        """
        config_str = config.to_directory_name()
        return cls.ensure_directory(cls.get_gaps_base_dir() / config_str)
    
    @classmethod 
    def get_gap_dir(cls, config: AnalysisConfiguration, gap_idx: int) -> Path:
        """Get directory for a specific gap.
        
        Args:
            config: AnalysisConfiguration object  
            gap_idx: Index of the gap (must be >= 0)
            
        Returns:
            Path to the gap directory
            
        Raises:
            ValueError: If gap_idx < 0
        """
        if gap_idx < 0:
            raise ValueError(f"gap_idx must be >= 0, got {gap_idx}")
        
        config_dir = cls.get_config_dir(config)
        return cls.ensure_directory(config_dir / f"gap_{gap_idx}")
    
    @classmethod
    def get_analysis_dir(cls, config: AnalysisConfiguration, gap_idx: int) -> Path:
        """Get analysis subdirectory for a specific gap."""
        return cls.ensure_directory(cls.get_gap_dir(config, gap_idx) / cls.ANALYSIS_SUBDIR)
    
    @classmethod
    def get_plots_dir(cls, config: AnalysisConfiguration, gap_idx: Optional[int] = None) -> Path:
        """Get plots subdirectory for a gap or config-level plots.
        
        Args:
            config: AnalysisConfiguration object
            gap_idx: Index of the gap (>= 0), or None for config-level plots
            
        Returns:
            Path to the plots directory
            
        Raises:
            ValueError: If gap_idx < 0
        """
        if gap_idx is not None:
            return cls.ensure_directory(cls.get_gap_dir(config, gap_idx) / cls.PLOTS_SUBDIR)
        else:
            return cls.ensure_directory(cls.get_config_dir(config) / cls.PLOTS_SUBDIR)
    
    @classmethod
    def get_plots_llm_dir(cls, config: AnalysisConfiguration, gap_idx: int) -> Path:
        """Get plots subdirectory for LLM-specific plots (plots that include proposals).
        
        These plots contain proposal data that varies by LLM configuration,
        so they need to be isolated per LLM config to prevent overwrites.
        
        Args:
            config: AnalysisConfiguration object
            gap_idx: Index of the gap (must be >= 0)
            
        Returns:
            Path to the LLM-specific plots directory
        """
        if gap_idx < 0:
            raise ValueError(f"gap_idx must be >= 0, got {gap_idx}")
        
        # Structure: gap_config/llm_config/gap_N/plots/
        llm_config_dir = cls.get_llm_config_dir(config)
        return cls.ensure_directory(llm_config_dir / f"gap_{gap_idx}" / cls.PLOTS_SUBDIR)
    
    @classmethod
    def get_prompts_dir(cls, config: AnalysisConfiguration, gap_idx: int) -> Path:
        """Get prompts subdirectory for a specific gap."""
        return cls.ensure_directory(cls.get_gap_dir(config, gap_idx) / cls.PROMPTS_SUBDIR)
    
    @classmethod
    def get_prompts_llm_dir(cls, config: AnalysisConfiguration, gap_idx: int) -> Path:
        """Get prompts subdirectory for specific LLM input counts.
        
        Args:
            config: AnalysisConfiguration object
            gap_idx: Index of the gap (must be >= 0)
            
        Returns:
            Path to the LLM-specific prompts directory
        """
        prompts_dir = cls.get_prompts_dir(config, gap_idx)
        llm_subdir_name = config.to_llm_subdir_name()
        return cls.ensure_directory(prompts_dir / llm_subdir_name)
    
    @classmethod
    def get_responses_dir(cls, config: AnalysisConfiguration, gap_idx: int) -> Path:
        """Get responses subdirectory for a specific gap."""
        return cls.ensure_directory(cls.get_gap_dir(config, gap_idx) / cls.RESPONSES_SUBDIR)
    
    @classmethod
    def get_responses_llm_dir(cls, config: AnalysisConfiguration, gap_idx: int) -> Path:
        """Get responses subdirectory for specific LLM input counts.
        
        Args:
            config: AnalysisConfiguration object
            gap_idx: Index of the gap (must be >= 0)
            
        Returns:
            Path to the LLM-specific responses directory
        """
        responses_dir = cls.get_responses_dir(config, gap_idx)
        llm_subdir_name = config.to_llm_subdir_name()
        return cls.ensure_directory(responses_dir / llm_subdir_name)
    
    @classmethod
    def get_batch_requests_dir(cls, config: AnalysisConfiguration) -> Path:
        """Get batch requests subdirectory for LLM-specific configuration."""
        config_dir = cls.get_config_dir(config)
        llm_subdir_name = config.to_llm_subdir_name()
        return cls.ensure_directory(config_dir / llm_subdir_name / cls.BATCH_REQUESTS_SUBDIR)
    
    @classmethod
    def get_batch_results_dir(cls, config: AnalysisConfiguration) -> Path:
        """Get batch results subdirectory for LLM-specific configuration."""
        config_dir = cls.get_config_dir(config)
        llm_subdir_name = config.to_llm_subdir_name()
        return cls.ensure_directory(config_dir / llm_subdir_name / cls.BATCH_RESULTS_SUBDIR)
    
    @classmethod
    def get_llm_config_dir(cls, config: AnalysisConfiguration) -> Path:
        """Get LLM-specific configuration directory.
        
        This directory stores files that should be isolated per LLM configuration,
        such as gap_analysis.csv, to prevent overwrites between different
        prompt configurations (e.g., 5,5 vs 10,5).
        
        Args:
            config: AnalysisConfiguration object
            
        Returns:
            Path to the LLM-specific directory
        """
        config_dir = cls.get_config_dir(config)
        llm_subdir_name = config.to_llm_subdir_name()
        return cls.ensure_directory(config_dir / llm_subdir_name)
    
    # Convenience methods for common file paths
    
    @classmethod
    def get_batch_request_file(cls, config: AnalysisConfiguration, timestamp: int) -> Path:
        """Get path for a batch request file in the LLM-specific configuration directory."""
        return cls.get_batch_requests_dir(config) / f"batch_request_{timestamp}.jsonl"
    
    @classmethod
    def get_batch_results_file(cls, config: AnalysisConfiguration, batch_id: str) -> Path:
        """Get path for a batch results file in the LLM-specific configuration directory."""
        return cls.get_batch_results_dir(config) / f"results_{batch_id}.jsonl"
    
    @classmethod
    def get_gap_analysis_file(cls, config: AnalysisConfiguration) -> Path:
        """Get path for gap_analysis.csv in the LLM-specific configuration directory.
        
        This ensures each LLM configuration (e.g., 5,5 vs 10,5) has its own
        gap_analysis.csv file to prevent data overwrites.
        
        Args:
            config: AnalysisConfiguration object
            
        Returns:
            Path to the LLM-specific gap_analysis.csv file
        """
        return cls.get_llm_config_dir(config) / "gap_analysis.csv"


