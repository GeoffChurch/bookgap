"""
Path and filename management utilities for bookgap analysis.

This module handles all path construction, parsing, and filename operations
without depending on LLM types, avoiding circular dependencies.
"""

import json
import re
from pathlib import Path
from typing import NamedTuple

from bookgap.util import create_unicode_safe_filename


class CustomIdComponents(NamedTuple):
    """
    Parsed components of a custom_id for batch operations.
    
    Moved from config_manager.py to avoid circular dependencies.
    """
    gap_config: str
    topic_name: str
    topic_id: str
    gap_idx: int
    denoiser: str
    drop_k: int
    radius_fraction: float
    llm_subdir: str


class CustomIdManager:
    """
    Manages building and parsing of custom IDs for batch operations.
    
    Custom ID format: "{gap_config}___{topic_name}___id={topic_id}___gap_{gap_idx}"
    Example: "NONE_k=5_radiusfrac=0.75___quantum_computing___id=1234___gap_3"
    """
    
    @staticmethod
    def build_custom_id(gap_config_str: str, llm_subdir_str: str, topic_name: str, 
                       topic_id: str, gap_idx: int) -> str:
        """
        Build a custom_id from configuration strings.
        
        Args:
            gap_config_str: Gap configuration string
            llm_subdir_str: LLM configuration subdirectory string
            topic_name: Topic name (should be already sanitized)
            topic_id: Topic ID string
            gap_idx: Gap index (must be >= 0)
            
        Returns:
            Custom ID string
            
        Raises:
            ValueError: If gap_idx < 0 or inputs are invalid
        """
        if gap_idx < 0:
            raise ValueError(f"gap_idx must be >= 0, got {gap_idx}")
        if not topic_name.strip():
            raise ValueError("topic_name cannot be empty")
        if not topic_id.strip():
            raise ValueError("topic_id cannot be empty")
        
        config_str = f"{gap_config_str}_{llm_subdir_str}"
        
        return f"{config_str}___{topic_name}___id={topic_id}___gap_{gap_idx}"
    
    @staticmethod
    def parse_custom_id(custom_id: str) -> CustomIdComponents:
        """
        Parse a custom_id into its components.
        
        Args:
            custom_id: Custom ID string to parse
            
        Returns:
            CustomIdComponents with all parsed fields
            
        Raises:
            ValueError: If custom_id format is invalid
        """
        if not custom_id.strip():
            raise ValueError("custom_id cannot be empty")
        
        # Split by triple underscore - expecting exactly 4 parts now
        parts = custom_id.split("___")
        if len(parts) != 4:
            raise ValueError(f"Expected 4 parts in custom_id separated by '___', got {len(parts)}: {custom_id}")
        
        config_str, topic_name, topic_id_part, gap_info = parts

        # Extract topic_id from "id=topic_id" format
        if not topic_id_part.startswith("id="):
            raise ValueError(f"topic_id part must start with 'id=', got: {topic_id_part}")
        topic_id = topic_id_part[3:]  # Remove "id=" prefix

        # Extract gap_idx from "gap_X" format
        if not gap_info.startswith("gap_"):
            raise ValueError(f"gap_info part must start with 'gap_', got: {gap_info}")
        try:
            gap_idx = int(gap_info[4:])  # Extract number after "gap_"
        except ValueError:
            raise ValueError(f"gap_idx must be an integer after 'gap_', got: {gap_info}")

        # Parse configuration string to extract components
        # We need to split gap config from llm config
        # Find where llm config starts (looks for "llm_" pattern)
        config_parts = config_str.split("_")
        llm_start_idx = None
        for i, part in enumerate(config_parts):
            if part == "llm" and i < len(config_parts) - 1:
                llm_start_idx = i
                break
        
        if llm_start_idx is None:
            raise ValueError(f"Could not find LLM config in custom_id: {custom_id}")
        
        gap_config_parts = config_parts[:llm_start_idx]
        llm_config_parts = config_parts[llm_start_idx:]
        
        gap_config_str = "_".join(gap_config_parts)
        llm_subdir = "_".join(llm_config_parts)
        
        # Extract individual gap config components for convenience
        # Parse denoiser, drop_k, radius_fraction from gap_config_str
        # Find k= parameter
        k_idx = None
        for i, part in enumerate(gap_config_parts):
            if part.startswith("k="):
                k_idx = i
                break
        
        if k_idx is None:
            raise ValueError(f"Could not find 'k=X' parameter in gap config: {gap_config_str}")
        
        denoiser = "_".join(gap_config_parts[:k_idx])
        drop_k = int(gap_config_parts[k_idx][2:])  # Remove "k=" prefix
        
        # Find radiusfrac parameter
        if k_idx + 1 >= len(gap_config_parts) or not gap_config_parts[k_idx + 1].startswith("radiusfrac="):
            raise ValueError(f"Could not find 'radiusfrac=X' parameter in gap config: {gap_config_str}")
        
        radius_fraction = float(gap_config_parts[k_idx + 1][len("radiusfrac="):])

        return CustomIdComponents(
            gap_config=gap_config_str,
            topic_name=topic_name,
            topic_id=topic_id,
            gap_idx=gap_idx,
            denoiser=denoiser,
            drop_k=drop_k,
            radius_fraction=radius_fraction,
            llm_subdir=llm_subdir
        )


class TopicPathManager:
    """
    Manages topic names, sanitization, and filename/path operations.
    
    Centralizes topic-related path handling with no dependencies on LLM types.
    """
    
    @staticmethod
    def sanitize_topic_name(topic_name: str) -> str:
        """
        Sanitize topic name for safe use in filenames and paths.
        
        Args:
            topic_name: Raw topic name that may contain problematic characters
            
        Returns:
            Sanitized topic name safe for filenames
            
        Raises:
            ValueError: If topic_name is empty after sanitization
        """
        if not topic_name or not topic_name.strip():
            raise ValueError("topic_name cannot be empty")
        
        sanitized = create_unicode_safe_filename(topic_name.strip())
        if not sanitized:
            raise ValueError(f"topic_name becomes empty after sanitization: '{topic_name}'")
        
        return sanitized
    
    @staticmethod
    def extract_topic_name_from_filename(filename: str) -> str:
        """
        Extract clean topic name from a proposal filename.
        
        Expected filename format: "Topic_Name_id=TopicID_gap_X_proposal.tex"
        
        Args:
            filename: Filename to parse
            
        Returns:
            Clean topic name with underscores replaced by spaces
            
        Raises:
            ValueError: If filename format is invalid
        """
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty")
        
        topic_name = filename.strip()
        
        # Remove .tex extension if present
        if topic_name.endswith('.tex'):
            topic_name = topic_name[:-4]  # len('.tex') = 4
        
        # Remove _proposal suffix if present
        if topic_name.endswith('_proposal'):
            topic_name = topic_name[:-9]  # len('_proposal') = 9
        
        # Extract just the topic name (everything before _id=)
        if '_id=' not in topic_name:
            raise ValueError(f"Expected '_id=' in filename: {filename}")
        
        topic_name = topic_name.split('_id=')[0]
        
        if not topic_name:
            raise ValueError(f"Topic name is empty after parsing filename: {filename}")
        
        return topic_name.replace('_', ' ')
    
    @staticmethod
    def extract_topic_id_from_filename(filename: str) -> str:
        """
        Extract topic ID from a filename.
        
        Expected filename format: "Topic_Name_id=TopicID_gap_X_proposal.tex"
        
        Args:
            filename: Filename to parse
            
        Returns:
            Topic ID string
            
        Raises:
            ValueError: If filename format is invalid or topic ID cannot be extracted
        """
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty")
        
        filename = filename.strip()
        
        # Remove extensions and suffixes
        if filename.endswith('.tex'):
            filename = filename[:-4]
        if filename.endswith('_proposal'):
            filename = filename[:-9]
        
        # Find the id= part
        if '_id=' not in filename:
            raise ValueError(f"Expected '_id=' in filename: {filename}")
        
        parts = filename.split('_id=')
        if len(parts) != 2:
            raise ValueError(f"Expected exactly one '_id=' in filename: {filename}")
        
        # Topic ID is everything after _id= until the next underscore or end
        id_part = parts[1]
        
        # Find where the topic ID ends (at next underscore)
        if '_' in id_part:
            topic_id = id_part.split('_')[0]
        else:
            topic_id = id_part
        
        if not topic_id:
            raise ValueError(f"Topic ID is empty after parsing filename: {filename}")
        
        return topic_id
    
    @staticmethod
    def build_analysis_filename(topic_name: str, topic_id: str) -> str:
        """
        Build analysis filename from topic name and ID.
        
        Format: "{sanitized_topic_name}_id={topic_id}_analysis.txt"
        
        Args:
            topic_name: Raw topic name
            topic_id: Topic ID
            
        Returns:
            Analysis filename
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not topic_id or not topic_id.strip():
            raise ValueError("topic_id cannot be empty")
        
        sanitized_name = TopicPathManager.sanitize_topic_name(topic_name)
        return f"{sanitized_name}_id={topic_id.strip()}_analysis.txt"
    
    @staticmethod
    def build_prompt_filename(topic_name: str, topic_id: str) -> str:
        """
        Build prompt filename from topic name and ID.
        
        Format: "{sanitized_topic_name}_id={topic_id}_prompt.txt"
        
        Args:
            topic_name: Raw topic name
            topic_id: Topic ID
            
        Returns:
            Prompt filename
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not topic_id or not topic_id.strip():
            raise ValueError("topic_id cannot be empty")
        
        sanitized_name = TopicPathManager.sanitize_topic_name(topic_name)
        return f"{sanitized_name}_id={topic_id.strip()}_prompt.txt"
    
    @staticmethod
    def build_plot_filename(topic_name: str, gap_idx: int, method: str, plot_type: str) -> str:
        """
        Build plot filename from components.
        
        Format: "{sanitized_topic_name}_gap_{gap_idx}_{method}_{plot_type}.png"
        
        Args:
            topic_name: Raw topic name
            gap_idx: Gap index
            method: Analysis method name
            plot_type: Type of plot
            
        Returns:
            Plot filename
            
        Raises:
            ValueError: If inputs are invalid
        """
        if gap_idx < 0:
            raise ValueError(f"gap_idx must be >= 0, got {gap_idx}")
        if not method or not method.strip():
            raise ValueError("method cannot be empty")
        if not plot_type or not plot_type.strip():
            raise ValueError("plot_type cannot be empty")
        
        sanitized_name = TopicPathManager.sanitize_topic_name(topic_name)
        sanitized_method = create_unicode_safe_filename(method.strip().lower())
        sanitized_plot_type = create_unicode_safe_filename(plot_type.strip())
        
        return f"{sanitized_name}_gap_{gap_idx}_{sanitized_method}_{sanitized_plot_type}.png"


class ConfigPathParser:
    """
    Centralized parsing of configuration information from file paths and directory structures.
    
    Handles the standard gap analysis directory structure without depending on configuration classes.
    """
    
    @staticmethod 
    def parse_prompt_file_path(prompt_file_path: str, gaps_base_dir: str) -> dict:
        """
        Parse configuration information from a prompt file path.
        
        Expected structure: {gaps_base_dir}/{gap_config}/gap_{idx}/prompts/{llm_subdir}/{filename}
        
        Args:
            prompt_file_path: Full path to prompt file
            gaps_base_dir: Base gaps directory path
            
        Returns:
            Dictionary with parsed components:
            - gap_config: Configuration string
            - llm_config: LLM configuration string  
            - topic_name: Extracted topic name
            - topic_id: Extracted topic ID
            - gap_idx: Gap index
            - filename: Original filename
            
        Raises:
            ValueError: If path structure is invalid or components cannot be parsed
            FileNotFoundError: If path doesn't exist relative to gaps_base_dir
        """
        prompt_path = Path(prompt_file_path)
        gaps_dir = Path(gaps_base_dir)
        
        # Get relative path from gaps directory
        try:
            relative_path = prompt_path.relative_to(gaps_dir)
        except ValueError:
            raise FileNotFoundError(f"Prompt file not in gaps directory: {prompt_file_path} (base: {gaps_base_dir})")
        
        # Expected structure: config/gap_X/prompts/articles_Y_books_Z/filename
        if len(relative_path.parts) != 5:
            raise ValueError(f"Expected 5 path parts (config/gap_X/prompts/llm_subdir/filename), got {len(relative_path.parts)}: {relative_path}")
        
        gap_config_str, gap_dir, prompts_subdir, llm_subdir, filename = relative_path.parts
        
        # Validate prompts subdirectory
        if prompts_subdir != "prompts":
            raise ValueError(f"Expected 'prompts' subdirectory, got '{prompts_subdir}' in {relative_path}")
        
        # Extract gap index from "gap_X" format
        gap_idx_pattern = re.compile(r'^gap_(\d+)$')
        gap_match = gap_idx_pattern.match(gap_dir)
        if not gap_match:
            raise ValueError(f"Invalid gap directory format, expected 'gap_X', got '{gap_dir}'")
        gap_idx = int(gap_match.group(1))
        
        # Extract topic name and ID from filename
        topic_id_pattern = re.compile(r'^(.+)_id=([^_]+)_prompt\.txt$')
        topic_match = topic_id_pattern.match(filename)
        if not topic_match:
            raise ValueError(f"Invalid prompt filename format, expected 'topic_id=ID_prompt.txt', got '{filename}'")
        
        topic_name = topic_match.group(1)
        topic_id = topic_match.group(2)
        
        return {
            'gap_config': gap_config_str,
            'llm_config': llm_subdir,
            'topic_name': topic_name,
            'topic_id': topic_id,
            'gap_idx': gap_idx,
            'filename': filename,
            'gap_config_str': gap_config_str,
            'llm_subdir': llm_subdir
        }
    
    @staticmethod
    def get_config_dir_from_custom_id(custom_id: str, gaps_base_dir: str) -> str:
        """
        Get configuration directory path from a custom ID.
        
        Args:
            custom_id: Custom ID to parse
            gaps_base_dir: Base gaps directory path
            
        Returns:
            Full path to configuration directory
            
        Raises:
            ValueError: If custom_id cannot be parsed
        """
        components = CustomIdManager.parse_custom_id(custom_id)
        return str(Path(gaps_base_dir) / components.gap_config)
    
    @staticmethod
    def get_config_dir_from_results_file(results_file_path: str, gaps_base_dir: str) -> str:
        """
        Get configuration directory path from a batch results file.
        
        Args:
            results_file_path: Path to JSONL batch results file
            gaps_base_dir: Base gaps directory path
            
        Returns:
            Full path to configuration directory
            
        Raises:
            ValueError: If results file is empty or custom_id cannot be parsed
            FileNotFoundError: If results file doesn't exist
        """
        results_file = Path(results_file_path)
        if not results_file.exists():
            raise FileNotFoundError(f"Results file does not exist: {results_file_path}")
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line:
                    raise ValueError(f"Results file is empty: {results_file_path}")
                
                result = json.loads(first_line)
                if 'custom_id' not in result:
                    raise ValueError(f"No custom_id found in results file: {results_file_path}")
                
                return ConfigPathParser.get_config_dir_from_custom_id(result['custom_id'], gaps_base_dir)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in results file {results_file_path}: {e}")
        except IOError as e:
            raise ValueError(f"Error reading results file {results_file_path}: {e}")
    
    @staticmethod
    def validate_directory_structure(base_path: str, expected_structure: list) -> bool:
        """
        Validate that a directory follows expected structure.
        
        Args:
            base_path: Base directory path to validate
            expected_structure: List of expected subdirectory patterns
            
        Returns:
            True if structure is valid
            
        Raises:
            ValueError: If structure is invalid (when fail_fast=True)
        """
        base_dir = Path(base_path)
        if not base_dir.exists():
            raise ValueError(f"Directory does not exist: {base_path}")
        
        if not base_dir.is_dir():
            raise ValueError(f"Path is not a directory: {base_path}")
        
        # This is a simple validation - can be extended based on needs
        for expected_subdir in expected_structure:
            subdir_path = base_dir / expected_subdir
            if not subdir_path.exists():
                raise ValueError(f"Expected subdirectory not found: {subdir_path}")
        
        return True
