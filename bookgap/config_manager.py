"""
Configuration management for bookgap analysis.

This module provides centralized configuration classes for gap analysis and LLM proposal generation.
"""

from dataclasses import dataclass
from typing import NamedTuple

from bookgap.types import CellType
from bookgap.llm.llm_implementation import Implementation
from bookgap.llm.providers import ProviderType
from bookgap.cloud import WorkDenoisers
from bookgap.util import create_unicode_safe_filename


@dataclass(frozen=True)
class GapConfiguration:
    """
    Core gap analysis parameters - independent of LLM processing.
    
    These parameters determine which research gaps are found and how the gap analysis runs.
    They correspond to the directory structure level: artifacts/gaps/{gap_config}/
    
    String format: "{denoiser}_k={drop_k}_radiusfrac={radius_fraction:.2f}_{cell_type}"
    """
    denoiser: str
    drop_k: int
    radius_fraction: float
    cell_type: CellType
    
    def __post_init__(self) -> None:
        """Validate gap configuration parameters."""
        # Gap configuration validation - denoiser must be a valid WorkDenoiser name
        denoisers = [d.name for d in WorkDenoisers]
        if self.denoiser not in denoisers:
            raise ValueError(f"denoiser must be one of {denoisers}, got '{self.denoiser}'")
        if self.drop_k < 0:
            raise ValueError(f"drop_k must be >= 0, got {self.drop_k}")
        if not (0.0 <= self.radius_fraction <= 1.0):
            raise ValueError(f"radius_fraction must be in [0.0, 1.0], got {self.radius_fraction}")
        
        # CellType validation is handled by the enum type
        if not isinstance(self.cell_type, CellType):
            raise ValueError(f"cell_type must be a CellType enum, got {type(self.cell_type)}")
    
    def to_directory_name(self) -> str:
        """Convert to directory name format."""
        return f"{self.denoiser}_k={self.drop_k}_radiusfrac={self.radius_fraction:.2f}_{self.cell_type.name.lower()}"
    
    @classmethod
    def from_directory_name(cls, dir_name: str) -> "GapConfiguration":
        """
        Parse gap configuration from directory name format.
        
        Args:
            dir_name: Directory name like "NONE_k=0_radiusfrac=0.75_kball"
            
        Returns:
            GapConfiguration instance
            
        Raises:
            ValueError: If directory name format is invalid
        """
        parts = dir_name.split("_")
        if len(parts) < 4:
            raise ValueError(f"Invalid gap configuration directory format. Expected at least 4 parts, got {len(parts)}: {dir_name}")
        
        try:
            # Find the index of the first parameter (k=X)
            k_idx = None
            for i, part in enumerate(parts):
                if part.startswith("k="):
                    k_idx = i
                    break
            
            if k_idx is None:
                raise ValueError(f"Could not find 'k=X' parameter in: {dir_name}")
            
            # Denoiser is everything before the k= parameter, joined with underscores
            denoiser = "_".join(parts[:k_idx])
            
            # Extract drop_k from "k=X" format
            if not parts[k_idx].startswith("k="):
                raise ValueError(f"Expected 'k=X', got: {parts[k_idx]}")
            drop_k = int(parts[k_idx][2:])
            
            # Extract radius_fraction from "radiusfrac=Y" format
            if not parts[k_idx + 1].startswith("radiusfrac="):
                raise ValueError(f"Expected 'radiusfrac=Y', got: {parts[k_idx + 1]}")
            radius_fraction = float(parts[k_idx + 1][len("radiusfrac="):])
            
            # Cell type
            cell_type_str = parts[k_idx + 2]
            if cell_type_str == "kball":
                cell_type = CellType.KBALL
            elif cell_type_str == "voronoi":
                cell_type = CellType.VORONOI
            else:
                raise ValueError(f"Cell type must be 'kball' or 'voronoi', got: {cell_type_str}")
            
            return cls(
                denoiser=denoiser,
                drop_k=drop_k,
                radius_fraction=radius_fraction,
                cell_type=cell_type
            )
            
        except (ValueError, IndexError) as e:
            if isinstance(e, ValueError) and "Invalid gap configuration directory format" in str(e):
                raise  # Re-raise our custom message
            raise ValueError(f"Invalid gap configuration directory format: {dir_name}. Error: {e}")


@dataclass(frozen=True)
class LLMConfiguration:
    """
    LLM processing parameters - independent of gap analysis.
    
    These parameters determine how research gaps are turned into LLM prompts.
    They correspond to the subdirectory structure level: prompts/{llm_config}/
    
    String format: "llm_{implementation}_{articles}_{books}" (breaking change from old format)
    """
    implementation: Implementation
    requested_num_prompt_articles: int  # Articles to include in prompt
    requested_num_prompt_books: int  # Books to include in prompt
    provider: ProviderType = ProviderType.OPENAI  # Changed from str to enum
    model: str = "gpt-5"  # Default model (supports structured outputs)
    temperature: float = 0.2  # Default temperature
    max_tokens: int = 2000  # Default max tokens
    chapters_min: int = 10  # Minimum chapters in proposal
    chapters_max: int = 15  # Maximum chapters in proposal
    
    def __post_init__(self) -> None:
        """Validate LLM configuration parameters."""
        if self.requested_num_prompt_articles < 0:
            raise ValueError(f"requested_num_prompt_articles must be non-negative, got {self.requested_num_prompt_articles}")
        if self.requested_num_prompt_books < 0:
            raise ValueError(f"requested_num_prompt_books must be non-negative, got {self.requested_num_prompt_books}")
        if self.provider != ProviderType.OPENAI:  # Updated validation
            raise ValueError(f"Only ProviderType.OPENAI is currently supported, got '{self.provider}'")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature must be in [0.0, 2.0], got {self.temperature}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if not (1 <= self.chapters_min <= self.chapters_max <= 50):
            raise ValueError(f"chapters must satisfy 1 <= min <= max <= 50, got min={self.chapters_min}, max={self.chapters_max}")
        
        # Validate that implementation is correct type (will be checked after registry is imported)
        # We defer this validation since Implementation enum may not be available yet
    
    def to_subdir_name(self) -> str:
        """Convert to LLM subdirectory name format."""
        return f"llm_{self.implementation.name.lower()}_articles_{self.requested_num_prompt_articles}_books_{self.requested_num_prompt_books}"
    
    @classmethod
    def from_subdir_name(cls, subdir_name: str) -> "LLMConfiguration":
        """
        Parse LLM configuration from subdirectory name format.
        
        Expected format: "llm_{implementation}_articles_{X}_books_{Y}"
        
        Args:
            subdir_name: Subdirectory name like "llm_structured_articles_10_books_5"
            
        Returns:
            LLMConfiguration instance
            
        Raises:
            ValueError: If subdirectory name format is invalid
        """
        parts = subdir_name.split("_")
        
        if len(parts) != 6:
            raise ValueError(f"Invalid LLM configuration subdirectory format. Expected 6 parts (llm_impl_articles_X_books_Y), got {len(parts)}: {subdir_name}")
        
        try:
            # Expected format: "llm_structured_articles_X_books_Y"
            if parts[0] != "llm":
                raise ValueError(f"Expected subdirectory to start with 'llm', got: {parts[0]}")
            if parts[2] != "articles":
                raise ValueError(f"Expected third part to be 'articles', got: {parts[2]}")
            if parts[4] != "books":
                raise ValueError(f"Expected fifth part to be 'books', got: {parts[4]}")
            
            # Parse implementation
            impl_name = parts[1].upper()
            try:
                implementation = Implementation[impl_name]
            except KeyError:
                raise ValueError(f"Unknown implementation: {impl_name}. Valid options: {[e.name for e in Implementation]}")
            
            requested_num_prompt_articles = int(parts[3])
            requested_num_prompt_books = int(parts[5])
            
            return cls(
                implementation=implementation,
                requested_num_prompt_articles=requested_num_prompt_articles,
                requested_num_prompt_books=requested_num_prompt_books
            )
            
        except (ValueError, IndexError) as e:
            if "Invalid LLM configuration subdirectory format" in str(e):
                raise  # Re-raise our custom message
            raise ValueError(f"Invalid LLM configuration subdirectory format: {subdir_name}. Error: {e}")


@dataclass(frozen=True)
class AnalysisConfiguration:
    """
    Consolidated configuration combining gap analysis and LLM processing parameters.
    
    String format: "{gap_config}_{llm_config}"
    """
    gap_config: GapConfiguration
    llm_config: LLMConfiguration
    
    def to_directory_name(self) -> str:
        """Convert to directory name format (gap config only)."""
        return self.gap_config.to_directory_name()
    
    def to_llm_subdir_name(self) -> str:
        """Convert to LLM subdirectory name format."""
        return self.llm_config.to_subdir_name()
    
    def to_string(self) -> str:
        """Convert to full configuration string format."""
        return f"{self.gap_config.to_directory_name()}_{self.llm_config.to_subdir_name()}"
    
    @classmethod
    def from_components(cls, gap_config: GapConfiguration, llm_config: LLMConfiguration) -> "AnalysisConfiguration":
        """Create from separate configuration components."""
        return cls(gap_config=gap_config, llm_config=llm_config)
    
    @classmethod 
    def from_string(cls, config_str: str) -> "AnalysisConfiguration":
        """
        Parse configuration from full string format.
        
        Args:
            config_str: Configuration string like
                "NONE_k=0_radiusfrac=0.75_kball_articles=50_books=10"
                "JAD_EQUAL_k=5_radiusfrac=0.5_voronoi_articles=100_books=25"
            
        Returns:
            AnalysisConfiguration instance
            
        Raises:
            ValueError: If string format is invalid
        """
        # Split on underscores and find where LLM config starts
        parts = config_str.split("_")
        if len(parts) < 6:
            raise ValueError(f"Invalid configuration string format. Expected at least 6 parts, got {len(parts)}: {config_str}")
        
        # Find the articles part to split gap config from LLM config
        articles_idx = None
        for i, part in enumerate(parts):
            if part.startswith("articles=") or part == "articles":
                articles_idx = i
                break
        
        if articles_idx is None:
            raise ValueError(f"Could not find 'articles' parameter in: {config_str}")
        
        # Split into gap config and LLM config parts
        gap_parts = parts[:articles_idx]
        llm_parts = parts[articles_idx:]
        
        gap_config_str = "_".join(gap_parts)
        llm_config_str = "_".join(llm_parts)
        
        try:
            gap_config = GapConfiguration.from_directory_name(gap_config_str)
            llm_config = LLMConfiguration.from_subdir_name(llm_config_str)
            
            return cls.from_components(gap_config, llm_config)
            
        except ValueError as e:
            raise ValueError(f"Invalid configuration string format: {config_str}. Error: {e}")


class CustomIdComponents(NamedTuple):
    """
    Parsed components of a custom_id for batch operations.
    
    Moved from batch_operations.py to centralize custom ID handling.
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
    
    Custom ID format: "{gap_config}___{topic_name}___id={topic_id}___gap_{gap_idx}___{llm_subdir}"
    Example: "NONE_k=5_radiusfrac=0.75___quantum_computing___id=1234___gap_3___articles_10_books_5"
    """
    
    @staticmethod
    def build_custom_id(analysis_config: AnalysisConfiguration, topic_name: str, topic_id: str, 
                       gap_idx: int) -> str:
        """
        Build a custom_id from AnalysisConfiguration components.
        
        Args:
            analysis_config: Consolidated analysis configuration
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
        
        config_str = analysis_config.to_string()
        
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

        # Parse configuration to extract individual components
        analysis_config = AnalysisConfiguration.from_string(config_str)

        return CustomIdComponents(
            gap_config=config_str,  # Full config string
            topic_name=topic_name,
            topic_id=topic_id,
            gap_idx=gap_idx,
            denoiser=analysis_config.gap_config.denoiser,
            drop_k=analysis_config.gap_config.drop_k,
            radius_fraction=analysis_config.gap_config.radius_fraction,
            llm_subdir=analysis_config.to_llm_subdir_name()
        )


class TopicPathManager:
    """
    Manages topic names, sanitization, and filename/path operations.
    
    Centralizes topic-related path handling previously scattered across modules.
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
    
    Handles the standard gap analysis directory structure:
    artifacts/gaps/{gap_config}/gap_{idx}/prompts/{llm_subdir}/{topic}_id={topic_id}_prompt.txt
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
        from pathlib import Path
        import re
        
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
        
        # Parse configuration components separately
        gap_config = GapConfiguration.from_directory_name(gap_config_str)
        llm_config = LLMConfiguration.from_subdir_name(llm_subdir)
        analysis_config = AnalysisConfiguration.from_components(gap_config, llm_config)
        
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
            'llm_config': analysis_config.to_llm_subdir_name(),
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
        from pathlib import Path
        
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
        import json
        from pathlib import Path
        
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
        from pathlib import Path
        
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
