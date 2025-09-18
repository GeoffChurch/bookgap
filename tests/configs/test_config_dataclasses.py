"""
Test suite for the new Hydra-based configuration system.

Tests all configuration dataclasses, ConfigStore registration, and preset configs.
This ensures our foundational configuration system works correctly before building
the rest of the pipeline.
"""

import pytest
from dataclasses import FrozenInstanceError

from bookgap.configs.config import ExperimentConfig
from bookgap.configs.gap.gap_config import GapConfig
from bookgap.configs.llm.llm_config import LLMConfig
from bookgap.configs.data.data_config import DataConfig
from bookgap.configs.batch.batch_config import BatchConfig
from bookgap.configs.output.output_config import OutputConfig
from bookgap.configs.output.export_formats import ExportFormat

# Import required enums
from bookgap.types import CellType
from bookgap.cloud import WorkDenoisers
from bookgap.index import IndexType
from bookgap.llm.providers import ProviderType
from bookgap.util import WorkType, SubField, Filters


class TestGapConfig:
    """Test GapConfig dataclass functionality."""
    
    def test_gap_config_creation(self):
        """Test basic GapConfig creation with required parameters."""
        config = GapConfig(
            cell_type=CellType.KBALL,
            k=10,
            radius_fraction=0.75,
            drop_k=0,
            hnsw_m=16,
            hnsw_efc=512,
            hnsw_efs=1024,
            denoiser=WorkDenoisers.NONE,
            index_type=IndexType.HNSW,
            num_gaps=1
        )
        
        assert config.cell_type == CellType.KBALL
        assert config.radius_fraction == 0.75
        assert config.denoiser == WorkDenoisers.NONE
        assert config.num_gaps == 1
    
    def test_gap_config_frozen(self):
        """Test that GapConfig is immutable (frozen)."""
        config = GapConfig(
            cell_type=CellType.KBALL,
            k=10,
            radius_fraction=0.75,
            drop_k=0,
            hnsw_m=16,
            hnsw_efc=512,
            hnsw_efs=1024,
            denoiser=WorkDenoisers.NONE,
            index_type=IndexType.HNSW,
            num_gaps=1
        )
        
        # Should not be able to modify frozen dataclass
        with pytest.raises(FrozenInstanceError):
            config.radius_fraction = 0.5
            

class TestLLMConfig:
    """Test LLMConfig dataclass functionality."""
    
    def test_llm_config_creation(self):
        """Test basic LLMConfig creation with required parameters."""
        config = LLMConfig(
            provider_type=ProviderType.OPENAI,
            model_name="gpt-5",
            num_articles=10,
            num_books=5,
            temperature=0.2,
            max_tokens=2000,
            chapters_min=10,
            chapters_max=15,
            min_book_coverage=0.5,
            use_structured_output=True,
            include_citations=True
        )
        
        assert config.provider_type == ProviderType.OPENAI
        assert config.model_name == "gpt-5"
        assert config.num_articles == 10
        assert config.num_books == 5
        assert config.temperature == 0.2
        assert config.max_tokens == 2000
        assert config.chapters_min == 10
        assert config.chapters_max == 15
        assert config.use_structured_output is True
    
    def test_llm_config_frozen(self):
        """Test that LLMConfig is immutable (frozen)."""
        config = LLMConfig(
            provider_type=ProviderType.OPENAI,
            model_name="gpt-5",
            num_articles=10,
            num_books=5,
            temperature=0.2,
            max_tokens=2000,
            chapters_min=10,
            chapters_max=15,
            min_book_coverage=0.5,
            use_structured_output=True,
            include_citations=True
        )
        
        with pytest.raises(FrozenInstanceError):
            config.temperature = 0.8


class TestDataConfig:
    """Test DataConfig dataclass functionality."""
    
    def test_data_config_creation(self):
        """Test basic DataConfig creation."""
        config = DataConfig(
            max_works=None,
            deduplicator=None
        )
        
        assert config.max_works is None
        assert config.deduplicator is None
    
    def test_data_config_with_max_works(self):
        """Test DataConfig with work limit."""
        config = DataConfig(
            subfield=SubField.AI,
            max_works=1000,
            work_types=(WorkType.BOOK, WorkType.ARTICLE),
            work_filters=(),
            filter_preset=Filters.NLP,
            custom_filter=None,
            deduplicator=None
        )
        
        assert config.max_works == 1000


class TestBatchConfig:
    """Test BatchConfig dataclass functionality."""
    
    def test_batch_config_creation(self):
        """Test basic BatchConfig creation."""
        config = BatchConfig(
            num_gaps=1,
            completion_window="24h",
            generate_plots=True,
            save_prompts=True,
            save_batch_file=True
        )
        
        assert config.num_gaps == 1
        assert config.completion_window == "24h"
        assert config.generate_plots is True
        assert config.save_prompts is True
        assert config.save_batch_file is True


class TestOutputConfig:
    """Test OutputConfig dataclass functionality."""
    
    def test_output_config_creation(self):
        """Test basic OutputConfig creation."""
        config = OutputConfig(
            export_formats=(ExportFormat.CSV, ExportFormat.JSON),
            create_subdirectories=True,
            include_metadata=True,
            use_timestamp=True
        )
        
        assert config.export_formats == (ExportFormat.CSV, ExportFormat.JSON)
        assert config.create_subdirectories is True
        assert config.include_metadata is True
        assert config.use_timestamp is True
    
    def test_output_config_minimal(self):
        """Test minimal OutputConfig creation."""
        config = OutputConfig(
            export_formats=(ExportFormat.JSON,),
            create_subdirectories=False,
            include_metadata=False,
            use_timestamp=False
        )
        
        assert config.export_formats == (ExportFormat.JSON,)
        assert config.create_subdirectories is False


class TestExperimentConfig:
    """Test top-level ExperimentConfig functionality."""
    
    def test_experiment_config_creation(self):
        """Test complete ExperimentConfig creation."""
        gap_config = GapConfig(
            subfield=1702,
            work_types=["book", "article"],
            cell_type=CellType.KBALL,
            k=10,
            radius_fraction=0.75,
            drop_k=0,
            hnsw_m=16,
            hnsw_efc=512,
            hnsw_efs=1024,
            denoiser=WorkDenoisers.NONE,
            index_type=IndexType.HNSW,
            num_gaps=1
        )
        
        llm_config = LLMConfig(
            provider_type=ProviderType.OPENAI,
            model_name="gpt-5",
            num_articles=10,
            num_books=5,
            temperature=0.2,
            max_tokens=2000,
            chapters_min=10,
            chapters_max=15,
            min_book_coverage=0.5,
            use_structured_output=True,
            include_citations=True
        )
        
        data_config = DataConfig(
            subfield=SubField.AI,
            max_works=None,
            work_types=(WorkType.BOOK, WorkType.ARTICLE),
            work_filters=(),
            filter_preset=Filters.NLP,
            custom_filter=None,
            deduplicator=None
        )
        
        batch_config = BatchConfig(
            num_gaps=1,
            completion_window="24h",
            generate_plots=True,
            save_prompts=True,
            save_batch_file=True
        )
        
        output_config = OutputConfig(
            export_formats=(ExportFormat.CSV, ExportFormat.JSON),
            create_subdirectories=True,
            include_metadata=True,
            use_timestamp=True
        )
        
        experiment_config = ExperimentConfig(
            gap=gap_config,
            llm=llm_config,
            data=data_config,
            batch=batch_config,
            output=output_config,
            experiment_name="test_experiment"
        )
        
        assert experiment_config.gap == gap_config
        assert experiment_config.llm == llm_config
        assert experiment_config.data == data_config
        assert experiment_config.batch == batch_config
        assert experiment_config.output == output_config
        assert experiment_config.experiment_name == "test_experiment"
    
    def test_experiment_config_frozen(self):
        """Test that ExperimentConfig is immutable."""
        # Create minimal configs for testing
        gap_config = GapConfig(
            subfield=1702, work_types=["book"], cell_type=CellType.KBALL,
            k=10, radius_fraction=0.75, drop_k=0, hnsw_m=16, hnsw_efc=512,
            hnsw_efs=1024, denoiser=WorkDenoisers.NONE, index_type=IndexType.HNSW,
            num_gaps=1
        )
        
        llm_config = LLMConfig(
            provider_type=ProviderType.OPENAI, model_name="gpt-5",
            num_articles=10, num_books=5, temperature=0.2, max_tokens=2000,
            chapters_min=10, chapters_max=15, min_book_coverage=0.5,
            use_structured_output=True, include_citations=True
        )
        
        data_config = DataConfig(
            subfield=SubField.AI, max_works=None, work_types=(WorkType.BOOK,),
            work_filters=(), filter_preset=Filters.NLP, custom_filter=None,
            deduplicator=None
        )
        
        batch_config = BatchConfig(
            num_gaps=1, completion_window="24h", generate_plots=True,
            save_prompts=True, save_batch_file=True
        )
        
        output_config = OutputConfig(
            export_formats=(ExportFormat.JSON,), create_subdirectories=False,
            include_metadata=False, use_timestamp=False
        )
        
        experiment_config = ExperimentConfig(
            gap=gap_config, llm=llm_config, data=data_config,
            batch=batch_config, output=output_config,
            experiment_name="test"
        )
        
        with pytest.raises(FrozenInstanceError):
            experiment_config.experiment_name = "modified"


class TestExportFormats:
    """Test ExportFormat enum functionality."""
    
    def test_export_format_values(self):
        """Test that ExportFormat enum has expected values."""
        assert ExportFormat.CSV.value == "csv"
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.PARQUET.value == "parquet"
        assert ExportFormat.JSONL.value == "jsonl"
    
    def test_export_format_iteration(self):
        """Test that we can iterate over all export formats."""
        formats = list(ExportFormat)
        assert len(formats) == 4  # CSV, JSON, PARQUET, JSONL
        assert ExportFormat.CSV in formats
        assert ExportFormat.JSON in formats
        assert ExportFormat.PARQUET in formats
        assert ExportFormat.JSONL in formats
