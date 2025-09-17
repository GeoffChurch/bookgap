"""
ConfigStore registration for BookGap configurations.

Registers preset configurations that combine existing codebase defaults
with common use case patterns.
"""

from hydra.core.config_store import ConfigStore

from .config import ExperimentConfig
from .gap.gap_config import GapConfig
from .llm.llm_config import LLMConfig
from .data.data_config import DataConfig
from .batch.batch_config import BatchConfig
from .output.output_config import OutputConfig
from .output.export_formats import ExportFormat

# Import required enums and types
from bookgap.types import CellType
from bookgap.cloud import WorkDenoisers
from bookgap.index import IndexType
from bookgap.llm.providers import ProviderType
from bookgap.util import WorkType, SubField, Filters


def register_configs():
    """Register all preset configurations with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    
    # ===== GAP CONFIGURATIONS =====
    
    # Production gap config (from parse_args.py + batch_workflow.py)
    production_gap = GapConfig(
        subfield=1702,  # Artificial Intelligence
        work_types=["book", "article"],
        cell_type=CellType.KBALL,  # From batch_workflow.py
        k=10,  # From parse_args.py
        radius_fraction=0.75,  # From parse_args.py
        drop_k=0,  # From parse_args.py
        hnsw_m=16,  # From parse_args.py
        hnsw_efc=512,  # From parse_args.py
        hnsw_efs=1024,  # From parse_args.py
        denoiser=WorkDenoisers.NONE,  # From parse_args.py
        index_type=IndexType.HNSW,  # From parse_args.py
        num_gaps=1
    )
    cs.store(group="gap", name="production", node=production_gap)
    
    # Debug gap config (faster settings)
    debug_gap = GapConfig(
        subfield=1702,
        work_types=["book", "article"],
        cell_type=CellType.KBALL,
        k=5,  # Smaller k for speed
        radius_fraction=0.5,  # Smaller radius for speed
        drop_k=0,
        hnsw_m=8,  # Smaller m for speed
        hnsw_efc=200,  # Lower efc for speed
        hnsw_efs=400,  # Lower efs for speed
        denoiser=WorkDenoisers.NONE,
        index_type=IndexType.HNSW,
        num_gaps=1
    )
    cs.store(group="gap", name="debug", node=debug_gap)
    
    # High quality gap config (thorough settings)
    high_quality_gap = GapConfig(
        subfield=1702,
        work_types=["book", "article"],
        cell_type=CellType.VORONOI,  # More thorough algorithm
        k=20,  # Larger k for more thorough analysis
        radius_fraction=0.9,  # Larger radius for more coverage
        drop_k=0,
        hnsw_m=32,  # Higher m for better quality
        hnsw_efc=1000,  # Higher efc for better quality
        hnsw_efs=2000,  # Higher efs for better quality
        denoiser=WorkDenoisers.JAD_EQUAL,  # Apply denoising
        index_type=IndexType.HNSW,
        num_gaps=3  # Find more gaps
    )
    cs.store(group="gap", name="high_quality", node=high_quality_gap)
    
    # ===== LLM CONFIGURATIONS =====
    
    # Production LLM config (from config_manager.py + batch_workflow.py)
    production_llm = LLMConfig(
        provider_type=ProviderType.OPENAI,
        model_name="gpt-5",  # From config_manager.py
        num_articles=10,  # From batch_workflow.py
        num_books=5,  # From batch_workflow.py
        temperature=0.2,  # From config_manager.py
        max_tokens=2000,  # From config_manager.py
        chapters_min=10,  # From config_manager.py
        chapters_max=15,  # From config_manager.py
        min_book_coverage=0.5,
        use_structured_output=True,
        include_citations=True
    )
    cs.store(group="llm", name="production", node=production_llm)
    
    # Debug LLM config (cheaper/faster)
    debug_llm = LLMConfig(
        provider_type=ProviderType.OPENAI,
        model_name="gpt-4o-mini",  # Cheaper model
        num_articles=5,  # Fewer articles for speed
        num_books=3,  # Fewer books for speed
        temperature=0.2,
        max_tokens=1000,  # Fewer tokens for speed
        chapters_min=5,  # Shorter proposals
        chapters_max=8,
        min_book_coverage=0.3,  # Lower threshold for speed
        use_structured_output=True,
        include_citations=False  # Skip citations for speed
    )
    cs.store(group="llm", name="debug", node=debug_llm)
    
    # Creative LLM config (more exploratory)
    creative_llm = LLMConfig(
        provider_type=ProviderType.OPENAI,
        model_name="gpt-5",
        num_articles=15,  # More context
        num_books=8,  # More context
        temperature=0.8,  # More creative
        max_tokens=4000,  # Longer proposals
        chapters_min=12,  # Longer proposals
        chapters_max=20,
        min_book_coverage=0.4,
        use_structured_output=True,
        include_citations=True
    )
    cs.store(group="llm", name="creative", node=creative_llm)
    
    # ===== DATA CONFIGURATIONS =====
    
    # Standard data config
    standard_data = DataConfig(
        subfield=SubField.AI,
        max_works=None,  # No limit
        work_types=(WorkType.BOOK, WorkType.ARTICLE),
        work_filters=(),  # No additional filters
        filter_preset=Filters.NLP,
        custom_filter=None,
        deduplicator=None
    )
    cs.store(group="data", name="standard", node=standard_data)
    
    # Debug data config (limited works for speed)
    debug_data = DataConfig(
        subfield=SubField.AI,
        max_works=1000,  # Limited for speed
        work_types=(WorkType.BOOK, WorkType.ARTICLE),
        work_filters=(),
        filter_preset=Filters.NLP,
        custom_filter=None,
        deduplicator=None
    )
    cs.store(group="data", name="debug", node=debug_data)
    
    # ===== BATCH CONFIGURATIONS =====
    
    # Standard batch config
    standard_batch = BatchConfig(
        num_gaps=1,
        completion_window="24h",
        generate_plots=True,
        save_prompts=True,
        save_batch_file=True
    )
    cs.store(group="batch", name="standard", node=standard_batch)
    
    # Debug batch config
    debug_batch = BatchConfig(
        num_gaps=1,
        completion_window="24h",
        generate_plots=False,  # Skip plots for speed
        save_prompts=True,
        save_batch_file=False  # Skip batch file for speed
    )
    cs.store(group="batch", name="debug", node=debug_batch)
    
    # ===== OUTPUT CONFIGURATIONS =====
    
    # Standard output config
    standard_output = OutputConfig(
        export_formats=(ExportFormat.CSV, ExportFormat.JSON),
        create_subdirectories=True,
        include_metadata=True,
        use_timestamp=True
    )
    cs.store(group="output", name="standard", node=standard_output)
    
    # Minimal output config
    minimal_output = OutputConfig(
        export_formats=(ExportFormat.JSON,),  # Just JSON
        create_subdirectories=False,
        include_metadata=False,
        use_timestamp=False
    )
    cs.store(group="output", name="minimal", node=minimal_output)
    
    # ===== FULL EXPERIMENT CONFIGURATIONS =====
    
    # Production experiment (existing codebase defaults)
    production_experiment = ExperimentConfig(
        gap=production_gap,
        llm=production_llm,
        data=standard_data,
        batch=standard_batch,
        output=standard_output,
        experiment_name="production"
    )
    cs.store(name="production", node=production_experiment)
    
    # Debug experiment (fast settings)
    debug_experiment = ExperimentConfig(
        gap=debug_gap,
        llm=debug_llm,
        data=debug_data,
        batch=debug_batch,
        output=minimal_output,
        experiment_name="debug"
    )
    cs.store(name="debug", node=debug_experiment)
    
    # High quality experiment (thorough analysis)
    high_quality_experiment = ExperimentConfig(
        gap=high_quality_gap,
        llm=creative_llm,
        data=standard_data,
        batch=standard_batch,
        output=standard_output,
        experiment_name="high_quality"
    )
    cs.store(name="high_quality", node=high_quality_experiment)


# Register configs when module is imported
register_configs()
