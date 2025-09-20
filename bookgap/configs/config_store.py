"""
ConfigStore registration for BookGap configurations.

Registers preset configurations that combine existing codebase defaults
with common use case patterns.
"""

from hydra.core.config_store import ConfigStore

from .config import ExperimentConfig
from .gap.gap_config import GapConfig
from .llm.llm_config import LLMConfig
from .data.data_config import DataConfig, SqDistDeduplicatorConfig
from .batch.batch_config import BatchConfig
from .output.output_config import OutputConfig
from .output.export_formats import ExportFormat

# Import required enums and types
from bookgap.types import CellType
from bookgap.cloud import WorkDenoisers
from bookgap.index import IndexType
from bookgap.llm.protocols import OpenAIModel


def register_configs():
    """Register all preset configurations with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    
    # ===== GAP CONFIGURATIONS =====
    
    # Production gap config - matches existing codebase defaults from parse_args.py
    production_gap = GapConfig(
        cell_type=CellType.KBALL,           # K-ball algorithm (vs VORONOI) - from batch_workflow.py
        radius_fraction=0.75,               # Gap radius = 75% of k-NN distance - from parse_args.py  
        drop_k=0,                          # No PCA dropping (keep all dimensions) - from parse_args.py
        hnsw_m=16,                         # HNSW connectivity (16 = good quality/speed balance) - from parse_args.py
        hnsw_efc=512,                      # HNSW construction candidates (512 = high quality) - from parse_args.py
        hnsw_efs=1024,                     # HNSW search candidates (1024 = high recall) - from parse_args.py
        denoiser=WorkDenoisers.NONE,       # No embedding denoising (raw embeddings) - from parse_args.py
        index_type=IndexType.HNSW          # HNSW index (fast approximate search) - from parse_args.py
    )
    cs.store(group="gap", name="production", node=production_gap)
    
    # ===== LLM CONFIGURATIONS =====
    
    # Production LLM config - combines config_manager.py and batch_workflow.py defaults
    production_llm = LLMConfig(
        _target_="bookgap.llm.generators.openai_batch.OpenAIBatchGenerator",  # Batch API generator
        model=OpenAIModel.GPT_5,           # GPT-5 model (structured output support) - from config_manager.py
        num_articles=10,                   # 10 gap articles per prompt - from batch_workflow.py
        num_books=5,                       # 5 covering books per prompt - from batch_workflow.py
        temperature=0.2,                   # Low temperature (focused responses) - from config_manager.py
        max_tokens=2000,                   # 2000 token limit for proposals - from config_manager.py
        chapters_min=10,                   # Minimum 10 chapters per textbook - from config_manager.py
        chapters_max=15,                   # Maximum 15 chapters per textbook - from config_manager.py
        use_structured_output=True         # Use JSON schema for structured responses - enables parsing
    )
    cs.store(group="llm", name="production", node=production_llm)
    
    # ===== DATA CONFIGURATIONS =====
    
    # Data config with deduplication - removes semantically similar papers
    deduped_data = DataConfig(
        max_works=None,                    # No limit on dataset size (use full OpenAlex data)
        deduplicator=SqDistDeduplicatorConfig(
            min_sqdist=0.01                # Remove papers with <1% embedding distance
        )
    )
    cs.store(group="data", name="deduped", node=deduped_data)
    
    # ===== BATCH CONFIGURATIONS =====
    
    standard_batch = BatchConfig(
        num_gaps=1,
        completion_window="24h",
        generate_plots=True,
        save_prompts=True,
        save_batch_file=True
    )
    cs.store(group="batch", name="standard", node=standard_batch)

    # ===== OUTPUT CONFIGURATIONS =====
    
    # Standard output config - comprehensive export formats for research use
    standard_output = OutputConfig(
        export_formats=(ExportFormat.CSV, ExportFormat.JSON, ExportFormat.JSONL, ExportFormat.PARQUET),
        create_subdirectories=True,    # Organize outputs by topic/gap for easy navigation
        include_metadata=True,         # Include configuration details in output files
        use_timestamp=True             # Add timestamps to filenames for version tracking
    )
    cs.store(group="output", name="standard", node=standard_output)
    
    # ===== FULL EXPERIMENT CONFIGURATIONS =====
    
    # Production experiment - uses existing codebase defaults from parse_args.py + config_manager.py
    production_experiment = ExperimentConfig(
        gap=production_gap,
        llm=production_llm,
        data=deduped_data,
        batch=standard_batch,
        output=standard_output,
    )
    cs.store(name="production", node=production_experiment)
