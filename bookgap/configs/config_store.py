"""
ConfigStore registration for BookGap configurations.

Registers preset configurations that combine existing codebase defaults
with common use case patterns.
"""

from hydra.core.config_store import ConfigStore

from .config import ExperimentConfig
from .gap.gap_config import GapConfig
from .llm.llm_config import LLMConfig
from .data.data_config import SqDistDeduplicatorConfig
from .batch.batch_config import BatchConfig
from .output.output_config import OutputConfig
from .output.export_formats import ExportFormat
from .index.index_config import HNSWConfig
from .cloud.cloud_config import CloudConfig

# Import required enums and types
from bookgap.types import CellType
from bookgap.cloud import WorkDenoisers, Embedder
from bookgap.util import SubField, WorkType
from .util.filter_config import FilterConfig, FilterPairConfig
from bookgap.llm.protocols import OpenAIModel


def register_configs():
    """Register all preset configurations with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    
    # ===== GAP CONFIGURATIONS =====
    
    # Production gap config - matches existing codebase defaults from parse_args.py
    hnsw_index = HNSWConfig(hnsw_m=16, hnsw_efc=512, hnsw_efs=512)

    standard_gap = GapConfig(
        cell_type=CellType.KBALL,           # K-ball algorithm (vs VORONOI) - from batch_workflow.py
        radius_fraction=0.75,               # Gap radius = 75% of k-NN distance - from parse_args.py  
    )

    ai_filter_pair = FilterPairConfig(
        pre=FilterConfig(
            subfield=SubField.AI.value,
            work_types=(WorkType.BOOK.value, WorkType.ARTICLE.value),
        ),
        post=FilterConfig(),
    )
    
    # ===== LLM CONFIGURATIONS =====
    
    # Production LLM config - combines config_manager.py and batch_workflow.py defaults
    standard_llm = LLMConfig(
        _target_="bookgap.llm.generators.openai_batch.OpenAIBatchGenerator",
        model=OpenAIModel.GPT_5,           # GPT-5 model (structured output support)
        num_articles=10,                   # 10 gap articles per prompt
        num_books=5,                       # 5 covering books per prompt
        temperature=0.2,                   # Low temperature (focused responses)
        max_tokens=4096,                   # Token limit for proposals
        use_structured_output=True         # Use JSON schema for structured responses - enables parsing
    )
    
    deduplicator_config = SqDistDeduplicatorConfig(min_sqdist=0.01)

    # ===== BATCH CONFIGURATIONS =====
    
    standard_batch = BatchConfig(
        num_gaps=1,
        completion_window="24h",
        generate_plots=False,
        save_prompts=True,
        save_batch_file=True
    )

    # ===== OUTPUT CONFIGURATIONS =====
    
    # Standard output config - comprehensive export formats for research use
    standard_output = OutputConfig(
        export_formats=(ExportFormat.CSV, ExportFormat.JSON, ExportFormat.JSONL, ExportFormat.PARQUET),
        create_subdirectories=True,    # Organize outputs by topic/gap for easy navigation
        include_metadata=True,         # Include configuration details in output files
        use_timestamp=True             # Add timestamps to filenames for version tracking
    )
    
    # ===== FULL EXPERIMENT CONFIGURATIONS =====
    
    # Production experiment - uses existing codebase defaults from parse_args.py + config_manager.py
    standard_experiment = ExperimentConfig(
        gap=standard_gap,
        llm=standard_llm,
        batch=standard_batch,
        output=standard_output,
        cloud=CloudConfig(
            filter_pair=ai_filter_pair,
            index_info=hnsw_index,
            denoiser=WorkDenoisers.NONE.name,
            drop_k=0,
            embedder_model=Embedder.ModelName.GIST_SMALL,
            max_works=None,
            deduplicator=deduplicator_config,
            normalize_embeddings=True,
        ),
    )
    cs.store(name="standard", node=standard_experiment)
