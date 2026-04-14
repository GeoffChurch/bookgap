"""
ConfigStore registration for BookGap configurations.

Registers preset configurations that combine existing codebase defaults
with common use case patterns.
"""

from hydra.core.config_store import ConfigStore

from .config import ExperimentConfig, SnapshotConfig
from .gap.gap_config import GapConfig
from .llm.llm_config import LLMConfig
from .data.data_config import SqDistDeduplicatorConfig
from .batch.batch_config import BatchConfig
from .output.output_config import OutputConfig
from .output.export_formats import ExportFormat
from .index.index_config import HNSWConfig
from .cloud.cloud_config import CloudConfig
from .embedder.embedder_config import SentenceTransformerEmbedderConfig

# Import required enums and types
from bookgap.types import CellType
from bookgap.cloud import WorkDenoisers
from bookgap.util import SubField, WorkType
from .util.filter_config import FilterConfig, FilterPairConfig
from bookgap.llm.protocols import OpenAIModel


def register_configs():
    """Register all preset configurations with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="gap", node=GapConfig())
    cs.store(name="cloud", node=CloudConfig())
    cs.store(group="snapshot", name="default", node=SnapshotConfig())
    # Provide a fully-instantiated default experiment configuration to avoid
    # missing-required-args errors when importing/using configs.
    cs.store(
        name="standard",
        node=ExperimentConfig(
            gap=GapConfig(),
            llm=LLMConfig(),
            batch=BatchConfig(),
            output=OutputConfig(),
            cloud=CloudConfig(),
        ),
    )
