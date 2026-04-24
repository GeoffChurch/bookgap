"""Cloud configuration structure for Hydra composition."""

from dataclasses import dataclass, field
from typing import Any, Optional

from bookgap.cloud_denoiser import WorkDenoisers
from bookgap.configs.data.data_config import SimHashDeduplicatorConfig, SqDistDeduplicatorConfig
from bookgap.configs.embedder.embedder_config import SentenceTransformerEmbedderConfig
from bookgap.configs.index.index_config import HNSWConfig
from bookgap.configs.util.filter_config import FilterPairConfig


@dataclass
class CloudConfig:
    """Hydra-friendly configuration for :class:`bookgap.cloud.Cloud`.

    The `_target_` attribute allows Hydra's instantiate utility to construct the
    actual :class:`bookgap.cloud.Cloud` object while keeping the configuration
    statically typed and reusable across presets.
    """

    _target_: str = field(default="bookgap.cloud.Cloud", init=False)
    filter_pair: FilterPairConfig = field(default_factory=FilterPairConfig)
    index_info: HNSWConfig = field(default_factory=HNSWConfig)
    denoiser: str = WorkDenoisers.NONE.name
    drop_k: int = 0
    embedder: Any = field(default_factory=SentenceTransformerEmbedderConfig)
    max_works: Optional[int] = None
    # Defaults to SimHash (k=32, L=20) since it's much faster than the
    # HNSW-based SqDist path at 42M-corpus scale while maintaining
    # 99.98-99.9998% recall across both sparse and dense duplicate
    # regimes. See artifacts_scripts/dedup_methods_bench.py.
    deduplicator: Optional[Any] = field(default_factory=SimHashDeduplicatorConfig)
    normalize_embeddings: bool = True
    snapshot_date: str = "2025-11-06"


__all__ = ["CloudConfig"]
