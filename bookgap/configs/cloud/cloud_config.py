"""Cloud configuration structure for Hydra composition."""

from dataclasses import dataclass, field
from typing import Optional

from bookgap.cloud import Embedder
from bookgap.configs.data.data_config import SqDistDeduplicatorConfig
from bookgap.configs.index.index_config import HNSWConfig
from bookgap.configs.util.filter_config import FilterPairConfig


@dataclass(frozen=True)
class CloudConfig:
    """Hydra-friendly configuration for :class:`bookgap.cloud.Cloud`.

    The `_target_` attribute allows Hydra's instantiate utility to construct the
    actual :class:`bookgap.cloud.Cloud` object while keeping the configuration
    statically typed and reusable across presets.
    """

    _target_: str = field(default="bookgap.cloud.Cloud", init=False)
    filter_pair: FilterPairConfig
    index_info: HNSWConfig
    denoiser: str
    drop_k: int
    embedder_model: Embedder.ModelName
    max_works: Optional[int] = None
    deduplicator: Optional[SqDistDeduplicatorConfig] = None
    normalize_embeddings: bool = True


__all__ = ["CloudConfig"]
