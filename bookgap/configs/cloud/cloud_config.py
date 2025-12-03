"""Cloud configuration structure for Hydra composition."""

from dataclasses import dataclass, field
from typing import Optional

from bookgap.cloud import Embedder
from bookgap.cloud_denoiser import WorkDenoisers
from bookgap.configs.data.data_config import SqDistDeduplicatorConfig
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
    embedder_model: Embedder.ModelName = Embedder.ModelName.GIST_SMALL
    max_works: Optional[int] = None
    deduplicator: Optional[SqDistDeduplicatorConfig] = field(default_factory=SqDistDeduplicatorConfig)
    normalize_embeddings: bool = True


__all__ = ["CloudConfig"]
