"""Index configuration structures for Hydra composition."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HNSWConfig:
    """Hydra-friendly configuration for :class:`bookgap.index.HNSW`.

    The `_target_` attribute allows Hydra's instantiate utility to construct the
    actual FAISS-backed index information object while keeping the configuration
    statically typed and reusable across presets.
    """

    _target_: str = field(default="bookgap.index.HNSW", init=False)
    hnsw_m:   int = 16  # Number of bidirectional links for each node (default: 16, higher=better recall)
    hnsw_efc: int = 512 # Size of candidate list during construction (default: 512, higher=better quality)
    hnsw_efs: int = 512 # Size of candidate list during search (default: 512, higher=better recall)


__all__ = ["HNSWConfig"]
