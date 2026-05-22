"""Index configuration structures for Hydra composition."""

from dataclasses import dataclass, field


@dataclass
class HNSWConfig:
    """Hydra-friendly configuration for :class:`bookgap.index.HNSW`.

    The `_target_` attribute allows Hydra's instantiate utility to construct the
    actual FAISS-backed index information object while keeping the configuration
    statically typed and reusable across presets.
    """

    _target_: str = field(default="bookgap.index.HNSW", init=False)
    # Defaults tuned on the domain=3 / 42M-work GIST corpus (April 2026):
    # see artifacts_scripts/hnsw_dedup_bench.py. efC dominates recall on
    # this geometry; efS only contributes marginal recall once efC is solid,
    # so we cut it harder than efC. At (M=16, efC=384, efS=128) recall@1
    # vs brute force is 99.84% with 0% SqDistDeduplicator(min_sqdist=0.01)
    # mismatches on a 5k-query sample, and the dedup phase projects to
    # ~1.4h on the full corpus vs ~3h at the prior (512, 512) defaults.
    hnsw_m:   int = 16  # Bidirectional links per node (higher = better recall, more RAM)
    hnsw_efc: int = 384 # Construction candidate list (higher = better graph quality)
    hnsw_efs: int = 128 # Search candidate list (higher = better recall at query time)


__all__ = ["HNSWConfig"]
