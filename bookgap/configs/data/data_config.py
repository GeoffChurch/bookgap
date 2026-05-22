"""Deduplicator configuration for BookGap analysis."""

from dataclasses import dataclass, field


@dataclass
class SqDistDeduplicatorConfig:
    """Configuration for SqDistDeduplicator.

    Removes semantically similar works based on squared embedding distance threshold.
    Fast range-based deduplication suitable for large datasets.
    """

    _target_: str = field(default="bookgap.deduplicators.SqDistDeduplicator", init=False)
    min_sqdist: float = 0.01  # Minimum squared distance threshold (e.g., 0.01 = 1% distance)

    def __post_init__(self):
        """Validate SqDistDeduplicator configuration parameters."""
        if self.min_sqdist <= 0:
            raise ValueError(f"min_sqdist must be positive, got {self.min_sqdist}")


@dataclass
class SimHashDeduplicatorConfig:
    """Configuration for SimHashDeduplicator.

    LSH-style dedup: random hyperplane SimHash + exact-L2 verify. Defaults
    (k_bits=32, n_tables=20) chosen from ``artifacts_scripts/dedup_methods_bench.py``:
    maintains 99.98-99.9998% recall vs IVFFlat-exact across both sparse and
    ultra-dense duplicate regimes, and runs much faster than FAISS HNSW's
    range_search at 42M-corpus scale (HNSW dedup was projected at 10+ h
    on the domain=3 corpus; SimHash k=32 L=20 projects to ~5 h).
    """

    _target_: str = field(default="bookgap.deduplicators.SimHashDeduplicator", init=False)
    min_sqdist: float = 0.01  # Squared distance threshold (0.01 ≈ cos > 0.995 on unit vectors)
    k_bits: int = 32           # Bits per signature (≤ 64, higher = tinier buckets, cheaper verify)
    n_tables: int = 20         # Hash tables; recall ≈ 1 - (1 - p)^L where p = per-table hit rate
    seed: int = 0              # Hyperplane RNG seed — change for variance measurement

    def __post_init__(self):
        if self.min_sqdist <= 0:
            raise ValueError(f"min_sqdist must be positive, got {self.min_sqdist}")
        if not 1 <= self.k_bits <= 64:
            raise ValueError(f"k_bits must be 1..64, got {self.k_bits}")
        if self.n_tables < 1:
            raise ValueError(f"n_tables must be >= 1, got {self.n_tables}")
