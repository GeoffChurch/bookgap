"""Deduplicator configuration for BookGap analysis."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SqDistDeduplicatorConfig:
    """Configuration for SqDistDeduplicator.

    Removes semantically similar works based on squared embedding distance threshold.
    Fast range-based deduplication suitable for large datasets.
    """

    _target_: str = field(default="bookgap.deduplicators.SqDistDeduplicator", init=False)
    min_sqdist: float  # Minimum squared distance threshold (e.g., 0.01 = 1% distance)

    def __post_init__(self):
        """Validate SqDistDeduplicator configuration parameters."""
        if self.min_sqdist <= 0:
            raise ValueError(f"min_sqdist must be positive, got {self.min_sqdist}")
