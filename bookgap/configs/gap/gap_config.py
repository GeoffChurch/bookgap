"""Gap detection configuration."""

from dataclasses import dataclass
from typing import Optional, Tuple

from bookgap.types import CellType, DeflationStrategy
from bookgap.util import WorkType

@dataclass
class GapConfig:
    """Configuration for gap detection parameters.

    All parameters are required - use ConfigStore presets for common configurations.
    Compatible with existing GapConfiguration in config_manager.py.
    """

    # Gap detection algorithm - determines which spatial algorithm to use
    cell_type: CellType = CellType.KBALL  # KBALL or VORONOI cell algorithm for finding research gaps

    # Gap-deflation strategy (K-ball only; Voronoi ignores it). Default is the
    # byte-identical greedy hard-subtraction; FL_BOOKDEMAND_RESID is opt-in.
    deflation_strategy: DeflationStrategy = DeflationStrategy.GREEDY_HARD

    # Spatial parameters - control gap detection sensitivity and coverage
    radius_fraction: float = 0.75  # Fraction of k-NN distance for gap radius (0.0-1.0, default: 0.75)
    
    # Work type parameters - define ambient (query) and landmark (reference) types
    ambient_work_type: str = WorkType.ARTICLE  # Query work type (e.g., articles defining gaps)
    landmark_work_type: str = WorkType.BOOK  # Reference work type (e.g., books covering gaps)

    # Gap-sizing metric used to rank candidate gaps for top-k selection.
    # "raw_kball" (default): number of articles in the bookless ball — historical, byte-identical.
    # "virtual_centroid": Variant 4 from docs/backlog/gap-detection-methods/kball-residual-mass.md —
    #   count articles whose nearest book (among real books + the gap's centroid as a virtual book) is the virtual centroid.
    # "mutual_nn": Variant 3 (partition-mutual-NN discount).
    # "book_knn_discount": Variant 1 (global k-NN discount).
    # "kde_ratio": Variant 2 (KDE-on-KDE soft).
    size_metric: str = "raw_kball"

    # When size_metric != "raw_kball", generate this many candidates from the gap generator
    # to rerank. Ignored when size_metric == "raw_kball" (raw-KBALL ranking == generator
    # order; the islice in _compute_gaps_for_topic is enough).
    candidate_pool_size: int = 10

    # Optional whitelist of topic IDs to restrict gap detection to.
    # None / empty (default) → run on all topics resolvable from
    # cloud.filter_pair. When set, only topics whose ID appears here are
    # gap-detected (and therefore proposal-generated against). Intended for
    # rapid-iteration sweeps on a representative subset; not for paper-headline
    # runs. Stored as Tuple[str, ...] for joblib hashability.
    topic_subset: Optional[Tuple[str, ...]] = None

