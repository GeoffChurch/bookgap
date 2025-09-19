"""Gap detection configuration."""

from dataclasses import dataclass

from bookgap.types import CellType
from bookgap.cloud import WorkDenoisers
from bookgap.index import IndexType


@dataclass(frozen=True)
class GapConfig:
    """Configuration for gap detection parameters.
    
    All parameters are required - use ConfigStore presets for common configurations.
    Compatible with existing GapConfiguration in config_manager.py.
    """
    
    # Gap detection algorithm - determines which spatial algorithm to use
    cell_type: CellType  # KBALL or VORONOI cell algorithm for finding research gaps
    
    # Spatial parameters - control gap detection sensitivity and coverage
    radius_fraction: float  # Fraction of k-NN distance for gap radius (0.0-1.0, default: 0.75)
    drop_k: int            # Number of top embedding components to drop for noise reduction (default: 0)
    
    # HNSW index parameters - control vector search performance vs accuracy tradeoffs
    hnsw_m: int    # Number of bidirectional links for each node (default: 16, higher=better recall)
    hnsw_efc: int  # Size of candidate list during construction (default: 512, higher=better quality)
    hnsw_efs: int  # Size of candidate list during search (default: 1024, higher=better recall)
    
    # Data processing - control embedding transformation and indexing strategy
    denoiser: WorkDenoisers  # Embedding denoising method (NONE, PCA, GPCA, REL_AMP, LPP, etc.)
    index_type: IndexType    # Vector index type (HNSW for large datasets, FLAT_L2 for exact search)
