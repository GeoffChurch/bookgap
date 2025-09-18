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
    
    # Gap detection algorithm
    cell_type: CellType
    
    # Spatial parameters
    k: int
    radius_fraction: float
    drop_k: int
    
    # HNSW index parameters
    hnsw_m: int
    hnsw_efc: int
    hnsw_efs: int
    
    # Data processing
    denoiser: WorkDenoisers
    index_type: IndexType
    
    # Analysis control
    num_gaps: int
