"""Hydra-friendly embedder configurations."""
from dataclasses import dataclass, field


@dataclass
class SentenceTransformerEmbedderConfig:
    _target_: str = field(default="bookgap.embedder.SentenceTransformerEmbedder", init=False)
    model_name: str = "avsolatorio/GIST-small-Embedding-v0"


@dataclass
class TfGepEmbedderConfig:
    _target_: str = field(default="bookgap.tf_gep_embedder.TfGepEmbedder", init=False)
    n_components: int = 768
    gep_type: str = "citation_error"
    ridge_fraction: float = 0.01
    min_df: int = 5
    max_df: float = 0.95
    add_self_loops: bool = True
    normalize_adjacency: bool = True
