"""Hydra-friendly embedder configurations.

These stay ``frozen=True`` because :class:`bookgap.cloud.Cloud` stores the
embedder config DIRECTLY (not an instantiated runtime object) so it can
participate in the Cloud hash key for joblib caching. See Cloud.__post_init__
which normalizes the embedder field to one of these two classes.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SentenceTransformerEmbedderConfig:
    _target_: str = field(default="bookgap.embedder.SentenceTransformerEmbedder", init=False)
    model_name: str = "avsolatorio/GIST-small-Embedding-v0"


@dataclass(frozen=True)
class TfGepEmbedderConfig:
    _target_: str = field(default="bookgap.tf_gep_embedder.TfGepEmbedder", init=False)
    n_components: int = 768
    gep_type: str = "citation_error"
    ridge_fraction: float = 0.01
    min_df: int = 5
    max_df: float = 0.95
    add_self_loops: bool = True
    normalize_adjacency: bool = True
