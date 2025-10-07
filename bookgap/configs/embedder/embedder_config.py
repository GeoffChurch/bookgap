"""
Embedder configuration for BookGap processing.

Defines embedding model settings shared across cloud construction and proposal
post-processing to ensure consistent cache reuse and reproducibility.
"""

from dataclasses import dataclass
from typing import Union

from bookgap.cloud import Embedder


@dataclass(frozen=True)
class EmbedderConfig:
    """Configuration for embedding model selection.

    Attributes:
        model_name: Name of the embedding model to load. Accepts either the
            :class:`Embedder.ModelName` enum or a plain string for flexibility
            with OmegaConf-based configs in tests.
    """

    model_name: Embedder.ModelName


__all__ = ["EmbedderConfig"]
