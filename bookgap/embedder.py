import abc
import enum
from functools import lru_cache
import gc
import torch
from typing import List, Optional, Union
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
import scipy.sparse as sp

from bookgap.util import LOGGER, MEMORY

@lru_cache(maxsize=1)
@MEMORY.cache
def _cached_embedding_space(model_name: str) -> tuple[int, str]:
    model = SentenceTransformer(model_name)
    dummy = model.encode(["dummy"], show_progress_bar=False)
    one, d = dummy.shape
    if one != 1:
        raise ValueError(f"Expected embedding of single input to have shape (1, d), got ({one}, {d})")
    reported = model.get_sentence_embedding_dimension()
    if reported is not None and d != reported:
        raise ValueError(f"Model's reported embedding dimension {reported} does not match actual dimension {d}")
    if dummy.dtype != np.float32:
        raise ValueError(f"Expected embedding dtype to be float32, got {dummy.dtype}")
    return d, str(dummy.dtype)


class BaseEmbedder(abc.ABC):
    """Abstract base class defining the fit/transform interface for embedders."""

    @abc.abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimensionality of the embeddings produced by this embedder."""
        ...

    @abc.abstractmethod
    def fit(
        self,
        works,
        embeddings_pretrained: Optional[NDArray[np.float32]] = None,
        citation_graph: Optional[sp.spmatrix] = None,
    ) -> None:
        """Fit the embedder on a corpus of works.

        For pretrained embedders (e.g. SentenceTransformerEmbedder) this is a no-op
        that merely marks the embedder as fitted.  Corpus-fitted embedders (e.g.
        TfGepEmbedder) will perform real training here.
        """
        ...

    @abc.abstractmethod
    def transform(self, title_abstract_pairs: List[tuple[str, str]]) -> NDArray[np.float32]:
        """Embed a list of (title, abstract) pairs and return an (N, D) float32 array."""
        ...

    @abc.abstractmethod
    def is_fitted(self) -> bool:
        """Return True iff fit() has been called at least once."""
        ...


class SentenceTransformerEmbedder(BaseEmbedder):
    class ModelName(enum.Enum):
        GIST_SMALL = 'avsolatorio/GIST-small-Embedding-v0'
        GEMMA_300M = 'google/embeddinggemma-300m'

    def __init__(self, model_name: Union[str, 'SentenceTransformerEmbedder.ModelName']):
        resolved_model: Union[str, 'SentenceTransformerEmbedder.ModelName'] = model_name
        if isinstance(resolved_model, SentenceTransformerEmbedder.ModelName):
            resolved_model = resolved_model.value

        self.model_name = str(resolved_model)
        self._model_: Optional[SentenceTransformer] = None
        self._embedding_dtype_: Optional[np.dtype[np.float32]] = None
        self._embedding_dimension_: Optional[int] = None
        self._fitted: bool = False

    def _model(self) -> SentenceTransformer:
        if self._model_ is None:
            self._model_ = SentenceTransformer(self.model_name)
        return self._model_

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup GPU resources."""
        if self._model_ is not None:
            # Delete the model reference
            del self._model_
            self._model_ = None

            # Aggressive GPU memory cleanup
            if torch.cuda.is_available():
                # Force garbage collection
                gc.collect()
                # Clear CUDA cache multiple times
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  # Clean up inter-process communication cache
                torch.cuda.empty_cache()  # Clear again after IPC cleanup
                LOGGER.info("Performed aggressive CUDA cache cleanup")

    def _determine_embedding_space(self):
        d, dtype_str = _cached_embedding_space(self.model_name)
        self._embedding_dimension_ = d
        self._embedding_dtype_ = np.dtype(dtype_str)

    def embedding_dimension(self) -> int:
        if self._embedding_dimension_ is None:
            self._determine_embedding_space()
        assert self._embedding_dimension_ is not None
        return self._embedding_dimension_

    def embedding_dtype(self) -> np.dtype[np.float32]:
        if self._embedding_dtype_ is None:
            self._determine_embedding_space()
        assert self._embedding_dtype_ is not None
        return self._embedding_dtype_

    def fit(
        self,
        works,
        embeddings_pretrained: Optional[NDArray[np.float32]] = None,
        citation_graph: Optional[sp.spmatrix] = None,
    ) -> None:
        """No-op fit: SentenceTransformerEmbedder uses a pretrained model.

        Calling fit() marks the embedder as ready for transform().
        """
        self._fitted = True

    def is_fitted(self) -> bool:
        return self._fitted

    def transform(self, title_abstract_pairs: List[tuple[str, str]]) -> NDArray[np.float32]:
        """Embed title/abstract pairs by delegating to embed_texts()."""
        return self.embed_texts(title_abstract_pairs)

    def embed_texts(self, title_abstract_pairs: List[tuple[str, str]], batch_size: int = 32) -> NDArray[np.float32]:
        if not title_abstract_pairs:
            return np.empty((0, self.embedding_dimension()), dtype=self.embedding_dtype())
        inputs = [f"{title}\n{abstract}" for title, abstract in title_abstract_pairs]

        embeddings = self._model().encode(inputs, batch_size=batch_size, show_progress_bar=False)
        if embeddings.shape != (len(title_abstract_pairs), self.embedding_dimension()):
            raise ValueError(f"Expected embeddings shape to be {(len(title_abstract_pairs), self.embedding_dimension())}, got {embeddings.shape}")
        return embeddings


# Backward compatibility alias — all existing code that imports `Embedder` continues to work.
Embedder = SentenceTransformerEmbedder
