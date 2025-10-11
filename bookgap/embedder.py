import enum
import gc
import torch
from typing import List, Optional, Union
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from bookgap.util import LOGGER

class Embedder:
    class ModelName(enum.Enum):
        GIST_SMALL = 'avsolatorio/GIST-small-Embedding-v0'
        GEMMA_300M = 'google/embeddinggemma-300m'

    def __init__(self, model_name: Union[str, 'Embedder.ModelName']):
        resolved_model: Union[str, 'Embedder.ModelName'] = model_name
        if isinstance(resolved_model, Embedder.ModelName):
            resolved_model = resolved_model.value

        self.model_name = str(resolved_model)
        self._model_: Optional[SentenceTransformer] = None
        self._embedding_dtype_: Optional[np.dtype[np.float32]] = None
        self._embedding_dimension_: Optional[int] = None

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
        dummy = self._model().encode(["dummy"], show_progress_bar=False)
        one, d = dummy.shape
        if one != 1:
            raise ValueError(f"Expected embedding of single input to have shape (1, d), got ({one}, {d})")
        if self._model().get_sentence_embedding_dimension() is not None:
            if d != self._model().get_sentence_embedding_dimension():
                raise ValueError(f"Model's reported embedding dimension {self._model().get_sentence_embedding_dimension()} does not match actual dimension {d}")
        if dummy.dtype != np.float32:
            raise ValueError(f"Expected embedding dtype to be float32, got {dummy.dtype}")
        self._embedding_dimension_ = dummy.shape[1]
        self._embedding_dtype_ = dummy.dtype

    def embedding_dimension(self):
        if self._embedding_dimension_ is None:
            self._determine_embedding_space()
        assert self._embedding_dimension_ is not None
        return self._embedding_dimension_
    
    def embedding_dtype(self) -> np.dtype[np.float32]:
        if self._embedding_dtype_ is None:
            self._determine_embedding_space()
        assert self._embedding_dtype_ is not None
        return self._embedding_dtype_
    
    def embed_texts(self, title_abstract_pairs: List[tuple[str, str]], batch_size: int = 32) -> NDArray[np.float32]:
        if not title_abstract_pairs:
            return np.empty((0, self.embedding_dimension()), dtype=self.embedding_dtype())
        inputs = [f"{title}\n{abstract}" for title, abstract in title_abstract_pairs]
        
        embeddings = self._model().encode(inputs, batch_size=batch_size, show_progress_bar=False)
        if embeddings.shape != (len(title_abstract_pairs), self.embedding_dimension()):
            raise ValueError(f"Expected embeddings shape to be {(len(title_abstract_pairs), self.embedding_dimension())}, got {embeddings.shape}")
        return embeddings