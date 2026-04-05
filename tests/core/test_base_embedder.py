from __future__ import annotations

import numpy as np
import pytest

from bookgap.embedder import BaseEmbedder, SentenceTransformerEmbedder, Embedder


class TestBaseEmbedderContract:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseEmbedder()

    def test_sentence_transformer_is_base_embedder(self):
        emb = SentenceTransformerEmbedder(model_name=SentenceTransformerEmbedder.ModelName.GIST_SMALL)
        assert isinstance(emb, BaseEmbedder)

    def test_embedder_alias(self):
        """Embedder should still be usable as an alias."""
        assert Embedder is SentenceTransformerEmbedder

    def test_fit_is_noop(self):
        emb = SentenceTransformerEmbedder(model_name=SentenceTransformerEmbedder.ModelName.GIST_SMALL)
        assert not emb.is_fitted()
        emb.fit(works=[])
        assert emb.is_fitted()

    def test_transform_delegates_to_embed_texts(self):
        emb = SentenceTransformerEmbedder(model_name=SentenceTransformerEmbedder.ModelName.GIST_SMALL)
        emb.fit(works=[])
        result = emb.transform([("hello", "world")])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, emb.embedding_dimension())
        assert result.dtype == np.float32
