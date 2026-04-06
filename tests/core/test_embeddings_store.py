from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest
from pathlib import Path
from numpy.testing import assert_allclose

from bookgap.cloud import (
    _embeddings_store_path,
    write_embeddings_store,
    read_embeddings_store,
)


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    import bookgap.cloud as mod
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    return tmp_path


class TestEmbeddingsStore:
    def test_roundtrip(self, tmp_cache):
        rng = np.random.default_rng(42)
        embeddings = rng.normal(size=(100, 32)).astype(np.float32)
        path = tmp_cache / "emb_test" / "embeddings.parquet"
        write_embeddings_store(path, embeddings)
        loaded = read_embeddings_store(path)
        assert_allclose(loaded, embeddings, atol=1e-7)
        assert loaded.dtype == np.float32

    def test_empty(self, tmp_cache):
        embeddings = np.empty((0, 16), dtype=np.float32)
        path = tmp_cache / "emb_empty" / "embeddings.parquet"
        write_embeddings_store(path, embeddings)
        loaded = read_embeddings_store(path)
        assert loaded.shape == (0, 16)
        assert loaded.dtype == np.float32

    def test_path_function(self, tmp_cache):
        from bookgap.configs.embedder.embedder_config import SentenceTransformerEmbedderConfig
        corpus_fp = "abc123"
        embedder_cfg = SentenceTransformerEmbedderConfig()
        path = _embeddings_store_path(corpus_fp, embedder_cfg)
        assert "abc123" in str(path)
        assert path.name == "embeddings.parquet"
