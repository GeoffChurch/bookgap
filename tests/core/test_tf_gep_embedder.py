from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from bookgap.tf_gep_embedder import TfGepEmbedder, TfGepProjection


def _make_works(n: int, rng: np.random.Generator) -> list[dict]:
    """Create minimal synthetic works with titles, abstracts, IDs, and references."""
    vocab_a = ["neural", "network", "deep", "learning", "training", "gradient", "backpropagation"]
    vocab_b = ["quantum", "entanglement", "qubit", "superposition", "measurement", "hamiltonian"]
    works = []
    ids = [f"W{i}" for i in range(n)]
    for i in range(n):
        vocab = vocab_a if i < n // 2 else vocab_b
        title = " ".join(rng.choice(vocab, size=3, replace=True))
        abstract = " ".join(rng.choice(vocab, size=20, replace=True))
        refs = [ids[j] for j in rng.choice(n, size=min(3, n), replace=False) if j != i]
        works.append({
            "id": ids[i],
            "display_name": title,
            "abstract": abstract,
            "referenced_works": refs,
        })
    return works


class TestTfGepEmbedderFitTransform:
    def test_fit_and_transform_shape(self):
        rng = np.random.default_rng(42)
        works = _make_works(60, rng)
        emb = TfGepEmbedder(n_components=10, gep_type="pca",
                            ridge_fraction=0.01, min_df=2, max_df=0.95)
        emb.fit(works)
        assert emb.is_fitted()
        assert emb.embedding_dimension() == 10
        pairs = [(w["display_name"], w["abstract"]) for w in works]
        result = emb.transform(pairs)
        assert result.shape == (60, 10)
        assert result.dtype == np.float32

    def test_citation_error_gep(self):
        rng = np.random.default_rng(42)
        works = _make_works(60, rng)
        emb = TfGepEmbedder(n_components=10, gep_type="citation_error",
                            ridge_fraction=0.01, min_df=2, max_df=0.95)
        emb.fit(works)
        pairs = [(w["display_name"], w["abstract"]) for w in works]
        result = emb.transform(pairs)
        assert result.shape == (60, 10)

    def test_lpp_gep(self):
        rng = np.random.default_rng(42)
        works = _make_works(60, rng)
        emb = TfGepEmbedder(n_components=10, gep_type="lpp",
                            ridge_fraction=0.01, min_df=2, max_df=0.95)
        emb.fit(works)
        pairs = [(w["display_name"], w["abstract"]) for w in works]
        result = emb.transform(pairs)
        assert result.shape == (60, 10)

    def test_transform_new_documents(self):
        rng = np.random.default_rng(42)
        works = _make_works(60, rng)
        emb = TfGepEmbedder(n_components=10, gep_type="pca",
                            ridge_fraction=0.01, min_df=2, max_df=0.95)
        emb.fit(works)
        result = emb.transform([("unseen neural quantum", "a completely new abstract about deep quantum")])
        assert result.shape == (1, 10)
        assert result.dtype == np.float32

    def test_projection_artifact(self):
        rng = np.random.default_rng(42)
        works = _make_works(60, rng)
        emb = TfGepEmbedder(n_components=10, gep_type="pca",
                            ridge_fraction=0.01, min_df=2, max_df=0.95)
        emb.fit(works)
        artifact = emb.projection_artifact()
        assert isinstance(artifact, TfGepProjection)
        assert artifact.projection_matrix.shape[1] == 10
        assert len(artifact.vocabulary) > 0

    def test_not_fitted_raises(self):
        emb = TfGepEmbedder(n_components=10, gep_type="pca",
                            ridge_fraction=0.01, min_df=2, max_df=0.95)
        with pytest.raises(RuntimeError, match="not fitted"):
            emb.transform([("hello", "world")])

    def test_n_components_capped_at_n(self):
        rng = np.random.default_rng(42)
        works = _make_works(10, rng)
        emb = TfGepEmbedder(n_components=100, gep_type="pca",
                            ridge_fraction=0.01, min_df=1, max_df=1.0)
        emb.fit(works)
        assert emb.embedding_dimension() <= 10
