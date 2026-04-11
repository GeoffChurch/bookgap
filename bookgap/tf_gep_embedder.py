"""TF-GEP Embedder: corpus-fitted dense embeddings via term-frequency GEP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from bookgap.denoisers import (
    CenteredMatrix,
    CitationErrorLDADenoiser,
    DenoiserContext,
    LPPDenoiser,
    PCADenoiser,
    scatter,
)
from bookgap.embedder import BaseEmbedder
from bookgap.util import LOGGER, build_citation_adjacency_from_ids


@dataclass(frozen=True)
class TfGepProjection:
    """Cached artifact for transforming new documents into the fitted TF-GEP space."""
    vocabulary: dict[str, int]
    projection_matrix: NDArray  # V x d
    fit_mean: NDArray | None    # (1, V)
    vectorizer_params: dict


class _TfDenoiserContext(DenoiserContext):
    """DenoiserContext backed by a sparse TF matrix."""

    def __init__(self, X_sparse: sp.csr_matrix, adjacency: sp.csr_matrix):
        super().__init__()
        self._X = X_sparse
        self._A = adjacency

    def A(self) -> sp.csr_matrix:
        return self._A

    def X(self) -> sp.csr_matrix:  # type: ignore[override]
        return self._X

    def X_n(self, n: int) -> NDArray:
        raise IndexError("TfDenoiserContext has no subsets")

    def num_subsets(self) -> int:
        return 0

    def S_C(self) -> NDArray:
        """Citation-error scatter matrix, dense-safe for sparse TF input."""
        X_dense = np.asarray(self._X.todense(), dtype=np.float64)
        cited_means = self._A @ X_dense  # N x V, dense
        D_error = X_dense - cited_means
        return scatter(D_error, centered=False)


_GEP_REGISTRY: dict[str, type] = {
    "citation_error": CitationErrorLDADenoiser,
    "lpp": LPPDenoiser,
    "pca": PCADenoiser,
}

# gep_types that bypass the GEP and use SVD directly
_SVD_TYPES = {"lsa"}


class TfGepEmbedder(BaseEmbedder):
    """Corpus-fitted embedder: raw TF -> GEP projection -> dense."""

    def __init__(
        self,
        n_components: int = 768,
        gep_type: str = "citation_error",
        ridge_fraction: float = 0.01,
        min_df: int = 5,
        max_df: float = 0.95,
        add_self_loops: bool = True,
        normalize_adjacency: bool = True,
    ):
        if gep_type not in _GEP_REGISTRY and gep_type not in _SVD_TYPES:
            raise ValueError(f"Unknown gep_type '{gep_type}'. Choose from: {list(_GEP_REGISTRY) + list(_SVD_TYPES)}")
        self._n_components = n_components
        self._gep_type = gep_type
        self._ridge_fraction = ridge_fraction
        self._min_df = min_df
        self._max_df = max_df
        self._add_self_loops = add_self_loops
        self._normalize_adjacency = normalize_adjacency

        self._vectorizer: Optional[CountVectorizer] = None
        self._projection_matrix: Optional[NDArray] = None
        self._fit_mean: Optional[NDArray] = None
        self._actual_n_components: Optional[int] = None

    def embedding_dimension(self) -> int:
        if self._actual_n_components is None:
            return self._n_components
        return self._actual_n_components

    def is_fitted(self) -> bool:
        return self._projection_matrix is not None

    def fit(
        self,
        works: list[dict[str, Any]],
        embeddings_pretrained: NDArray | None = None,
        citation_graph: sp.csr_matrix | None = None,
    ) -> None:
        from scipy.sparse.linalg import svds

        texts = [f"{w['display_name']} {w['abstract']}" for w in works]

        if self._gep_type in _SVD_TYPES:
            return self._fit_svd(texts, works)

        # 1. Build sparse TF matrix
        self._vectorizer = CountVectorizer(
            min_df=self._min_df, max_df=self._max_df, dtype=np.float64,
        )
        X_sparse: sp.csr_matrix = self._vectorizer.fit_transform(texts)
        N, V = X_sparse.shape
        LOGGER.info(f"TfGepEmbedder: built {N}x{V} TF matrix")

        # Cap n_components at N (dual formulation can produce at most N directions)
        actual_k = min(self._n_components, N, V)
        self._actual_n_components = actual_k

        # 2. Build citation graph
        if citation_graph is None:
            work_ids = [w["id"] for w in works]
            work_refs = [w.get("referenced_works", []) for w in works]
            citation_graph, _ = build_citation_adjacency_from_ids(
                work_ids, work_refs,
                add_self_loops=self._add_self_loops,
                normalize=self._normalize_adjacency,
            )

        # 3. Build context and run GEP
        ctx = _TfDenoiserContext(X_sparse, citation_graph)
        denoiser = _GEP_REGISTRY[self._gep_type]()
        P, eigvals = denoiser.projector(ctx, actual_k)

        # P is V x actual_k from primal or recovered from dual
        self._projection_matrix = np.ascontiguousarray(P.astype(np.float32))
        self._fit_mean = np.asarray(X_sparse.mean(axis=0)).astype(np.float32)  # (1, V)

        LOGGER.info(
            f"TfGepEmbedder fitted: gep_type={self._gep_type}, "
            f"V={V}, d={actual_k}, eigvals range=[{eigvals[0]:.4g}, {eigvals[-1]:.4g}]"
        )

    def _fit_svd(self, texts: list[str], works: list[dict[str, Any]]) -> None:
        """LSA path: TF-IDF + truncated SVD, no citation graph."""
        from scipy.sparse.linalg import svds

        self._vectorizer = TfidfVectorizer(
            min_df=self._min_df, max_df=self._max_df, dtype=np.float64,
        )
        X_sparse: sp.csr_matrix = self._vectorizer.fit_transform(texts)
        N, V = X_sparse.shape
        LOGGER.info(f"TfGepEmbedder(lsa): built {N}x{V} TF-IDF matrix")

        actual_k = min(self._n_components, min(N, V) - 1)
        self._actual_n_components = actual_k

        LOGGER.info(f"TfGepEmbedder(lsa): computing rank-{actual_k} truncated SVD ...")
        # svds returns (U, s, Vt) with Vt being (k, V)
        _, s, Vt = svds(X_sparse, k=actual_k, which="LM")
        # svds returns in ascending order; reverse to descending
        s = s[::-1]
        Vt = Vt[::-1, :]

        # Projection matrix: V x k (right singular vectors)
        P = Vt.T
        self._projection_matrix = np.ascontiguousarray(P.astype(np.float32))
        # For TF-IDF, centering is not needed (IDF already handles global weighting)
        self._fit_mean = None

        LOGGER.info(
            f"TfGepEmbedder(lsa) fitted: V={V}, d={actual_k}, "
            f"top s-vals: {' '.join(f'{v:.1f}' for v in s[:5])}"
        )

    def transform(self, title_abstract_pairs: list[tuple[str, str]]) -> NDArray[np.float32]:
        if not self.is_fitted():
            raise RuntimeError("TfGepEmbedder is not fitted. Call fit() first.")
        assert self._vectorizer is not None
        assert self._projection_matrix is not None

        texts = [f"{title} {abstract}" for title, abstract in title_abstract_pairs]
        X_sparse = self._vectorizer.transform(texts)

        # Center and project (densify per-document, safe for small batches)
        X_dense = np.asarray(X_sparse.todense(), dtype=np.float32)
        if self._fit_mean is not None:
            X_dense -= self._fit_mean
        return X_dense @ self._projection_matrix

    def projection_artifact(self) -> TfGepProjection:
        if not self.is_fitted():
            raise RuntimeError("TfGepEmbedder is not fitted.")
        assert self._vectorizer is not None
        assert self._projection_matrix is not None
        return TfGepProjection(
            vocabulary=self._vectorizer.vocabulary_,
            projection_matrix=self._projection_matrix,
            fit_mean=self._fit_mean,
            vectorizer_params={
                "min_df": self._min_df,
                "max_df": self._max_df,
            },
        )
