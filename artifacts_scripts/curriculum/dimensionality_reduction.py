"""Dimensionality reduction methods for curriculum visualization.

This module provides projection methods (PCA, LDA, UMAP) for reducing high-dimensional
embeddings to 2D for visualization. Methods can use topic labels for supervised separation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import umap

from bookgap.util import LOGGER


class DimensionalityReductionMethod(str, Enum):
    """Method for reducing embeddings to 2D for visualization."""
    
    PCA = "pca"
    """Principal Component Analysis - fast, unsupervised, linear."""
    
    LDA = "lda"
    """Linear Discriminant Analysis - supervised, maximizes topic separation."""
    
    UMAP = "umap"
    """Uniform Manifold Approximation and Projection - supervised, preserves structure."""

    LPP = "lpp"
    """Locality Preserving Projection - linear, uses citation graph as locality."""

def project_to_2d(
    embeddings: NDArray[np.floating],
    labels: Optional[NDArray[np.int64]] = None,
    method: DimensionalityReductionMethod = DimensionalityReductionMethod.LDA,
    umap_config: Optional[Any] = None,
    lpp_config: Optional[Any] = None,
    citation_graph: Optional[sp.csr_matrix] = None,
) -> tuple[NDArray[np.floating], Any]:
    """Project embeddings to 2D using specified method.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim).
        labels: Optional topic labels for supervised methods (n_samples,).
        method: Dimensionality reduction method.
        umap_config: UMAPConfig instance with n_neighbors, min_dist, use_citation_graph.
        citation_graph: Optional citation adjacency matrix for UMAP precomputed metric.
        
    Returns:
        Tuple of (projected_embeddings, reducer_object).
        - projected_embeddings: Array of shape (n_samples, 2).
        - reducer_object: The fitted reducer (PCA, LDA, or UMAP object).
    """
    n_samples = embeddings.shape[0]
    
    if method == DimensionalityReductionMethod.PCA:
        LOGGER.info("Projecting to 2D via PCA...")
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(embeddings)
        explained_var = reducer.explained_variance_ratio_.sum()
        LOGGER.info(f"PCA explained variance: {explained_var:.2%}")
        
    elif method == DimensionalityReductionMethod.LDA:
        LOGGER.info("Projecting to 2D via LDA (supervised)...")
        if labels is None:
            raise ValueError("LDA requires topic labels")
        
        # Filter out unassigned points (label == -1)
        assigned_mask = labels != -1
        if not assigned_mask.any():
            raise ValueError("No assigned points for LDA")
        
        embeddings_assigned = embeddings[assigned_mask]
        labels_assigned = labels[assigned_mask]
        
        n_classes = len(np.unique(labels_assigned))
        n_components = min(2, n_classes - 1)
        
        if n_components < 2:
            LOGGER.warning(f"Only {n_classes} classes, LDA can produce at most {n_components} components. Padding with zeros.")
        
        reducer = LinearDiscriminantAnalysis(n_components=n_components)
        embeddings_assigned_2d = reducer.fit_transform(embeddings_assigned, labels_assigned)
        
        # Pad to 2D if needed
        if n_components < 2:
            embeddings_assigned_2d = np.column_stack([
                embeddings_assigned_2d.ravel(),
                np.zeros(len(embeddings_assigned_2d))
            ])
        
        # Project unassigned points using the same transformation
        embeddings_2d = np.zeros((n_samples, 2), dtype=np.float64)
        embeddings_2d[assigned_mask] = embeddings_assigned_2d
        
        if (~assigned_mask).any():
            embeddings_unassigned = embeddings[~assigned_mask]
            embeddings_unassigned_2d = reducer.transform(embeddings_unassigned)
            if n_components < 2:
                embeddings_unassigned_2d = np.column_stack([
                    embeddings_unassigned_2d.ravel(),
                    np.zeros(len(embeddings_unassigned_2d))
                ])
            embeddings_2d[~assigned_mask] = embeddings_unassigned_2d
        
        LOGGER.info(f"LDA used {n_classes} classes, {n_components} components")
        
    elif method == DimensionalityReductionMethod.UMAP:
        LOGGER.info("Projecting to 2D via UMAP (supervised)...")
        
        if umap_config is None:
            raise ValueError("UMAP requires umap_config")
        
        # Extract parameters
        n_neighbors = min(umap_config.n_neighbors, n_samples - 1)
        min_dist = umap_config.min_dist
        use_citation_graph = umap_config.use_citation_graph
        
        if use_citation_graph and citation_graph is not None:
            # Use citation graph as precomputed metric
            LOGGER.info(f"Using citation graph connectivity (shape={citation_graph.shape})")
            
            # Ensure rows are L1-normalized (row-stochastic). If not, normalize them.
            row_sums = np.array(citation_graph.sum(axis=1)).ravel()
            approx_norm = np.mean((row_sums > 0).astype(float) * np.abs(row_sums - 1.0))
            if approx_norm > 1e-5:
                LOGGER.info("Citation graph rows not normalized; applying L1 normalization.")
                citation_graph = normalize(citation_graph, norm='l1', axis=1)

            # Symmetrize adjacency to capture co-citation / bi-directional ties
            connectivity = citation_graph + citation_graph.T
            connectivity.eliminate_zeros()

            # L1 row-normalize to probability mass per row
            P = normalize(connectivity, norm='l1', axis=1)

            # Apply elementwise sqrt to data for Hellinger geometry
            P_sqrt = P.tocsr().copy()
            P_sqrt.data = np.sqrt(P_sqrt.data)

            # Cosine similarity over sqrt-normalized probability rows (Bhattacharyya-like)
            # Works with sparse; then distance = 1 - similarity
            sim = cosine_similarity(P_sqrt)
            distance_matrix = 1.0 - sim
            np.fill_diagonal(distance_matrix, 0.0)
            
            if labels is not None:
                LOGGER.info("UMAP with cosine(Hellinger) distance on symmetrized L1 rows + supervised labels")
                reducer = umap.UMAP(
                    n_components=2,
                    metric='precomputed',
                    n_neighbors=n_neighbors,
                    random_state=42,
                )
                embeddings_2d = reducer.fit_transform(distance_matrix, y=labels)
            else:
                LOGGER.info("UMAP with cosine(Hellinger) distance on symmetrized L1 rows (unsupervised)")
                reducer = umap.UMAP(
                    n_components=2,
                    metric='precomputed',
                    n_neighbors=n_neighbors,
                    random_state=42,
                )
                embeddings_2d = reducer.fit_transform(distance_matrix)
        else:
            # Use embeddings directly
            if use_citation_graph and citation_graph is None:
                LOGGER.warning("use_citation_graph=True but citation_graph not provided, falling back to embeddings")
            
            if labels is not None:
                LOGGER.info("UMAP with embedding space + supervised labels")
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    # random_state=42, # Setting random_state disables parallelism
                )
                embeddings_2d = reducer.fit_transform(embeddings, y=labels)
            else:
                LOGGER.info("UMAP with embedding space (unsupervised)")
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    # random_state=42, # Setting random_state disables parallelism
                )
                embeddings_2d = reducer.fit_transform(embeddings)
    elif method == DimensionalityReductionMethod.LPP:
        LOGGER.info("Projecting to 2D via LPP (supervised by citation locality)...")

        if lpp_config is None:
            raise ValueError("LPP requires lpp_config")

        if citation_graph is None:
            raise ValueError("LPP requires citation_graph")

        # Ensure rows are L1-normalized (row-stochastic). If not, normalize them.
        row_sums = np.array(citation_graph.sum(axis=1)).ravel()
        approx_norm = np.mean((row_sums > 0).astype(float) * np.abs(row_sums - 1.0))
        if approx_norm > 1e-5:
            LOGGER.info("Citation graph rows not normalized; applying L1 normalization for LPP.")
            citation_graph = normalize(citation_graph, norm='l1', axis=1)

        # Symmetrize and build S = (A + A.T) / 2
        S = (citation_graph + citation_graph.T) * 0.5
        S.eliminate_zeros()

        # Degree vector and Laplacian action via sparse matmul
        d = np.array(S.sum(axis=1)).ravel()

        # Optionally center the embeddings
        X = embeddings
        if lpp_config.center:
            Xc = X - X.mean(axis=0)
        else:
            Xc = X

        # Compute L @ X = D X - S X
        SX = S.dot(Xc)
        LX = (d[:, None] * Xc) - SX

        # Form small matrices M = X^T L X and B = X^T D X
        M = Xc.T.dot(LX)
        B = Xc.T.dot(d[:, None] * Xc)

        try:
            eigvals, eigvecs = eigh(M, B)
        except Exception as e:
            # Fall back to tiny ridge on B if it's not positive definite
            LOGGER.warning("Generalized eigenproblem failed (B not positive-definite). Applying tiny ridge and retrying.")
            eps = 1e-8 * np.trace(B) / max(1, B.shape[0])
            B_reg = B + eps * np.eye(B.shape[0])
            eigvals, eigvecs = eigh(M, B_reg)
        idx = np.argsort(eigvals)
        k = min(2, eigvecs.shape[1])
        A = eigvecs[:, idx[:k]]

        embeddings_2d = Xc.dot(A)
        reducer = {'name': 'lpp', 'components': A, 'eigenvalues': eigvals[idx[:k]]}
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    
    return embeddings_2d, reducer
