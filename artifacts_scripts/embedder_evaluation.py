"""Embedder evaluation: citation recall@k, MAP, and scatter metrics."""

from __future__ import annotations

from typing import Optional

import faiss
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp


def citation_recall_at_k(
    embeddings: NDArray[np.float32],
    citation_graph: sp.spmatrix,
    k: int = 20,
) -> float:
    """Mean fraction of actual citations found among k nearest embedding neighbors.

    Documents with no citations (zero rows in citation_graph) are excluded.
    """
    N, d = embeddings.shape
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    # k+1 because nearest neighbor of a point is itself
    _, I = index.search(embeddings, k + 1)

    recalls = []
    for i in range(N):
        true_citations = set(citation_graph[i].nonzero()[1])
        if not true_citations:
            continue
        # Exclude self from neighbor list
        neighbors = set(int(j) for j in I[i] if j != i)
        # Take only k neighbors (excluding self may have given us k from k+1)
        if len(neighbors) > k:
            neighbors = set(list(neighbors)[:k])
        recall = len(true_citations & neighbors) / len(true_citations)
        recalls.append(recall)

    return float(np.mean(recalls)) if recalls else 0.0


def citation_map(
    embeddings: NDArray[np.float32],
    citation_graph: sp.spmatrix,
) -> float:
    """Mean Average Precision: how well does embedding distance rank actual citations?"""
    N, d = embeddings.shape
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    _, I = index.search(embeddings, N)  # full ranking

    aps = []
    for i in range(N):
        true_citations = set(citation_graph[i].nonzero()[1])
        if not true_citations:
            continue
        ranking = [int(j) for j in I[i] if j != i]
        hits = 0
        precision_sum = 0.0
        for rank, j in enumerate(ranking, 1):
            if j in true_citations:
                hits += 1
                precision_sum += hits / rank
        ap = precision_sum / len(true_citations)
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def scatter_metrics(
    embeddings: NDArray,
    citation_graph: sp.spmatrix,
    book_mask: Optional[NDArray] = None,
) -> dict[str, float]:
    """Compute scatter-based embedding quality metrics."""
    N, d = embeddings.shape
    X_c = embeddings - embeddings.mean(axis=0, keepdims=True)

    S_T_trace = float(np.sum(X_c ** 2) / N)

    # Citation error
    cited_means = citation_graph @ embeddings
    D_error = embeddings - cited_means
    D_error_c = D_error - D_error.mean(axis=0, keepdims=True)
    S_C_trace = float(np.sum(D_error_c ** 2) / N)

    # Laplacian
    A_sym = citation_graph + citation_graph.T
    degrees = np.asarray(A_sym.sum(axis=1)).ravel()
    DX = degrees[:, None] * X_c
    S_D_trace = float(np.sum(X_c * DX) / N)
    AX = A_sym @ X_c
    S_A_trace = float(np.sum(X_c * AX) / N)
    S_L_trace = S_D_trace - S_A_trace

    result = {
        "variance_retention": S_T_trace,
        "citation_error_ratio": S_C_trace / S_T_trace if S_T_trace > 0 else 0.0,
        "laplacian_roughness": S_L_trace / S_T_trace if S_T_trace > 0 else 0.0,
    }

    if book_mask is not None:
        article_mask = ~book_mask
        n_b, n_a = int(book_mask.sum()), int(article_mask.sum())
        if n_b > 0 and n_a > 0:
            mu_b = X_c[book_mask].mean(axis=0)
            mu_a = X_c[article_mask].mean(axis=0)
            mu_diff = mu_b - mu_a
            S_B_trace = float((n_b * n_a / N) * np.dot(mu_diff, mu_diff))
            S_W_trace = float(np.sum(X_c[book_mask] ** 2) / N + np.sum(X_c[article_mask] ** 2) / N)
            result["class_separability"] = S_B_trace / S_W_trace if S_W_trace > 0 else 0.0

    return result
