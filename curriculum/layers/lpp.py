"""Spectral / link-LPP node geometry from the graph (Phase-0-bis).

Two related ways to derive a clustering subspace from the affinity:

1. :func:`spectral_embedding_from_graph` — the **pure-graph** path
   (``embeddings_mode='spectral_from_graph'``). Identity node features ⟹ the
   link-LPP generalized eigenproblem ``X L Xᵀ a = λ X D Xᵀ a`` collapses to
   ``L a = λ D a`` — i.e. Laplacian Eigenmaps on the (hub-suppressed) graph. We
   compute the bottom-``k`` eigenvectors of the **normalized** Laplacian
   directly via sparse ``scipy.sparse.linalg.eigsh`` (shift-invert at σ=0).
   We deliberately do NOT form any dense ``X L Xᵀ`` (the efficiency point the
   plan calls out). The result is a per-node ``(n, k)`` spectral embedding;
   cluster in it.

2. :func:`lpp_projection` — the **embedding-rich** path: fit a linear
   projection ``P`` (d×k) of an EXISTING embedding matrix that pulls
   link-connected nodes together, by solving the dense GEP
   ``X_cᵀ L X_c a = λ X_cᵀ D X_c a`` (the classic LPP). Used when you have
   semantic embeddings and want to reproject them through the link affinity
   (the ``wiki_pedagogical_lpp`` experiment).

Both share :func:`hub_suppressed_link_affinity`, which down-weights each edge
into a high-in-degree hub by ``1/log(indeg + e)`` before symmetrizing — the
fix for the DOI/ISBN/ISO bibliographic-hub Laplacian pathology.

The dense GEP solver is a self-contained lift of ``bookgap.denoisers``'
``_solve_eigh_robustly`` / ``subset_range`` (scipy/numpy only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from curriculum._logging import get_logger

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Affinities
# ---------------------------------------------------------------------------

def hub_suppressed_link_affinity(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Symmetric, hub-suppressed link affinity ``W``.

    Booleanize the directed adjacency, weight each directed edge ``i->j`` by
    ``1/log(indeg_j + e)`` (j's global in-degree), then symmetrize
    ``W = (W_dir + W_dirᵀ)/2``. High-in-degree hubs get heavily damped inbound
    edges so they no longer pin the Laplacian.
    """
    A = adj.copy().tocsr()
    A.data[:] = 1.0
    indeg = np.asarray(A.sum(axis=0)).ravel()
    supp = 1.0 / np.log(indeg + np.e)
    W_dir = A.multiply(supp[np.newaxis, :]).tocsr()
    W = (W_dir + W_dir.T) * 0.5
    return W.tocsr()


# ---------------------------------------------------------------------------
# Pure-graph spectral embedding (identity-feature link-LPP = Laplacian Eigenmaps)
# ---------------------------------------------------------------------------

def spectral_embedding_from_graph(
    adj: sp.csr_matrix,
    n_components: int = 64,
    *,
    hub_suppress: bool = True,
) -> NDArray[np.floating]:
    """Laplacian-Eigenmaps node embedding from the adjacency alone.

    Solves ``L a = λ D a`` for the bottom (non-trivial) ``n_components``
    eigenvectors of the normalized Laplacian of the (optionally hub-suppressed,
    symmetrized) affinity, via sparse ``eigsh`` — NEVER forming a dense
    ``X L Xᵀ``. Returns an ``(n_nodes, n_components)`` real embedding to cluster
    in. This is the Phase-0-bis pure-graph geometry (no semantic embedder).
    """
    n = adj.shape[0]
    if hub_suppress:
        W = hub_suppressed_link_affinity(adj)
    else:
        A = adj.copy().tocsr()
        A.data[:] = 1.0
        W = ((A + A.T) * 0.5).tocsr()

    deg = np.asarray(W.sum(axis=1)).ravel()
    deg_safe = np.where(deg > 0, deg, 1e-12)
    D = sp.diags(deg)
    L = (D - W).tocsr()

    # Solve the generalized symmetric problem L a = λ D a for the smallest
    # eigenvalues. eigsh's sigma=0 shift-invert targets the bottom of the
    # spectrum; we request a few extra to drop the trivial near-constant mode(s).
    k = min(n_components, max(1, n - 2))
    n_extra = min(k + 8, n - 1)
    try:
        eigvals, eigvecs = eigsh(L, k=n_extra, M=D, sigma=0, which="LM")
    except Exception as exc:  # pragma: no cover - numerical fallback
        LOGGER.warning("eigsh shift-invert failed (%s); retrying which='SM'", exc)
        eigvals, eigvecs = eigsh(L, k=n_extra, M=D, which="SM")

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Drop trivial near-DC eigenvector(s) (λ ≈ 0, constant mode).
    dc_floor = 1e-8 * max(1.0, float(np.abs(eigvals).max()))
    n_dc = int((eigvals < dc_floor).sum())
    n_dc = max(n_dc, 1) if eigvals[0] < dc_floor else n_dc
    keep = np.arange(n_dc, min(n_dc + k, eigvecs.shape[1]))
    Y = eigvecs[:, keep]
    # Row-scale by 1/sqrt(deg) is implicit in the D-orthonormal eigvecs; return
    # as-is (the geometry the clustering needs).
    if Y.shape[1] < k:
        LOGGER.warning(
            "spectral embedding produced %d < %d components", Y.shape[1], k
        )
    return np.ascontiguousarray(Y, dtype=np.float64)


# ---------------------------------------------------------------------------
# Dense LPP (reproject existing embeddings through the link affinity)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _EighResult:
    eigvals: NDArray
    vecs: NDArray


def _subset_range(n: int, subset_size: int, smallest: bool) -> Tuple[int, int]:
    subset_size = min(subset_size, n)
    if smallest:
        return 0, subset_size - 1
    return n - subset_size, n - 1


def _solve_eigh_robustly(
    a_matrix: NDArray,
    b_matrix: Optional[NDArray],
    subset: Optional[Tuple[int, int]] = None,
) -> _EighResult:
    """Dense symmetric (generalized) eigensolve with PSD-regularization retry.

    Self-contained lift of ``bookgap.denoisers._solve_eigh_robustly``: if
    ``b_matrix`` is indefinite/singular, add a ridge to push it PSD and retry.
    """
    try:
        eigvals, vecs = eigh(a=a_matrix, b=b_matrix, subset_by_index=subset)
    except LinAlgError:
        LOGGER.warning("eigh failed (singular/indefinite b); retrying with ridge")
        if b_matrix is not None:
            min_eigval = float(np.linalg.eigvalsh(b_matrix).min())
            if min_eigval < 0:
                ridge_val = -min_eigval + 1e-8
            else:
                ridge_val = float(np.mean(np.abs(np.diag(b_matrix)))) * 1e-4 + 1e-8
            b_matrix = b_matrix + np.eye(b_matrix.shape[0]) * ridge_val
            eigvals, vecs = eigh(a=a_matrix, b=b_matrix, subset_by_index=subset)
        else:
            raise
    return _EighResult(eigvals=eigvals, vecs=vecs)


def lpp_projection(
    X: NDArray[np.float64],
    W: sp.csr_matrix,
    k: int,
    *,
    dc_floor: float = 1e-6,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve LPP ``X_cᵀ L X_c a = λ X_cᵀ D X_c a`` for the k smallest non-trivial
    eigenvectors. Returns ``(P (d×k), eigvals (k,))``.

    Mirrors ``wiki_pedagogical_lpp.lpp_projection`` (centered ``X``, ``L=D-W``,
    smallest-eigenvalue ``eigh``), discarding any near-DC eigenvector
    (``λ < dc_floor``). Use this when you HAVE embeddings and want to reproject
    them through a link affinity; for no-embedding domains use
    :func:`spectral_embedding_from_graph` instead.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    deg = np.asarray(W.sum(axis=1)).ravel()
    D = sp.diags(deg)
    L = D - W

    A_mat = Xc.T @ (L @ Xc)
    B_mat = Xc.T @ (D @ Xc)
    A_mat = 0.5 * (A_mat + A_mat.T)
    B_mat = 0.5 * (B_mat + B_mat.T)

    res = _solve_eigh_robustly(a_matrix=A_mat, b_matrix=B_mat, subset=None)
    eigvals, V = res.eigvals, res.vecs
    n_dc = int((eigvals < dc_floor).sum())
    if n_dc:
        LOGGER.info("discarding %d near-DC eigenvector(s) (lambda<%.0e)", n_dc, dc_floor)
    keep = np.arange(n_dc, n_dc + k)
    P = V[:, keep]
    return P, eigvals[keep]
