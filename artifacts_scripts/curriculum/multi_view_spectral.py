"""Multi-view spectral clustering for the curriculum pipeline.

Combines embedding-similarity and citation-similarity views via the
co-regularized spectral framework of Kumar-Rai-Daumé (2011), with
CSLS hubness correction (Conneau et al. 2017) applied to each view
first.

Operates on dense (n, n) similarity matrices; assumes n ≤ ~10k.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp


def csls_correct(
    sim_matrix: NDArray[np.floating],
    k: int,
    clip: bool = True,
) -> NDArray[np.floating]:
    """Apply CSLS hubness correction to a dense similarity matrix.

    For each row i, computes ``r_K(i) = mean of i's top-k similarities to
    other points (self excluded)``. The CSLS-corrected similarity is::

        CSLS(i, j) = 2 * sim(i, j) - r_K(i) - r_K(j)

    Reference: Conneau, Lample, Ranzato, Denoyer, Jégou (2017),
    "Word Translation Without Parallel Data".

    Parameters
    ----------
    sim_matrix : (n, n) symmetric similarity matrix.
    k : Number of neighbours to average for r_K. Must be < n.
    clip : If True, clip negative entries to 0 (PSD-friendly for
        downstream Laplacian operations).
    """
    n = sim_matrix.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be less than n={n}")
    sim_no_self = sim_matrix.astype(np.float64, copy=True)
    np.fill_diagonal(sim_no_self, -np.inf)
    # argpartition leaves the top-k unsorted in the last k columns.
    top_k_idx = np.argpartition(sim_no_self, -k, axis=1)[:, -k:]
    top_k_vals = np.take_along_axis(sim_no_self, top_k_idx, axis=1)
    r_k = top_k_vals.mean(axis=1)
    csls = 2.0 * sim_matrix - r_k[:, None] - r_k[None, :]
    if clip:
        csls = np.maximum(csls, 0.0)
    return csls


def build_embedding_similarity_view(
    embeddings: NDArray[np.floating],
    csls_k: int = 10,
) -> NDArray[np.floating]:
    """Build a CSLS-corrected dense embedding-similarity view.

    Embeddings are L2-normalised; pairwise similarity is cosine; CSLS
    hubness correction is applied with the given k. Result is symmetric,
    non-negative, shape (n, n).
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # guard zero-norm rows
    emb_norm = embeddings / norms
    sim = emb_norm @ emb_norm.T
    return csls_correct(sim, k=csls_k, clip=True)


def build_citation_similarity_view(
    sub_adj: sp.spmatrix,
    csls_k: int = 10,
    mode: str = "weighted_unified",
) -> NDArray[np.floating]:
    """Build a CSLS-corrected dense citation-similarity view.

    Uses ``_citation_jaccard_distance_matrix`` to produce a Ruzicka
    distance matrix (one of three modes: ``binary``, ``weighted_unified``,
    ``weighted_per_direction``); converts to similarity; applies CSLS
    hubness correction.

    Default mode is ``weighted_unified`` because its row-normalisation
    already de-hubs the citation graph along the out-direction; CSLS
    then complements with in-direction de-hubbing.
    """
    # Lazy import to avoid circular dependency: build_curriculum imports
    # this module after we add the dispatch branch (Task 7).
    from artifacts_scripts.curriculum.build_curriculum import (
        _citation_jaccard_distance_matrix,
    )

    dist = _citation_jaccard_distance_matrix(sub_adj, mode=mode)
    sim = 1.0 - dist
    np.fill_diagonal(sim, 1.0)
    return csls_correct(sim, k=csls_k, clip=True)


def _normalized_laplacian(W: NDArray[np.floating]) -> NDArray[np.floating]:
    """Symmetric normalised Laplacian L = I - D^{-1/2} W D^{-1/2}.

    Floors degrees to a tiny positive value so isolated nodes don't break
    the inverse; the resulting row/column for an isolated node is zero
    off-diagonal and 1 on the diagonal.
    """
    deg = W.sum(axis=1)
    deg = np.maximum(deg, 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    L = np.eye(W.shape[0]) - d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]
    L = 0.5 * (L + L.T)  # numerical symmetrisation
    return L


def _subspace_distance(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
) -> float:
    """Frobenius distance between subspaces spanned by columns of A and B.

    Computes ``||A Aᵀ - B Bᵀ||_F``, which is invariant to column
    reorderings and sign flips since the projector ``A Aᵀ`` depends only
    on the column space of A. Used as the convergence metric for the
    co-regularized iteration.
    """
    PA = A @ A.T
    PB = B @ B.T
    return float(np.linalg.norm(PA - PB, ord="fro"))


def _bottom_k_eigvecs(M: NDArray[np.floating], k: int) -> NDArray[np.floating]:
    """Return the k eigenvectors of M with the smallest eigenvalues.

    Uses dense ``numpy.linalg.eigh`` (assumes M is symmetric and n is
    moderate, n ≤ ~10k). Symmetrises numerically before decomposing.
    """
    M_sym = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    # eigh returns ascending; first k are the smallest.
    return eigvecs[:, :k]


def cluster_with_multi_view_spectral_views(
    views: list[NDArray[np.floating]],
    *,
    n_clusters: int,
    co_reg_lambda: Optional[float] = None,
    auto_lambda_scale: float = 1.5,
    max_iters: int = 30,
    tol: float = 1e-5,
    seed: int = 42,
) -> NDArray[np.intp]:
    """Co-regularized multi-view spectral clustering for V≥2 views.

    Generalises Kumar-Rai-Daumé (2011) to any number of views. Solves for
    per-view embeddings ``U_1 … U_V`` of shape ``(n, K)`` that each
    minimise their own view's normalised-Laplacian quadratic form while
    maximising pairwise agreement::

        min  Σ_v tr(U_vᵀ L_v U_v) - λ Σ_{v<w} tr(U_v U_vᵀ U_w U_wᵀ)
        s.t. U_vᵀ U_v = I

    Solved by Gauss-Seidel alternation: with the other views fixed, ``U_v``
    is the bottom-K eigenvectors of ``L_v - λ · mean_{w≠v}(U_w U_wᵀ)``. The
    coupling is the **mean** (not the sum) of the other views' projectors,
    so that (a) a single λ scale transfers across V — the auto-tuned λ
    validated at V=2 stays calibrated at V=3 instead of the coupling
    growing ∝ (V−1) and collapsing the clustering; and (b) for V=2 this is
    byte-identical to the two-view path (mean of one term = that term).

    K-means on the row-normalised concatenation ``[U_1 | … | U_V]`` gives
    the final clustering.

    Parameters
    ----------
    views : list of V ``(n, n)`` symmetric similarity matrices.
    n_clusters : Number of clusters K.
    co_reg_lambda : Co-regularisation strength λ. If ``None``, auto-tune via
        ``auto_lambda_scale * mean_v(op_norm(L_v)) / K`` (the V=2 formula
        ``scale·(op1+op2)/(2K)`` written view-count-agnostically).
    auto_lambda_scale : Multiplicative constant; default ``1.5`` validated
        cross-corpus at V=2. Ignored if ``co_reg_lambda`` is set.
    max_iters, tol, seed : as for the two-view entry point.
    """
    from sklearn.cluster import KMeans

    V = len(views)
    if V < 2:
        raise ValueError(f"need ≥2 views, got {V}")
    n = views[0].shape[0]
    for i, W in enumerate(views):
        if W.shape != (n, n):
            raise ValueError(f"view {i} shape {W.shape} != ({n}, {n})")
    if n_clusters >= n:
        return np.arange(n, dtype=np.intp)

    K = n_clusters
    Ls = [_normalized_laplacian(W) for W in views]

    # Full eigendecomposition per view once: initial U_v + op_norms for λ.
    Us: list[NDArray[np.floating]] = []
    op_norms: list[float] = []
    for L in Ls:
        L_sym = 0.5 * (L + L.T)
        eigvals, eigvecs = np.linalg.eigh(L_sym)
        Us.append(eigvecs[:, :K])
        op_norms.append(float(eigvals[-1]))

    if co_reg_lambda is None:
        co_reg_lambda = auto_lambda_scale * (sum(op_norms) / V) / K

    for _ in range(max_iters):
        change = 0.0
        new_Us = list(Us)
        for v in range(V):
            # mean of the OTHER views' projectors (latest available, Gauss-Seidel)
            coupling = sum(
                new_Us[w] @ new_Us[w].T for w in range(V) if w != v
            ) / (V - 1)
            U_v_new = _bottom_k_eigvecs(Ls[v] - co_reg_lambda * coupling, K)
            change = max(change, _subspace_distance(U_v_new, Us[v]))
            new_Us[v] = U_v_new
        Us = new_Us
        if change < tol:
            break

    U_joint = np.concatenate(Us, axis=1)
    # Row-normalise (Ng-Jordan-Weiss spectral pre-processing).
    norms = np.linalg.norm(U_joint, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    U_joint = U_joint / norms

    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    labels = km.fit_predict(U_joint)
    return labels.astype(np.intp)


def cluster_with_multi_view_spectral(
    W1: NDArray[np.floating],
    W2: NDArray[np.floating],
    *,
    n_clusters: int,
    co_reg_lambda: Optional[float] = None,
    auto_lambda_scale: float = 1.5,
    max_iters: int = 30,
    tol: float = 1e-5,
    seed: int = 42,
) -> NDArray[np.intp]:
    """Two-view co-regularized spectral clustering (Kumar-Rai-Daumé 2011).

    Convenience entry point over
    :func:`cluster_with_multi_view_spectral_views` for the embedding +
    citation case. See that function for the full objective and the λ
    auto-tune. The two-view path is byte-identical to passing
    ``[W1, W2]`` to the general solver.

    Parameters
    ----------
    W1, W2 : (n, n) symmetric similarity matrices, one per view.
    n_clusters : Number of clusters K.
    co_reg_lambda : Co-regularisation strength λ. If ``None`` (default),
        auto-tune via
        ``auto_lambda_scale * (op_norm(L_1) + op_norm(L_2)) / (2 * K)``.
        This op-norm-based heuristic was empirically validated cross-corpus
        on the QC quantum-crypto and vision→robotics curriculum runs (both
        had modularity-Q optimum at λ=0.1; the auto-tuned λ ≈ 0.097 lands
        within −0.002 of the optimum on robotics, −0.019 on QC). Bulk-
        spectrum heuristics are robust to bottom-K eigenvalue degeneracy,
        which is common on near-rank-deficient Laplacians arising from
        CSLS-clipped similarity matrices. ``0.0`` recovers per-view
        independent spectral; large values force full agreement.
    auto_lambda_scale : Multiplicative constant ``c`` for the auto-tune
        formula. Default ``1.5`` is empirically validated cross-corpus.
        Ignored if ``co_reg_lambda`` is set explicitly.
    max_iters : Maximum alternations.
    tol : Subspace-distance tolerance for convergence.
    seed : Random seed for K-means.
    """
    return cluster_with_multi_view_spectral_views(
        [W1, W2],
        n_clusters=n_clusters,
        co_reg_lambda=co_reg_lambda,
        auto_lambda_scale=auto_lambda_scale,
        max_iters=max_iters,
        tol=tol,
        seed=seed,
    )
