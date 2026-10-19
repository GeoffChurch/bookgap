import abc
from dataclasses import dataclass
import enum
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, svd, qr
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.decomposition import PCA
from numpy.linalg import pinv, LinAlgError, lstsq
from bookgap.util import TIMER, LOGGER
from abc import ABC, abstractmethod


def subset_range(n: int, subset_size: int, smallest: bool) -> tuple[int, int]:
    subset_size = min(subset_size, n)
    if smallest:
        return 0, subset_size - 1
    else:
        return n - subset_size, n - 1

def adj(A):
    return A.conj().T

def scatter(M: NDArray, centered: bool = False) -> NDArray:
    """
    Computes the scatter matrix (unnormalized population covariance).
    If centered=False, it computes the mean and centers the data.
    If centered=True, it assumes M is already centered.
    """
    M_centered = M
    if not centered and M.shape[0] > 0:
        M_centered = M - M.mean(axis=0, keepdims=True)
    
    n = M_centered.shape[0]
    if n == 0:
        return np.zeros((M_centered.shape[1], M_centered.shape[1]), dtype=M.dtype)
        
    # S = Xc.T @ Xc / n
    return adj(M_centered) @ M_centered / n

@dataclass
class CenteredMatrix:
    """Stores X and its mean separately to avoid materializing dense centered matrices.

    For sparse X (N x V with V >> N), computing X_c = X - mean would destroy
    sparsity. Instead, operations like gram() use algebraic identities to stay
    in N x N space.
    """
    X: NDArray | sp.spmatrix  # N x d
    mean: NDArray             # (1, d) dense

    def gram(self) -> NDArray:
        """X_c @ X_c.T  (N x N) without forming the dense centered matrix.

        Uses the identity:
            (X - 1*mu)(X - 1*mu).T = X@X.T - (X@mu.T)@1.T - 1@(X@mu.T).T + ||mu||^2 * 1@1.T
        where 1 is (N,1) column of ones, mu is (1,d), and ||mu||^2 = mu@mu.T is a scalar.
        """
        n = self.X.shape[0]
        XXT = self.X @ self.X.T                   # N x N  (sparse-safe)
        Xmu = self.X @ self.mean.T                 # N x 1  (X dot mean^T)
        # Convert to dense arrays to handle both sparse and dense inputs uniformly
        XXT_dense = np.asarray(XXT.todense()) if sp.issparse(XXT) else np.asarray(XXT)
        Xmu_dense = np.asarray(Xmu.todense()) if sp.issparse(Xmu) else np.asarray(Xmu)  # (N, 1)
        mu_sq = float(np.squeeze(self.mean @ self.mean.T))  # scalar: ||mean||^2
        # Xmu_dense is (N,1); broadcast: Xmu_dense @ ones.T = (N,N) where row i is Xmu[i] repeated
        ones = np.ones((n, 1))
        return XXT_dense - Xmu_dense @ ones.T - ones @ Xmu_dense.T + np.full((n, n), mu_sq)

    def X_centered_dense(self) -> NDArray:
        """Materialize the centered matrix. Only safe when d is small."""
        X_dense = self.X.toarray() if sp.issparse(self.X) else np.asarray(self.X)
        return X_dense - self.mean


# def orthonormalize(P: NDArray) -> NDArray:
#     Q: NDArray
#     R: NDArray
#     Q, R = qr(P, mode='economic') # type: ignore
    
#     assert Q.shape == P.shape, f"{Q.shape=} != expected {P.shape=}"
    
#     # Check for numerical issues in R diagonal
#     r_diag = np.abs(np.diag(R))
#     tol = max(1e-12, r_diag[0] * 1e-12) if len(r_diag) > 0 else 1e-12
#     n_flat_dimensions = np.sum(r_diag < tol)
    
#     if n_flat_dimensions:
#         LOGGER.warning(f"Found {n_flat_dimensions} additional flat dimensions in projector, which may indicate numerical instability.")
    
#     return Q

# def get_coordinate_transform(P: NDArray) -> NDArray:
#     return P @ np.linalg.inv(P.T @ P)

class DenoiserContext(abc.ABC):
    """
    Abstract interface for providing data to denoiser strategies.
    Provides data and consistently centered scatter matrices.
    """
    def __init__(self):
        self._centered: Optional[CenteredMatrix] = None

    def _init_centering(self):
        """Initializes and caches the centered data."""
        if self._centered is None:
            X = self.X()
            mean = X.mean(axis=0, keepdims=True) if not sp.issparse(X) else np.asarray(X.mean(axis=0))
            self._centered = CenteredMatrix(X=X, mean=np.atleast_2d(mean))
            LOGGER.info(f"DenoiserContext: Centered, X shape={X.shape}")

    def centered(self) -> CenteredMatrix:
        """The CenteredMatrix wrapping X and its mean."""
        self._init_centering()
        assert self._centered is not None
        return self._centered

    def X_centered(self) -> NDArray:
        """The dense centered matrix. Only safe when d is small."""
        return self.centered().X_centered_dense()

    def X_mean(self) -> NDArray:
        """The global mean of X."""
        return self.centered().mean

    @abc.abstractmethod
    def A(self) -> sp.csr_matrix:
        """The citation graph affinity matrix."""
        ...

    @abc.abstractmethod
    def X(self) -> NDArray:
        """The full embeddings matrix."""
        ...

    @abc.abstractmethod
    def X_n(self, n: int) -> NDArray:
        """The embeddings for the n-th subset (0-indexed)."""
        ...

    @abc.abstractmethod
    def num_subsets(self) -> int:
        """The number of subsets available."""
        ...

    def A_sym(self) -> sp.csr_matrix:
        """The symmetrization of the affinity matrix A."""
        return self.A() + adj(self.A())
    
    def D(self) -> sp.dia_matrix:
        """The degree matrix D, where D[i, i] = sum(A[i, :])."""
        return sp.diags(np.asarray(self.A_sym().sum(axis=1)).squeeze(1))

    def L(self) -> sp.spmatrix:
        """The Laplacian matrix L = D - A."""
        return self.D() - self.A_sym()
    
    def S_T(self) -> NDArray:
        """The total scatter matrix (covariance matrix of X)."""
        return scatter(self.X_centered(), centered=True)
    
    def S_L(self) -> NDArray:
        """The graph Laplacian scatter matrix X_c^T L X_c."""
        return adj(self.X_centered()) @ self.L() @ self.X_centered()
    
    def S_B(self) -> NDArray:
        """The between-class scatter matrix for two subsets X1 and X2 (backward compatibility)."""
        assert self.num_subsets() == 2, "S_B is only defined for exactly two subsets."
        return self.S_B_multi([0, 1])
    
    def S_B_multi(self, subset_indices: list[int]) -> NDArray:
        """The between-class scatter matrix for multiple subsets."""
        if len(subset_indices) < 2:
            d = self.X().shape[1]
            return np.zeros((d, d))
        
        # Get all subset data and sizes
        subsets = [self.X_n(i) for i in subset_indices]
        sizes = [len(subset) for subset in subsets]
        total_size = sum(sizes)
        
        if total_size == 0:
            d = self.X().shape[1]
            return np.zeros((d, d))
        
        # Use the pre-computed global mean for all calculations
        overall_mean = self.X_mean().squeeze()
        
        # Compute between-class scatter matrix
        S_B = np.zeros((self.X().shape[1], self.X().shape[1]), dtype=self.X().dtype)
        for subset, size in zip(subsets, sizes):
            if size > 0:
                # subset.mean(axis=0) is the mean of the uncentered subset
                # (subset.mean(axis=0) - overall_mean) is the mean of the centered subset
                mu_diff_centered = subset.mean(axis=0) - overall_mean
                S_B += size * np.outer(mu_diff_centered, mu_diff_centered)
        
        return S_B / total_size
    
    def S_subset(self, subset_index: int) -> NDArray:
        """The scatter matrix for a specific subset, relative to the global mean."""
        X_n_centered = self.X_n(subset_index) - self.X_mean()
        return scatter(X_n_centered, centered=True)

    @TIMER.time()
    def S_C(self) -> NDArray:
        """The citation-error scatter matrix"""        
        # Sparse matmul gives the mean of cited embeddings per work because rows of A are normalized.
        X = self.X()
        cited_means = self.A() @ X # A[i, j] > 0 means i cites j
        D_error = X - cited_means

        # S_C should be the covariance of the error vectors
        return scatter(D_error, centered=False) # D_error may not be zero-mean

@dataclass(frozen=True)
class SolveEighRobustlyResult:
    eigvals: NDArray
    P_or_U: NDArray

    def __post_init__(self):
        if self.eigvals.ndim != 1:
            raise ValueError(f"eigvals must be 1D, got {self.eigvals.ndim}D array.")
        d, = self.eigvals.shape
        if self.P_or_U.ndim != 2 or self.P_or_U.shape[1] != d:
            raise ValueError(f"P_or_U must be 2D with second dimension matching eigvals, got {self.P_or_U.shape}.")

@TIMER.time(formatters=dict(a_matrix=lambda a: f"<shape={a.shape}>", b_matrix=lambda b: f"<shape={b.shape}>" if b is not None else "None"))
def _solve_eigh_robustly(a_matrix: NDArray, b_matrix: NDArray | None, subset: Optional[tuple[int, int]] = None) -> SolveEighRobustlyResult:
    """Solves eigh, retrying with regularization upon failure.

    If b_matrix is indefinite (has negative eigenvalues), clips it to PSD by
    adding a ridge equal to |min_eigenvalue| + a small epsilon.
    """
    try:
        eigvals, P_or_U = eigh(a=a_matrix, b=b_matrix, subset_by_index=subset)
    except LinAlgError:
        LOGGER.warning("eigh failed, likely due to a singular or indefinite b_matrix. Retrying with regularization.")
        if b_matrix is not None:
            # Compute the minimum eigenvalue to determine the required ridge.
            # If b_matrix is indefinite (min_eigval < 0), we need to shift it
            # to PSD; if it's just near-singular, a small epsilon suffices.
            min_eigval = float(np.linalg.eigvalsh(b_matrix).min())
            if min_eigval < 0:
                ridge_val = -min_eigval + 1e-8
                LOGGER.warning(f"b_matrix is indefinite (min_eigval={min_eigval:.4e}). Adding ridge: {ridge_val:.2e}")
            else:
                ridge_val = np.mean(np.abs(np.diag(b_matrix))) * 1e-4 + 1e-8
                LOGGER.warning(f"Retrying with adaptive regularization: {ridge_val:.2e}")
            b_matrix = b_matrix + np.eye(b_matrix.shape[0]) * ridge_val
            eigvals, P_or_U = eigh(a=a_matrix, b=b_matrix, subset_by_index=subset)
        else:
            raise
    
    # # If this was a generalized eigenvalue problem, orthogonalize the result
    # if b_matrix is not None:
    #     LOGGER.info(f"Orthogonalizing generalized eigenvectors from shape {P_or_U.shape}")
    #     P_or_U = orthonormalize(P_or_U)
    #     LOGGER.info(f"Orthogonalized projector shape: {P_or_U.shape}")
    
    LOGGER.info(f"{eigvals.shape=} {P_or_U.shape=}")
    return SolveEighRobustlyResult(eigvals=eigvals, P_or_U=P_or_U)


def _centered_matvec(X: sp.spmatrix | NDArray, mean: NDArray, v: NDArray) -> NDArray:
    """Compute X_c.T @ (X_c @ v) without materializing X_c.

    X_c = X - 1@mean where mean is (1, d).
    X_c @ v = X @ v - mean @ v * ones
    X_c.T @ u = X.T @ u - mean.T * sum(u)
    """
    N = X.shape[0]
    # Forward: X_c @ v
    Xv = X @ v  # (N,) or (N,1)
    mean_dot_v = mean.ravel() @ v.ravel()  # scalar
    Xc_v = np.asarray(Xv).ravel() - mean_dot_v  # (N,)
    # Backward: X_c.T @ (X_c @ v)
    XtXcv = X.T @ Xc_v  # (d,)
    result = np.asarray(XtXcv).ravel() - mean.ravel() * Xc_v.sum()  # (d,)
    return result


def _solve_gep_sparse(
    X: sp.spmatrix,
    A: sp.spmatrix,
    n_components: int,
) -> SolveEighRobustlyResult:
    """Solve a citation-error GEP for sparse X via truncated SVD reduction.

    Solves  S_C w = λ (S_T - S_C) w  for the smallest eigenvalues, where:
      S_T = X_c.T @ X_c / N       (total scatter)
      S_C = E_c.T @ E_c / N       (citation-error scatter, E = X - A@X)

    Strategy: use truncated SVD to find the top-r right singular vectors of
    X_c, project both scatter matrices into that r-dimensional subspace, then
    solve the small r×r dense GEP.  This is mathematically identical to the
    existing dual-path approach in CitationErrorLDADenoiser but uses
    scipy.sparse.linalg.svds to avoid materializing dense N×V matrices.

    Memory: O(N*r + r^2), never O(N*V) or O(V^2).
    Time:   O(nnz(X)*r) for the truncated SVD, O(r^3) for the dense GEP.
    """
    from scipy.sparse.linalg import svds

    N, V = X.shape
    LOGGER.info(f"Sparse GEP: N={N}, V={V}, k={n_components}")

    # Choose r: enough dimensions to capture signal, but small enough for dense GEP.
    # Mirrors the dual-path logic: keep directions with s_i/s_max > 1e-2.
    # Start with a generous r and trim after seeing the singular values.
    r = min(min(N, V) - 1, max(n_components * 3, 512))
    LOGGER.info(f"Sparse GEP: computing rank-{r} truncated SVD ...")

    # svds works directly on sparse X (only needs matvecs).
    # We subtract the mean via a LinearOperator to stay sparse-safe.
    X_mean = np.asarray(X.mean(axis=0)).ravel()  # (V,)

    def Xc_matvec(v):
        """(X - 1@mu) @ v = X@v - mu * sum(v)"""
        v = np.asarray(v).ravel()
        Xv = np.asarray(X @ v).ravel()  # (N,)
        return Xv - X_mean.dot(v)  # scalar broadcast to (N,)

    def Xc_rmatvec(u):
        """(X - 1@mu).T @ u = X.T@u - mu * sum(u)"""
        u = np.asarray(u).ravel()
        Xtu = np.asarray(X.T @ u).ravel()  # (V,)
        return Xtu - X_mean * u.sum()  # (V,)

    Xc_op = LinearOperator((N, V), matvec=Xc_matvec, rmatvec=Xc_rmatvec, dtype=np.float64)

    U_svd, s_svd, Vt_svd = svds(Xc_op, k=r, which="LM")
    # svds returns in ascending order; reverse to descending
    U_svd = U_svd[:, ::-1]
    s_svd = s_svd[::-1]
    Vt_svd = Vt_svd[::-1, :]

    # Trim to significant dimensions (s_i/s_max > 1e-2)
    sig_threshold = 1e-2 * s_svd[0]
    r_eff = max(int((s_svd > sig_threshold).sum()), n_components + 1)
    r_eff = min(r_eff, r)
    V_r = Vt_svd[:r_eff, :].T  # V x r_eff

    LOGGER.info(
        f"Sparse GEP: SVD done, r_eff={r_eff}, "
        f"top s-vals: {' '.join(f'{v:.1f}' for v in s_svd[:5])}"
    )

    # Project into r_eff-dimensional space
    Z = np.asarray(X @ V_r)  # N x r_eff (dense)
    if sp.issparse(Z):
        Z = Z.toarray()
    Z_c = Z - Z.mean(axis=0, keepdims=True)

    # Citation error in reduced space
    AZ = A @ Z  # N x r_eff
    AZ = np.asarray(AZ.todense()) if sp.issparse(AZ) else np.asarray(AZ)
    E_r = Z - AZ
    E_r_c = E_r - E_r.mean(axis=0, keepdims=True)

    # Build r_eff × r_eff scatter matrices
    S_T_r = Z_c.T @ Z_c / N
    S_C_r = E_r_c.T @ E_r_c / N
    S_signal_r = S_T_r - S_C_r

    LOGGER.info(f"Sparse GEP: solving {r_eff}x{r_eff} dense GEP ...")

    subset = subset_range(n=r_eff, subset_size=n_components, smallest=True)
    res = _solve_eigh_robustly(a_matrix=S_C_r, b_matrix=S_signal_r, subset=subset)

    # Recover V-dimensional projector: P = V_r @ P_reduced
    P_raw = V_r @ res.P_or_U  # V x n_components
    P_norms = np.linalg.norm(P_raw, axis=0)
    P_norms[P_norms == 0] = 1
    P = P_raw / P_norms

    LOGGER.info(f"Sparse GEP done: eigvals range=[{res.eigvals[0]:.4g}, {res.eigvals[-1]:.4g}]")
    return SolveEighRobustlyResult(eigvals=res.eigvals, P_or_U=P)


@TIMER.time(formatters=dict(X=lambda X: f"<shape={X.shape}>", U=lambda U: f"<shape={U.shape}>"))
def _recover_projector_from_dual(X: NDArray, U: NDArray) -> NDArray:
    """Recovers the d-dimensional projector P from the N-dimensional dual eigenvectors U."""
    P_raw = adj(X) @ U
    P_norms = np.linalg.norm(P_raw, axis=0)

    LOGGER.info(f"{X.shape=} @ {U.shape=} = {P_raw.shape=}")
    LOGGER.info(f"np.linalg.norm({P_raw.shape=}, axis=0) = {P_norms.shape=}")
    
    # Avoid division by zero for null projectors without adding a ridge.
    # If a norm is zero, the resulting column will be all zeros, which is correct.
    P_norms[P_norms == 0] = 1
    ret = P_raw / P_norms
    LOGGER.info(f"{P_raw.shape=} {P_norms.shape=} {ret.shape=}")
    return ret

class PCADenoiser:
    def projector(self, context: DenoiserContext, n_components: int) -> tuple[NDArray, Optional[NDArray]]:
        X_c = context.X_centered() # Use centered data
        min_cs = min(X_c.shape)
        if n_components > min_cs:
            n_components = min_cs
        # PCA in sklearn handles centering, but we provide centered data
        # which it will just re-center (no harm)
        pca = PCA(n_components=n_components)
        pca.fit(X_c)
        P = pca.components_.T
        return P, pca.singular_values_

class GPCADenoiser:
    def projector(self, context: DenoiserContext, n_components: int):
        X_c = context.X_centered() # Use centered data
        B = context.A_sym()
        M = adj(X_c) @ B @ X_c # Use centered data
        d = X_c.shape[1]
        subset = subset_range(n=d, subset_size=n_components, smallest=False)
        eigvals, P = eigh(M, subset_by_index=subset)
        return P, eigvals

class RelAmpDenoiser:
    def projector(self, context: DenoiserContext, n_components: int):
        X_c = context.X_centered() # Use centered data
        A = context.A()
        N, d = X_c.shape
        B = context.A_sym()
        
        if N < d:
            LOGGER.info(f"RelAmp: N < d, using dual formulation. N={N}, d={d}")
            K = X_c @ adj(X_c) # Use centered data
            subset = subset_range(n=N, subset_size=n_components, smallest=False)
            a_dual = adj(K) @ B @ K
            b_dual = adj(K) @ K
            res = _solve_eigh_robustly(a_dual, b_dual, subset)
            P = _recover_projector_from_dual(X_c, res.P_or_U) # Use centered data
            return P, res.eigvals
        else:
            subset = subset_range(n=d, subset_size=n_components, smallest=False)
            a_matrix = adj(X_c) @ B @ X_c # Use centered data
            b_matrix = adj(X_c) @ X_c # Use centered data
            res = _solve_eigh_robustly(a_matrix, b_matrix, subset)
            P = res.P_or_U
            return P, res.eigvals

# class NeighborDistMinDenoiser:
#     def projector(self, X: NDArray, A: sp.spmatrix, n_components: int):
#         N, d = X.shape
#         B = A + adj(A)
#         M = sp.eye(N) - B
#         a_mat = adj(X) @ adj(M) @ M @ X
        
#         if N < d:
#             LOGGER.info(f"NeighborDistMin: N < d, using dual formulation. N={N}, d={d}")
#             K = X @ adj(X)
#             subset = subset_range(n=N, subset_size=n_components, smallest=True)
#             a_dual = adj(K) @ adj(M) @ M @ K
#             b_dual = adj(K) @ K
#             res = _solve_eigh_robustly(a_dual, b_dual, subset)
#             P = _recover_projector_from_dual(X, res.P_or_U)
#             return P, res.eigvals
#         else:
#             subset = subset_range(n=d, subset_size=n_components, smallest=True)
#             b_mat = adj(X) @ X
#             res = _solve_eigh_robustly(a_mat, b_mat, subset)
#             P = res.P_or_U
#             return P, res.eigvals

class LPPDenoiser:
    def projector(self, context: DenoiserContext, n_components: int):
        X_c = context.X_centered() # Use centered data
        N, d = X_c.shape
        B = context.A_sym()

        negatives = B < 0
        num_negatives = np.sum(negatives)
        if num_negatives > 0:
            LOGGER.warning(f"LPPDenoiser received an affinity matrix with {num_negatives} negative weights ({num_negatives / B.size:.2%} of total). Setting negative weights to zero.")
            B[B < 0] = 0
            LOGGER.info(f"After setting negatives to zero, {type(B)=} {B.shape=}")

        B_sum = np.asarray(B.sum(axis=1)).squeeze(1)
        LOGGER.info(f"{np.mean(B_sum)=:.2f} (mean), {np.std(B_sum)=:.2f} (std), {np.min(B_sum)=:.2f} (min), {np.max(B_sum)=:.2f} (max) {np.percentile(B_sum, 10)=:.2f}, {np.percentile(B_sum, 90)=:.2f}")
        D = sp.diags(B_sum)
        L = D - B

        if N < d:
            LOGGER.info(f"LPP: N < d, using dual formulation. N={N}, d={d}")
            K = X_c @ adj(X_c) # Use centered data
            subset = subset_range(n=N, subset_size=n_components, smallest=True)
            a_dual = adj(K) @ L @ K
            LOGGER.info(f"{type(a_dual)=} {a_dual.shape=}")
            b_dual = adj(K) @ D @ K
            LOGGER.info(f"{type(b_dual)=} {b_dual.shape=}")
            res = _solve_eigh_robustly(a_dual, b_dual, subset)
            P = _recover_projector_from_dual(X_c, res.P_or_U) # Use centered data
            return P, res.eigvals
        else:
            subset = subset_range(n=d, subset_size=n_components, smallest=True)
            a_matrix = adj(X_c) @ L @ X_c # Use centered data
            b_matrix = adj(X_c) @ D @ X_c # Use centered data
            res = _solve_eigh_robustly(a_matrix, b_matrix, subset)
            P = res.P_or_U
            return P, res.eigvals

class SymmetricNormalizedLPPDenoiser:
    """Normalized LPP using the SYMMETRIC normalized Laplacian L_sym with
    the identity constraint --- the principled normalized variant.

    Solves the generalized eigenproblem:
        X_c^T L_sym X_c a = lambda X_c^T X_c a
    where L_sym = I - D^{-1/2} W D^{-1/2}. Equivalent to plain LPP on
    degree-scaled data Y = D^{-1/2} X via the identity
    X_c^T L_sym X_c = Y_c^T L Y_c.

    Contrast with :class:`NormalizedLPPDenoiser`, which uses L_sym + X^T D X
    constraint --- a non-standard hybrid that doesn't correspond to any
    published variant. This class is the standard normalized-spectral
    formulation.
    """
    def projector(self, context: DenoiserContext, n_components: int):
        X_c = context.X_centered()
        N, d = X_c.shape
        B = context.A_sym()
        degrees = np.array(B.sum(axis=1)).ravel()
        safe_indices = degrees > 0
        inv_sqrt_deg = np.zeros_like(degrees, dtype=float)
        inv_sqrt_deg[safe_indices] = np.power(degrees[safe_indices], -0.5)
        D_inv_sqrt = sp.diags(inv_sqrt_deg)
        if not np.any(safe_indices):
            LOGGER.warning(
                "SymmetricNormalizedLPPDenoiser: all nodes have zero degree; "
                "falling back to identity-Laplacian regularization."
            )
            L_sym = sp.eye(N) - (D_inv_sqrt @ B @ D_inv_sqrt)
        else:
            L_sym = sp.eye(N) - (D_inv_sqrt @ B @ D_inv_sqrt)

        if N < d:
            LOGGER.info(f"SymmetricNormalizedLPP: N < d, dual. N={N}, d={d}")
            K = X_c @ adj(X_c)
            subset = subset_range(n=N, subset_size=n_components, smallest=True)
            a_dual = adj(K) @ L_sym @ K
            b_dual = adj(K) @ K
            res = _solve_eigh_robustly(a_dual, b_dual, subset)
            P = _recover_projector_from_dual(X_c, res.P_or_U)
            return P, res.eigvals

        subset = subset_range(n=d, subset_size=n_components, smallest=True)
        a_matrix = adj(X_c) @ L_sym @ X_c
        b_matrix = adj(X_c) @ X_c
        assert np.allclose(adj(a_matrix), a_matrix, atol=1e-6), "a_matrix is not Hermitian!"
        res = _solve_eigh_robustly(a_matrix, b_matrix, subset)
        return res.P_or_U, res.eigvals


class NormalizedLPPDenoiser:
    def projector(self, context: DenoiserContext, n_components: int):
        X_c = context.X_centered() # Use centered data
        B = context.A_sym()
        N, d = X_c.shape
        degrees = np.array(B.sum(axis=1)).ravel()
        inv_sqrt_deg = np.zeros_like(degrees, dtype=float)
        safe_indices = degrees > 0
        inv_sqrt_deg[safe_indices] = np.power(degrees[safe_indices], -0.5)
        D_inv_sqrt = sp.diags(inv_sqrt_deg)
        D = sp.diags(degrees)
        L_sym = sp.eye(N) - (D_inv_sqrt @ B @ D_inv_sqrt)
        if not np.any(safe_indices):
            LOGGER.warning("Normalized LPP: All nodes have zero degree; falling back to identity constraint.")
            D = sp.eye(N)
            D_inv_sqrt = sp.eye(N)
            L_sym = sp.eye(N) - (D_inv_sqrt @ B @ D_inv_sqrt)
        if N < d:
            LOGGER.info(f"Normalized LPP: N < d, using dual formulation. N={N}, d={d}")
            K = X_c @ adj(X_c) # Use centered data
            subset = subset_range(n=N, subset_size=n_components, smallest=True)
            a_dual = adj(K) @ L_sym @ K
            b_dual = adj(K) @ D @ K
            res = _solve_eigh_robustly(a_dual, b_dual, subset)
            P = _recover_projector_from_dual(X_c, res.P_or_U) # Use centered data
            return P, res.eigvals

        a_matrix = adj(X_c) @ L_sym @ X_c # Use centered data
        b_matrix = adj(X_c) @ D @ X_c # Use centered data
        subset = subset_range(n=d, subset_size=n_components, smallest=True)
        assert np.allclose(adj(a_matrix), a_matrix, atol=1e-6), "a_matrix is not Hermitian!"
        res = _solve_eigh_robustly(a_matrix, b_matrix, subset)
        return res.P_or_U, res.eigvals

class WDenoiser:
    """
    Learns W where X_c W ≈ A X_c, then uses rank-reduced W for projection.
    """
    def projector(self, context: DenoiserContext, n_components: int):
        X_c = context.X_centered() # Use centered data
        A = context.A()
        # Solve for W in X_c W ≈ A X_c
        target = A @ X_c # Use centered data
        W, _, _, _ = lstsq(X_c, target, rcond=None) # Use centered data
        U, s, Vt = svd(W)
        P = U[:, :n_components]
        return P, s[:n_components]

# class LDADenoiser:
#     """
#     Maximizes the ratio of total variance to the variance of difference vectors AX-X.
#     This is solved as the generalized eigenvalue problem S_T g = λ S_W g.
#     """
#     def projector(self, context: DenoiserContext, n_components: int):
#         X = context.X()
#         A = context.A()
#         N, d = X.shape
#         res = _solve_eigh_robustly(a_matrix=scatter(X), b_matrix=scatter(A @ X - X), subset=subset_range(n=d, subset_size=n_components, smallest=False))
#         P_basis = res.P_or_U
#         LOGGER.info(f"LDADenoiser: {P_basis.shape=} {res.eigvals.shape=}")
#         # Return the projector and the eigenvalues as extra info
#         return P_basis, res.eigvals
    
# class SymmetrizedLDADenoiser(LDADenoiser):
#     def projector(self, X: NDArray, A: sp.spmatrix, n_components: int):
#         return super().projector(X, A + adj(A), n_components)

class TwoSubsetScatterDenoiser(ABC):
    """
    Abstract base for denoisers that operate on two subsets and require within-class scatter.
    Subclasses must implement get_within_class_scatter.
    """
    @abstractmethod
    def get_within_class_scatter(self, X1_c: NDArray, X2_c: NDArray, X_c: NDArray) -> NDArray:
        ...

    def projector(self, context: DenoiserContext, n_components: int) -> tuple[NDArray, Optional[NDArray]]:
        # Get data and global mean from context
        X_mean = context.X_mean()
        X1 = context.X_n(0)
        X2 = context.X_n(1)
        X_c = context.X_centered()
        
        N1, d = X1.shape
        N2, _ = X2.shape
        N_total = N1 + N2
        
        # Center all data using the pre-computed global mean
        X1_c = X1 - X_mean
        X2_c = X2 - X_mean
        
        # Within-class scatter (subclass-specific)
        S_W = self.get_within_class_scatter(X1_c, X2_c, X_c)
        
        # Between-class scatter (already defined in context, but re-calc for clarity)
        mu1_c = X1_c.mean(axis=0)
        mu2_c = X2_c.mean(axis=0)
        mu_diff_c = mu1_c - mu2_c
        S_B = (N1 * N2 / N_total) * np.outer(mu_diff_c, mu_diff_c)
        
        subset = subset_range(n=d, subset_size=n_components, smallest=True)
        res = _solve_eigh_robustly(a_matrix=S_B, b_matrix=S_W, subset=subset)
        P_basis = res.P_or_U
        eigvals = res.eigvals
        return P_basis, eigvals

class AntiLDADenoiser(TwoSubsetScatterDenoiser):
    """
    Anti-LDA denoiser that removes the directions that best discriminate between two groups.
    It minimizes the ratio of between-class scatter to within-class scatter.
    """
    def get_within_class_scatter(self, X1_c: NDArray, X2_c: NDArray, X_c: NDArray) -> NDArray:
        # S_W = S_subset1 + S_subset2
        # Use scatter(centered=True) since X1_c and X2_c are already centered
        return scatter(X1_c, centered=True) + scatter(X2_c, centered=True)

class DomainInvariantDenoiser(TwoSubsetScatterDenoiser):
    """
    Removes directions that discriminate between two groups by minimizing the ratio of 
    between-class scatter to total scatter.
    """
    def get_within_class_scatter(self, X1_c: NDArray, X2_c: NDArray, X_c: NDArray) -> NDArray:
        # S_W = S_T
        return scatter(X_c, centered=True)

# class EmbeddedGraphDenoiser:
#     def __init__(self, wrapped_denoiser):
#         self.wrapped_denoiser = wrapped_denoiser

#     def projector(self, context: DenoiserContext, n_components: int):
#         X = context.X()
#         A_sym = context.A_sym()
#         target = A_sym @ X
#         W, residuals, rank, s = lstsq(X, target, rcond=None)
        
#         X_smoothed = X @ W
#         A_w = X_smoothed @ adj(X_smoothed)
#         return self.wrapped_denoiser.projector(X, A_w, n_components)

#     def __repr__(self):
#         return f"EmbeddedGraph({self.wrapped_denoiser.__class__.__name__})"

class AugmentedLPPDenoiser:
    """
    Improves LPP by augmenting its objective.
    It combines the standard LPP goal (preserving neighbor distances) with the goal of
    minimizing the distance between a point and its A-transformed image.
    This is achieved by summing their respective scatter matrices in the numerator.

    When ``normalize_traces=True``, the two summed scatter matrices (S_W_lpp and S_W_new)
    are each pre-rescaled to tr=d before the gamma-weighted sum. Without this,
    tr(S_W_lpp) is ~6 orders of magnitude larger than tr(S_W_new) on our embeddings,
    so the augmentation term contributes nothing regardless of gamma and the method
    collapses to plain LPP.
    """
    def __init__(self, gamma: float = 1.0, normalize_traces: bool = False):
        self.gamma = gamma
        self.normalize_traces = normalize_traces

    def projector(self, context: DenoiserContext, n_components: int):
        X_c = context.X_centered() # Use centered data
        B = context.A_sym()
        N, d = X_c.shape
        D = sp.diags(np.asarray(B.sum(axis=1)).squeeze())
        L = D - B
        gamma = self.gamma
        if N < d:
            LOGGER.info(f"AugmentedLPP: N < d, using dual formulation. N={N}, d={d}")
            K = X_c @ adj(X_c) # Use centered data
            M_lpp = L
            M_new = adj(B - sp.eye(N)) @ (B - sp.eye(N))
            a_lpp = K @ M_lpp @ K
            a_new = K @ M_new @ K
            if self.normalize_traces:
                tr_lpp = float(np.trace(a_lpp))
                tr_new = float(np.trace(a_new))
                if tr_lpp > 0:
                    a_lpp = a_lpp * (N / tr_lpp)
                if tr_new > 0:
                    a_new = a_new * (N / tr_new)
            a_dual = a_lpp + gamma * a_new
            b_dual = K @ D @ K
            subset = subset_range(n=N, subset_size=n_components, smallest=True)
            res = _solve_eigh_robustly(a_dual, b_dual, subset)
            P = _recover_projector_from_dual(X_c, res.P_or_U) # Use centered data
            return P, res.eigvals
        else:
            # Primal formulation
            S_W_lpp = adj(X_c) @ L @ X_c # Use centered data
            # S_W_new must also be centered
            B_X_c = B @ X_c
            B_X_c_centered = B_X_c - B_X_c.mean(axis=0, keepdims=True)
            X_c_centered = X_c # already centered
            S_W_new = adj(B_X_c_centered - X_c_centered) @ (B_X_c_centered - X_c_centered) / N

            if self.normalize_traces:
                tr_lpp = float(np.trace(S_W_lpp))
                tr_new = float(np.trace(S_W_new))
                if tr_lpp > 0:
                    S_W_lpp = S_W_lpp * (d / tr_lpp)
                if tr_new > 0:
                    S_W_new = S_W_new * (d / tr_new)

            S_A_combined = S_W_lpp + gamma * S_W_new
            S_B_lpp = adj(X_c) @ D @ X_c # Use centered data
            subset = subset_range(n=d, subset_size=n_components, smallest=True)
            res = _solve_eigh_robustly(S_A_combined, S_B_lpp, subset)
            P = res.P_or_U
            return P, res.eigvals

class OrthogonalManifoldDenoiser:
    """
    A monolithic approach that balances preserving global, non-local structure against
    minimizing the error introduced by graph-smoothing (the AX-X distance).

    The "pure" objective is to find a subspace that maximizes "signal" (S_T)
    while minimizing "noise" (S_W).
    Objective: max (w^T S_T w) / (w^T S_W w)
    
    We solve the generalized eigenvalue problem: S_T w = λ S_W w
    ...and take the eigenvectors with the LARGEST eigenvalues.
    """
    def projector(self, context: DenoiserContext, n_components: int):
        X_c = context.X_centered() # Use centered data
        B = context.A_sym()
        N, d = X_c.shape
        
        # Numerator: S_T represents total signal (now centered)
        S_T = context.S_T()
        
        # Denominator: S_W represents the smoothing error (noise)
        # We must use centered data here too.
        B_X_c = B @ X_c
        # We need to center (B @ X_c) and (X_c) relative to their own means
        # before taking the scatter of their difference.
        B_X_c_centered = B_X_c - B_X_c.mean(axis=0, keepdims=True)
        X_c_centered = X_c # already centered
        
        S_W = scatter(B_X_c_centered - X_c_centered, centered=True) # S_W = scatter( (AX_c - mean(AX_c)) - (X_c) )
        
        subset = subset_range(n=d, subset_size=n_components, smallest=False)
        res = _solve_eigh_robustly(a_matrix=S_T, b_matrix=S_W, subset=subset)
        P_basis = res.P_or_U
        return P_basis, res.eigvals

class _PLSDenoiser(ABC):
    """
    Superclass for fast PLS/CCA-style projection using direct SVD of a cross-matrix.
    Subclasses should override get_cross_matrix.
    """
    def projector(self, context: DenoiserContext, n_components: int):
        X_c = context.X_centered() # Use centered data
        B = context.A_sym()
        N, d = X_c.shape
        
        Y = B @ X_c # Target is A @ X_c
        
        X_centered = X_c # Already centered
        Y_centered = Y - Y.mean(axis=0, keepdims=True) # Center the target
        
        # Get the cross-matrix (covariance or correlation)
        C = self.get_cross_matrix(X_centered, Y_centered, N)
        U, s, Vt = svd(C)
        P = U[:, :n_components]
        assert P.shape == (d, n_components), f"Expected shape ({d}, {n_components}), got {P.shape}"
        projector = P
        return projector, s[:n_components]

    @abstractmethod
    def get_cross_matrix(self, X_centered, Y_centered, N):
        ...

class PLSModeADenoiser(_PLSDenoiser):
    """
    Fast PLS approximation using direct SVD of cross-covariance matrix.
    """
    def get_cross_matrix(self, X_centered, Y_centered, N):
        # Default: cross-covariance
        return X_centered.T @ Y_centered / N

class PLSModeBDenoiser(_PLSDenoiser):
    """
    Fast CCA approximation using direct SVD of cross-correlation matrix.
    """
    def get_cross_matrix(self, X_centered, Y_centered, N):
        # Normalize by standard deviations for correlation
        X_std = np.std(X_centered, axis=0, ddof=1)
        Y_std = np.std(Y_centered, axis=0, ddof=1)
        X_std = np.where(X_std == 0, 1, X_std)
        Y_std = np.where(Y_std == 0, 1, Y_std)
        X_normalized = X_centered / X_std
        Y_normalized = Y_centered / Y_std
        return X_normalized.T @ Y_normalized / (N - 1)

class _CCLIPBaseDenoiser(ABC):
    """
    Base class for CCLIP-style denoisers that minimize a penalty matrix relative to 
    a constraint matrix.
    
    Objective: min (w^T A w) / (w^T B w)
    where:
        A = (1-alpha)*S_B + alpha*S_L  (penalty: class separation + graph inconsistency)
        B = constraint matrix (subclass-specific: S_T or S_W)
    
    Subclasses must implement get_constraint_matrix() to specify the denominator.
    """
    def __init__(self, alpha: float = 0.5, class_weights: Optional[tuple[float, ...]] = None):
        """
        Args:
            alpha (float): Balance between class-conflation (0) and locality (1).
                           alpha = 0 -> Purely class-conflating projection.
                           alpha = 1 -> Purely locality-inducing projection.
            class_weights (Optional[tuple[float, ...]]): Weights for each class when 
                           computing within-class scatter. None means equal weights.
                           Weights are normalized to sum to 1.
        """
        assert 0 <= alpha <= 1, "alpha must be in the range [0, 1]"
        self.alpha = alpha
        self.class_weights = class_weights

    @abstractmethod
    def get_constraint_matrix(self, context: DenoiserContext) -> NDArray:
        """Return the constraint matrix B for the generalized eigenvalue problem."""
        ...
    
    def _get_normalized_class_weights(self, context: DenoiserContext) -> NDArray:
        """Get normalized class weights, defaulting to equal weights if not specified."""
        num_subsets = context.num_subsets()
        if self.class_weights is None:
            return np.ones(num_subsets) / num_subsets
        if len(self.class_weights) != num_subsets:
            raise ValueError(f"class_weights length {len(self.class_weights)} != num_subsets {num_subsets}")
        weights = np.array(self.class_weights, dtype=np.float64)
        return weights / weights.sum()

    def projector(self, context: DenoiserContext, n_components: int) -> tuple[NDArray, Optional[NDArray]]:
        """Computes the projection matrix."""
        d = context.X().shape[1]
        
        # Get precomputed scatter matrices from context
        S_B = context.S_B_multi(list(range(context.num_subsets()))) 
        S_L = context.S_L()

        # Construct the numerator matrix (penalty) using the alpha mixing parameter
        A_matrix = (1 - self.alpha) * S_B + self.alpha * S_L
        
        # Get the constraint matrix from subclass
        B_matrix = self.get_constraint_matrix(context)
        
        # Find smallest eigenvalues and corresponding eigenvectors
        subset = subset_range(n=d, subset_size=n_components, smallest=True)
        res = _solve_eigh_robustly(a_matrix=A_matrix, b_matrix=B_matrix, subset=subset)

        return res.P_or_U, res.eigvals


class CCLIPDenoiser(_CCLIPBaseDenoiser):
    """
    Class-Conflating Locality-Inducing Projections (CCLIP) Denoiser.

    This denoiser seeks a projection that simultaneously:
    1. Minimizes the separation between two classes (e.g., books and articles).
    2. Minimizes fluctuation across a citation graph (induces locality).
    3. Preserves the maximum amount of total variance of the original data.

    The objective is to find a projection P that minimizes the Rayleigh quotient:
        R(P) = Trace(Pᵀ * ((1-α)*S_B + α*S_L) * P) / Trace(Pᵀ * S_T * P)

    The optimal projection directions are the eigenvectors corresponding to the
    smallest eigenvalues.

    Constraint matrix: S_T (Total Scatter)

    When ``normalize_traces=True``, each of S_B and S_L is pre-rescaled so its
    trace equals d (the embedding dimensionality) before the convex combination
    is formed. Without this, tr(S_L) can be 10^7× tr(S_B) at scale, so α=0.5
    does not actually balance the two terms; with it, α has its natural
    interpretation as the mixing weight. Formula matches the exploratory
    `_denoise_cclip` implementation in artifacts_scripts/hdbscan_gap_probe.py.
    """
    def __init__(self, alpha: float = 0.5, normalize_traces: bool = False):
        super().__init__(alpha=alpha, class_weights=None)
        self.normalize_traces = normalize_traces

    def get_constraint_matrix(self, context: DenoiserContext) -> NDArray:
        return context.S_T()

    def projector(self, context: DenoiserContext, n_components: int):
        if not self.normalize_traces:
            return super().projector(context, n_components)
        # Trace-normalized variant: rescale each scatter matrix to trace=d
        # before the convex combination. Guards against tr=0 (degenerate
        # topics with identical class means or empty citation graphs).
        d = context.X().shape[1]
        S_B = context.S_B_multi(list(range(context.num_subsets())))
        S_L = context.S_L()
        tr_B = float(np.trace(S_B))
        tr_L = float(np.trace(S_L))
        S_B_norm = S_B * (d / tr_B) if tr_B > 0 else S_B
        S_L_norm = S_L * (d / tr_L) if tr_L > 0 else S_L
        A_matrix = (1 - self.alpha) * S_B_norm + self.alpha * S_L_norm
        B_matrix = self.get_constraint_matrix(context)
        subset = subset_range(n=d, subset_size=n_components, smallest=True)
        res = _solve_eigh_robustly(a_matrix=A_matrix, b_matrix=B_matrix, subset=subset)
        return res.P_or_U, res.eigvals


class WithinClassCCLIPDenoiser(_CCLIPBaseDenoiser):
    """
    A variant of CCLIP that regularizes by Within-Class Scatter (S_W) instead of 
    Total Scatter (S_T).
    
    This avoids the objective conflict where S_T implicitly rewards the very 
    class separation (S_B) that the numerator tries to minimize.
    
    Objective:
        min (w^T [ (1-alpha)S_B + alpha*S_L ] w) / (w^T S_W w)
        
    This is effectively "Laplacian-Regularized Anti-LDA".
    
    Constraint matrix: S_W (Within-Class Scatter)
    
    Note on class weighting: Since scatter() normalizes by sample size, each class's
    scatter matrix represents its variance structure independent of sample count.
    Equal weights (default) treat each class's structure as equally important.
    To weight by sample count, pass class_weights proportional to (n_1, n_2, ...).
    """
    def __init__(self, alpha: float = 0.5, class_weights: Optional[tuple[float, ...]] = None,
                 normalize_traces: bool = False):
        """
        Args:
            alpha (float): 0.0 = Pure Anti-LDA (Class Conflation).
                           1.0 = Pure LPP (Locality Preserving) relative to within-class variance.
            class_weights (Optional[tuple[float, ...]]): Weights for each class scatter.
                           None = equal weights (recommended for imbalanced classes).
                           e.g., (1.0, 1.0) for equal emphasis on books and articles.
            normalize_traces (bool): If True, trace-normalize S_B and S_L to tr=d before the
                           convex combination, so alpha actually balances the two terms.
                           Without this, tr(S_L) >> tr(S_B) and the convex combination
                           collapses to pure locality regardless of alpha. See
                           ``CCLIPDenoiser`` for details.
        """
        super().__init__(alpha=alpha, class_weights=class_weights)
        self.normalize_traces = normalize_traces

    def get_constraint_matrix(self, context: DenoiserContext) -> NDArray:
        """Compute weighted within-class scatter S_W = sum(w_i * S_subset_i)."""
        weights = self._get_normalized_class_weights(context)
        d = context.X().shape[1]
        S_W = np.zeros((d, d), dtype=np.float64)
        for i in range(context.num_subsets()):
            S_W += weights[i] * context.S_subset(i)
        return S_W

    def projector(self, context: DenoiserContext, n_components: int):
        if not self.normalize_traces:
            return super().projector(context, n_components)
        d = context.X().shape[1]
        S_B = context.S_B_multi(list(range(context.num_subsets())))
        S_L = context.S_L()
        tr_B = float(np.trace(S_B))
        tr_L = float(np.trace(S_L))
        S_B_norm = S_B * (d / tr_B) if tr_B > 0 else S_B
        S_L_norm = S_L * (d / tr_L) if tr_L > 0 else S_L
        A_matrix = (1 - self.alpha) * S_B_norm + self.alpha * S_L_norm
        B_matrix = self.get_constraint_matrix(context)
        subset = subset_range(n=d, subset_size=n_components, smallest=True)
        res = _solve_eigh_robustly(a_matrix=A_matrix, b_matrix=B_matrix, subset=subset)
        return res.P_or_U, res.eigvals

class AntiCSPDenoiser:
    """
    Finds the least discriminative directions between two classes by finding
    eigenvectors of the generalized system S1*v = λ*(S1+S2)*v where λ is
    closest to 0.5.

    When ``normalize_traces=True``, each of S_1 and S_2 is pre-rescaled so its
    trace equals d (the embedding dimensionality) before the GEP is formed.
    This matches the standard CSP normalization convention and prevents
    class-size or per-class-variance imbalance from tilting the ratio.
    """
    def __init__(self, normalize_traces: bool = False):
        self.normalize_traces = normalize_traces

    def projector(self, context: DenoiserContext, n_components: int) -> tuple[NDArray, Optional[NDArray]]:
        S1 = context.S_subset(0) # Use context to get CENTERED subset scatter
        S2 = context.S_subset(1) # Use context to get CENTERED subset scatter

        if self.normalize_traces:
            d = context.X().shape[1]
            tr_1 = float(np.trace(S1))
            tr_2 = float(np.trace(S2))
            if tr_1 > 0:
                S1 = S1 * (d / tr_1)
            if tr_2 > 0:
                S2 = S2 * (d / tr_2)

        # 1. Use the robust solver to get the full set of orthogonal eigenvectors.
        result = _solve_eigh_robustly(a_matrix=S1, b_matrix=(S1 + S2), subset=None)
        P_full = result.P_or_U
        eigvals_full = result.eigvals

        # The eigenvalues λ are between 0 and 1. We want the ones closest to 0.5,
        # as this indicates that v.T*S1*v ≈ v.T*S2*v.
        scores = np.abs(eigvals_full - 0.5)
        
        # 2. Sort by score in ascending order to find components where λ is closest to 0.5
        order = np.argsort(scores)
        best_indices = order[:n_components]
        
        # 3. Select the best columns from the already-orthogonal projector.
        P_final = P_full[:, best_indices]
        
        # Return the corresponding variance ratios as the quality score for context.
        eigvals_final = eigvals_full[best_indices]
        eigvals_ratio = eigvals_final / (1 - eigvals_final + 1e-9)

        return P_final, eigvals_ratio

class JADDenoiser:
    """
    Joint Approximate Diagonalization (JAD) Denoiser.
    
    Uses joint approximate diagonalization of scatter matrices:
    - S_T: Total scatter matrix
    - S_L: Laplacian scatter matrix  
    - S_subset_i: Scatter matrix of each subset
    
    Based on the RNOJD algorithm from:
    He, H., & Kressner, D. (2024). A randomized algorithm for simultaneously 
    diagonalizing symmetric matrices by congruence. arXiv preprint arXiv:2402.16557.
    
    Uses Manopt+RNOJD for superior convergence performance.
    """
    
    def __init__(self, trials: int = 30, use_positive_definite: bool = False,
                 wt: float = 1.0, wl: float = 1.0, w_conflate: float = 1.0,
                 normalize_traces: bool = False):
        """
        Initialize the JAD denoiser.

        Args:
            trials: Number of random trials for the RNOJD algorithm
            use_positive_definite: If True, use the positive definite formulation
            wt: Weight for total scatter matrix S_T
            wl: Weight for Laplacian scatter matrix S_L
            w_conflate: Weight for the class-conflation objective (1 - normalized_entropy of subset variances)
            normalize_traces: If True, pre-rescale each input scatter matrix (S_T, S_L, S_subset_i)
                              so its trace equals d before applying weights. Without this, the raw
                              tr(S_L) is ~6 orders of magnitude larger than tr(S_T) or tr(S_subset_i)
                              on our embeddings, so S_L dominates the joint-diagonalization objective
                              regardless of nominal weights. The same rescaling is applied to the
                              per-component quadratic forms used in the scoring step, so that the
                              wt/wl/w_conflate exponents combine dimensionless quantities.
        """
        self.trials = trials
        self.use_positive_definite = use_positive_definite
        self.wt = wt
        self.wl = wl
        self.w_conflate = w_conflate
        self.normalize_traces = normalize_traces
    
    def projector(self, context: DenoiserContext, n_components: int) -> tuple[NDArray, Optional[NDArray]]:
        """
        Compute the JAD projection using RNOJD.
        
        Args:
            context: The denoiser context providing access to data
            n_components: Number of components to keep
            
        Returns:
            Tuple of (projection_matrix, quality_scores)
        """
        X = context.X()
        N, d = X.shape
        num_subsets = context.num_subsets()
        
        # Get the scatter matrices (all are now centered)
        S_T = context.S_T()
        S_L = context.S_L()

        # Get all subset scatter matrices (all are now centered)
        subset_scatter_matrices = []
        for i in range(num_subsets):
            subset_scatter_matrices.append(context.S_subset(i))

        if self.normalize_traces:
            def _tn(S):
                tr = float(np.trace(S))
                return S * (d / tr) if tr > 0 else S
            S_T = _tn(S_T)
            S_L = _tn(S_L)
            subset_scatter_matrices = [_tn(S) for S in subset_scatter_matrices]

        # Apply weights and stack the matrices for joint diagonalization.
        matrices_list = [
            self.wt * S_T,
            self.wl * S_L
        ]

        # Add weighted subset matrices
        for i, S_subset in enumerate(subset_scatter_matrices):
            matrices_list.append(self.w_conflate * S_subset)
        
        matrices = np.stack(matrices_list, axis=0)
        
        LOGGER.info(f"JAD: Joint diagonalization of {matrices.shape[0]} weighted scatter matrices of size {d}x{d}")
        LOGGER.info(f"JAD: {num_subsets} subsets, weights - wt={self.wt}, wl={self.wl}, w_conflate={self.w_conflate}")
        LOGGER.info(f"JAD: trials={self.trials}, use_positive_definite={self.use_positive_definite}")
        
        # Imported here, not at module level: rnojd.py is unlicensed third-party code
        # that is not distributed, so only JAD may need it (bookgap/rsdc/README.md).
        try:
            from bookgap.rsdc.rnojd import manopt_rnojd
        except ModuleNotFoundError as e:
            if e.name != "bookgap.rsdc.rnojd":
                raise
            raise ModuleNotFoundError(
                "The JAD denoiser needs bookgap/rsdc/rnojd.py, which is not distributed "
                "because its upstream carries no licence. See bookgap/rsdc/README.md "
                "for how to obtain it."
            ) from e

        # Apply the chosen SDC solver
        try:
            # Use Manopt+RNOJD for superior optimization performance
            result = manopt_rnojd(matrices, trials=self.trials, max_iter=50, trivial_init=False, pd=self.use_positive_definite)
            X_transform = result[0] if isinstance(result, tuple) else result
            LOGGER.info(f"JAD: Solver returned transformation matrix with shape {X_transform.shape}")
            
            if X_transform is None:
                raise ValueError("SDC algorithm returned None")
                
        except Exception as e:
            LOGGER.error(f"JAD/SDC solver failed: {e}")
            raise RuntimeError(f"JAD denoiser failed with SDC solver error: {e}") from e
        
        # Score the full basis before truncating
        P_full_basis = X_transform
        all_scores = []
        
        # Iterate over ALL basis vectors returned by the solver
        for i in range(P_full_basis.shape[1]):
            p_i = P_full_basis[:, i:i+1]
            
            # Get the unweighted quadratic forms (variances along the basis vector)
            t = (p_i.T @ S_T @ p_i).item()
            l = (p_i.T @ S_L @ p_i).item()
            
            # Get subset variances
            subset_variances = []
            for S_subset in subset_scatter_matrices:
                s = (p_i.T @ S_subset @ p_i).item()
                subset_variances.append(s)
            
            # # Add a small epsilon for numerical stability to prevent log(0)
            epsilon = 1e-9
            if t <= 0 or l <= 0 or any(s <= 0 for s in subset_variances):
                LOGGER.warning(f"JAD: Non-positive variance encountered for basis vector {i}: t={t}, l={l}, subset_variances={subset_variances}. Adjusting with epsilon.")
                t = max(t, epsilon)
                l = max(l, epsilon)
                subset_variances = [max(s, epsilon) for s in subset_variances]
            
            # Compute conflation measure for multiple subsets
            if len(subset_variances) > 1:
                # Normalized entropy conflation penalty (most statistically sound)
                # Based on analysis: ranges from 0.0000 (perfect) to ~0.43 (extreme outlier)
                subset_vars_normalized = np.array(subset_variances) / np.sum(subset_variances)
                # Compute entropy, skipping zero probabilities naturally
                entropy = -sum(p * np.log(p) for p in subset_vars_normalized if p > 0)
                max_entropy = np.log(len(subset_variances))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
                # Conflation penalty: 0=perfect conflation, 1=worst conflation
                conflation_measure = 1.0 - normalized_entropy
            else:
                # Single subset: conflation is irrelevant, use neutral value
                conflation_measure = 1.0

            # The final score is a weighted multiplicative objective.
            score = (t ** self.wt) / ((l ** self.wl) * (conflation_measure ** self.w_conflate))
                     
            all_scores.append(score)
        
        all_scores_array = np.array(all_scores)
        
        # Sort by the final, weighted score to find the best components
        sort_indices = np.argsort(all_scores_array)[::-1]
        
        # Truncate to the desired number of components
        n_components = min(n_components, len(sort_indices))
        best_indices = sort_indices[:n_components]
        
        # The final projector is built from the highest-scoring columns
        P_final = P_full_basis[:, best_indices]
        scores_final = all_scores_array[best_indices]

        # P_final_orthonormalized = orthonormalize(P_final)
        # return P_final_orthonormalized, scores_final
        
        LOGGER.info(f"JAD: Final projection shape {P_final.shape}, quality scores: {scores_final}")
        return P_final, scores_final
        

class CitationErrorDenoiser:
    """
    Finds the subspace that minimizes the "citation error" (distance between
    a work and the centroid of its citations).
    
    This is the "null-space" approach and may find low-variance noise dimensions.
    Objective: min_w (w^T S_C w)
    Solution: Eigenvectors of S_C with the SMALLEST eigenvalues.
    """
    def projector(self, context: DenoiserContext, n_components: int):
        S_C = context.S_C() # This is now centered
        d = S_C.shape[0]
        
        # We want the k SMALLEST eigenvectors
        subset = subset_range(n=d, subset_size=n_components, smallest=True)
        res = _solve_eigh_robustly(a_matrix=S_C, b_matrix=None, subset=subset)
        
        P = res.P_or_U
        eigvals = res.eigvals
        # No need to reverse, as we want the smallest
        return P, eigvals

class CitationErrorLDADenoiser:
    """
    Finds the subspace that balances minimizing "citation error" (S_C) against
    maximizing "citation signal" (S_T - S_C).

    Solves the generalized eigenvalue problem: S_C w = λ (S_T - S_C) w
    We seek the eigenvectors with the SMALLEST eigenvalues.
    Objective: min (w^T S_C w) / (w^T (S_T - S_C) w)
    """
    def projector(self, context: DenoiserContext, n_components: int):
        X = context.X()
        N, d = X.shape

        # Sparse path: when X is sparse, use iterative eigensolver to avoid
        # materializing dense V×V scatter matrices.
        if sp.issparse(X):
            LOGGER.info(f"CitationErrorLDA: sparse X ({N}x{d}), using iterative eigensolver")
            res = _solve_gep_sparse(X, context.A(), n_components)
            return res.P_or_U, res.eigvals

        if N < d:
            LOGGER.info(f"CitationErrorLDA: N < d, using dual (SVD) formulation. N={N}, d={d}")
            cm = context.centered()
            X_c_dense = cm.X_centered_dense()  # N x d, dense

            # Truncated SVD of X_c: keep directions where s_i/s_max > threshold
            # This discards noise subspace so the reduced scatter matrices are well-conditioned.
            U_svd, s_svd, Vt_svd = svd(X_c_dense, full_matrices=False)
            sig_threshold = 1e-2 * s_svd[0]
            r = max(int((s_svd > sig_threshold).sum()), n_components + 1)
            V_r = Vt_svd[:r, :].T  # d x r: significant right singular vectors

            # Compute citation error E = X - A@X, then center
            A = context.A()
            E = X - A @ X  # N x d (may be dense even if X is sparse)
            E_mean = np.asarray(E.mean(axis=0)) if sp.issparse(E) else E.mean(axis=0, keepdims=True)
            E_dense = np.asarray(E.todense()) if sp.issparse(E) else np.asarray(E)
            E_c = E_dense - E_mean  # N x d, centered (dense)

            # Project scatter matrices onto the r-dimensional signal subspace
            F = E_c @ V_r  # N x r: E_c projected onto row space of X_c
            S_C_red = F.T @ F  # r x r
            S_T_red = np.diag(s_svd[:r] ** 2)  # r x r (diagonal by construction)
            S_Signal_red = S_T_red - S_C_red  # r x r

            LOGGER.info(f"CitationErrorLDA dual: r={r}, S_Signal_red min eigval={np.linalg.eigvalsh(S_Signal_red).min():.4e}")
            subset = subset_range(n=r, subset_size=n_components, smallest=True)
            res = _solve_eigh_robustly(a_matrix=S_C_red, b_matrix=S_Signal_red, subset=subset)
            # Recover d-dimensional projector: w = V_r @ v for each eigenvector v
            P_raw = V_r @ res.P_or_U  # d x n_components
            P_norms = np.linalg.norm(P_raw, axis=0)
            P_norms[P_norms == 0] = 1
            P = P_raw / P_norms
            return P, res.eigvals
        else:
            S_C = context.S_C()
            S_T = context.S_T()
            S_Signal = S_T - S_C
            subset = subset_range(n=d, subset_size=n_components, smallest=True)
            res = _solve_eigh_robustly(a_matrix=S_C, b_matrix=S_Signal, subset=subset)
            return res.P_or_U, res.eigvals