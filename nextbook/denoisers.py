from dataclasses import dataclass
import enum
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, svd, qr
import scipy.sparse as sp
from sklearn.decomposition import PCA
from numpy.linalg import pinv, LinAlgError, lstsq
from nextbook.util import TIMER, LOGGER
from abc import ABC, abstractmethod

def subset_range(n: int, subset_size: int, smallest: bool) -> tuple[int, int]:
    subset_size = min(subset_size, n)
    if smallest:
        return 0, subset_size - 1
    else:
        return n - subset_size, n - 1

def adj(A):
    return A.conj().T

def scatter(M: NDArray) -> NDArray:
    M = M - M.mean(axis=0, keepdims=True)
    return M.T @ M

def orthonormalize(P: NDArray) -> NDArray:
    Q: NDArray
    R: NDArray
    Q, R = qr(P, mode='economic') # type: ignore
    
    assert Q.shape == P.shape, f"{Q.shape=} != expected {P.shape=}"
    
    # Check for numerical issues in R diagonal
    r_diag = np.abs(np.diag(R))
    tol = max(1e-12, r_diag[0] * 1e-12) if len(r_diag) > 0 else 1e-12
    n_flat_dimensions = np.sum(r_diag < tol)
    
    if n_flat_dimensions:
        LOGGER.warning(f"Found {n_flat_dimensions} additional flat dimensions in projector, which may indicate numerical instability.")
    
    return Q

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
def _solve_eigh_robustly(a_matrix: NDArray, b_matrix: NDArray | None, subset: tuple[int, int]) -> SolveEighRobustlyResult:
    """Solves eigh, retrying with regularization upon failure."""
    try:
        eigvals, P_or_U = eigh(a=a_matrix, b=b_matrix, subset_by_index=subset)
    except LinAlgError:
        LOGGER.warning("eigh failed, likely due to a singular b_matrix. Retrying with regularization.")
        if b_matrix is not None:
            # b_matrix += np.eye(b_matrix.shape[0]) * 1e-6
            ridge_val = np.mean(np.abs(np.diag(b_matrix))) * 1e-4 + 1e-8
            LOGGER.warning(f"Retrying with adaptive regularization: {ridge_val:.2e}")
            b_matrix += np.eye(b_matrix.shape[0]) * ridge_val
            eigvals, P_or_U = eigh(a=a_matrix, b=b_matrix, subset_by_index=subset)
        else:
            raise
    
    # If this was a generalized eigenvalue problem, orthogonalize the result
    if b_matrix is not None:
        expected_rank = subset[1] - subset[0] + 1
        LOGGER.info(f"Orthogonalizing generalized eigenvectors from shape {P_or_U.shape}")
        P_or_U = orthonormalize(P_or_U)
        LOGGER.info(f"Orthogonalized projector shape: {P_or_U.shape}")
    
    LOGGER.info(f"{eigvals.shape=} {P_or_U.shape=}")
    return SolveEighRobustlyResult(eigvals=eigvals, P_or_U=P_or_U)

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
    def projector(self, X: NDArray, n_components: int) -> tuple[NDArray, Optional[NDArray]]:
        min_cs = min(X.shape)
        if n_components > min_cs:
            n_components = min_cs
        pca = PCA(n_components=n_components)
        pca.fit(X)
        P = pca.components_.T
        return P, pca.singular_values_

class GPCADenoiser:
    def projector(self, X: NDArray, A: sp.spmatrix, n_components: int):
        B = (A + adj(A)) / 2
        M = adj(X) @ B @ X
        d = X.shape[1]
        subset = subset_range(n=d, subset_size=n_components, smallest=False)
        eigvals, P = eigh(M, subset_by_index=subset)
        return P, eigvals

class RelAmpDenoiser:
    def projector(self, X: NDArray, A: sp.spmatrix, n_components: int):
        N, d = X.shape
        B = (A + adj(A)) / 2
        
        if N < d:
            LOGGER.info(f"RelAmp: N < d, using dual formulation. N={N}, d={d}")
            K = X @ adj(X)
            subset = subset_range(n=N, subset_size=n_components, smallest=False)
            a_dual = adj(K) @ B @ K
            b_dual = adj(K) @ K
            res = _solve_eigh_robustly(a_dual, b_dual, subset)
            P = _recover_projector_from_dual(X, res.P_or_U)
            return P, res.eigvals
        else:
            subset = subset_range(n=d, subset_size=n_components, smallest=False)
            a_matrix = adj(X) @ B @ X
            b_matrix = adj(X) @ X
            res = _solve_eigh_robustly(a_matrix, b_matrix, subset)
            P = res.P_or_U
            return P, res.eigvals

class NeighborDistMinDenoiser:
    def projector(self, X: NDArray, A: sp.spmatrix, n_components: int):
        N, d = X.shape
        B = (A + adj(A)) / 2
        M = sp.eye(N) - B
        a_mat = adj(X) @ adj(M) @ M @ X
        
        if N < d:
            LOGGER.info(f"NeighborDistMin: N < d, using dual formulation. N={N}, d={d}")
            K = X @ adj(X)
            subset = subset_range(n=N, subset_size=n_components, smallest=True)
            a_dual = adj(K) @ adj(M) @ M @ K
            b_dual = adj(K) @ K
            res = _solve_eigh_robustly(a_dual, b_dual, subset)
            P = _recover_projector_from_dual(X, res.P_or_U)
            return P, res.eigvals
        else:
            subset = subset_range(n=d, subset_size=n_components, smallest=True)
            b_mat = adj(X) @ X
            res = _solve_eigh_robustly(a_mat, b_mat, subset)
            P = res.P_or_U
            return P, res.eigvals

class LPPDenoiser:
    def projector(self, X: NDArray, A: sp.spmatrix, n_components: int):
        assert A is not None
        LOGGER.info(f"Computing LPP projector for {X.shape=} and {A.shape=}.")
        N, d = X.shape
        B = (A + adj(A)) / 2
        LOGGER.info(f"{type(A)=} {A.shape=} {type(B)=} {B.shape=}")

        # LPP requires non-negative weights for a valid Degree matrix.
        negatives = B < 0
        num_negatives = np.sum(negatives)
        if num_negatives > 0:
            LOGGER.warning(f"LPPDenoiser received an affinity matrix with {num_negatives} negative weights ({num_negatives / B.size:.2%} of total). Setting negative weights to zero.")
            B[B < 0] = 0

        LOGGER.info(f"After setting negatives to zero, {type(B)=} {B.shape=}")
        B_sum = np.asarray(B.sum(axis=1)).squeeze(1)
        # Print statistics about the sum of rows
        LOGGER.info(f"{np.mean(B_sum)=:.2f} (mean), {np.std(B_sum)=:.2f} (std), {np.min(B_sum)=:.2f} (min), {np.max(B_sum)=:.2f} (max) {np.percentile(B_sum, 10)=:.2f}, {np.percentile(B_sum, 90)=:.2f}")
        D = sp.diags(B_sum)
        L = D - B

        if N < d:
            LOGGER.info(f"LPP: N < d, using dual formulation. N={N}, d={d}")
            K = X @ adj(X)
            subset = subset_range(n=N, subset_size=n_components, smallest=True)
            a_dual = adj(K) @ L @ K
            LOGGER.info(f"{type(a_dual)=} {a_dual.shape=}")
            b_dual = adj(K) @ D @ K
            LOGGER.info(f"{type(b_dual)=} {b_dual.shape=}")
            res = _solve_eigh_robustly(a_dual, b_dual, subset)
            P = _recover_projector_from_dual(X, res.P_or_U)
            return P, res.eigvals
        else:
            subset = subset_range(n=d, subset_size=n_components, smallest=True)
            a_matrix = adj(X) @ L @ X
            b_matrix = adj(X) @ D @ X
            res = _solve_eigh_robustly(a_matrix, b_matrix, subset)
            P = res.P_or_U
            return P, res.eigvals

class NormalizedLPPDenoiser:
    def projector(self, X: NDArray, A: sp.spmatrix, n_components: int):
        B = (A + adj(A)) / 2
        N, d = X.shape
        degrees = np.array(B.sum(axis=1)).ravel()
        inv_sqrt_deg = np.zeros_like(degrees, dtype=float)
        safe_indices = degrees > 0
        inv_sqrt_deg[safe_indices] = np.power(degrees[safe_indices], -0.5)
        D_inv_sqrt = sp.diags(inv_sqrt_deg)
        L_sym = sp.eye(N) - (D_inv_sqrt @ B @ D_inv_sqrt)
        a_matrix = adj(X) @ L_sym @ X
        subset = subset_range(n=d, subset_size=n_components, smallest=True)
        assert np.allclose(adj(a_matrix), a_matrix, atol=1e-6), "a_matrix is not Hermitian!"
        eigvals, P = eigh(a_matrix, subset_by_index=subset)
        return P, eigvals

class WDenoiser:
    """
    Learns W where XW ≈ AX, then uses rank-reduced W for projection.
    """
    def projector(self, X: NDArray, A: sp.spmatrix, n_components: int):
        # Solve for W in XW ≈ AX
        target = A @ X
        W, _, _, _ = lstsq(X, target, rcond=None)
        U, s, Vt = svd(W)
        P = U[:, :n_components]
        return P, s[:n_components]

class LDADenoiser:
    """
    Maximizes the ratio of total variance to the variance of difference vectors AX-X.
    This is solved as the generalized eigenvalue problem S_T g = λ S_W g.
    """
    def projector(self, X: NDArray, A: sp.spmatrix, n_components: int):
        def scatter(M: NDArray) -> NDArray:
            return adj(M) @ M
        N, d = X.shape
        res = _solve_eigh_robustly(a_matrix=scatter(X), b_matrix=scatter(A @ X - X), subset=subset_range(n=d, subset_size=n_components, smallest=False))
        P_basis = res.P_or_U
        LOGGER.info(f"LDADenoiser: {P_basis.shape=} {res.eigvals.shape=}")
        # Return the projector and the eigenvalues as extra info
        return P_basis, res.eigvals
    
class SymmetrizedLDADenoiser(LDADenoiser):
    def projector(self, X: NDArray, A: sp.spmatrix, n_components: int):
        return super().projector(X, (A + adj(A)) / 2, n_components)

class TwoSubsetScatterDenoiser(ABC):
    """
    Abstract base for denoisers that operate on two subsets and require within-class scatter.
    Subclasses must implement get_within_class_scatter.
    """
    @abstractmethod
    def get_within_class_scatter(self, X1: NDArray, X2: NDArray, X: NDArray) -> NDArray:
        ...

    def projector(self, X1: NDArray, X2: NDArray, n_components: int, X: Optional[NDArray] = None) -> tuple[NDArray, Optional[NDArray]]:
        # Use entire embeddings set if provided, otherwise combine the two groups for centering
        if X is None:
            X_combined = np.vstack([X1, X2])
        else:
            X_combined = X
        N1, d = X1.shape
        N2, _ = X2.shape
        N_total = N1 + N2
        # Center all data using the full embeddings set (or combined data if X is None)
        X_mean = X_combined.mean(axis=0)
        X1_centered = X1 - X_mean
        X2_centered = X2 - X_mean
        X_combined_centered = X_combined - X_mean
        # Within-class scatter (subclass-specific)
        S_W = self.get_within_class_scatter(X1_centered, X2_centered, X_combined_centered)
        # Between-class scatter
        mu1_centered = X1_centered.mean(axis=0)
        mu2_centered = X2_centered.mean(axis=0)
        mu_diff = mu1_centered - mu2_centered
        S_B = (N1 * N2 / N_total) * np.outer(mu_diff, mu_diff)
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
    def get_within_class_scatter(self, X1: NDArray, X2: NDArray, X: NDArray) -> NDArray:
        return scatter(X1) + scatter(X2)
    # get_scatter_matrices uses default (Anti-LDA)

class DomainInvariantDenoiser(TwoSubsetScatterDenoiser):
    """
    Removes directions that discriminate between two groups by minimizing the ratio of 
    between-class scatter to total scatter.
    """
    def get_within_class_scatter(self, X1: NDArray, X2: NDArray, X: NDArray) -> NDArray:
        return scatter(X)

class EmbeddedGraphDenoiser:
    def __init__(self, wrapped_denoiser):
        self.wrapped_denoiser = wrapped_denoiser

    def projector(self, X: NDArray, A: sp.spmatrix, n_components: int):
        N, d = X.shape
        LOGGER.info(f"Embedding A into embedding space by solving AX ≈ XW.")
        A_sym = (A + adj(A)) / 2
        target = A_sym @ X
        W, residuals, rank, s = lstsq(X, target, rcond=None)
        
        X_smoothed = X @ W
        A_w = X_smoothed @ adj(X_smoothed)
        return self.wrapped_denoiser.projector(X, A_w, n_components)

    def __repr__(self):
        return f"EmbeddedGraph({self.wrapped_denoiser.__class__.__name__})"

class AugmentedLPPDenoiser:
    """
    Improves LPP by augmenting its objective.
    It combines the standard LPP goal (preserving neighbor distances) with the goal of
    minimizing the distance between a point and its A-transformed image.
    This is achieved by summing their respective scatter matrices in the numerator.
    """
    def projector(self, X: NDArray, A: sp.spmatrix, n_components: int):
        N, d = X.shape
        B = (A + adj(A)) / 2
        
        # Standard LPP components
        D = sp.diags(np.asarray(B.sum(axis=1)).squeeze())
        L = D - B

        # Tuning parameter to balance the two objectives
        gamma = 1.0
        
        if N < d:
            LOGGER.info(f"AugmentedLPP: N < d, using dual formulation. N={N}, d={d}")
            K = X @ adj(X)
            
            # Numerator: S_A_combined = (XᵀLX) + gamma * ((BX - X)ᵀ(BX - X))
            # The dual of XᵀMX is KMK.
            M_lpp = L
            M_new = adj(B - sp.eye(N)) @ (B - sp.eye(N))
            a_dual = K @ (M_lpp + gamma * M_new) @ K

            # Denominator: S_B_lpp = XᵀDX
            b_dual = K @ D @ K

            subset = subset_range(n=N, subset_size=n_components, smallest=True)
            res = _solve_eigh_robustly(a_dual, b_dual, subset)
            P = _recover_projector_from_dual(X, res.P_or_U)
            return P, res.eigvals
        else:
            # Primal formulation
            S_W_lpp = adj(X) @ L @ X
            S_W_new = adj(B @ X - X) @ (B @ X - X)
            S_A_combined = S_W_lpp + gamma * S_W_new
            S_B_lpp = adj(X) @ D @ X

            subset = subset_range(n=d, subset_size=n_components, smallest=True)
            res = _solve_eigh_robustly(S_A_combined, S_B_lpp, subset)
            P = res.P_or_U
            return P, res.eigvals

class OrthogonalManifoldDenoiser:
    """
    A monolithic approach that balances preserving global, non-local structure against
    minimizing the error introduced by graph-smoothing (the AX-X distance).
    S_B = Xᵀ(I - L)X  (preserves global variance while shrinking neighbor distances)
    S_W = (AX - X)ᵀ(AX - X) (minimizes the smoothing error)
    """
    def projector(self, X: NDArray, A: sp.spmatrix, n_components: int):
        N, d = X.shape
        B = (A + adj(A)) / 2

        # Numerator: S_B represents non-local scatter (Total Scatter - Local Scatter)
        D = sp.diags(np.asarray(B.sum(axis=1)).squeeze())
        L = D - B
        S_B = adj(X) @ (sp.eye(N) - L) @ X

        # Denominator: S_W represents the smoothing error to be minimized
        S_W = adj(B @ X - X) @ (B @ X - X)

        subset = subset_range(n=d, subset_size=n_components, smallest=False)
        res = _solve_eigh_robustly(a_matrix=S_B, b_matrix=S_W, subset=subset)
        P_basis = res.P_or_U
        
        # Form the projector using the pseudo-inverse for a non-orthonormal basis
        return P_basis, res.eigvals

class _PLSDenoiser(ABC):
    """
    Superclass for fast PLS/CCA-style projection using direct SVD of a cross-matrix.
    Subclasses should override get_cross_matrix.
    """
    def projector(self, X: NDArray, A: sp.spmatrix, n_components: int):
        N, d = X.shape
        B = (A + adj(A)) / 2
        Y = B @ X
        X_centered = X - X.mean(axis=0)
        Y_centered = Y - Y.mean(axis=0)
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
