import abc
from dataclasses import dataclass
import enum
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, svd, qr
import scipy.sparse as sp
from sklearn.decomposition import PCA
from numpy.linalg import pinv, LinAlgError, lstsq
from bookgap.rsdc import rnojd, manopt_rnojd
from bookgap.util import TIMER, LOGGER
from abc import ABC, abstractmethod

# Import RNOJD from the submodule
import sys
from pathlib import Path


def subset_range(n: int, subset_size: int, smallest: bool) -> tuple[int, int]:
    subset_size = min(subset_size, n)
    if smallest:
        return 0, subset_size - 1
    else:
        return n - subset_size, n - 1

def adj(A):
    return A.conj().T

def scatter(M: NDArray) -> NDArray:
    # M_centered = M - M.mean(axis=0, keepdims=True)
    # n = M.shape[0]
    # if n <= 1:
    #     # Avoid division by zero for single-sample case
    #     return M_centered.T @ M_centered
    # return M_centered.T @ M_centered / (n - 1)
    return adj(M) @ M / M.shape[0]

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
    """
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
        return scatter(self.X())
    
    def S_L(self) -> NDArray:
        """The graph Laplacian scatter matrix X^T L X."""
        X = self.X()
        X_centered = X - X.mean(axis=0)
        return adj(X_centered) @ self.L() @ X_centered
    
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
        
        # Compute weighted overall mean
        overall_mean = np.zeros(self.X().shape[1])
        for subset, size in zip(subsets, sizes):
            if size > 0:
                overall_mean += size * subset.mean(axis=0)
        overall_mean /= total_size
        
        # Compute between-class scatter matrix
        S_B = np.zeros((self.X().shape[1], self.X().shape[1]))
        for subset, size in zip(subsets, sizes):
            if size > 0:
                mu_diff = subset.mean(axis=0) - overall_mean
                S_B += size * np.outer(mu_diff, mu_diff)
        
        return S_B / total_size
    
    def S_subset(self, subset_index: int) -> NDArray:
        """The scatter matrix for a specific subset."""
        return scatter(self.X_n(subset_index))

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
    
    # # If this was a generalized eigenvalue problem, orthogonalize the result
    # if b_matrix is not None:
    #     LOGGER.info(f"Orthogonalizing generalized eigenvectors from shape {P_or_U.shape}")
    #     P_or_U = orthonormalize(P_or_U)
    #     LOGGER.info(f"Orthogonalized projector shape: {P_or_U.shape}")
    
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
    def projector(self, context: DenoiserContext, n_components: int) -> tuple[NDArray, Optional[NDArray]]:
        X = context.X()
        min_cs = min(X.shape)
        if n_components > min_cs:
            n_components = min_cs
        pca = PCA(n_components=n_components)
        pca.fit(X)
        P = pca.components_.T
        return P, pca.singular_values_

class GPCADenoiser:
    def projector(self, context: DenoiserContext, n_components: int):
        X = context.X()
        B = context.A_sym()
        M = adj(X) @ B @ X
        d = X.shape[1]
        subset = subset_range(n=d, subset_size=n_components, smallest=False)
        eigvals, P = eigh(M, subset_by_index=subset)
        return P, eigvals

class RelAmpDenoiser:
    def projector(self, context: DenoiserContext, n_components: int):
        X = context.X()
        A = context.A()
        N, d = X.shape
        B = context.A_sym()
        
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
        X = context.X()
        N, d = X.shape
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
            K = X @ adj(X)
            subset = subset_range(n=N, subset_size=n_components, smallest=True)
            a_dual = adj(K) @ L @ K
            LOGGER.info(f"{type(a_dual)=} {a_dual.shape=}")
            b_dual = adj(K) @ D @ K
            LOGGER.info(f"{type(b_dual)=} {b_dual.shape=}")
            res = _solve_eigh_robustly(a_dual, b_dual, subset)
            P = _recover_projector_from_dual(X, res.P_or_U)
            P = P[:, ::-1]  # Reverse to put most important components first
            eigvals = res.eigvals[::-1]  # Reverse to match projector order
            return P, eigvals
        else:
            subset = subset_range(n=d, subset_size=n_components, smallest=True)
            a_matrix = adj(X) @ L @ X
            b_matrix = adj(X) @ D @ X
            res = _solve_eigh_robustly(a_matrix, b_matrix, subset)
            P = res.P_or_U
            P = P[:, ::-1]  # Reverse to put most important components first
            eigvals = res.eigvals[::-1]  # Reverse to match projector order
            return P, eigvals

class NormalizedLPPDenoiser:
    def projector(self, context: DenoiserContext, n_components: int):
        X = context.X()
        B = context.A_sym()
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
        P = P[:, ::-1]  # Reverse to put most important components first
        eigvals = eigvals[::-1]  # Reverse to match projector order
        return P, eigvals

class WDenoiser:
    """
    Learns W where XW ≈ AX, then uses rank-reduced W for projection.
    """
    def projector(self, context: DenoiserContext, n_components: int):
        X = context.X()
        A = context.A()
        # Solve for W in XW ≈ AX
        target = A @ X
        W, _, _, _ = lstsq(X, target, rcond=None)
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
    
# class SymmetrizedLDADenoiser::
#     def projector(self, X: NDArray, A: sp.spmatrix, n_components: int):
#         return super().projector(X, A + adj(A), n_components)

class TwoSubsetScatterDenoiser(ABC):
    """
    Abstract base for denoisers that operate on two subsets and require within-class scatter.
    Subclasses must implement get_within_class_scatter.
    """
    @abstractmethod
    def get_within_class_scatter(self, X1: NDArray, X2: NDArray, X: NDArray) -> NDArray:
        ...

    def projector(self, context: DenoiserContext, n_components: int) -> tuple[NDArray, Optional[NDArray]]:
        X1 = context.X_n(0)
        X2 = context.X_n(1)
        X = context.X()
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
        P_basis = P_basis[:, ::-1]  # Reverse to put most important components first
        eigvals = eigvals[::-1]  # Reverse to match projector order
        return P_basis, eigvals

class AntiLDADenoiser(TwoSubsetScatterDenoiser):
    """
    Anti-LDA denoiser that removes the directions that best discriminate between two groups.
    It minimizes the ratio of between-class scatter to within-class scatter.
    """
    def get_within_class_scatter(self, X1: NDArray, X2: NDArray, X: NDArray) -> NDArray:
        return scatter(X1) + scatter(X2)

class DomainInvariantDenoiser(TwoSubsetScatterDenoiser):
    """
    Removes directions that discriminate between two groups by minimizing the ratio of 
    between-class scatter to total scatter.
    """
    def get_within_class_scatter(self, X1: NDArray, X2: NDArray, X: NDArray) -> NDArray:
        return scatter(X)

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
    """
    def projector(self, context: DenoiserContext, n_components: int):
        X = context.X()
        B = context.A_sym()
        N, d = X.shape
        D = sp.diags(np.asarray(B.sum(axis=1)).squeeze())
        L = D - B
        # Tuning parameter to balance the two objectives
        gamma = 1.0
        if N < d:
            LOGGER.info(f"AugmentedLPP: N < d, using dual formulation. N={N}, d={d}")
            K = X @ adj(X)
            M_lpp = L
            M_new = adj(B - sp.eye(N)) @ (B - sp.eye(N))
            a_dual = K @ (M_lpp + gamma * M_new) @ K
            b_dual = K @ D @ K
            subset = subset_range(n=N, subset_size=n_components, smallest=True)
            res = _solve_eigh_robustly(a_dual, b_dual, subset)
            P = _recover_projector_from_dual(X, res.P_or_U)
            P = P[:, ::-1]  # Reverse to put most important components first
            eigvals = res.eigvals[::-1]  # Reverse to match projector order
            return P, eigvals
        else:
            # Primal formulation
            S_W_lpp = adj(X) @ L @ X
            S_W_new = adj(B @ X - X) @ (B @ X - X)
            S_A_combined = S_W_lpp + gamma * S_W_new
            S_B_lpp = adj(X) @ D @ X
            subset = subset_range(n=d, subset_size=n_components, smallest=True)
            res = _solve_eigh_robustly(S_A_combined, S_B_lpp, subset)
            P = res.P_or_U
            P = P[:, ::-1]  # Reverse to put most important components first
            eigvals = res.eigvals[::-1]  # Reverse to match projector order
            return P, eigvals

class OrthogonalManifoldDenoiser:
    """
    A monolithic approach that balances preserving global, non-local structure against
    minimizing the error introduced by graph-smoothing (the AX-X distance).
    S_B = Xᵀ(I - L)X  (preserves global variance while shrinking neighbor distances)
    S_W = (AX - X)ᵀ(AX - X) (minimizes the smoothing error)
    """
    def projector(self, context: DenoiserContext, n_components: int):
        X = context.X()
        B = context.A_sym()
        N, d = X.shape
        # Numerator: S_B represents non-local scatter (Total Scatter - Local Scatter)
        D = sp.diags(np.asarray(B.sum(axis=1)).squeeze())
        L = D - B
        S_B = adj(X) @ (sp.eye(N) - L) @ X
        # Denominator: S_W represents the smoothing error to be minimized
        S_W = adj(B @ X - X) @ (B @ X - X)
        subset = subset_range(n=d, subset_size=n_components, smallest=False)
        res = _solve_eigh_robustly(a_matrix=S_B, b_matrix=S_W, subset=subset)
        P_basis = res.P_or_U
        return P_basis, res.eigvals

class _PLSDenoiser(ABC):
    """
    Superclass for fast PLS/CCA-style projection using direct SVD of a cross-matrix.
    Subclasses should override get_cross_matrix.
    """
    def projector(self, context: DenoiserContext, n_components: int):
        X = context.X()
        B = context.A_sym()
        N, d = X.shape
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

class CCLIPDenoiser:
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
    """
    def __init__(self, alpha: float = 0.5):
        """
        Initializes the CCLIPDenoiser.

        Args:
            alpha (float): A tuning parameter in [0, 1] to balance the objectives.
                           alpha = 0 -> Purely class-conflating projection.
                           alpha = 1 -> Purely locality-inducing projection.
        """
        assert 0 <= alpha <= 1, "alpha must be in the range [0, 1]"
        self.alpha = alpha

    def projector(self, context: DenoiserContext, n_components: int) -> tuple[NDArray, Optional[NDArray]]:
        """
        Computes the projection matrix for CCLIP.
        """
        X = context.X()
        N, d = X.shape
        
        # Get precomputed scatter matrices from context
        S_T = context.S_T()
        S_B = context.S_B()
        S_L = context.S_L()

        # Construct the numerator matrix using the alpha mixing parameter
        A_matrix = (1 - self.alpha) * S_B + self.alpha * S_L
        
        # Find smallest eigenvalues and corresponding eigenvectors
        subset = subset_range(n=d, subset_size=n_components, smallest=True)
        res = _solve_eigh_robustly(a_matrix=A_matrix, b_matrix=S_T, subset=subset)

        P = res.P_or_U
        eigvals = res.eigvals
        P = P[:, ::-1]  # Ensure the projector is in descending order of eigenvalues
        eigvals = eigvals[::-1]  # Reverse to match projector order
        return P, eigvals

class AntiCSPDenoiser:
    """
    Finds the least discriminative directions between two classes by finding
    eigenvectors of the generalized system S1*v = λ*(S1+S2)*v where λ is
    closest to 0.5.
    """
    def projector(self, context: DenoiserContext, n_components: int) -> tuple[NDArray, Optional[NDArray]]:
        S1 = scatter(context.X_n(0))
        S2 = scatter(context.X_n(1))
        
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
                 wt: float = 1.0, wl: float = 1.0, w_conflate: float = 1.0):
        """
        Initialize the JAD denoiser.
        
        Args:
            trials: Number of random trials for the RNOJD algorithm
            use_positive_definite: If True, use the positive definite formulation
            wt: Weight for total scatter matrix S_T
            wl: Weight for Laplacian scatter matrix S_L
            w_conflate: Weight for the class-conflation objective (1 - normalized_entropy of subset variances)
        """
        self.trials = trials
        self.use_positive_definite = use_positive_definite
        self.wt = wt
        self.wl = wl
        self.w_conflate = w_conflate
    
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
        
        # Get the scatter matrices
        S_T = context.S_T()
        S_L = context.S_L()
        
        # Get all subset scatter matrices
        subset_scatter_matrices = []
        for i in range(num_subsets):
            subset_scatter_matrices.append(context.S_subset(i))
        
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
            t = float(p_i.T @ S_T @ p_i)
            l = float(p_i.T @ S_L @ p_i)
            
            # Get subset variances
            subset_variances = []
            for S_subset in subset_scatter_matrices:
                s = float(p_i.T @ S_subset @ p_i)
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
        