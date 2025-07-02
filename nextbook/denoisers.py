from dataclasses import dataclass
import enum
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from numpy.linalg import pinv, LinAlgError, lstsq
from util import TIMER, logger
from abc import ABC, abstractmethod

def adj(A: NDArray) -> NDArray:
    return A.conj().T

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
def _solve_eigh_robustly(a_matrix: NDArray, b_matrix: NDArray | None, subset: list[int]) -> SolveEighRobustlyResult:
    """Solves eigh, retrying with regularization upon failure."""
    try:
        eigvals, P_or_U = eigh(a=a_matrix, b=b_matrix, subset_by_index=subset)
    except LinAlgError:
        logger.warning("eigh failed, likely due to a singular b_matrix. Retrying with regularization.")
        if b_matrix is not None:
            # b_matrix += np.eye(b_matrix.shape[0]) * 1e-6
            ridge_val = np.mean(np.abs(np.diag(b_matrix))) * 1e-4 + 1e-8
            logger.warning(f"Retrying with adaptive regularization: {ridge_val:.2e}")
            b_matrix += np.eye(b_matrix.shape[0]) * ridge_val
            eigvals, P_or_U = eigh(a=a_matrix, b=b_matrix, subset_by_index=subset)
        else:
            raise
    logger.info(f"{eigvals.shape=} {P_or_U.shape=}")
    return SolveEighRobustlyResult(eigvals=eigvals, P_or_U=P_or_U)

@TIMER.time(formatters=dict(X=lambda X: f"<shape={X.shape}>", U=lambda U: f"<shape={U.shape}>"))
def _recover_projector_from_dual(X: NDArray, U: NDArray) -> NDArray:
    """Recovers the d-dimensional projector P from the N-dimensional dual eigenvectors U."""
    P_raw = adj(X) @ U
    P_norms = np.linalg.norm(P_raw, axis=0)

    logger.info(f"{X.shape=} @ {U.shape=} = {P_raw.shape=}")
    logger.info(f"np.linalg.norm({P_raw.shape=}, axis=0) = {P_norms.shape=}")
    
    # Avoid division by zero for null projectors without adding a ridge.
    # If a norm is zero, the resulting column will be all zeros, which is correct.
    P_norms[P_norms == 0] = 1
    ret = P_raw / P_norms
    logger.info(f"{P_raw.shape=} {P_norms.shape=} {ret.shape=}")
    return ret

class DenoiserImpl(ABC):
    def __call__(self, X: NDArray, A: NDArray, n_components: int) -> tuple[NDArray, NDArray]:
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean
        P, extra = self.projector(X_centered, A, n_components)

        if np.allclose(adj(P) @ P, np.eye(P.shape[1])):
            P_inv = adj(P)
        else:
            P_inv = pinv(P)

        X_denoised = X_centered @ (P @ P_inv) + X_mean
        return X_denoised, extra

    @TIMER.time(formatters=dict(X=lambda X: f"<shape={X.shape}>", A=lambda A: f"<shape={A.shape}>"))
    @abstractmethod
    def projector(self, X: NDArray, A: NDArray, n_components: int) -> tuple[NDArray, NDArray]:
        pass

class IdDenoiser(DenoiserImpl):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        d = X.shape[1]
        if n_components > d: n_components = d
        return np.eye(d, n_components), np.array([])

class PCADenoiser(DenoiserImpl):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        min_cs = min(X.shape)
        if n_components > min_cs:
            n_components = min_cs
        pca = PCA(n_components=n_components)
        pca.fit(X)
        return pca.components_.T, pca.singular_values_

class GPCADenoiser(DenoiserImpl):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        B = (A + adj(A)) / 2
        M = adj(X) @ B @ X
        d = X.shape[1]
        subset_size = min(n_components, d - 1) if d > 1 else 0
        eigvals, P = eigh(M, subset_by_index=[d - 1 - subset_size, d - 1])
        return P, eigvals

class RelAmpDenoiser(DenoiserImpl):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        N, d = X.shape
        B = (A + adj(A)) / 2
        
        if N < d:
            logger.info(f"RelAmp: N < d, using dual formulation. N={N}, d={d}")
            K = X @ adj(X)
            a_dual = adj(K) @ B @ K
            b_dual = adj(K) @ K
            subset_size = min(n_components, N - 1) if N > 1 else 0
            subset = [N - 1 - subset_size, N - 1]
            res = _solve_eigh_robustly(a_dual, b_dual, subset)
            P = _recover_projector_from_dual(X, res.P_or_U)
            return P, res.eigvals
        else:
            a_matrix = adj(X) @ B @ X
            b_matrix = adj(X) @ X
            subset_size = min(n_components, d - 1) if d > 1 else 0
            subset = [d - 1 - subset_size, d - 1]
            res = _solve_eigh_robustly(a_matrix, b_matrix, subset)
            return res.P_or_U, res.eigvals

class NeighborDistMinDenoiser(DenoiserImpl):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        N, d = X.shape
        B = (A + adj(A)) / 2
        M = np.eye(N) - B
        a_mat = adj(X) @ adj(M) @ M @ X
        
        if N < d:
            logger.info(f"NeighborDistMin: N < d, using dual formulation. N={N}, d={d}")
            K = X @ adj(X)
            a_dual = adj(K) @ adj(M) @ M @ K
            b_dual = adj(K) @ K
            subset_size = min(n_components, N - 1) if N > 1 else 0
            subset = [0, subset_size]
            res = _solve_eigh_robustly(a_dual, b_dual, subset)
            P = _recover_projector_from_dual(X, res.P_or_U)
            return P, res.eigvals
        else:
            b_mat = adj(X) @ X
            subset_size = min(n_components, d - 1) if d > 1 else 0
            subset = [0, subset_size]
            res = _solve_eigh_robustly(a_mat, b_mat, subset)
            return res.P_or_U, res.eigvals

class LPPDenoiser(DenoiserImpl):
    @TIMER.time(formatters=dict(X=lambda X: f"<shape={X.shape}>", A=lambda A: f"<shape={A.shape}>"))
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        N, d = X.shape
        B = (A + adj(A)) / 2

        # LPP requires non-negative weights for a valid Degree matrix.
        negatives = B < 0
        num_negatives = np.sum(negatives)
        if num_negatives > 0:
            logger.warning(f"LPPDenoiser received an affinity matrix with {num_negatives} negative weights ({num_negatives / B.size:.2%} of total). Setting negative weights to zero.")
            B[B < 0] = 0

        D = np.diag(np.sum(B, axis=1))
        L = D - B

        if N < d:
            logger.info(f"LPP: N < d, using dual formulation. N={N}, d={d}")
            K = X @ adj(X)
            a_dual = adj(K) @ L @ K
            b_dual = adj(K) @ D @ K
            subset_size = min(n_components, N - 1) if N > 1 else 0
            subset = [0, subset_size]
            res = _solve_eigh_robustly(a_dual, b_dual, subset)
            P = _recover_projector_from_dual(X, res.P_or_U)
            return P, res.eigvals
        else:
            a_matrix = adj(X) @ L @ X
            b_matrix = adj(X) @ D @ X
            subset_size = min(n_components, d - 1) if d > 1 else 0
            subset = [0, subset_size]
            res = _solve_eigh_robustly(a_matrix, b_matrix, subset)
            return res.P_or_U, res.eigvals

class NormalizedLPPDenoiser(DenoiserImpl):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        B = (A + adj(A)) / 2
        N, d = X.shape
        degrees = np.sum(B, axis=1)
        
        # Handle zero-degree nodes robustly without a ridge
        inv_sqrt_deg = np.zeros_like(degrees, dtype=float)
        safe_indices = degrees > 0
        inv_sqrt_deg[safe_indices] = np.power(degrees[safe_indices], -0.5)
        
        D_inv_sqrt = np.diag(inv_sqrt_deg)
        L_sym = np.eye(N) - (D_inv_sqrt @ B @ D_inv_sqrt)
        a_matrix = adj(X) @ L_sym @ X
        subset_size = min(n_components, d - 1) if d > 1 else 0
        eigvals, P = eigh(a_matrix, subset_by_index=[0, subset_size])
        return P, eigvals

class EmbeddedGraphDenoiser(DenoiserImpl):
    def __init__(self, wrapped_denoiser: DenoiserImpl):
        self.wrapped_denoiser = wrapped_denoiser

    def projector(self, X: NDArray, A: NDArray, n_components: int) -> tuple[NDArray, NDArray]:
        N, d = X.shape
        logger.info(f"Embedding A into embedding space by solving AX ≈ XW.")
        A_sym = (A + adj(A)) / 2
        target = A_sym @ X
        W, residuals, rank, s = lstsq(X, target, rcond=None)
        
        X_smoothed = X @ W
        if N > 5000:
            logger.warning(f"Graph embedding is creating a dense {N}x{N} affinity matrix. This may be slow or cause memory issues.")
        A_w = X_smoothed @ adj(X_smoothed)
        return self.wrapped_denoiser.projector(X, A_w, n_components)

    def __repr__(self):
        return f"EmbeddedGraph({self.wrapped_denoiser.__class__.__name__})"

class Denoiser(enum.Enum):
    NONE = IdDenoiser()
    PCA = PCADenoiser()
    GPCA = GPCADenoiser()
    REL_AMP = RelAmpDenoiser()
    NEIGHBOR_DIST_MIN = NeighborDistMinDenoiser()
    LPP = LPPDenoiser()
    NORM_LPP = NormalizedLPPDenoiser()
    W_LPP = EmbeddedGraphDenoiser(LPP)
    W_NEIGHBOR_DIST_MIN = EmbeddedGraphDenoiser(NEIGHBOR_DIST_MIN)
    
    def __call__(self, X: NDArray, A: NDArray, n_components: int) -> tuple[NDArray, NDArray]:
        return self.value(X, A, n_components)

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return str(self)
