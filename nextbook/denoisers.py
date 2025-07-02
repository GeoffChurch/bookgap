import enum
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from util import TIMER, logger
from abc import ABC, abstractmethod

def adj(A: NDArray) -> NDArray:
    return A.conj().T

class DenoiserImpl(ABC):
    def __call__(self, X: NDArray, A: NDArray, n_components: int) -> tuple[NDArray, NDArray]:
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean
        P, extra = self.projector(X_centered, A, n_components)

        if np.allclose(adj(P) @ P, np.eye(P.shape[1])):
            P_inv = adj(P)
            logger.info(f"Projector P is orthonormal, using adj(P) as P_inv.")
        else:
            P_inv = np.linalg.pinv(P)
            logger.info(f"Projector P is not orthonormal, using np.linalg.pinv(P) as P_inv.")

        X_denoised = X_centered @ (P @ P_inv) + X_mean
        return X_denoised, extra

    @TIMER.time(formatters=dict(X=lambda X: f"<shape={X.shape}>", A=lambda A: f"<shape={A.shape}>"))
    @abstractmethod
    def projector(self, X: NDArray, A: NDArray, n_components: int) -> tuple[NDArray, NDArray]:
        pass

class IdDenoiser(DenoiserImpl):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        return np.eye(X.shape[1]), np.array([])

class PCADenoiser(DenoiserImpl):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        min_cs = min(X.shape)
        if n_components > min_cs:
            logger.warning(f"n_components ({n_components}) is greater than the minimum of the dimensions of X ({min_cs}). Using {min_cs} components instead.")
            n_components = min_cs
        pca = PCA(n_components=n_components)
        pca.fit(X)
        return pca.components_.T, np.array([])

class GPCADenoiser(DenoiserImpl):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        B = (A + adj(A)) / 2
        M = adj(X) @ B @ X
        N, d = X.shape
        eigvals, P = eigh(M, subset_by_index=[d-n_components, d-1])
        # P = eigvecs[:, -n_components:]
        return P, eigvals

class InvariantPCADenoiser(DenoiserImpl):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        B = (A + adj(A)) / 2
        C = (np.eye(X.shape[0]) - B) @ X
        M = adj(C) @ C
        eigvals, P = eigh(M, subset_by_index=[0, n_components-1])
        # P = eigvecs[:, :n_components]
        return P, eigvals
    
class RelInvariantPCADenoiser(InvariantPCADenoiser):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        B = (A + adj(A)) / 2
        N, d = X.shape
        C = (np.eye(N) - B) @ X
        a_mat = adj(C) @ C
        b_mat = adj(X) @ X
        if N < d:
            nonzero_vals = b_mat[np.nonzero(b_mat)]
            median = np.median(nonzero_vals) if nonzero_vals.size > 0 else 1.0
            logger.warning(f"Number of samples ({N}) is less than number of features ({d}). Using median abs({median})={np.abs(median)} for regularization.")
            b_mat += np.eye(d) * np.abs(median)
        logger.info(f"Solving generalized eigenvalue problem for RelInvariantPCADenoiser with shapes: {X.shape=}, {A.shape=}, {C.shape=}, {B.shape=}")
        eigvals, P = eigh(
            a=a_mat,
            b=b_mat,
            subset_by_index=[0, n_components-1])
        # P = eigvecs[:, :n_components]
        logger.info(f"RelInvariantPCADenoiser projector shape: {P.shape}, eigvals shape: {eigvals.shape}")
        return P, eigvals

    
class RelAmpDenoiser(DenoiserImpl):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        B = (A + adj(A)) / 2
        N, d = X.shape
        if n_components > d:
            raise ValueError(f"n_components ({n_components}) cannot be greater than the number of features ({d}).")
        logger.info(f"Computing {n_components} components for RelAmpDenoiser on data with shape {X.shape} and A shape {A.shape}. {N=}, {d=}")
        b_matrix = adj(X) @ X
        if N < d:
            nonzero_vals = b_matrix[np.nonzero(b_matrix)]
            median = np.median(nonzero_vals) if nonzero_vals.size > 0 else 1.0
            logger.warning(f"Number of samples ({N}) is less than number of features ({d}). Using median abs({median})={np.abs(median)} for regularization.")
            b_matrix += np.eye(d) * np.abs(median)  # Regularization to avoid singularity
        eigvals, P = eigh(
            a=adj(X) @ B @ X,
            b=b_matrix,
            # driver="gvd",
            subset_by_index=[d-n_components, d-1])
        logger.info(f"{P.shape=}, {eigvals.shape=}")
        # P = P[:, -n_components:]
        logger.info(f"RelAmpDenoiser truncated projector shape: {P.shape}, eigvals shape: {eigvals.shape}")
        return P, eigvals

class LPPDenoiser(DenoiserImpl):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        B = (A + adj(A)) / 2
        # Define L and D
        D = np.diag(np.sum(B, axis=1))
        L = D - B
        # Compute the generalized eigenvalue problem
        N, d = X.shape
        if n_components > d:
            raise ValueError(f"n_components ({n_components}) cannot be greater than the number of features ({d}).")
        logger.info(f"Computing {n_components} components for LPPDenoiser on data with shape {X.shape} and A shape {A.shape}. {N=}, {d=}")
        a_matrix = adj(X) @ L @ X
        b_matrix = adj(X) @ D @ X
        if N < d:
            nonzero_vals = b_matrix[np.nonzero(b_matrix)]
            median = np.median(nonzero_vals) if nonzero_vals.size > 0 else 1.0
            logger.warning(f"Number of samples ({N}) is less than number of features ({d}). Using median abs({median})={np.abs(median)} for regularization.")
            b_matrix += np.eye(d) * np.abs(median)
        logger.info(f"Solving generalized eigenvalue problem for LPPDenoiser with shapes: {X.shape=}, {A.shape=}, {D.shape=}, {L.shape=}")
        eigvals, P = eigh(
            a=a_matrix,
            b=b_matrix,
            subset_by_index=[0, n_components-1])
        return P, eigvals

class NormalizedLPPDenoiser(DenoiserImpl):
    """
    Denoising using Locality Preserving Projections with a
    symmetrically normalized graph Laplacian.
    """
    def projector(self, X: NDArray, A: NDArray, n_components: int) -> tuple[NDArray, NDArray]:
        # 1. Ensure a symmetric adjacency matrix with no self-loops
        B = (A + adj(A)) / 2
        np.fill_diagonal(B, 0)
        N, d = X.shape

        # 2. Calculate the inverse square root of the degree matrix D^(-1/2)
        # This is the key normalization step.
        degrees = np.sum(B, axis=1)
        assert not any(degrees == 0), "Degrees must be non-zero for normalization."
        # # Add a small epsilon for numerical stability in case of isolated nodes (degree=0)
        # inv_sqrt_deg = np.power(degrees + 1e-9, -0.5)
        inv_sqrt_deg = np.power(degrees, -0.5)
        D_inv_sqrt = np.diag(inv_sqrt_deg)

        # 3. Compute the symmetrically normalized Laplacian: L_sym = I - D^(-1/2) A D^(-1/2)
        I = np.eye(N)
        L_sym = I - (D_inv_sqrt @ B @ D_inv_sqrt)

        # 4. Form the matrix for the standard eigenvalue problem
        # The objective is to minimize w^T * (X^T * L_sym * X) * w
        a_matrix = adj(X) @ L_sym @ X

        # 5. Solve the standard eigenvalue problem.
        # We select eigenvectors with the smallest eigenvalues to preserve locality.
        eigvals, P = eigh(
            a=a_matrix,
            subset_by_index=[0, n_components - 1]
        )
        
        # Because this is a standard eigenproblem (b=I), the resulting P is orthonormal
        # and works correctly with the base DenoiserImpl.__call__ method.
        return P, eigvals

class Denoiser(enum.Enum):
    NONE = IdDenoiser()
    PCA = PCADenoiser()
    GPCA = GPCADenoiser()
    INVAR_PCA = InvariantPCADenoiser()
    REL_INVAR_PCA = RelInvariantPCADenoiser()
    REL_AMP = RelAmpDenoiser()
    LPP = LPPDenoiser()
    NORM_LPP = NormalizedLPPDenoiser()

    def __init__(self, func):
        self._func = func

    def __call__(self, X: NDArray, A: NDArray, n_components: int) -> tuple[NDArray, NDArray]:
        return self._func(X, A, n_components)

    def __str__(self):
        return self._func.__class__.__name__
    
    def __repr__(self):
        return str(self)
