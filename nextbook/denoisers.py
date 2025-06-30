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
        X_projected = X_centered @ P
        X_centered_denoised = X_projected @ adj(P)
        X_denoised = X_centered_denoised + X_mean
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
        pca = PCA(n_components=n_components)
        pca.fit(X)
        return pca.components_.T, pca.singular_values_

class GPCADenoiser(DenoiserImpl):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        B = (A + adj(A)) / 2
        M = adj(X) @ B @ X
        eigvals, eigvecs = eigh(M)
        P = eigvecs[:, -n_components:]
        return P, eigvals

class InvariantPCADenoiser(DenoiserImpl):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        B = (A + adj(A)) / 2
        C = (np.eye(X.shape[0]) - B) @ X
        M = adj(C) @ C
        eigvals, eigvecs = eigh(M)
        P = eigvecs[:, :n_components]
        return P, eigvals

class RelAmpDenoiser(DenoiserImpl):
    def projector(self, X: NDArray, A: NDArray, n_components: int):
        B = (A + adj(A)) / 2
        _N, d = X.shape
        if n_components > d:
            raise ValueError(f"n_components ({n_components}) cannot be greater than the number of features ({d}).")
        logger.info(f"Computing {n_components} components for RelAmpDenoiser on data with shape {X.shape} and A shape {A.shape}. {_N=}, {d=}")
        b_matrix = adj(X) @ X
        if _N < d:
            b_matrix += np.eye(d) * 1e-6  # Regularization to avoid singularity
        eigvals, P = eigh(
            a=adj(X) @ B @ X,
            b=b_matrix,
            driver="gvd",
            # subset_by_index=[d-n_components, d-1])
        )
        logger.info(f"{P.shape=}, {eigvals.shape=}")
        P = P[:, -n_components:]
        logger.info(f"RelAmpDenoiser truncated projector shape: {P.shape}, eigvals shape: {eigvals.shape}")
        return P, eigvals

class Denoiser(enum.Enum):
    NONE = IdDenoiser()
    PCA = PCADenoiser()
    GPCA = GPCADenoiser()
    INVAR_PCA = InvariantPCADenoiser()
    REL_AMP = RelAmpDenoiser()

    def __init__(self, func):
        self._func = func

    def __call__(self, X: NDArray, A: NDArray, n_components: int) -> tuple[NDArray, NDArray]:
        return self._func(X, A, n_components)

    def __str__(self):
        return self._func.__class__.__name__
    
    def __repr__(self):
        return str(self)
