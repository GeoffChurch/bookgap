from dataclasses import dataclass
import enum
from typing import Any, Dict, List, Optional, Protocol, TypeVar, runtime_checkable
import numpy as np
from numpy.typing import NDArray
from scipy import sparse as sp
from sklearn.preprocessing import normalize

import bookgap.denoisers as denoisers
from bookgap.util import Filter, WorkType, TIMER, LOGGER

T = TypeVar('T', bound=np.generic)

@dataclass(frozen=True)
class ProjectiveDenoiserProjection:
    """Serialized payload describing a projective denoiser transformation."""

    projection_matrix: NDArray
    fit_mean: Optional[NDArray]

    def apply(self, embeddings: NDArray) -> NDArray:
        """Project embeddings into the cached denoised space."""

        if self.fit_mean is not None:
            embeddings = embeddings - self.fit_mean

        return embeddings @ self.projection_matrix


class CloudDenoiserContext(denoisers.DenoiserContext):
    def __init__(self, works: List[Dict[str, Any]], embeddings: NDArray, *filters: Filter):
        self.works = works
        self.filters = list(filters)
        self._A: Optional[sp.csr_matrix] = None
        self._X: NDArray = embeddings
        self._X_subsets: List[Optional[NDArray]] = [None] * len(self.filters)

    def A(self) -> sp.csr_matrix:
        if self._A is None:
            self._A = build_citation_graph(self.works)
        return self._A
    
    def X(self) -> NDArray:
        return self._X
    
    def X_n(self, n: int) -> NDArray:
        """Get embeddings for the n-th subset (0-indexed)."""
        if n >= len(self.filters):
            raise IndexError(f"Subset {n} requested, but only {len(self.filters)} subsets available")
        
        if self._X_subsets[n] is None:
            self._X_subsets[n] = self._X[self._get_mask(self.filters[n])]
        
        result = self._X_subsets[n]
        assert result is not None, f"Subset {n} should not be None after initialization"
        return result
    
    def num_subsets(self) -> int:
        """Return the number of subsets available."""
        return len(self.filters)
    
    def _get_mask(self, filter: Filter) -> NDArray:
        """Get a boolean mask for works that match the filter."""
        return np.array([filter.check_work(work) for work in self.works], dtype=bool)
    
@runtime_checkable
class WorkDenoiser(Protocol):
    """Protocol for denoising embeddings."""

    def compute_projection_artifact(
        self,
        drop_k: int,
        normalize_embeddings: bool,
        works: List[Dict[str, Any]],
        embeddings: NDArray[T],
    ) -> ProjectiveDenoiserProjection:
        """Return the projection metadata for downstream reuse."""
        ...

class NoneDenoiser:
    """No-op denoiser that returns original embeddings unchanged."""
    def compute_projection_artifact(
        self,
        drop_k: int,
        _works: List[Dict[str, Any]],
        embeddings: NDArray[T],
    ) -> ProjectiveDenoiserProjection:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}")
        if embeddings.shape[0] == 0:
            raise ValueError(f"Cannot build projection with zero embeddings")
        if drop_k != 0:
            raise ValueError(f"NoneDenoiser requires drop_k=0, got drop_k={drop_k}")

        original_dim = embeddings.shape[1]
        identity = np.eye(original_dim, dtype=embeddings.dtype)

        return ProjectiveDenoiserProjection(
            projection_matrix=identity,
            fit_mean=None,
        )




@TIMER.time(formatters=dict(works=lambda works: f"<{len(works)} works>"))
def build_citation_graph(works: List[Dict[str, Any]]) -> sp.csr_matrix:
    id_to_idx = {work["id"]: i for i, work in enumerate(works)}
    n = len(works)
    row = []
    col = []
    for i, work in enumerate(works):
        # Add self-reference for safe normalization and stability. Don't use append on the original list!
        refs: list[str] = work.get("referenced_works", []) + [work["id"]]  
        for ref_id in refs:
            if ref_id in id_to_idx:
                row.append(i)
                col.append(id_to_idx[ref_id])
    data = np.ones(len(row), dtype=np.float32)
    citation_graph = sp.csr_matrix((data, (row, col)), shape=(n, n))
    row_sums = np.asarray(citation_graph.sum(axis=1)).squeeze(1)
    assert row_sums.min() >= 1 # This is guaranteed by the self-references

    # Normalize rows so each citing work distributes equal total weight.
    citation_graph = normalize(citation_graph, norm="l1", axis=1, copy=False)
    
    assert type(citation_graph) is sp.csr_matrix, f"Expected csr_matrix, got {type(citation_graph)}"
    LOGGER.info(f"Built citation graph with shape {citation_graph.shape} and {citation_graph.nnz} non-zero entries")
    return citation_graph





class ProjectiveDenoiser:
    """Projective denoiser with full orchestration including plotting and analysis."""
    
    def __init__(self, denoiser_impl, *filters: Filter):
        self.denoiser_impl = denoiser_impl
        self.filters = filters

    def compute_projection_artifact(
        self,
        drop_k: int,
        works: List[Dict[str, Any]],
        embeddings: NDArray[T],
    ) -> ProjectiveDenoiserProjection:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}")
        if embeddings.shape[0] == 0:
            raise ValueError(f"Cannot build projection with zero embeddings")

        original_dim = embeddings.shape[1]
        if drop_k >= original_dim:
            raise ValueError(f"drop_k={drop_k} must be smaller than embedding dimension {original_dim}")

        n_components = original_dim - drop_k
        context = CloudDenoiserContext(works, embeddings, *self.filters)
        projector, eigvals = self.denoiser_impl.projector(context, n_components)

        projection_matrix = np.asarray(projector, dtype=np.float64, order="F")
        Q, R = np.linalg.qr(projection_matrix, mode="reduced")
        r_diag = np.abs(np.diag(R))
        if r_diag.size:
            tol = max(r_diag.max() * 1e-12, 1e-12)
            num_flat = int(np.count_nonzero(r_diag < tol))
            if num_flat:
                raise ValueError(f"Projector is rank deficient; {num_flat} of {r_diag.size} diagonal entries are < {tol}")
        projection_matrix = Q

        projection_matrix = np.ascontiguousarray(np.asarray(projection_matrix, dtype=embeddings.dtype))
        if projection_matrix.ndim != 2:
            raise ValueError(f"Expected projector to be 2D, got {projection_matrix.ndim}D array")
        if projection_matrix.shape[0] != original_dim:
            raise ValueError(
                f"Projector row dimension {projection_matrix.shape[0]} does not match embeddings dimension {original_dim}"
            )

        projected_dim = projection_matrix.shape[1]
        if projected_dim != n_components:
            raise ValueError(f"Requested {n_components} components but projector produced {projected_dim} components")

        mean_vector = np.ascontiguousarray(np.asarray(embeddings.mean(axis=0), dtype=embeddings.dtype))

        return ProjectiveDenoiserProjection(
            projection_matrix=projection_matrix,
            fit_mean=mean_vector,
        )
    


class WorkDenoisers(enum.Enum):
    # No-op denoiser
    NONE = NoneDenoiser()
    
    # Simple denoisers
    PCA = ProjectiveDenoiser(denoisers.PCADenoiser())
    GPCA = ProjectiveDenoiser(denoisers.GPCADenoiser())
    REL_AMP = ProjectiveDenoiser(denoisers.RelAmpDenoiser())
    # NEIGHBOR_DIST_MIN = ProjectiveDenoiser(denoisers.NeighborDistMinDenoiser())
    LPP = ProjectiveDenoiser(denoisers.LPPDenoiser())
    AUG_LPP = ProjectiveDenoiser(denoisers.AugmentedLPPDenoiser())
    ORTHO_MANIFOLD = ProjectiveDenoiser(denoisers.OrthogonalManifoldDenoiser())
    NORM_LPP = ProjectiveDenoiser(denoisers.NormalizedLPPDenoiser())
    # W = ProjectiveDenoiser(denoisers.WDenoiser())
    # LDA = ProjectiveDenoiser(denoisers.LDADenoiser())
    # SYM_LDA = ProjectiveDenoiser(denoisers.SymmetrizedLDADenoiser())
    PLS = ProjectiveDenoiser(denoisers.PLSModeADenoiser())
    CCA = ProjectiveDenoiser(denoisers.PLSModeBDenoiser())
    # W_LPP = ProjectiveDenoiser(denoisers.EmbeddedGraphDenoiser(denoisers.LPPDenoiser()))
    # W_NEIGHBOR_DIST_MIN = ProjectiveDenoiser(denoisers.EmbeddedGraphDenoiser(denoisers.NeighborDistMinDenoiser()))
    
    ANTI_CSP = ProjectiveDenoiser(
        denoisers.AntiCSPDenoiser(),
        Filter(work_types=(WorkType.BOOK.value,)),
        Filter(work_types=(WorkType.ARTICLE.value,)),
    )

    ANTI_LDA = ProjectiveDenoiser(denoisers.AntiLDADenoiser(), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    DOM_INVAR = ProjectiveDenoiser(denoisers.DomainInvariantDenoiser(), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,))) 
    
    CCLIP = ProjectiveDenoiser(denoisers.CCLIPDenoiser(alpha=0.5), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    CCLIP_1 = ProjectiveDenoiser(denoisers.CCLIPDenoiser(alpha=0.1), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    CCLIP_01 = ProjectiveDenoiser(denoisers.CCLIPDenoiser(alpha=0.01), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    CCP = ProjectiveDenoiser(denoisers.CCLIPDenoiser(alpha=0), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    LIP = ProjectiveDenoiser(denoisers.CCLIPDenoiser(alpha=1), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    
    # Joint Approximate Diagonalization denoisers
    # JAD_1_4 = ProjectiveDenoiser(denoisers.JADDenoiser(wt=1.0, wl=0.5, wsubset1=0.25, wsubset2=1.0), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))  # Optimal 1:4 book:article ratio
    # JAD_EQUAL = ProjectiveDenoiser(denoisers.JADDenoiser(wt=1.0, wl=1.0, wsubset1=1.0, wsubset2=1.0), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))  # Pre-sample-covariance optimal
    # JAD_PRIO_T = ProjectiveDenoiser(denoisers.JADDenoiser(wt=1.0, wl=0.5, wsubset1=0.5, wsubset2=0.5), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))  # Equal emphasis
    JAD_EQUAL = ProjectiveDenoiser(denoisers.JADDenoiser(wt=1.0, wl=1.0, w_conflate=1.0), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    JAD_HALF_L = ProjectiveDenoiser(denoisers.JADDenoiser(wt=1.0, wl=0.5, w_conflate=1.0), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    JAD_CCP = ProjectiveDenoiser(denoisers.JADDenoiser(wt=1.0, wl=0.0, w_conflate=1.0), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    JAD_LIP = ProjectiveDenoiser(denoisers.JADDenoiser(wt=1.0, wl=1.0, w_conflate=0.0), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return str(self)