from dataclasses import dataclass
import time
from typing import Collection, Optional, Union
import numpy as np
from numpy.typing import NDArray
from nextbook.cloud import Cloud, SubCloud
from nextbook.index import range_search_, range_search_radii_, FlatWithLims, from_flat_with_lims
from nextbook.util import TIMER, MEMORY, LOGGER, get_stats

@dataclass
class GetBallCoverageResult:
    index_works_per_query_work: NDArray
    ball_counts: NDArray  # Number of index works within the ball for each query work
    covered_points: list[tuple[NDArray[np.float32], NDArray[np.intp]]]  # List of (dists, ids) tuples

@TIMER.time()
@MEMORY.cache
def get_ball_coverage(index_subcloud: SubCloud, query_subcloud: SubCloud, radius_fraction: float) -> GetBallCoverageResult:
    """Get ball coverage statistics between two SubClouds."""
    # Both must be SubClouds of the same parent
    assert isinstance(index_subcloud, SubCloud) and isinstance(query_subcloud, SubCloud), "Both arguments must be SubClouds"
    assert index_subcloud.parent is query_subcloud.parent, "SubClouds must share the same parent Cloud"
    LOGGER.info(f"Finding closest query work to each index work (each query work gets a voronoi cell of index works)")

    # For each index_subcloud item, find the nearest query_subcloud item
    res = faiss_search_outside_epsilon(query_subcloud, index_subcloud, is_intra_cloud_search=False, k=1, epsilon=0)
    D = res.D_outside.squeeze(1)
    I = res.I_outside.squeeze(1)
    if query_subcloud.index().ntotal == 0:
        index_works_per_query_work = np.array([], dtype=np.intp)
    else:
        index_works_per_query_work = np.bincount(I, minlength=query_subcloud.index().ntotal)
    title = f"Number of index works covered by each query work (radius fraction: {radius_fraction})"
    LOGGER.info(f"{title} (voronoi):\n{get_stats(index_works_per_query_work)}")

    # Now, for each article, count how many other articles are within a radius defined by its distance to the nearest book.
    D_scaled = D * radius_fraction
    res = range_search_radii_(index_subcloud.index(), index_subcloud.embeddings(), D_scaled)
    assert res.lims is not None and res.dists is not None and res.ids is not None
    ball_counts = res.lims[1:] - res.lims[:-1]
    LOGGER.info(f"Number of index works covered by each index work (ball):\n{get_stats(ball_counts)}")
    dists_list = from_flat_with_lims(FlatWithLims(res.dists, res.lims))
    ids_list = from_flat_with_lims(FlatWithLims(res.ids, res.lims))
    covered_points = list(zip(dists_list, ids_list))
    return GetBallCoverageResult(
        index_works_per_query_work=index_works_per_query_work,
        ball_counts=ball_counts,
        covered_points=covered_points)

@dataclass(frozen=True)
class FaissSearchOutsideEpsilonResult:
    D_outside: NDArray
    I_outside: NDArray
    lims_within: NDArray
    dists_within: NDArray
    ids_within: NDArray

@MEMORY.cache
def faiss_search_outside_epsilon(
    index: Union[Cloud, SubCloud],
    query: Union[Cloud, SubCloud],
    is_intra_cloud_search: bool,
    k: int,
    epsilon: float,
) -> FaissSearchOutsideEpsilonResult:
    """
    Find the k nearest neighbors of query that are not (strictly) within epsilon of the query.
    """
    query_embeddings = query.embeddings()
    if epsilon == 0:
        if is_intra_cloud_search:
            # For intra-cloud, we need k+1 neighbors to discard the point itself.
            num_to_find = k + 1
            D_raw, I_raw = index.index().search(query_embeddings, num_to_find) # type: ignore
            
            # Check that the first neighbor is the point itself.
            # TODO may need to allow for some tolerance.
            if query.index().ntotal > 0:
                dist_to_self = D_raw[:, 0]
                max_dist_to_self = np.max(dist_to_self)
                atol = 1e-5
                if not np.isclose(max_dist_to_self, 0, atol=atol):
                    LOGGER.warning(f"Max distance to self was not close to 0: {max_dist_to_self:.6f}")
            
            # Check that distances are non-decreasing.
            assert np.all(np.diff(D_raw, axis=1) >= 0)

            # The results "outside epsilon" are the columns from index 1 onwards.
            D_outside = D_raw[:, 1:]
            I_outside = I_raw[:, 1:]

        else:
            # For inter-cloud search, a standard k-NN search is correct.
            D_outside, I_outside = index.index().search(query_embeddings, k) # type: ignore

        return FaissSearchOutsideEpsilonResult(
            D_outside=D_outside,
            I_outside=I_outside,
            lims_within=np.zeros((len(query_embeddings) + 1,), dtype=np.uint64),
            dists_within=np.empty((0,), dtype=np.float32),
            ids_within=np.empty((0,), dtype=np.int64))

    # Range search for all points strictly within epsilon
    LOGGER.info(f"Range searching for all points strictly within epsilon")
    start_time = time.time()
    range_search_result = range_search_(index.index(), query.embeddings(), epsilon)
    lims_within, dists_within, ids_within = range_search_result.lims, range_search_result.dists, range_search_result.ids
    assert lims_within is not None
    assert dists_within is not None
    assert ids_within is not None
    LOGGER.info(f"Range searching took {time.time() - start_time:.2f} seconds")
    assert lims_within[0] == 0
    num_within = lims_within[1:] - lims_within[:-1]
    assert len(num_within) == len(query_embeddings)
    assert np.all(num_within >= 0), f"Negative number of points within epsilon: {num_within}"

    # Batch search all queries with the same number of points "j" within epsilon
    D_outside = np.empty((len(query_embeddings), k), dtype=np.float32)
    I_outside = np.empty((len(query_embeddings), k), dtype=np.int64)
    for j in np.unique(num_within).tolist():
        query_indices = num_within == j # Could pre-sort, but index.search is the bottleneck
        D, I = index.index().search(query_embeddings[query_indices], j + k) # type: ignore
        # Discard the first j points, which are within epsilon
        D_outside[query_indices] = D[:, j:]
        I_outside[query_indices] = I[:, j:]

    return FaissSearchOutsideEpsilonResult(
        D_outside=D_outside,
        I_outside=I_outside,
        lims_within=lims_within,
        dists_within=dists_within,
        ids_within=ids_within,
    )

@dataclass(frozen=True)
class CrossCloudNNRanksResult:
    ranks: NDArray
    dists: Optional[NDArray]
    ids: Optional[NDArray]

    def __post_init__(self):
        assert self.ranks.ndim == 1
        if self.dists is not None:
            assert self.dists.ndim == 1
            assert len(self.dists) == len(self.ranks)
        if self.ids is not None:
            assert self.ids.ndim == 1
            assert len(self.ids) == len(self.ranks)

@TIMER.time()
@MEMORY.cache
def within_cloud_nn_ranks(
        ambient_subcloud: SubCloud,
        query_subcloud: SubCloud,
        return_dists: bool = False,
        return_ids: bool = False,
        k: int = 1):
    # Find nearest neighbor in query_subcloud for each query_subcloud point
    nns = faiss_search_outside_epsilon(query_subcloud, query_subcloud, is_intra_cloud_search=True, k=k, epsilon=0)
    D = nns.D_outside[:, -1]
    I = nns.I_outside[:, -1]
    
    # Sanity check: calculate squared distance to each such nearest neighbor
    query_embeddings = query_subcloud.embeddings()
    num_query_points = query_subcloud.index().ntotal
    if num_query_points > 0:
        sq_dists = np.linalg.norm(query_embeddings[I] - query_embeddings, axis=1) ** 2
        assert len(D) == len(I) == len(sq_dists) == num_query_points
        assert np.allclose(D, sq_dists)
    
    balls = range_search_radii_(
        ambient_subcloud.index(),
        query_embeddings,
        D,
        return_lims=True,
        return_dists=return_dists,
        return_ids=return_ids
    )
    assert balls.lims is not None
    ranks = balls.lims[1:] - balls.lims[:-1]
    assert len(ranks) == num_query_points, f"Ranks length mismatch: {len(ranks)} != {num_query_points}"
    assert np.all(ranks >= 0), f"Negative ranks found: {ranks[ranks < 0]}"
    return CrossCloudNNRanksResult(
        ranks=ranks,
        dists=balls.dists,
        ids=balls.ids
    )


@TIMER.time()
@MEMORY.cache
def cross_cloud_nn_ranks(
        ambient_subcloud: SubCloud,
        query_subcloud: SubCloud,
        return_dists: bool = False,
        return_ids: bool = False,
        k: int = 1):
    # Find nearest neighbor in index_subcloud for each query_subcloud point
    nns = faiss_search_outside_epsilon(ambient_subcloud, query_subcloud, is_intra_cloud_search=False, k=k, epsilon=0)
    D = nns.D_outside[:, -1]
    I = nns.I_outside[:, -1]
    # Calculate squared distance to each such nearest neighbor
    if ambient_subcloud.index().ntotal > 0:
        sq_dists = np.linalg.norm(ambient_subcloud.embeddings()[I] - query_subcloud.embeddings(), axis=1) ** 2
        assert len(D) == len(I) == len(sq_dists) == query_subcloud.index().ntotal
        assert np.allclose(D, sq_dists)
    balls = range_search_radii_(
        query_subcloud.index(),
        query_subcloud.embeddings(),
        D,
        return_lims=True,
        return_dists=return_dists,
        return_ids=return_ids
    )
    assert balls.lims is not None
    ranks = balls.lims[1:] - balls.lims[:-1]
    assert len(ranks) == query_subcloud.index().ntotal, f"Ranks length mismatch: {len(ranks)} != {query_subcloud.index().ntotal}"
    assert np.all(ranks >= 0), f"Negative ranks found: {ranks[ranks < 0]}"
    return CrossCloudNNRanksResult(
        ranks=ranks,
        dists=balls.dists,
        ids=balls.ids
    )

@dataclass(frozen=True)
class AvgrefsResult:
    avgref_embeddings: NDArray
    corresponding_query_idxs: NDArray[np.uint64]

@TIMER.time(formatters=dict(index_clouds=lambda x: f"<{len(x)} clouds>"))
@MEMORY.cache
def get_avgrefs(index_clouds: Collection[Cloud | SubCloud], query_cloud: Cloud | SubCloud, normalize_avgref_embeddings: bool) -> AvgrefsResult:
    index_id_to_cloud_idx = {work_id: (index_cloud, i) for index_cloud in index_clouds for i, work_id in enumerate(index_cloud.work_ids())}
    avgref_embeddings = []
    corresponding_query_idxs = []
    query_works = query_cloud.works()
    for query_idx, work in enumerate(query_works):
        referenced_embeddings = []
        for ref_id in work["referenced_works"]:
            if ref_id in index_id_to_cloud_idx:
                cloud, idx = index_id_to_cloud_idx[ref_id]
                referenced_embeddings.append(cloud.embeddings()[idx])
        if referenced_embeddings:
            avgref_embeddings.append(np.mean(referenced_embeddings, axis=0))
            corresponding_query_idxs.append(query_idx)
    frac_str = f"{len(avgref_embeddings) / len(query_works):.2%}" if len(query_works) > 0 else "N/A"
    LOGGER.info(f"Found {len(avgref_embeddings)} avgrefs for {len(query_works)} query works ({frac_str})")

    d = query_cloud.embeddings().shape[1]
    if len(avgref_embeddings) == 0:
        LOGGER.warning(f"No avgrefs found, returning empty array with shape (0, {d})")
        avgref_embeddings = np.empty((0, d), dtype=np.float32)
    else:
        avgref_embeddings = np.array(avgref_embeddings)
    assert avgref_embeddings.ndim == 2, f"Expected avgref_embeddings to be 2D, got {avgref_embeddings.ndim}D"
    assert avgref_embeddings.shape[1] == d, f"Expected avgref_embeddings shape[1] to be {d}, got {avgref_embeddings.shape[1]}"

    # Normalize avgref embeddings if requested
    avgref_norms = np.linalg.norm(avgref_embeddings, axis=1)
    assert avgref_norms.shape == (len(avgref_embeddings),), f"Expected avgref_norms shape {(len(avgref_embeddings),)}, got {avgref_norms.shape}"
    LOGGER.info(f"avgref norms: {get_stats(avgref_norms)}")
    if normalize_avgref_embeddings:
        threshold = 1e-6
        near_zero = avgref_norms < threshold
        if np.any(near_zero):
            small_norms = avgref_norms[near_zero]
            LOGGER.warning(f"{len(small_norms)} of {len(avgref_norms)} avgref embeddings have norms <= {threshold:.2e} and will not be normalized. Small norms: {get_stats(small_norms)}")
        avgref_embeddings[~near_zero] /= avgref_norms[~near_zero, np.newaxis]

    return AvgrefsResult(
        avgref_embeddings=avgref_embeddings,
        corresponding_query_idxs=np.array(corresponding_query_idxs, dtype=np.uint64)
    )

@dataclass(frozen=True)
class AvgrefRanksResult:
    avgref_res: AvgrefsResult
    per_cloud_ranks: list[NDArray[np.uint64]]

@MEMORY.cache
def avgref_ranks(index_clouds: Collection[Cloud | SubCloud], query: Cloud | SubCloud, normalize_avgref_embeddings: bool) -> AvgrefRanksResult:
    avgref_res = get_avgrefs(index_clouds, query, normalize_avgref_embeddings=normalize_avgref_embeddings)
    avgref_embeddings = avgref_res.avgref_embeddings
    corresponding_query_embeddings = query.embeddings()[avgref_res.corresponding_query_idxs]
    assert avgref_embeddings.shape == corresponding_query_embeddings.shape, f"{avgref_embeddings.shape} != {corresponding_query_embeddings.shape}"
    
    sq_dists = np.sum((avgref_embeddings - corresponding_query_embeddings) ** 2, axis=1)

    # For each index cloud, find avgref embedding ranks
    per_cloud_ranks: list[NDArray[np.uint64]] = []
    for index_cloud in index_clouds:
        res = range_search_radii_(index_cloud.index(), corresponding_query_embeddings, sq_dists, return_lims=True, return_dists=False, return_ids=False)
        assert res.lims is not None
        cur_cloud_ranks = res.lims[1:] - res.lims[:-1]
        assert cur_cloud_ranks.shape == (len(corresponding_query_embeddings),)
        per_cloud_ranks.append(cur_cloud_ranks)

    return AvgrefRanksResult(
        avgref_res=avgref_res,
        per_cloud_ranks=per_cloud_ranks
    )