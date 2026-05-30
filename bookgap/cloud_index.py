from dataclasses import dataclass
import time
from typing import Collection, Optional, Union, List
import numpy as np
from numpy.typing import NDArray
import tqdm

from bookgap.cloud import Cloud
from bookgap.corpus import Corpus
from bookgap.index import range_search_, range_search_radii_, FlatWithLims, from_flat_with_lims
from bookgap.subcloud import SubCloud
from bookgap.util import TIMER, MEMORY, LOGGER, get_stats

def get_kball_coverage(ambient: Corpus, landmark: Corpus, query: Corpus,
                       radius_fraction: float, k: int, is_intra_cloud: bool):
    """
    Get k-ball coverage statistics between three corpora.

    For each query point, find its k-th nearest landmark point, then find all ambient
    points within radius_fraction of that distance from the query point.

    Args:
        ambient: The ambient space to cover points in (e.g., articles)
        landmark: The landmarks used to define distances (e.g., books)
        query: The query points to analyze (e.g., articles for gap analysis)
        radius_fraction: Fraction of distance to k-th nearest landmark to use as radius
        k: Which nearest landmark to use (1 = nearest, 2 = second nearest, etc.)
        is_intra_cloud_search: Whether landmark and query are the same cloud

    Returns:
        GetKBallCoverageResult with coverage statistics
    """
    # ambient/landmark/query only need to share one embedding space — the same
    # contract detect_gaps enforces (equal embedder_fingerprint). The k-th-nearest
    # search and the range search consume nothing but .index()/.embeddings(), and
    # the returned ids are ambient-LOCAL (0..len(ambient)-1), so no shared .parent
    # or parent-wide index is required. This admits any Corpus (incl. UnionCorpus,
    # which has no .parent) while staying byte-identical on the SubCloud path.
    assert ambient.embedder_fingerprint == landmark.embedder_fingerprint == query.embedder_fingerprint, \
        "ambient/landmark/query must share one embedder (shared embedding space)"

    # For each query point, find the k-th nearest landmark point
    # res = kth_nearest(landmark, query, k=k)
    if is_intra_cloud:
        k += 1  # +1 to exclude self-neighbor
    res = knn(landmark, query, k=k)
    expected_shape = (len(query.embeddings()), k)
    assert res.sq_dists.shape == res.I.shape == expected_shape, f"Unexpected shape(s) {res.sq_dists.shape=}, {res.I.shape=}, {expected_shape=}"
    D = res.sq_dists[:, -1]  # Squared distance to k-th nearest landmark
    I = res.I[:, -1]  # Index of k-th nearest landmark
    assert D.shape == I.shape == (expected_shape[0],), f"Unexpected shape(s) {D.shape=}, {I.shape=}, {(expected_shape[0],)=}"

    # For each query point, find ambient points within radius_fraction of the squared distance to k-th nearest landmark
    res = range_search_radii_(ambient.index(), query.embeddings(),
                              radii=radius_fraction * D)
    
    assert res.lims is not None and res.dists is not None and res.ids is not None
    dists_list = from_flat_with_lims(FlatWithLims(res.dists, res.lims))
    ids_list = from_flat_with_lims(FlatWithLims(res.ids, res.lims))
    covered_points = list(zip(dists_list, ids_list))
    
    return covered_points

# def kballs_with(ambient: SubCloud, landmark: SubCloud, query: SubCloud, radius_fraction: float, k: int, is_intra_cloud: bool, kernel_fn: Callable[[int, NDArray, NDArray], None]) -> None:

@dataclass(frozen=True)
class FaissSearchOutsideEpsilonResult:
    sq_dists: NDArray[np.float32]
    I: NDArray[np.int64]
    lims_excluded: NDArray[np.uint64]
    sq_dists_excluded: NDArray[np.float32]
    I_excluded: NDArray[np.int64]

@MEMORY.cache
def knn(
    index: Union[Cloud, SubCloud, Corpus],
    query: Union[Cloud, SubCloud, Corpus],
    k: int = 1,
    radius_of_exclusion: float = 0.0,
) -> FaissSearchOutsideEpsilonResult:
    """
    Find the k nearest neighbors of query that are not (strictly) within radius_of_exclusion of the query.
    """
    query_embeddings = query.embeddings()
    if radius_of_exclusion == 0:
        D, I = index.index().search(query_embeddings, k) # type: ignore
        return FaissSearchOutsideEpsilonResult(
            sq_dists=D,
            I=I,
            lims_excluded=np.zeros((len(query_embeddings) + 1,), dtype=np.uint64),
            sq_dists_excluded=np.zeros((0,), dtype=np.float32),
            I_excluded=np.zeros((0,), dtype=np.int64))
    else:
        # Range search for all points strictly within epsilon
        LOGGER.info(f"Range searching for all points strictly within epsilon")
        start_time = time.time()
        range_search_result = range_search_(index.index(), query.embeddings(), radius_of_exclusion)
        lims_excluded, dists_excluded, ids_excluded = range_search_result.lims, range_search_result.dists, range_search_result.ids
        assert lims_excluded is not None
        assert dists_excluded is not None
        assert ids_excluded is not None
        LOGGER.info(f"Range searching took {time.time() - start_time:.2f} seconds")
        assert lims_excluded[0] == 0
        num_within = lims_excluded[1:] - lims_excluded[:-1]
        assert len(num_within) == len(query_embeddings)
        assert np.all(num_within >= 0), f"Negative number of points within epsilon: {num_within}"

        # Batch search all queries with the same number of points "j" within epsilon
        sq_dists = np.full(shape=(len(query_embeddings), k), fill_value=-1, dtype=np.float32)
        I = np.full(shape=(len(query_embeddings), k), fill_value=-1, dtype=np.int64)
        for j in np.unique(num_within).tolist():
            query_indices = num_within == j # Could pre-sort, but index.search is the bottleneck
            D, I = index.index().search(query_embeddings[query_indices], j + k) # type: ignore
            # Discard the first j points, which are within epsilon
            sq_dists[query_indices] = D[:, j:]
            I[query_indices] = I[:, j:]

        return FaissSearchOutsideEpsilonResult(
            sq_dists=sq_dists,
            I=I,
            lims_excluded=lims_excluded,
            sq_dists_excluded=dists_excluded,
            I_excluded=ids_excluded,
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
        ambient: SubCloud,
        query: SubCloud,
        return_dists: bool = False,
        return_ids: bool = False,
        k: int = 1,
        p: float = 1.0):
    assert p > 0, f"p must be positive, got {p}"
    # Find nearest neighbor in query for each query point
    nns = knn(query, query, k=k+1) # +1 to exclude self-neighbor
    D = nns.sq_dists[:, -1]
    I = nns.I[:, -1]
    
    # Sanity check: calculate squared distance to each such nearest neighbor
    query_embeddings = query.embeddings()
    num_query_points = query.index().ntotal
    if num_query_points > 0:
        sq_dists = np.sum((query_embeddings[I] - query_embeddings) ** 2, axis=1)
        assert len(D) == len(I) == len(sq_dists) == num_query_points, f"Length mismatch: {len(D)=}, {len(I)=}, {len(sq_dists)=}, {num_query_points=}"
        # assert np.allclose(D, sq_dists), f"Distance mismatch between FAISS and recomputed distances: max abs diff = {float(np.nanmax(np.abs(D - sq_dists)))}"
        diff = D - sq_dists

        # Allow tiny numeric drift between FAISS cached distances and recomputed values.
        FAISS_DISTANCE_RTOL = 1e-6
        FAISS_DISTANCE_ATOL = 1e-9
        if not np.allclose(D, sq_dists, rtol=FAISS_DISTANCE_RTOL, atol=FAISS_DISTANCE_ATOL):
            stats = get_stats(diff)
            msg = (
                "within_cloud_nn_ranks: Distance mismatch between FAISS and recomputed "
                "distances: stats="
                f"{stats} rtol={FAISS_DISTANCE_RTOL} atol={FAISS_DISTANCE_ATOL}"
            )
            LOGGER.error(msg)
            raise AssertionError(msg)
    
    radii = p * D

    balls = range_search_radii_(
        ambient.index(),
        query_embeddings,
        radii,
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
        ambient: SubCloud,
        query: SubCloud,
        return_dists: bool = False,
        return_ids: bool = False,
        k: int = 1):
    # Find nearest neighbor in index_subcloud for each query point
    nns = knn(ambient, query, k=k)
    D = nns.sq_dists[:, -1]
    I = nns.I[:, -1]
    # Calculate squared distance to each such nearest neighbor
    if ambient.index().ntotal > 0:
        sq_dists = np.sum((ambient.embeddings()[I] - query.embeddings()) ** 2, axis=1)
        assert len(D) == len(I) == len(sq_dists) == query.index().ntotal
        assert np.allclose(D, sq_dists)
    balls = range_search_radii_(
        query.index(),
        query.embeddings(),
        D,
        return_lims=True,
        return_dists=return_dists,
        return_ids=return_ids
    )
    assert balls.lims is not None
    ranks = balls.lims[1:] - balls.lims[:-1]
    assert len(ranks) == query.index().ntotal, f"Ranks length mismatch: {len(ranks)} != {query.index().ntotal}"
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
        avgref_embeddings_array = np.empty((0, d), dtype=np.float32)
    else:
        avgref_embeddings_array = np.array(avgref_embeddings)
    assert avgref_embeddings_array.ndim == 2, f"Expected avgref_embeddings_array to be 2D, got {avgref_embeddings_array.ndim}D"
    assert avgref_embeddings_array.shape[1] == d, f"Expected avgref_embeddings_array shape[1] to be {d}, got {avgref_embeddings_array.shape[1]}"

    # Normalize avgref embeddings if requested
    avgref_norms = np.linalg.norm(avgref_embeddings_array, axis=1)
    assert avgref_norms.shape == (len(avgref_embeddings_array),), f"Expected avgref_norms shape {(len(avgref_embeddings_array),)}, got {avgref_norms.shape}"
    LOGGER.info(f"avgref norms: {get_stats(avgref_norms)}")
    if normalize_avgref_embeddings:
        threshold = 1e-6
        near_zero = avgref_norms < threshold
        if np.any(near_zero):
            small_norms = avgref_norms[near_zero]
            LOGGER.warning(f"{len(small_norms)} of {len(avgref_norms)} avgref embeddings have norms <= {threshold:.2e} and will not be normalized. Small norms: {get_stats(small_norms)}")
        avgref_embeddings_array[~near_zero] /= avgref_norms[~near_zero, np.newaxis]

    return AvgrefsResult(
        avgref_embeddings=avgref_embeddings_array,
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

@dataclass(frozen=True)
class RangeSearchWithinMaxResult:
    """Result from range_search_within_max_radius."""
    distances: NDArray[np.float32]
    sorted_distances: NDArray[np.float32]
    max_radius: float

def range_search_within_max_radius(
    index: Union[Cloud, SubCloud],
    query_point: NDArray[np.float32],
    max_radius: float,
    sort_distances: bool = True
) -> RangeSearchWithinMaxResult:
    """
    Range search for all points within maximum radius from a single query point.
    
    Args:
        index: Index cloud to search in
        query_point: Single query point (1D array or 2D with shape (1, d))
        max_radius: Maximum radius to search within
        sort_distances: Whether to sort the returned distances
        
    Returns:
        RangeSearchWithinMaxResult with distances and optional sorted distances
    """
    if query_point.ndim == 1:
        query_point = query_point[np.newaxis, :]
    assert query_point.shape[0] == 1, f"Expected single query point, got shape {query_point.shape}"
    assert query_point.dtype == np.float32, f"query_point must be float32, got {query_point.dtype}"
    
    res = range_search_radii_(
        index.index(), 
        query_point, 
        np.array([max_radius], dtype=np.float32),
        return_dists=True, 
        return_ids=False, 
        return_lims=False
    )
    
    distances = np.asarray(res.dists, dtype=np.float32) if res.dists is not None else np.empty((0,), dtype=np.float32)
    sorted_distances = np.sort(distances) if sort_distances else distances
    
    return RangeSearchWithinMaxResult(
        distances=distances,
        sorted_distances=sorted_distances,
        max_radius=max_radius
    )

def counts_within_radii_linear(
    sorted_distances: NDArray[np.float32], 
    radii: NDArray[np.float32], 
    strict: bool = True
) -> NDArray[np.int32]:
    """
    Linear-time counts of distances within radii using two-pointer merge.
    Both arrays must be float32. Returns counts in original radii order.
    
    Args:
        sorted_distances: Sorted array of distances (float32)
        radii: Array of radii to count within (float32)
        strict: If True, count distances < radius (strict inequality)
               If False, count distances <= radius (non-strict inequality)
               
    Returns:
        Array of counts for each radius in original order
    """
    assert isinstance(sorted_distances, np.ndarray) and isinstance(radii, np.ndarray)
    assert sorted_distances.dtype == np.float32, f"sorted_distances dtype must be float32, got {sorted_distances.dtype}"
    assert radii.dtype == np.float32, f"radii dtype must be float32, got {radii.dtype}"
    
    if radii.size == 0:
        return np.empty((0,), dtype=np.int32)
    
    # Sort radii to enable linear sweep
    order = np.argsort(radii, kind='mergesort')
    r_sorted = radii[order]
    counts_sorted = np.empty_like(r_sorted, dtype=np.int32)
    
    # Two-pointer sweep
    i = 0
    n = sorted_distances.size
    for j, r in enumerate(r_sorted):
        if strict:
            while i < n and sorted_distances[i] < r:
                i += 1
        else:
            while i < n and sorted_distances[i] <= r:
                i += 1
        counts_sorted[j] = i
    
    # Restore original order
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return counts_sorted[inv]

@dataclass(frozen=True)
class BatchRangeSearchResult:
    """Result from batch_range_search_individual_radii."""
    all_neighbor_ids: List[List[int]]
    num_neighbors: NDArray[np.int32]

def batch_range_search_individual_radii(
    index: Union[Cloud, SubCloud],
    query_points: NDArray[np.float32],
    individual_radii: NDArray[np.float32],
    progress_desc: Optional[str] = None
) -> BatchRangeSearchResult:
    """
    Perform range search with individual radius for each query point.
    
    Args:
        index: Index cloud to search in
        query_points: Query points (2D array)
        individual_radii: Individual radius for each query point
        progress_desc: Optional description for progress bar
        
    Returns:
        BatchRangeSearchResult with neighbor lists and counts
    """
    assert query_points.shape[0] == individual_radii.shape[0], f"Mismatch: {query_points.shape[0]} queries vs {individual_radii.shape[0]} radii"
    assert query_points.dtype == np.float32, f"query_points must be float32, got {query_points.dtype}"
    assert individual_radii.dtype == np.float32, f"individual_radii must be float32, got {individual_radii.dtype}"
    
    all_neighbor_ids = []
    
    iterator = tqdm.tqdm(
        zip(query_points, individual_radii),
        total=len(individual_radii),
        desc=progress_desc or "Batch range search"
    )
    
    for point, radius in iterator:
        lims, dists, ids = index.index().range_search(point[np.newaxis], radius) # type: ignore
        zero, num = lims
        assert zero == 0, f"Range search lims[0] != 0: {zero}"
        assert num == len(dists) == len(ids), f"Range search: not {num=} == {len(dists)=} == {len(ids)=}"
        all_neighbor_ids.append(ids.tolist())
    
    num_neighbors = np.array([len(neighbors) for neighbors in all_neighbor_ids], dtype=np.int32)
    
    return BatchRangeSearchResult(
        all_neighbor_ids=all_neighbor_ids,
        num_neighbors=num_neighbors
    )

@dataclass(frozen=True)
class ProposalRankingResult:
    """Result from compute_proposal_rankings_for_gap."""
    book_ranks: NDArray[np.int32]
    article_ranks: NDArray[np.int32]
    proposal_distances: NDArray[np.float32]

def compute_proposal_rankings_for_gap(
    books: SubCloud,
    articles: SubCloud,
    gap_center: NDArray[np.float32],
    proposal_embeddings: NDArray[np.float32]
) -> ProposalRankingResult:
    """
    Compute rankings for proposal embeddings relative to books and articles from a gap center.
    
    Args:
        books: Book subcloud for ranking
        articles: Article subcloud for ranking  
        gap_center: Center point of the gap (single point)
        proposal_embeddings: Array of proposal embeddings to rank
        
    Returns:
        ProposalRankingResult with book ranks, article ranks, and distances
    """
    assert gap_center.shape[0] == 1, f"Expected single gap center, got shape {gap_center.shape}"
    assert gap_center.dtype == np.float32, f"gap_center must be float32, got {gap_center.dtype}"
    assert proposal_embeddings.dtype == np.float32, f"proposal_embeddings must be float32, got {proposal_embeddings.dtype}"
    assert gap_center.shape[1] == proposal_embeddings.shape[1], f"Dimension mismatch: {gap_center.shape[1]} vs {proposal_embeddings.shape[1]}"
    
    # Compute proposal distances from gap center
    diffs = proposal_embeddings - gap_center  # broadcast
    proposal_distances = np.einsum('ij,ij->i', diffs, diffs).astype(np.float32)
    max_radius = float(proposal_distances.max()) if len(proposal_distances) > 0 else 0.0
    
    # Range search in books
    if len(books.embeddings()) > 0 and max_radius > 0:
        book_result = range_search_within_max_radius(books, gap_center[0], max_radius, sort_distances=True)
        book_ranks = counts_within_radii_linear(book_result.sorted_distances, proposal_distances, strict=True)
    else:
        book_ranks = np.full(proposal_distances.shape, -1, dtype=np.int32)
    
    # Range search in articles
    if len(articles.embeddings()) > 0 and max_radius > 0:
        article_result = range_search_within_max_radius(articles, gap_center[0], max_radius, sort_distances=True)
        article_ranks = counts_within_radii_linear(article_result.sorted_distances, proposal_distances, strict=True)
    else:
        article_ranks = np.full(proposal_distances.shape, -1, dtype=np.int32)
    
    return ProposalRankingResult(
        book_ranks=book_ranks,
        article_ranks=article_ranks,
        proposal_distances=proposal_distances
    )