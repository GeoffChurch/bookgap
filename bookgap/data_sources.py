import abc
import argparse
from collections import defaultdict, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
import enum
from functools import lru_cache, partial
import logging
import matplotlib.pyplot as plt
import faiss
import itertools
import json
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import plotly.express as px
from bookgap.pyalex_client import pyalex
import re
# from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import torch
import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # type: ignore[reportPrivateImportUsage]
from typing import Collection, Dict, Iterable, List, Optional, Literal, Union#, override (Python 3.12+)
import pandas as pd
from scipy import sparse

from bookgap.cloud import Cloud, SubCloud, WorkDenoisers
from bookgap.cloud_index import get_avgrefs, get_kball_coverage, knn, batch_range_search_individual_radii
from bookgap.index import HNSW, FlatL2, IndexInfo, IndexType, build_index, range_search_radii_
from bookgap.parse_args import get_explicit_args, parse_args
from bookgap.util import ARTIFACTS_DIR, SCOPES_DIR, CACHE_DIR, LOGGER, TIMER, MEMORY, Filters, WorkType, Filter, create_unicode_safe_filename, get_stats, setup_logging
from bookgap.mp_ks_test import ks_2samp
from bookgap.deduplicators import FractionalDeduplicator, SqDistDeduplicator
# Import plotting utilities to eliminate duplication  
from bookgap.plot_utils import PlotManager, StatisticalPlotUtils, DPI

FIGSIZE = (10, 6)

@TIMER.time(formatters=dict(index_info=lambda _: "<...>", clouds=lambda clouds: f"<{len(clouds)} clouds, num works={[len(cloud.works()) for cloud in clouds]}>"))
@MEMORY.cache
def build_index_from_clouds(index_info: IndexInfo, clouds: Sequence[Cloud]) -> faiss.Index:
    """
    Build a single index from multiple clouds.
    This is useful for combining multiple topics or work types into a single index.
    """
    if len(clouds) == 1 and clouds[0].index_info == index_info:
        # If there's only one cloud and it matches the index_info, return its index directly
        LOGGER.info(f"Only one cloud provided, returning its index: {clouds[0].name}")
        return clouds[0].index()
    all_embeddings = np.concatenate([cloud.embeddings() for cloud in clouds], axis=0)
    LOGGER.info(f"Building index from {len(clouds)} clouds with {index_info.index_type} index. Total embeddings shape: {all_embeddings.shape}")
    index = build_index(index_info, all_embeddings)
    return index

@TIMER.time()
@MEMORY.cache
def get_nearest_neighbor(index_cloud: Cloud, query_cloud: Cloud, min_dist: float) -> tuple[NDArray, NDArray]:
    LOGGER.info(f"get_nearest_neighbor({index_cloud.name=}, {query_cloud.name=}, {min_dist=})")
    start_time = time.time()
    if min_dist == 0:
        # Search with k=2 to include self point and nearest non-self neighbor.
        D, I = index_cloud.index().search(index_cloud.embeddings(), k=2) # type: ignore
        D_first, D_second = D.T
        I_first, _I_second = I.T

        # Diagnostics for HNSW search quality
        identity_indices = np.arange(len(index_cloud.works()))
        
        # Discrepancy 1: Nearest neighbor is not the point itself.
        mismatch_mask = (I_first != identity_indices)
        
        # Discrepancy 2: Distance to first neighbor is not close to zero.
        atol = 1e-5
        nonzero_dist_mask = np.logical_not(np.isclose(D_first, 0, atol=atol))

        if np.any(mismatch_mask | nonzero_dist_mask):
            problem_indices = np.where(mismatch_mask | nonzero_dist_mask)[0]
            LOGGER.warning(f"{index_cloud.name} -- Found {len(problem_indices)} points where the first nearest neighbor was not the point itself or had distance not close to 0.")

            # Log the largest distance to the first neighbor
            if len(problem_indices) > 0:
                worst_case_idx = problem_indices[np.argmax(D_first[problem_indices])]
                work_i = index_cloud.works()[worst_case_idx]
                work_found = index_cloud.works()[I_first[worst_case_idx]]
                LOGGER.warning(
                    f"{index_cloud.name} -- Worst case: point {worst_case_idx} found neighbor {I_first[worst_case_idx]} "
                    f"with distance {D_first[worst_case_idx]:.4f} (should be close to 0).\n"
                    f"work_{worst_case_idx}:\n{json.dumps(work_i, indent=2)}\n"
                    f"found_neighbor_{I_first[worst_case_idx]}:\n{json.dumps(work_found, indent=2)}"
                )

            is_tie = np.isclose(D_first[problem_indices], D_second[problem_indices])
            significant_error_indices = problem_indices[~is_tie]

            if len(significant_error_indices) > 0:
                LOGGER.warning(f"{index_cloud.name} -- Found {len(significant_error_indices)} significant HNSW errors (mismatches that are not near-ties).")
                # Log the most egregious case (largest distance to the incorrect first neighbor)
                worst_case_idx = significant_error_indices[np.argmax(D_first[significant_error_indices])]
                
                work_i = index_cloud.works()[worst_case_idx]
                work_found = index_cloud.works()[I_first[worst_case_idx]]
                LOGGER.warning(
                    f"{index_cloud.name} -- Worst error example: point {worst_case_idx} found neighbor {I_first[worst_case_idx]} "
                    f"with distance {D_first[worst_case_idx]:.4f} (second neighbor dist: {D_second[worst_case_idx]:.4f}).\n"
                    f"work_{worst_case_idx}:\n{json.dumps(work_i, indent=2)}\n"
                    f"found_neighbor_{I_first[worst_case_idx]}:\n{json.dumps(work_found, indent=2)}"
                )
        
        # The actual nearest neighbor is the second result.
        Dnn = D_second
        # The original return signature is preserved for the caller in main().
        return Dnn, I
    else:
        res = knn(index_cloud, query_cloud, radius_of_exclusion=min_dist)
        LOGGER.info(f"{index_cloud.name} -- Nearest neighbor search took {time.time() - start_time:.2f} seconds")
        # Dnn, = res.D_outside.T
        # I = res.I_outside
        Dnn, I = res.sq_dists.squeeze(1), res.I.squeeze(1)
    return Dnn, I

@TIMER.time()
@MEMORY.cache
def get_rank_of_nearest_incloud_neighbor(selfcloud: Cloud, othercloud: Cloud, min_dist: float):
    nn_dir = CACHE_DIR / "nearest_neighbors" / create_unicode_safe_filename(f"{selfcloud.name} vs {othercloud.name}, min_dist: {min_dist}")
    nn_dir.mkdir(parents=True, exist_ok=True)
    
    selfcloud_ids = [work["id"] for work in selfcloud.works()]
    othercloud_ids = [work["id"] for work in othercloud.works()]

    selfcloud_ids_file = nn_dir / "selfcloud_ids.json"
    othercloud_ids_file = nn_dir / "othercloud_ids.json"
    def validate_ids() -> bool:
        if not selfcloud_ids_file.exists() or not othercloud_ids_file.exists():
            return False
        with selfcloud_ids_file.open("r") as sci_f, othercloud_ids_file.open("r") as oci_f:
            loaded_selfcloud_ids = json.load(sci_f)
            loaded_othercloud_ids = json.load(oci_f)
        if len(loaded_selfcloud_ids) != len(selfcloud_ids):
            LOGGER.warning(f"{selfcloud.name} -- Selfcloud ID length mismatch: {len(loaded_selfcloud_ids)} != {len(selfcloud_ids)}")
            return False
        if len(loaded_othercloud_ids) != len(othercloud_ids):
            LOGGER.warning(f"{selfcloud.name} -- Othercloud ID length mismatch: {len(loaded_othercloud_ids)} != {len(othercloud_ids)}")
            return False
        if (num_mismatches := np.count_nonzero(np.array(loaded_selfcloud_ids) != np.array(selfcloud_ids))) > 0:
            LOGGER.warning(f"{selfcloud.name} -- Selfcloud mismatched IDs count: {num_mismatches}")
            return False
        if (num_mismatches := np.count_nonzero(np.array(loaded_othercloud_ids) != np.array(othercloud_ids))) > 0:
            LOGGER.warning(f"{selfcloud.name} -- Othercloud mismatched IDs count: {num_mismatches}")
            return False
        LOGGER.debug(f"{selfcloud.name} -- IDs match")
        return True
    ids_match = validate_ids()
    if not ids_match:
        selfcloud_ids_file.unlink(missing_ok=True)
        othercloud_ids_file.unlink(missing_ok=True)

    # Get nearest neighbors
    Dnn, I = get_nearest_neighbor(selfcloud, selfcloud, min_dist)

    closer_points_file = nn_dir / "closer_points.json"
    def load_range_search():
        if not ids_match or not closer_points_file.exists():
            return None
        with closer_points_file.open("r") as f:
            return json.load(f)
    def build_range_search():
        LOGGER.info(f"{selfcloud.name} -- Range searching for all points in {othercloud.name} < distance to nearest neighbor")
        closer_points_file.unlink(missing_ok=True)
        start_time = time.time()
        
        # Use consolidated batch range search function
        result = batch_range_search_individual_radii(
            othercloud, 
            selfcloud.embeddings(), 
            Dnn.astype(np.float32), 
            progress_desc=f"Range search in {othercloud.name} of {selfcloud.name} points"
        )
        closer_points = result.all_neighbor_ids
        
        LOGGER.info(f"{selfcloud.name} -- Range searching took {time.time() - start_time:.2f} seconds")
        # Save range search results
        with closer_points_file.open("w") as f:
            json.dump(closer_points, f)
        return closer_points
    closer_points = load_range_search() or build_range_search()
    
    num_closer = np.array([len(closer) for closer in closer_points])
    
    artifacts_dir = SCOPES_DIR / create_unicode_safe_filename(f"{selfcloud.name} vs {othercloud.name}") / f"min_dist: {min_dist}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Save statistics to file
    with (artifacts_dir / f"stats.json").open("w") as f:
        json.dump({
            "mean": np.mean(num_closer).item(),
            "median": np.median(num_closer).item(),
            "min": np.min(num_closer).item(),
            "max": np.max(num_closer).item(),
            "std": np.std(num_closer).item(),
            "num_points": len(num_closer),
            "num_points_in_other_cloud": len(othercloud.works()),
        }, f, indent=2)
    # # Log statistics
    # logger.info(f"Cloud {selfcloud.name} -- Statistics of number of points in {othercloud.name} within distance to nearest neighbor in {selfcloud.name}:\n"
    #              f"Mean: {np.mean(num_closer):.2f}\n"
    #              f"Median: {np.median(num_closer):.2f}\n"
    #              f"Min: {np.min(num_closer):.2f}\n"
    #              f"Max: {np.max(num_closer):.2f}\n"
    #              f"Std: {np.std(num_closer):.2f}\n"
    #              f"Num points: {len(num_closer)}\n"
    #              f"Num points in other cloud: {len(othercloud.items)}\n")
    counts = np.bincount(num_closer)
    # Build histogram of num_closer
    fig, hist_ax = PlotManager.create_figure(figsize=FIGSIZE)
    hist_ax.bar(range(len(counts)), counts)
    hist_ax.set_xlabel(f"Number of {othercloud} works between neighboring {selfcloud} works")
    hist_ax.set_ylabel("Count")
    hist_ax.set_xscale("log")
    hist_ax.set_yscale("log")
    title = f"{othercloud} works between neighboring {selfcloud} works"
    if othercloud.filter_pair != selfcloud.filter_pair:
        LOGGER.warning(f"{selfcloud.name} -- Topic mismatch: {selfcloud.filter_pair} != {othercloud.filter_pair}")
        title = f"{othercloud.filter_pair} {title}"
    hist_ax.set_title(title)
    kde_ax = hist_ax.twinx()
    kde = gaussian_kde(num_closer)
    x = np.linspace(0, len(counts), 1000)
    kde_ax.plot(x, kde(x), color="red", label="KDE")
    # kde_ax.legend()
    PlotManager.save_and_close(fig, artifacts_dir / f"hist.png")

    # Save ids
    with selfcloud_ids_file.open("w") as sci_f, othercloud_ids_file.open("w") as oci_f:
        json.dump(selfcloud_ids, sci_f)
        json.dump(othercloud_ids, oci_f)

# def get_points_by_coverage_voronoi(index_cloud: Cloud, query_cloud: Cloud, min_dist: float, output_dir: Path):
#     if index_cloud.topic != query_cloud.topic:
#         logger.warning(f"Topic mismatch: {index_cloud.topic} != {query_cloud.topic}")
#     logger.info(f"{index_cloud.topic} -- Calculating coverage of hypothetical {query_cloud.work_type}s in {index_cloud.work_type}s")
#     I_orig = faiss_search_outside_epsilon(index_cloud.index, query_cloud.embeddings, 1, min_dist).I_outside
#     logger.info(f"Statistics of nearest neighbors in {index_cloud.work_type}:\n{get_stats(np.unique(I_orig, return_counts=True)[1])}")
#     query = np.empty((query_cloud.embeddings.shape[0] + 1, query_cloud.embeddings.shape[1]), dtype=np.float32)
#     query[:-1] = query_cloud.embeddings
#     covered_points = []
#     for i, embedding in enumerate(tqdm.tqdm(index_cloud.embeddings, desc="Voronoi coverage")):
#         query[-1] = embedding
#         I = faiss_search_outside_epsilon(index_cloud.index, query, 1, min_dist).I_outside.squeeze(1)
#         covered_points.append(np.where(I == i)[0])
#         if covered_points[-1].size > 1:
#             logger.debug(f"Point {i} -- {len(covered_points[-1])} {index_cloud.work_type}s covered by {index_point['display_name']} {index_point['id']}")
#         index_point = index_cloud.items[i]
#     argsort = np.argsort([len(covered) for covered in covered_points])[::-1]
#     # Log the top 10 points
#     for i in argsort[:10]:
#         index_point = index_cloud.items[i]
#         covered = covered_points[i]
#         logger.info(f"Point {i} -- {len(covered)} {index_cloud.work_type}s covered by {index_point['display_name']} {index_point['id']}")



# def plot_avgref(index_clouds: Collection[Cloud], query_cloud: Cloud, method: L
#     logger.info(f"plot_avgref({query_cloud.name=}) for {len(index_clouds)} index clouds")
#     avgref = get_avgref(index_clouds, query_cloud)
#     # Plotly interactive plot
#     query_embeddings, pca = reduce_dimensions_pca([query_cloud], dimensions=2)
#     avgref_embeddings = pca.transform(avgref)
    
@TIMER.time(formatters=dict(index_clouds=lambda x: f"<{len(x)} clouds>"))
def do_avgref_analysis(index_clouds: Sequence[Cloud], query_cloud: Cloud) -> None:
    assert len(index_clouds) == 1
    LOGGER.info(f"do_avgref_analysis({query_cloud.name=}) for {len(index_clouds)} index clouds")
    
    artifacts_dir = SCOPES_DIR / str(query_cloud.filter_pair) / "avgref_analysis"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    avgref_result = get_avgrefs(index_clouds, query_cloud, normalize_avgref_embeddings=query_cloud.normalize_embeddings)
    avgref_embeddings = avgref_result.avgref_embeddings
    avgref_idxs = avgref_result.corresponding_query_idxs
    
    # avgref_norms = np.linalg.norm(avgref_embeddings, axis=1, keepdims=True)
    # LOGGER.info(f"{query_cloud.name} -- Avgref norms: {get_stats(avgref_norms)}")
    # if query_cloud.normalize_embeddings:
    #     LOGGER.info(f"{query_cloud.name} -- Normalizing avgref embeddings")
    #     if np.any(avgref_norms < 1e-6):
    #         LOGGER.warning(f"{query_cloud.name} -- Some avgref embeddings have zero norm, will not normalize those.")
    #         non_zero_mask = (avgref_norms > 1e-6).squeeze(1) # Use a small epsilon for floating point
    #         LOGGER.info(f"{avgref_embeddings.shape=}, {avgref_norms.shape=}, {non_zero_mask.shape=}")
    #         LOGGER.info(f"{avgref_embeddings.dtype=}, {avgref_norms.dtype=}, {non_zero_mask.dtype=}")
    #         avgref_embeddings[non_zero_mask] /= avgref_norms[non_zero_mask]
    #     else:
    #         avgref_embeddings /= avgref_norms

    LOGGER.info(f"{query_cloud.name} -- Found {len(avgref_embeddings)} avgrefs for {len(query_cloud.works())} query works ({len(avgref_embeddings) / len(query_cloud.works()) * 100:.2f}%)")
    embeddings_with_avgrefs = query_cloud.embeddings()[avgref_idxs]

    # Average distance from centroid
    centroid = np.mean(embeddings_with_avgrefs, axis=0)
    embedding_dists_to_centroid = np.linalg.norm(embeddings_with_avgrefs - centroid, axis=1)
    avgref_dists_to_centroid = np.linalg.norm(avgref_embeddings - centroid, axis=1) ** 2
    fig, ax = PlotManager.create_figure(figsize=FIGSIZE)
    all_data = np.concatenate([embedding_dists_to_centroid, avgref_dists_to_centroid])
    bins = np.histogram_bin_edges(all_data, bins=200)
    ax.stairs(*np.histogram(embedding_dists_to_centroid, bins=bins, density=False), fill=True, color='lightblue', alpha=0.7, label='Squared distance from works to centroid of works')
    ax.stairs(*np.histogram(avgref_dists_to_centroid, bins=bins, density=False), color='red', alpha=0.7, label='Squared distance from avgrefs to centroid of works')
    ax.set_xlabel("Squared Euclidean distance to centroid")
    ax.set_ylabel("Count")
    ax.set_title(f"Squared Euclidean distance to centroid for {query_cloud.name}")
    ax.legend()
    PlotManager.save_and_close(fig, artifacts_dir / "squared_distance_to_centroid.png")

    # Distribution of squared distances to avgref
    sq_dists = np.linalg.norm(embeddings_with_avgrefs - avgref_embeddings, axis=1) ** 2
    fig, ax = PlotManager.create_figure(figsize=FIGSIZE)
    ax.hist(sq_dists, bins=50, density=False, alpha=0.7, color='green', label='Squared Distance to avgref')

    # Overlay histogram of randomized permutation of idxs
    random_idxs = np.random.permutation(len(sq_dists))
    # random_embeddings_with_avgrefs = embeddings_with_avgrefs[random_idxs]
    # assert sorted(random_embeddings_with_avgrefs) == sorted(embeddings_with_avgrefs), "Randomized embeddings are not a permutation of original embeddings"
    random_avgref_embeddings = avgref_embeddings[random_idxs]
    assert np.all(random_avgref_embeddings.sort() == avgref_embeddings.sort()), "Randomized avgref embeddings are not a permutation of original avgref embeddings"
    # random_sq_dists = np.linalg.norm(random_embeddings_with_avgrefs - avgref_embeddings, axis=1) ** 2
    random_sq_dists = np.linalg.norm(embeddings_with_avgrefs - random_avgref_embeddings, axis=1) ** 2
    assert random_sq_dists.shape == sq_dists.shape, f"Randomized squared distances shape mismatch: {random_sq_dists.shape} != {sq_dists.shape}"
    ax.hist(random_sq_dists, bins=50, density=False, alpha=0.5, color='orange', label='Squared distance to random avgref')
    ax.legend()
    ax.set_xlabel("Squared Euclidean distance to avgref")
    ax.set_ylabel("Count")
    ax.set_title(f"Squared Euclidean distance to avgref for {query_cloud.name} ({len(avgref_embeddings)} avgrefs, {len(query_cloud.works())} query works)")
    fig.savefig(artifacts_dir / "euclidean_distance_distribution.png", dpi=DPI)
    plt.close(fig)
    # Distribution of rank of avgref in index clouds
    # Range search for each query point using the distance to avgref
    # index_cloud, = index_clouds
    
    # # Find subset of index_cloud that has at least one reference in index_cloud
    # _, index_avgref_ids = get_avgref(index_clouds, index_cloud)
    # index_id_to_idx = {work["id"]: i for i, work in enumerate(index_cloud.works())}
    # index_avgref_idxs = [index_id_to_idx[ref_id] for ref_id in index_avgref_ids if ref_id in index_id_to_idx]
    # # Build index on that subset
    # subset_index = build_index(index_cloud.index_info, index_cloud.embeddings()[index_avgref_idxs])
    # # ORIGINAL
    # # res = range_search_radii_(index_cloud.index(), embeddings_with_avgrefs, dists)
    # # SUBSET
    # res = range_search_radii_(subset_index, embeddings_with_avgrefs, random_sq_dists)

    # Build index on the full index cloud
    index_info = index_clouds[0].index_info
    assert all(cloud.index_info == index_info for cloud in index_clouds), "Index info mismatch across clouds"
    full_index = build_index_from_clouds(index_info, index_clouds)
    LOGGER.info(f"{query_cloud.name} -- Range searching for true avgrefs in index cloud {index_clouds[0].name} with {full_index.ntotal} items")
    res = range_search_radii_(full_index, embeddings_with_avgrefs, sq_dists, return_lims=True, return_dists=False, return_ids=False)
    assert res.lims is not None # For type checking

    # ORIGINAL
    # random_res = range_search_radii_(index_cloud.index(), random_embeddings_with_avgrefs, random_dists)
    # SUBSET
    # random_res = range_search_radii_(subset_index, random_embeddings_with_avgrefs, random_sq_dists)
    LOGGER.info(f"{query_cloud.name} -- Range searching for random avgrefs in index cloud {index_clouds[0].name} with {full_index.ntotal} items")
    random_res = range_search_radii_(full_index, embeddings_with_avgrefs, random_sq_dists, return_lims=True, return_dists=False, return_ids=False)
    assert random_res.lims is not None # For type checking

    ranks = res.lims[1:] - res.lims[:-1]
    random_ranks = random_res.lims[1:] - random_res.lims[:-1]
    LOGGER.info(f"{query_cloud.name} -- Rank of avgref with respect to index works:\n{get_stats(ranks)}")
    LOGGER.info(f"{query_cloud.name} -- Rank of random avgref with respect to index works:\n{get_stats(random_ranks)}")

    # --- Write statistics to global CSV ---
    # Prepare stats for both avgref and random avgref
    key_columns = [
        'topic',
        'index_info',
        'deduplicator',
        'normalize_embeddings',
        'drop_k',
        'denoiser',
    ]
    csv_path = Path("artifacts") / "avgref_rank_stats" / "avgref_rank_stats.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame()
    existing_keys = set(tuple(row) for row in df[key_columns].to_numpy()) if not df.empty else set()

    # Compute stats for both nonrandom and random
    stats_nonrand = get_stats(ranks)
    stats_rand = get_stats(random_ranks)
    ks = ks_2samp(ranks, random_ranks, method='exact')
    LOGGER.info(f"{query_cloud.name} -- KS test between ranks of avgref and random avgref: {ks}")
    row = {
        'topic': str(query_cloud.filter_pair),
        'index_info': str(index_info),
        'deduplicator': str(query_cloud.deduplicator) if query_cloud.deduplicator else 'None',
        'normalize_embeddings': query_cloud.normalize_embeddings,
        'drop_k': query_cloud.drop_k,
        'denoiser': str(query_cloud.denoiser),
        'mean_nonrand': stats_nonrand['mean'],
        'median_nonrand': stats_nonrand['median'],
        'min_nonrand': stats_nonrand['min'],
        'max_nonrand': stats_nonrand['max'],
        'std_nonrand': stats_nonrand['std'],
        'count_nonrand': stats_nonrand['count'],
        'num_avgrefs': len(avgref_embeddings),
        'num_query_works': len(query_cloud.works()),
        'mean_rand': stats_rand['mean'],
        'median_rand': stats_rand['median'],
        'min_rand': stats_rand['min'],
        'max_rand': stats_rand['max'],
        'std_rand': stats_rand['std'],
        'count_rand': stats_rand['count'],
        'ks_log10_pvalue': ks.log10_pvalue,
    }
    key_tuple = tuple(row[k] for k in key_columns)
    # if key_tuple not in existing_keys:
    if key_tuple in existing_keys:
        LOGGER.info(f"{query_cloud.name} -- Found existing row for {' '.join(f'{k}={row[k]}' for k in key_columns)}")
    new_df = pd.DataFrame([row])
    if not df.empty:
        assert list(df.columns) == list(new_df.columns), f"Column mismatch between existing and new DataFrame: {list(df.columns)=} != {list(new_df.columns)=}"
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df
    df = df.sort_values(by="mean_nonrand", ascending=False)
    df.to_csv(csv_path, index=False, float_format='%.3f')
    export_avgref_stats_to_latex()
    # --- End CSV output ---

    # CDF of ranks
    cumsum = np.cumsum(np.bincount(ranks.astype(np.int64))) / len(ranks)
    cumsum_random = np.cumsum(np.bincount(random_ranks.astype(np.int64))) / len(random_ranks)

    # Plot CDF of ranks
    fig, ax = PlotManager.create_figure(figsize=FIGSIZE)
    # ax.plot(np.arange(len(cumsum)), cumsum, label='Rank of avgref', color='green')
    # ax.plot(np.arange(len(cumsum_random)), cumsum_random, label='Rank of random avgref', color='orange')
    ax.scatter(np.arange(len(cumsum)), cumsum, label='Rank of avgref', color='green', s=10, zorder=2)
    ax.scatter(np.arange(len(cumsum_random)), cumsum_random, label='Rank of random avgref', color='orange', s=10, zorder=1)

    # Overlay the ks statistic
    # ks = kstest(ranks, random_ranks)
    LOGGER.info(f"{query_cloud.name} -- KS test between ranks of avgref and random avgref: {ks}")
    lower, higher = (cumsum_random, cumsum) if ks.statistic_sign == 1 else (cumsum, cumsum_random)
    ymin = lower[ks.statistic_location].item()
    ymax = higher[ks.statistic_location].item()
    assert np.isclose(ymax - ymin, ks.statistic), f"KS statistic mismatch: {ymax - ymin} != {ks.statistic} at location {ks.statistic_location}"
    LOGGER.info(f"{query_cloud.name} -- KS statistic location: {ks.statistic_location}, ymin: {ymin}, ymax: {ymax}")
    # Overlay KS max diff
    ax.vlines(ks.statistic_location,
               ymin=ymin,
               ymax=ymax,
               color='purple', linestyle='dashed', linewidth=1, label=f"Max diff of {ymax-ymin:.2f} at rank {ks.statistic_location} (KS p-value = 10^{ks.log10_pvalue:.1f})",
                zorder=0)

    # # Overlay average ranks
    # ax.axvline(average_rank, color='blue', linestyle='dashed', linewidth=1, label=f"Average Rank = {average_rank:.2f} ({average_rank / max(ranks) * 100:.1f}% of max)", zorder=3)
    # ax.axvline(average_rank_random, color='red', linestyle='dashed', linewidth=1, label=f"Average Random Rank = {average_rank_random:.2f} ({average_rank_random / max(random_ranks) * 100:.1f}% of max)", zorder=3)


    ax.set_xlabel("Rank of avgref in index cloud")
    ax.set_ylabel("Cumulative count")
    ax.set_title(f"CDF of ranks of avgref for {len(avgref_embeddings)} {query_cloud.name} works")
    ax.legend()
    fig.savefig(artifacts_dir / "rank_cdf.png", dpi=DPI)
    ax.set_xscale("log")
    fig.savefig(artifacts_dir / "rank_cdf_logx.png", dpi=DPI)
    # ax.set_yscale("log")
    plt.close(fig)

    
    fig, ax = PlotManager.create_figure(figsize=FIGSIZE)
    
    # ax.hist(ranks, bins=200, density=False, alpha=0.7, color='green', label='Rank of avgref')
    # ax.hist(random_ranks, bins=200, density=False, alpha=0.5, color='orange', label='Rank of random avgref')
    all_data = np.concatenate([ranks, random_ranks])
    bins = np.histogram_bin_edges(all_data, bins=200)
    ax.stairs(*np.histogram(ranks, bins=bins, density=False), fill=True, alpha=0.7, color='green', label='Rank of avgref')
    ax.stairs(*np.histogram(random_ranks, bins=bins, density=False), fill=True, alpha=0.5, color='orange', label='Rank of random avgref')
    # # Overlay expected rank
    # expected_rank = len(index_avgref_idxs) / 2
    # ax.axvline(expected_rank, color='red', linestyle='dashed', linewidth=1, label=f"Expected Rank ({len(index_avgref_idxs)} / 2 = {expected_rank})")
    ax.legend()
    ax.set_xlabel("Rank of avgref in index cloud")
    ax.set_ylabel("Count")
    ax.set_title(f"# works closer than avgref for {len(avgref_embeddings)} {query_cloud.name} works in {full_index.ntotal} works")
    fig.savefig(artifacts_dir / "rank_distribution.png", dpi=DPI)
    ax.set_xscale("log")
    fig.savefig(artifacts_dir / "rank_distribution_logx.png", dpi=DPI)
    # ax.set_yscale("log")
    plt.close(fig)


    # # Sanity check: the rank of a random work wrt another random work should be distributed uniformly from 0 to len(index_cloud.works())
    # # sanity_choices1 = np.random.choice(len(query_cloud.works()), size=len(query_cloud.works()), replace=True)
    # # sanity_choices2 = np.random.choice(len(index_cloud.works()), size=len(index_cloud.works()), replace=True)
    # sanity_choices1 = np.arange(len(query_cloud.works())) # Use all query works
    # sanity_choices2 = np.random.permutation(len(index_cloud.works())) # Randomly permute index works
    # assert sanity_choices1.shape == sanity_choices2.shape, f"Sanity choices shape mismatch: {sanity_choices1.shape} != {sanity_choices2.shape}"
    # assert sanity_choices1.shape == (len(index_cloud.works()),), f"Sanity choices shape mismatch: {sanity_choices1.shape} != ({len(query_cloud.works()),})"
    # assert index_cloud.embeddings().shape == index_cloud.embeddings()[sanity_choices1].shape, f"Index cloud embeddings shape mismatch: {index_cloud.embeddings().shape} != {index_cloud.embeddings()[sanity_choices1].shape}"
    # assert index_cloud.embeddings().shape == index_cloud.embeddings()[sanity_choices2].shape, f"Index cloud embeddings shape mismatch: {index_cloud.embeddings().shape} != {index_cloud.embeddings()[sanity_choices2].shape}"
    # sanity_sq_dists = np.linalg.norm(index_cloud.embeddings()[sanity_choices1] - index_cloud.embeddings()[sanity_choices2], axis=1) ** 2
    # assert sanity_sq_dists.shape == (len(index_cloud.works()),), f"Sanity squared distances shape mismatch: {sanity_sq_dists.shape} != ({len(index_cloud.works()),})"
    # sanity_res = range_search_radii_(
    #     index_cloud.index(),
    #     index_cloud.embeddings()[sanity_choices1],
    #     sanity_sq_dists
    # )
    # logger.info(f"{query_cloud.name} -- Sanity check rank of random work with respect to {index_cloud.name} works:\nlims ({sanity_res.lims.shape=}): {sanity_res.lims[:10]} ... {sanity_res.lims[-10:]}")
    # sanity_ranks = sanity_res.lims[1:] - sanity_res.lims[:-1]
    # logger.info(f"{query_cloud.name} -- Sanity check rank of random work with respect to {index_cloud.name} works:\n{get_stats(sanity_ranks)}")
    # fig, ax = plt.subplots(figsize=FIGSIZE)
    # ax.hist(sanity_ranks, bins=200, density=False, alpha=0.7, color='blue', label='Rank of random work')
    # sanity_expected_rank = len(index_cloud.works()) / 2
    # ax.axvline(sanity_expected_rank, color='red', linestyle='dashed', linewidth=1, label=f'Expected Rank ({len(index_cloud.works())} / 2 = {sanity_expected_rank})') 
    # ax.legend()
    # ax.set_xlabel("Rank of random work in index cloud")
    # ax.set_ylabel("Count")
    # ax.set_title(f"# works closer than random work for {len(index_cloud.works())} {index_cloud.name} works")
    # fig.savefig(artifacts_dir / "sanity_rank_distribution.png", dpi=DPI)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # fig.savefig(artifacts_dir / "sanity_rank_distribution_log.png", dpi=DPI)
    # plt.close(fig)


class InferenceManager:
    def __init__(self):
        self.pipe = loadPipeline()

    def run(self, prompt: str, output_path: Path):
        start_time = time.time()
        llama_output = self.pipe(prompt, max_new_tokens=4096)
        elapsed_time = time.time() - start_time
        assert isinstance(llama_output, list)
        assert len(llama_output) == 1
        assert isinstance(llama_output[0], dict)
        assert set(llama_output[0].keys()) == {"generated_text"}
        assert len(llama_output[0]['generated_text']) >= len(prompt)
        generated_text = llama_output[0]['generated_text'][len(prompt):]
        with open(output_path, "w") as f:
            f.write(generated_text)
        LOGGER.info(f"Inference took {elapsed_time:.2f} seconds for prompt of length {len(prompt)}, output length {len(generated_text)}. Saved to {output_path}")

@TIMER.time()
@lru_cache(maxsize=1)
def loadPipeline():
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Use "nf4" for NormalFloat 4-bit
        bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computation
        bnb_4bit_use_double_quant=True, # Optional: for slightly higher precision
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    LOGGER.info(f"Loading {model_id}")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16, # Match the compute dtype
    )
    LOGGER.info(f"Model {model_id} loaded in {time.time() - start_time:.2f} seconds")

    # 5. Create the text-generation pipeline with your pre-loaded, quantized model
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return pipe

def output_covered_points(hole_rank: int, highest_coverage_id: np.intp, covered_dists: NDArray, covered_ids: NDArray, original_coverage: int, index_cloud: Union[Cloud, SubCloud], query_cloud: Union[Cloud, SubCloud], artifacts_dir: Path):
    assert len(covered_dists) == len(covered_ids), f"Covered dists and ids length mismatch: {len(covered_dists)} != {len(covered_ids)}"
    topic_dir = artifacts_dir / "highest coverage (ball)"
    topic_dir.mkdir(parents=True, exist_ok=True)
    
    order = np.argsort(covered_dists)
    # order = np.random.permutation(len(covered_dists))
    covered_ids = covered_ids[order]
    covered_dists = covered_dists[order]

    sample_idxs = np.random.choice(len(covered_ids) - 1, size=min(29, len(covered_ids) - 1), replace=False)
    sample_idxs = np.concatenate(([0], sample_idxs + 1)) # + 1 to skip 0
    assert np.unique(sample_idxs).size == sample_idxs.size, f"Sample indices are not unique: {sample_idxs}"
    sample_ids = covered_ids[sample_idxs]
    sample_dists = covered_dists[sample_idxs]

    # Collect titles and abstracts of the top 10 covered articles
    top_articles = []
    for sample_id in sample_ids:
        covered_point = index_cloud.works()[sample_id]
        # top_articles.append(f"TITLE: {covered_point['display_name']}\nABSTRACT: {covered_point['abstract']}")
        top_articles.append({"title": covered_point['display_name'], "abstract": covered_point['abstract']})
    articles_text = json.dumps(top_articles, indent=2, ensure_ascii=False)

    with open(topic_dir / f"{hole_rank}_articles.txt", "w") as f:
        f.write(articles_text)

    # Write details file for the center
    with open(topic_dir / f"{hole_rank}_details.txt", "w") as f:
        # Write the original coverage
        f.write(f"Original gap size: {original_coverage}\n")
        f.write(f"Gap size: {len(covered_ids)}\n")
        f.write(f"Centroid item (id={index_cloud.works()[highest_coverage_id]['id']}):\n")
        f.write(f"TITLE: {index_cloud.works()[highest_coverage_id]['display_name']}\n")
        f.write(f"ABSTRACT: {index_cloud.works()[highest_coverage_id]['abstract']}\n\n")

        center_embedding = index_cloud.embeddings()[highest_coverage_id]
        # # Find the nearest book to the center (highest_coverage_id)
        # book_res = faiss_search_outside_epsilon(query_cloud, index_cloud, k=1, epsilon=0)
        # nearest_book_idx = book_res.I_outside[highest_coverage_id, 0]
        # nearest_book = query_cloud.works()[nearest_book_idx]
        # f.write(f"Nearest book to center (id={nearest_book['id']}):\n")
        # f.write(f"TITLE: {nearest_book['display_name']}\n")
        # f.write(f"ABSTRACT: {nearest_book['abstract']}\n\n")

        # # Find the nearest in-ball article to that book
        # book_embedding = query_cloud.embeddings()[nearest_book_idx]
        # # Only consider in-ball articles (covered_ids)
        # in_ball_embeddings = index_cloud.embeddings()[covered_ids]
        # dists = np.linalg.norm(in_ball_embeddings - book_embedding, axis=1)
        # nearest_inball_idx = np.argmin(dists)
        # nearest_inball_id = covered_ids[nearest_inball_idx]
        # nearest_inball_article = index_cloud.works()[nearest_inball_id]
        # f.write(f"Nearest in-ball article to this book (id={nearest_inball_article['id']}):\n")
        # f.write(f"TITLE: {nearest_inball_article['display_name']}\n")
        # f.write(f"ABSTRACT: {nearest_inball_article['abstract']}\n\n")

        # For every in-ball article, find its nearest book
        f.write("Distances to nearest in-ball article and to center for each book:\n")
        book_to_nearest_inball = defaultdict(list)
        book_to_idx: dict[str, int] = dict()
        for in_ball_id in covered_ids:
            book_res = knn(query_cloud, index_cloud)
            nearest_book_idx = book_res.I[in_ball_id, 0]
            nearest_book = query_cloud.works()[nearest_book_idx]
            book_to_nearest_inball[nearest_book['id']].append(in_ball_id)
            if nearest_book['id'] not in book_to_idx:
                book_to_idx[nearest_book['id']] = nearest_book_idx
        

        num_books = len(book_to_nearest_inball)
        f.write(f"Number of unique books that are the nearest book to at least one in-ball article (number of book Voronoi cells intersected by ball): {num_books}\n")
        
        TOP_K = 5

        # Print the top closest such books to the center embedding
        f.write(f"Top {TOP_K} books closest to the center item:\n")
        book_to_nearest_inball_increasing_distance = sorted(book_to_nearest_inball.items(),
                                                            key=lambda x: np.linalg.norm(query_cloud.embeddings()[book_to_idx[x[0]]] - center_embedding))
        for i, (book_id, in_ball_ids) in enumerate(book_to_nearest_inball_increasing_distance[:TOP_K]):
            book_idx = book_to_idx[book_id]
            book = query_cloud.works()[book_idx]
            f.write(f"Book {i+1} of {num_books} (id={book['id']}):\n")
            f.write(f"\tNumber of in-ball articles in Voronoi cell: {len(in_ball_ids)} ({len(in_ball_ids) / len(covered_ids) * 100:.2f}% of ball):\n")
            f.write(f"\tTITLE: {book['display_name']}\n")
            f.write(f"\tABSTRACT: {book['abstract']}\n")
            # Sort in-ball articles by distance to this book
            in_ball_embeddings = index_cloud.embeddings()[in_ball_ids]
            dists = np.linalg.norm(in_ball_embeddings - query_cloud.embeddings()[book_idx], axis=1)
            sorted_in_ball_ids = np.array(in_ball_ids)[np.argsort(dists)]
            nearest_article = index_cloud.works()[sorted_in_ball_ids[0]]
            f.write(f"Nearest in-ball article to this book (id={nearest_article['id']}):\n")
            f.write(f"\tTITLE: {nearest_article['display_name']}\n")
            f.write(f"\tABSTRACT: {nearest_article['abstract']}\n")
            f.write("\n")
        f.write("\n\n")
        # Print the books that cover the most in-ball articles
        book_to_nearest_inball_decreasing_coverage = sorted(book_to_nearest_inball.items(),
                                                            key=lambda x: len(x[1]),
                                                            reverse=True)
        f.write("Books that cover the most in-ball articles:\n")
        for i, (book_id, in_ball_ids) in enumerate(book_to_nearest_inball_decreasing_coverage[:TOP_K]):
            book = query_cloud.works()[book_to_idx[book_id]]
            f.write(f"Book {i+1} of {num_books} (id={book['id']}):\n")
            f.write(f"\tNumber of in-ball articles in Voronoi cell: {len(in_ball_ids)} ({len(in_ball_ids) / len(covered_ids) * 100:.2f}% of ball):\n")
            f.write(f"TITLE: {book['display_name']}\n")
            f.write(f"ABSTRACT: {book['abstract']}\n")
            # Find the nearest in-ball article to this book
            in_ball_embeddings = index_cloud.embeddings()[in_ball_ids]
            dists = np.linalg.norm(in_ball_embeddings - query_cloud.embeddings()[book_to_idx[book_id]], axis=1)
            nearest_in_ball_idx = np.argmin(dists)
            nearest_article = index_cloud.works()[in_ball_ids[nearest_in_ball_idx]]
            f.write(f"\tNearest in-ball article to this book (id={nearest_article['id']}):\n")
            f.write(f"\tTITLE: {nearest_article['display_name']}\n")
            f.write(f"\tABSTRACT: {nearest_article['abstract']}\n")
            f.write("\n")

    # Get top works closest to centroid
    with open(topic_dir / f"{hole_rank}_center.txt", "w") as f:
        for idx in range(min(20, len(covered_dists))):
            dist = covered_dists[idx]
            article = index_cloud.works()[covered_ids[idx]]
            f.write(f"{dist:.4f} -- {article['display_name']} -- {article['abstract']}\n")
    # # Compute convex hull
    # covered_embeddings = index_cloud.embeddings()[covered_ids]
    # if covered_embeddings.shape[0] > covered_embeddings.shape[1]:
    #     try:
    #         hull = ConvexHull(covered_embeddings)
    #         hull_idxs = hull.vertices
    #     except Exception as e:
    #         logger.warning(f"Convex hull computation failed: {e}")
    #         hull_idxs = np.arange(len(covered_embeddings))
    # else:
    #     logger.warning("Not enough points for convex hull; using all points.")
    #     hull_idxs = np.arange(len(covered_embeddings))
    # with open(top10_dir / f"{hole_rank}_convex_hull.txt", "w") as f:
    #     for idx in hull_idxs:
    #         dist = covered_dists[idx]
    #         article = index_cloud.works()[covered_ids[idx]]
    #         f.write(f"{dist:.4f} -- {article['display_name']} -- {article['abstract']}\n")


@TIMER.time(formatters=dict(cloud=lambda cloud: f"<{cloud.name}>"))
@MEMORY.cache
def get_points_by_coverage_ball(cloud: Cloud, index_work_type: str, query_work_type: str, radius_fraction: float):
    # Create SubClouds for the two work types
    index_subcloud = SubCloud(parent=cloud, post_filter=Filter(work_types=(index_work_type,)))
    query_subcloud = SubCloud(parent=cloud, post_filter=Filter(work_types=(query_work_type,)))
    
    filter_pair = cloud.filter_pair
    artifacts_dir = SCOPES_DIR / f"{filter_pair}" / f"{index_work_type}s covered by {query_work_type}s"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # index_cloud = get_cloud(topic, index_work_type, **kwargs) 
    # query_cloud = get_cloud(topic, query_work_type, **kwargs)
    # index_cloud = Cloud(topic=topic, work_type=index_work_type, index_info=kwargs["index_info"], max_works=kwargs["max_works"])
    # query_cloud = Cloud(topic=topic, work_type=query_work_type, index_info=kwargs["index_info"], max_works=kwargs["max_works"])
    # output_dir = ARTIFACTS_DIR / create_unicode_safe_filename(f"{index_cloud.name} vs {query_cloud.name}") / f"min_dist: {min_dist}, radius_fraction: {radius_fraction}"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # if index_cloud.topic != query_cloud.topic:
    #     logger.warning(f"Topic mismatch: {index_cloud.topic} != {query_cloud.topic}")
    # logger.info(f"{index_cloud.topic} -- Calculating coverage of hypothetical {query_cloud.work_type}s in {index_cloud.work_type}s")
    # logger.info(f"{topic} -- Finding closest {query_work_type} to each {index_work_type} (each {query_work_type} gets a voronoi cell of {index_work_type}s)")
    # res = faiss_search_outside_epsilon(query_cloud, index_cloud, k=1, epsilon=0)
    # D = res.D_outside.squeeze(1)
    # I = res.I_outside.squeeze(1)
    # _idxs, index_works_per_query_work = np.unique(I, return_counts=True)
    # # Add in zero counts for empty voronoi cells
    # index_works_per_query_work = np.concatenate((index_works_per_query_work, np.zeros(index_cloud.index().ntotal - len(index_works_per_query_work), dtype=np.int32)))
    # title = f"{topic} - Number of {index_work_type}s covered by each {query_work_type}"
    # logger.info(f"{title} (voronoi):\n{get_stats(index_works_per_query_work)}")
    
    

    # # Now get stats for the balls
    # D_scaled = D * radius_fraction
    # res = range_search_radii_(index_cloud.index(), query_cloud.embeddings(), D_scaled)
    # ball_counts = res.lims[1:] - res.lims[:-1]
    # logger.info(f"{title} (ball):\n{get_stats(ball_counts)}")

    covered_points = get_kball_coverage(ambient=index_subcloud, landmark=query_subcloud, 
                                        query=index_subcloud, radius_fraction=radius_fraction, 
                                        k=1, is_intra_cloud=False)
    # title = f"{filter_pair} - Number of {index_work_type}s covered by each {query_work_type}"

    # fig, ax = plt.subplots(figsize=FIGSIZE)
    # all_data = np.concatenate([index_works_per_query_work, ball_counts])
    # bins = np.histogram_bin_edges(all_data, bins=50)
    # ax.stairs(*np.histogram(index_works_per_query_work, bins=bins), color="lightblue", fill=True, zorder=1, label=f"Number of {index_work_type}s covered by each {query_work_type} (voronoi)")
    # ax.stairs(*np.histogram(ball_counts, bins=bins), color="orange", fill=False, zorder=2, label=f"Number of {index_work_type}s covered by each {index_work_type} (ball of radius {radius_fraction} * dist to nearest {query_work_type})")
    # ax.set_xlabel(f"Number of {index_work_type}s covered")
    # ax.set_ylabel(f"Frequency")
    # ax.set_title(title)
    # ax.set_yscale("log")
    # ax.legend(loc="upper right")
    # fig.savefig(artifacts_dir / f"{index_work_type}s covered per {query_work_type}.png", dpi=DPI)
    # plt.close(fig)


    # Iterate over the top 10 points, deleting covered points from the list
    LOGGER.info(f"Determining top {index_work_type}s by number of {index_work_type}s within {radius_fraction} * distance to nearest {query_work_type}")
    num_centers = 5
    # original_coverage = np.array([len(ids) for _dists, ids in covered_points])
    def coverage(covered_points) -> NDArray[np.int64]:
        return np.array([len(ids) for _dists, ids in covered_points])
    coverages = [coverage(covered_points)]
    for rank in tqdm.tqdm(range(num_centers), desc=f"Outputting highest coverage {index_work_type}s"):
        highest_coverage_id = np.argmax([len(ids) for _dists, ids in covered_points])
        dists, ids = covered_points[highest_coverage_id]
        output_covered_points(rank, highest_coverage_id, dists, ids, coverages[0][highest_coverage_id], index_subcloud, query_subcloud, artifacts_dir)
        # Remove the covered points from the list
        if highest_coverage_id not in ids:
            LOGGER.warning(f"Top point {highest_coverage_id} not in its own covered points")
        # Remove the covered points from the list
        for covered_i in ids:
            covered_points[covered_i] = (np.empty(0, dtype=dists.dtype), np.empty(0, dtype=ids.dtype))
        # Remove the covered points from each remaining point's list
        ids = np.unique(ids)
        for coverage_to_plot in range(len(covered_points)):
            i_dists, i_ids = covered_points[coverage_to_plot]
            to_keep = np.isin(i_ids, ids)
            covered_points[coverage_to_plot] = (i_dists[~to_keep], i_ids[~to_keep])
        coverages.append(coverage(covered_points))

    # Plot a histogram comparing old to new ball coverages
    fig, ax = PlotManager.create_figure(figsize=FIGSIZE)
    # pruned_coverage = np.array([len(ids) for _dists, ids in covered_points])
    # all_data = np.concatenate([original_coverage, pruned_coverage])
    all_data = np.concatenate(coverages)
    assert all_data.ndim == 1, f"all_data should be 1D, got {all_data.shape=}"
    bins = np.histogram_bin_edges(all_data, bins=50)
    # ax.stairs(*np.histogram(pruned_coverage, bins=bins), color="lightblue", fill=True, zorder=1, label="New coverage")
    # ax.stairs(*np.histogram(original_coverage, bins=bins), color="black", fill=False, zorder=2, label="Original coverage")

    # def i_to_rgb(i: int) -> tuple[float, float, float, float]:
    #     v = i / (len(coverages) - 1)
    #     return (v, v, v, 0.9)
    # for i, cov in enumerate(coverages):
    #     # ax.stairs(*np.histogram(cov, bins=bins), color=f"C{i}", fill=True, alpha=0.7, zorder=i+1, label=f"Coverage after pruning {i} {index_work_type}s")
    #     # Grayscale
    #     ax.stairs(*np.histogram(cov, bins=bins), color=i_to_rgb(i), fill=True, zorder=i+1, label=f"Coverage after pruning {i} {index_work_type}s")

    coverages_to_plot = sorted(set([min(i, len(coverages) - 1)
                                    for i in [0, 1, 2, len(coverages) - 1]]))

    cmap = plt.get_cmap("coolwarm")
    def i_to_color(i: int) -> tuple[float, float, float, float]:
        num_cycles = 1
        mod = len(coverages_to_plot) // num_cycles
        v = (i % mod) / (mod - 1)
        ret = cmap(v, alpha=0.75)
        # darkener = (i // mod) / (num_cycles - 1)
        # darkener *= 0.5
        # scalar = 1 - darkener
        # return (ret[0] * scalar, ret[1] * scalar, ret[2] * scalar, ret[3])
        return ret
    for i, coverage_to_plot in enumerate(coverages_to_plot):
        ax.stairs(*np.histogram(coverages[coverage_to_plot], bins=bins), color=i_to_color(i), fill=True, zorder=coverage_to_plot+1, label=f"Coverage after pruning {coverage_to_plot} {index_work_type}s")

    ax.set_xlabel(f"Number of {index_work_type}s covered")
    ax.set_ylabel(f"Number of {index_work_type}s with a given coverage of other {index_work_type}s")
    ax.set_title(f"{filter_pair} - Number of {index_work_type}s covered by each {index_work_type}")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    fig.savefig(artifacts_dir / f"{index_work_type}s covered per {index_work_type} (pruning overlaid).png", dpi=DPI)
    plt.close(fig)

    # # Plot a histogram comparing old to new voronoi coverages
    # fig, ax = plt.subplots(figsize=FIGSIZE)
    # ax.stairs(*np.histogram(original_coverage, bins=100), color="lightblue", fill=True, zorder=1, label="Original coverage")
    # # ax1 = ax.twinx()

def export_avgref_stats_to_latex():
    directory = Path("artifacts") / "avgref_rank_stats"
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "avgref_rank_stats.csv"
    if not csv_path.exists():
        LOGGER.info(f"CSV file {csv_path} does not exist, skipping LaTeX export.")
        return
    df = pd.read_csv(csv_path)
    LOGGER.info(f"df:\n{df.head()}")
    LOGGER.info(f"columns: {df.columns.tolist()}")
    
    plot_x_column = "drop_k"
    plot_series_column = "denoiser"
    plot_y_column = "mean_nonrand"

    key_columns = ["topic"]
    
    col_map = {
        "denoiser": "denoiser",
        "drop_k": "dimensions dropped",
        "mean_nonrand": "mean rank",
        "median_nonrand": "median rank",
    }
    
    # For each combination of other_columns, export LaTeX and plot
    for combo in df[key_columns].drop_duplicates().itertuples(index=False):
        subtable = df.copy()
        for col, value in zip(key_columns, combo):
            subtable = subtable[subtable[col] == value]
        LOGGER.info(f"Exporting LaTeX for combination {combo}")
        tex_path = directory / f"avgref_rank_stats_{'_'.join([f'{col}={value}' for col, value in zip(key_columns, combo)])}.tex"
        subtable.drop(columns=key_columns).rename(columns=col_map).to_latex(tex_path, index=False, float_format="%.1f", escape=False)
        LOGGER.info(f"Exported LaTeX to {tex_path}")
        LOGGER.info(f"Exporting plot for combination {combo}")
        fig, ax = PlotManager.create_figure(figsize=FIGSIZE)
        for denoiser, group in subtable.groupby(plot_series_column):
            group = group.sort_values(by=plot_x_column)
            ax.plot(group[plot_x_column], group[plot_y_column], marker='o', alpha=0.7, label=denoiser)
        ax.set_xlabel(col_map[plot_x_column])
        ax.set_ylabel(col_map[plot_y_column])
        ax.set_title(f"{combo.topic} - {col_map[plot_x_column]} vs {col_map[plot_y_column]} by {col_map[plot_series_column]}")
        ax.legend(title=col_map[plot_series_column])
        PlotManager.save_and_close(fig, directory / f"avgref_rank_stats_{'_'.join(map(str, combo))}.png")
    # # De-dup based on key columns
    # df = df.drop_duplicates(subset=["topic", "drop_k", "denoiser"])

    # df = df[list(col_map.keys())].rename(columns=col_map)
    # LOGGER.info(f"Topics: {df['topic'].unique()}")
    # for topic in df["topic"].unique():
    #     LOGGER.info(f"Exporting avgref stats to LaTeX for topic {topic}")
    #     topic_df = df[df["topic"] == topic].drop(columns=["topic"])
    #     tex_path = Path("artifacts") / f"avgref_rank_stats_{topic}.tex"
    #     topic_df.to_latex(tex_path, index=False, float_format="%.1f", escape=False)
    #     LOGGER.info(f"Exported avgref stats to LaTeX for topic {topic}: {tex_path}")

# @MEMORY.cache
# def reduce_dimensions(clouds: Iterable[Cloud], dimensions: int, method: Literal["pca", "tsne"]) -> NDArray:
#     embeddings = np.concatenate([cloud.embeddings() for cloud in clouds], axis=0)
#     logger.info(f"Reducing {embeddings.shape[0]} embeddings from {embeddings.shape[1]} to {dimensions} dimensions using {method.upper()}")
#     start_time = time.time()
#     if method == "pca":
#         pca = PCA(n_components=dimensions)
#         transformed_embeddings = pca.fit_transform(embeddings)
#     elif method == "tsne":
#         tsne = TSNE(n_components=dimensions, perplexity=30, max_iter=1000)
#         transformed_embeddings = tsne.fit_transform(embeddings)
#     else:
#         raise ValueError(f"Unknown method: {method}")
#     logger.info(f"{method.upper()} took {time.time() - start_time:.2f} seconds")
#     return transformed_embeddings

@TIMER.time()
@MEMORY.cache
def reduce_dimensions_pca(clouds: Iterable[Cloud], dimensions: int) -> tuple[NDArray, PCA]:
    embeddings = np.concatenate([cloud.embeddings() for cloud in clouds], axis=0)
    LOGGER.info(f"Reducing {embeddings.shape[0]} embeddings from {embeddings.shape[1]} to {dimensions} dimensions using PCA")
    pca = PCA(n_components=dimensions)
    transformed_embeddings = pca.fit_transform(embeddings)
    return transformed_embeddings, pca

@MEMORY.cache
def reduce_dimensions_tsne(clouds: Iterable[Cloud], dimensions: int) -> tuple[NDArray, TSNE]:
    embeddings = np.concatenate([cloud.embeddings() for cloud in clouds], axis=0)
    LOGGER.info(f"Reducing {embeddings.shape[0]} embeddings from {embeddings.shape[1]} to {dimensions} dimensions using t-SNE")
    start_time = time.time()
    tsne = TSNE(n_components=dimensions, perplexity=30, max_iter=1000)
    transformed_embeddings = tsne.fit_transform(embeddings)
    LOGGER.info(f"t-SNE took {time.time() - start_time:.2f} seconds")
    return transformed_embeddings, tsne

def plot_embeddings(clouds: Iterable[Cloud], colored_feature: str, method: Literal["pca", "tsne"], dimensions: int = 2, color_f = lambda colors: colors, **kwargs):
    """
    Plot the embeddings of the works in a 2D or 3D space using PCA or t-SNE, colored by the specified feature.
    Outputs a plotly html file.
    """
    assert dimensions in {2, 3}, f"Dimensions must be 2 or 3, got {dimensions}"
    # topics, work_types = list(topics), list(work_types)
    # assert topics == sorted(topics, key=lambda t: t.value), f"Topics must be sorted: {topics}"
    # assert work_types == sorted(work_types, key=lambda t: t.value), f"Work types must be sorted: {work_types}"
    artifacts_dir = ARTIFACTS_DIR / "plots" / create_unicode_safe_filename(colored_feature)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # clouds = {
    #     (topic, work_type): Cloud(topic=topic, work_type=work_type, index_info=kwargs["index_info"], max_works=kwargs["max_works"])
    #     for topic, work_type in itertools.product(topics, work_types)
    # }
    works = [work for cloud in clouds for work in cloud.works()]
    
    if method == "pca":
        transformed_embeddings, _ = reduce_dimensions_pca(clouds, dimensions)
    elif method == "tsne":
        transformed_embeddings, _ = reduce_dimensions_tsne(clouds, dimensions)
    else:
        raise ValueError(f"Unknown method: {method}")

    plot_title_prefix = ""
    labels = {}

    cloud_names = ", ".join(sorted(cloud.name() for cloud in clouds))

    if method == "pca":
        plot_title_prefix = f"{cloud_names} PCA embeddings colored by {colored_feature}"
        labels = {"x": "PCA 1", "y": "PCA 2"}
        if dimensions == 3:
            labels["z"] = "PCA 3"
        output_file_suffix = f"_pca_{dimensions}d.html"
    elif method == "tsne":
        plot_title_prefix = f"{cloud_names} t-SNE embeddings colored by {colored_feature}"
        labels = {"x": "t-SNE 1", "y": "t-SNE 2"}
        output_file_suffix = "_tsne.html"
    else:
        raise ValueError(f"Unknown method: {method}")

    scatter_args = dict(
        x=transformed_embeddings[:, 0],
        y=transformed_embeddings[:, 1],
        color=color_f([work[colored_feature] for work in works]),
        hover_name=[f"{work['display_name']} ({work['id']})\\n{colored_feature}: {work[colored_feature]}" for work in works],
        title=plot_title_prefix,
        labels=labels,
    )
    if dimensions == 3:
        scatter_args["z"] = transformed_embeddings[:, 2]
    
    scatter_method = px.scatter if dimensions == 2 else px.scatter_3d

    # fig = scatter_method(**{k: v for k, v in scatter_args.items() if k in inspect.signature(scatter_method).parameters})
    fig = scatter_method(**scatter_args) # type: ignore

    fig.update_traces(marker=dict(size=5) if dimensions == 2 else dict(size=3, opacity=0.7))

    output_file = artifacts_dir / f"{create_unicode_safe_filename(cloud_names)}{output_file_suffix}"
    fig.write_html(output_file)

def plot_norms(clouds: Collection[Cloud]) -> None:
    # Plot histogram of norms of all embeddings in the clouds
    embeddings = np.concatenate([cloud.embeddings() for cloud in clouds], axis=0)
    LOGGER.info(f"Plotting norms of {embeddings.shape} embeddings from {len(clouds)} clouds")
    norms = np.linalg.norm(embeddings, axis=1)
    LOGGER.info(f"Norms stats:\n"
                f"\tmin: {norms.min()}\n"
                f"\tmed: {np.median(norms)}\n"
                f"\tmax: {norms.max()}\n"
                f"\tmean: {norms.mean()}\n"
                f"\tstd: {norms.std()}\n"
                )
    assert norms.shape == (embeddings.shape[0],)
    fig, ax = PlotManager.create_figure(figsize=FIGSIZE)
    ax.hist(norms, bins=100)
    ax.set_xlabel("Norm")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Norms of Embeddings")
    PlotManager.save_and_close(fig, ARTIFACTS_DIR / "norms_histogram.png")

def do_embedding_plots(config, clouds):
    color_f = lambda colors: [lang if lang in {"en", "fr", "ja", "ko", "nl", "ru", "tr"} else "other" for lang in colors]

    for work_type, topic in itertools.product(WorkType, Filters):
        cloud = clouds[(topic, work_type)]
        plot_embeddings([cloud], "language", method="pca", dimensions=2, color_f=color_f, **vars(config))
        # plot_embeddings([cloud], "language", method="pca", dimensions=3, color_f=color_f, **vars(config))
        plot_embeddings([cloud], "language", method="tsne", dimensions=2, color_f=color_f, **vars(config))

    # for work_type in WorkType:
    #     topic_clouds = [clouds[(topic, work_type)] for topic in Topic]
    #     plot_embeddings(topic_clouds, "topic", method="pca", dimensions=2, **vars(config))
    #     plot_embeddings(topic_clouds, "topic", method="pca", dimensions=3, **vars(config))
    #     plot_embeddings(topic_clouds, "topic", method="tsne", **vars(config))
    #     plot_embeddings(topic_clouds, "language", method="pca", dimensions=2, color_f=color_f, **vars(config))
    #     plot_embeddings(topic_clouds, "language", method="pca", dimensions=3, color_f=color_f, **vars(config))
    #     plot_embeddings(topic_clouds, "language", method="tsne", color_f=color_f, **vars(config))


def do_inference():
    # Synchronous inference for all *_articles.txt files
    inference_manager = InferenceManager()
    for articles_file in tqdm.tqdm(ARTIFACTS_DIR.glob("**/*_articles.txt"), desc="Running inference on articles files"):
        output_path = articles_file.with_name(articles_file.name.replace("_articles.txt", "_book_info.txt"))
        if output_path.exists():
            LOGGER.info(f"Skipping {articles_file} as {output_path} already exists.")
            continue
        with open(articles_file, "r") as f:
            articles_text = f.read()
        prompt = (
            "Given the following list of research articles (titles and abstracts), "
            "generate a suitable title, abstract, and table of contents for a textbook "
            "that would prepare a student to understand the topics in these articles. "
            "Format your answer as:\n"
            "==TITLE: <title goes here>\n"
            "==ABSTRACT: <abstract goes here>\n"
            "==TABLE OF CONTENTS:\n"
            "<table of contents goes here>\n"
            "==END TABLE OF CONTENTS\n\n"
            "Provide no extraneous output, just the formatted answer.\n\n"
            f"Articles:\n\n{articles_text}\n\n"
        )
        LOGGER.info(f"Running inference for {articles_file} -> {output_path}")
        inference_manager.run(prompt, output_path)


@dataclass(frozen=True)
class Config:
    max_works: Optional[int]
    duplicate_expectation: float
    min_sqdist: Optional[float]
    radius_fraction: float
    index_info: IndexInfo
    denoiser: Optional[str]
    drop_k: int
    normalize_embeddings: bool = field(repr=False)

@TIMER.time()
def main(drop_k: int, max_works: Optional[int], duplicate_expectation: float, radius_fraction: float, denoiser: Optional[str],
         index_type: IndexType, hnsw_m: int, hnsw_efc: int, hnsw_efs: int, min_sqdist: Optional[float]):

    # Construct IndexInfo
    index_info_obj: IndexInfo
    if index_type == IndexType.HNSW:
        index_info_obj = HNSW(
            hnsw_m=hnsw_m,
            hnsw_efc=hnsw_efc,
            hnsw_efs=hnsw_efs,
        )
    elif index_type == IndexType.FLAT_L2:
        index_info_obj = FlatL2()
    else:
        raise ValueError(f"Unhandled index type: {index_type}")

    config = Config(
        max_works=max_works,
        duplicate_expectation=duplicate_expectation,
        radius_fraction=radius_fraction,
        index_info=index_info_obj,
        denoiser=denoiser,
        drop_k=drop_k,
        # normalize_embeddings=normalize_embeddings
        normalize_embeddings=True,
        min_sqdist=min_sqdist,
    )

    global ARTIFACTS_DIR
    # ARTIFACTS_DIR /= create_unicode_safe_filename("_".join(f"{k}:{v}" for k, v in sorted(vars(config).items())))
    ARTIFACTS_DIR = SCOPES_DIR / create_unicode_safe_filename(" ".join(f"{k}={v}" for k, v in sorted(vars(config).items())))
    LOGGER.info(f"Artifacts directory: {ARTIFACTS_DIR}")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Starting run with config: {config}")

    export_avgref_stats_to_latex()

    # Create appropriate deduplicator based on parameters
    if config.min_sqdist is not None:
        deduplicator = SqDistDeduplicator(min_sqdist=config.min_sqdist)
    else:
        deduplicator = FractionalDeduplicator(duplicate_fraction=config.duplicate_expectation)

    clouds = {
        topic: Cloud(
            filter_pair=topic.value,
            index_info=config.index_info,
            max_works=config.max_works,
            deduplicator=deduplicator,
            normalize_embeddings=config.normalize_embeddings,
            denoiser=config.denoiser,
            drop_k=config.drop_k,
        )
        for topic in Filters
    }

    # Log the closest pairs of works in each cloud
    # for work_type, topic in itertools.product(WorkType, Topic):
    for topic in Filters:
        work_type = WorkType.ARTICLE
        fname = SCOPES_DIR / f"{topic}" / f"{work_type}" / "closest_pairs.txt"
        fname.parent.mkdir(parents=True, exist_ok=True)
        with open(fname, "w") as f:
            cloud = clouds[topic]
            # Log the nearest neighbors in order of increasing distance
            Dnn, I = get_nearest_neighbor(cloud, cloud, 0)
            I = I[:, 1]
            sorted_indices = np.argsort(Dnn)
            # logger.info(f"Nearest neighbors in {cloud.name} ({work_type})\n")
            f.write(f"Nearest neighbors in {cloud.name} ({work_type})\n")
            for i in sorted_indices[:20]:
                selfwork, otherwork = cloud.works()[i], cloud.works()[I[i]]
                if otherwork["id"] == selfwork["id"]:
                    msg = f"Self match nearest neighbor: {selfwork['id']} == {otherwork['id']}"
                    LOGGER.warning(msg)
                    f.write(f"WARNING: {msg}\n")
                if otherwork["id"] < selfwork["id"]:
                    continue
                # logger.info(f"Distance {i}: {Dnn[i]:.4f}\n"
                #             f"{selfwork['id']} | {selfwork['publication_date']} | {selfwork['display_name']} | {' '.join(selfwork['abstract'].split())}\n"
                #             f"{otherwork['id']} | {otherwork['publication_date']} | {otherwork['display_name']} | {' '.join(otherwork['abstract'].split())}\n")
                f.write(f"Distance {i}: {Dnn[i]:.4f}\n"
                        f"{selfwork['id']} | {selfwork['publication_date']} | {selfwork['display_name']} | {' '.join(selfwork['abstract'].split())}\n"
                        f"{otherwork['id']} | {otherwork['publication_date']} | {otherwork['display_name']} | {' '.join(otherwork['abstract'].split())}\n\n")

    for topic in Filters:
        cloud = clouds[topic]
        get_points_by_coverage_ball(cloud, WorkType.ARTICLE.value, WorkType.BOOK.value, config.radius_fraction)

    # for work_type, topic in itertools.product(WorkType, Topic):
    for topic in Filters:
        work_type = WorkType.ARTICLE
        cloud = clouds[topic]
        do_avgref_analysis([cloud], cloud)

    # for work_type, topic in itertools.product(WorkType, Topic):
    for topic in Filters:
        work_type = WorkType.ARTICLE
        # Compute distribution of number of referenced_works
        cloud = clouds[topic]
        referenced_work_counts = np.array([len(work["referenced_works"]) for work in cloud.works()])
        LOGGER.info(f"Number of referenced works in {topic} - {work_type}:\n{get_stats(referenced_work_counts)}")
        fig, ax = PlotManager.create_figure(figsize=FIGSIZE)
        ax.hist(referenced_work_counts, bins=100, color="lightblue", alpha=0.7)
        ax.set_xlabel("Number of referenced works")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{topic} - {work_type} - Number of referenced works")
        ax.set_yscale("log")
        fig_dir = SCOPES_DIR / f"{topic}" / f"{work_type}"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir / "referenced_works.png", dpi=DPI)
        plt.close(fig)


    # do_embedding_plots(config, clouds)

    # do_inference()

if __name__ == "__main__":
    from bookgap.script_utils import run_script_main
    run_script_main(main, __file__)
