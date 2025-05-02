import abc
import argparse
import csv
from itertools import product
import json
import os
import textwrap
import requests
import time
from typing import Iterator, Dict, Any, Optional, List, Literal, Tuple
from collections import defaultdict
from dataclasses import dataclass
import enum
import numpy as np
from pathlib import Path
import pickle
import plotly.graph_objects as go
import pyalex
from pyalex import Works, Topics
import re
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tqdm import tqdm
import colorsys # Added for color manipulation
# from sklearn.neighbors import NearestNeighbors, BallTree # Remove BallTree
import faiss # Import faiss
import faiss.contrib.torch_utils
import faiss.contrib.inspect_tools # For reconstructing vectors
from sklearn.preprocessing import StandardScaler # Keep if you might re-enable scaling
from sklearn.manifold import TSNE
from collections import Counter # Added for histogram counting
import math # Needed for sqrt, inf
import time # Explicitly import time if not already done at top
import hashlib # Added for hashing content
import csv
import logging # Use logging for better output control

pyalex.config.email = "gchurchill@cs.stonybrook.edu"

# --- Configuration ---
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = OUTPUT_DIR / "api_data" # Raw OpenAlex page caches
DATA_DIR.mkdir(exist_ok=True)

# NEW: Global cache for threshold-agnostic data
GLOBAL_CACHE_DIR = OUTPUT_DIR / "global_cache"
GLOBAL_CACHE_DIR.mkdir(exist_ok=True)

# RENAMED: Cache for threshold-specific deduplication *results* (unique ID lists)
DEDUP_RESULTS_CACHE_DIR = OUTPUT_DIR / "dedup_results_cache"
DEDUP_RESULTS_CACHE_DIR.mkdir(exist_ok=True)

KEY_DIR = OUTPUT_DIR / "api_keys"
KEY_DIR.mkdir(exist_ok=True)

EMBEDDING_DIR = OUTPUT_DIR / "embeddings"
EMBEDDING_DIR.mkdir(exist_ok=True)

LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Setup basic logging
# Correct the format string: remove the extra %s after asctime
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(levelname)s - %(message)s'
    format='{asctime} - {levelname} - {message}',
    style='{',
)
log = logging.getLogger(__name__)

# --- Enums and Constants ---
class Topic(enum.Enum):
    # NLP      = ("Natural Language Processing Techniques",      "T10181")
    ROBOTICS = ("Robotics and Sensor-Based Localization",      "T10191")
    # MYTHS    = ("Folklore, Mythology, and Literature Studies", "T12586")
    # MEDIEVAL = ("Medieval Literature and History",             "T10595")
    # ALG_GEOM = ("Algebraic Geometry and Number Theory",        "T10061")

    def __init__(self, display_name: str, topic_id: str):
        self._display_name = display_name
        self._topic_id = topic_id
        # Validate topic_id
        topic = Topics()[self._topic_id]
        assert topic['display_name'] == self._display_name, f"Display name mismatch: {topic.display_name} != {self._display_name}"

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def topic_id(self) -> str:
        return self._topic_id

class WorkTypes(enum.Enum):
    ARTICLE = "article"
    BOOK    = "book"
    def __str__(self):
        return self.value

# --- Utility Functions ---
def create_unicode_safe_filename(s):
    return re.sub(r'[^a-zA-Z0-9 _.-]', '', str(s)).rstrip()

def load_api_key(service_name: str) -> str:
    key_path = KEY_DIR / f"{service_name}.key"
    assert key_path.parts[-1] == f"{service_name}.key", f"Invalid key path: {key_path}"
    with open(key_path, 'r') as f:
        return f.read()

def resolve_abstract_inverted_index(abstract_inverted_index: Optional[Dict[str, List[int]]]) -> str:
    if not abstract_inverted_index:
        return ""
    pos_to_word = {}
    for word, positions in abstract_inverted_index.items():
        # assert " " not in word, f"Word contains space: '{word}'" # Relax this assertion
        for pos in positions:
            # assert pos not in pos_to_word, f"Duplicate position {pos} for word {word}" # Relax this assertion
            pos_to_word[pos] = word
    max_pos = max(pos_to_word.keys()) if pos_to_word else -1
    # Handle potential missing positions gracefully
    words = [pos_to_word.get(i, "[[MISSING]]") for i in range(max_pos + 1)]
    # Filter out potential empty strings resulting from relaxed assertions if needed
    return " ".join(filter(None, words))


def calculate_normalized_string(work: Dict[str, Any]) -> Optional[str]:
    """
    Generates a normalized string for content comparison.
    Normalization involves lowercasing and stripping all whitespace from the
    concatenation of a work's title and abstract.

    Args:
        work: The work dictionary, expected to have 'display_name' and 'abstract'.
              The 'abstract' should be the resolved string.

    Returns:
        The normalized string, or None if title or abstract is missing/empty.
    """
    title = work.get('display_name', '')
    abstract = work.get('abstract', '') # Assumes abstract is already resolved
    if not title or not abstract:
        return None
    content_str_normalized = ''.join(f"{title}{abstract}".lower().split())
    return content_str_normalized

# --- Core Classes ---

class OpenAlexDownloader:
    """Handles fetching and caching raw work data from OpenAlex."""
    def __init__(self, work_type: WorkTypes, topic: Topic, base_data_dir: Path = DATA_DIR, max_works_to_fetch: Optional[int] = None):
        self.work_type = work_type
        self.topic = topic
        self.output_dir = base_data_dir / create_unicode_safe_filename(topic.display_name) / create_unicode_safe_filename(str(work_type))
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.state_file = self.output_dir / "downloader_state.json"
        self.max_works_to_fetch = max_works_to_fetch # Store the limit
        self._load_state()

    def _load_state(self):
        state_loaded = False
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    loaded_state = json.load(f)
                # Validate and set state from loaded data
                self.state = loaded_state
                self.next_cursor = self.state.get("next_cursor", "*") # Use .get for robustness
                self.next_page = self.state.get("next_page", 1)
                # Ensure processed_ids is loaded as a set for instance use
                self.processed_ids = set(self.state.get('processed_ids', []))
                state_loaded = True
                log.info(f"Loaded state from {self.state_file}: cursor={self.next_cursor}, page={self.next_page}, processed={len(self.processed_ids)}")
            except (json.JSONDecodeError, FileNotFoundError, KeyError, TypeError) as e: # Catch potential issues
                log.warning(f"Failed to load or parse state from {self.state_file}, re-initializing: {e}")
                # Fall through to initialize state below

        if not state_loaded:
            # Initialize state dictionary AND instance attributes if load failed or file missing
            self.state = dict(next_cursor="*", next_page=1, processed_ids=[]) # Use list for JSON compatibility
            self.next_cursor = self.state["next_cursor"]
            self.next_page = self.state["next_page"]
            self.processed_ids = set(self.state["processed_ids"]) # Convert to set for instance use
            log.info(f"Initializing new state file: {self.state_file}")
            try:
                # Now self.processed_ids exists before _save_state is called
                self._save_state() # Attempt to save the initial state file
                log.info(f"Successfully saved initial state to {self.state_file}")
            except Exception as save_e:
                log.error(f"CRITICAL: Failed to save initial state file {self.state_file}: {save_e}")
                # Consider if the program should exit if it cannot save initial state

    def _save_state(self):
        # Convert set to list for JSON serialization
        save_state = self.state.copy()
        # Ensure self.processed_ids exists and is a set before converting
        if hasattr(self, 'processed_ids') and isinstance(self.processed_ids, set):
             save_state['processed_ids'] = list(self.processed_ids)
        else:
             log.error("Attempted to save state but self.processed_ids is missing or not a set.")
             # Handle error: maybe save an empty list or skip saving processed_ids?
             save_state['processed_ids'] = [] # Save empty list as fallback

        try:
            with open(self.state_file, "w") as f:
                json.dump(save_state, f)
        except IOError as e:
            log.error(f"Failed to save state to {self.state_file}: {e}")
        except Exception as e:
            log.error(f"Unexpected error saving state to {self.state_file}: {e}")

    def get_total_count(self) -> Optional[int]:
        """
        Gets the estimated total count of works matching the query from OpenAlex,
        respecting the max_works_to_fetch limit if set.
        """
        log.info(f"Querying OpenAlex for total count: {self.topic.display_name} - {self.work_type}")
        try:
            # Replicate the query setup used in stream_works
            total_expected = Works() \
                .filter(has_abstract=True,
                        type=str(self.work_type),
                        **{"primary_topic.id": self.topic.topic_id}).count()
            log.info(f"OpenAlex reports total matching works: {total_expected}")

            # Apply the limit if it's set
            if self.max_works_to_fetch is not None:
                effective_total = min(total_expected, self.max_works_to_fetch)
                if effective_total < total_expected:
                    log.info(f"Applying limit: effective total count is {effective_total} (max_works_to_fetch={self.max_works_to_fetch})")
                return effective_total
            else:
                return total_expected
        except Exception as e:
            log.warning(f"Could not get total count from OpenAlex: {e}")
            # If limit is set, return it even if API fails? Or return None? Let's return None.
            # If the API fails, we don't know the real total, so applying the limit is less meaningful.
            # However, the progress bar needs *a* total. Let's return the limit if set and API fails.
            if self.max_works_to_fetch is not None:
                log.warning(f"Returning max_works_to_fetch ({self.max_works_to_fetch}) as total due to API error.")
                return self.max_works_to_fetch
            return None

    def stream_works(self) -> Iterator[Dict[str, Any]]:
        """
        Yields works from OpenAlex, handling pagination and caching raw pages.
        Resolves abstracts but does not perform deduplication.
        Respects the instance's max_works_to_fetch limit.
        """
        log.info(f"Streaming works for {self.topic.display_name} - {self.work_type}" + (f" (limit: {self.max_works_to_fetch})" if self.max_works_to_fetch is not None else ""))

        # --- Query Setup ---
        query = Works() \
            .filter(has_abstract=True, type=str(self.work_type), **{"primary_topic.id": self.topic.topic_id}) \
            .select(", ".join(["display_name", "type", "publication_date", "ids", "abstract_inverted_index"]))

        # Use get_total_count which already incorporates the limit
        total_for_pbar = self.get_total_count()

        pbar = tqdm(total=total_for_pbar, desc=f"Fetching {self.topic.display_name} ({self.work_type})", unit="work")

        # --- Process Cached Pages ---
        log.info(f"Processing cached pages up to page {self.next_page - 1}...")
        yielded_count = 0 # Track total yielded works against the limit
        for i in range(1, self.next_page):
            page_file = self.output_dir / f"page_{i}.json"
            if page_file.exists():
                try:
                    with open(page_file, 'r') as f:
                        cached_page = json.load(f)
                    for work in cached_page:
                        work_id = work.get("ids", {}).get("openalex")
                        if work_id and work_id not in self.processed_ids:
                            # Resolve abstract if not already done (e.g., older cache format)
                            if "abstract_inverted_index" in work:
                                work["abstract"] = resolve_abstract_inverted_index(work["abstract_inverted_index"])
                                # del work["abstract_inverted_index"] # Keep for potential reprocessing if needed

                            # Check limit *before* yielding
                            if self.max_works_to_fetch is not None and yielded_count >= self.max_works_to_fetch:
                                log.info(f"Reached max_works_to_fetch limit ({self.max_works_to_fetch}) from cache.")
                                pbar.close()
                                return

                            yield work
                            yielded_count += 1
                            self.processed_ids.add(work_id)
                            # No need to check limit again here, checked before yield

                        pbar.update(1) # Update progress even for skipped/already processed items
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    log.warning(f"Skipping corrupted/missing cache file: {page_file} ({e})")
                    # Attempt to estimate items in missing page if total is known
                    # This is tricky, might lead to inaccurate progress bars.
                    # Best to just log and continue.
            else:
                log.warning(f"Cache file not found: {page_file}. Progress bar might be inaccurate.")

        log.info(f"Yielded {yielded_count} works from cache.")
        pbar.n = yielded_count # Reset progress bar count based on cache yield

        # Check limit again before fetching new pages
        if self.max_works_to_fetch is not None and yielded_count >= self.max_works_to_fetch:
            log.info(f"Reached max_works_to_fetch limit ({self.max_works_to_fetch}) after processing cache.")
            pbar.close()
            return

        # --- Fetch New Pages ---
        log.info(f"Fetching new pages starting from cursor {self.next_cursor}...")
        last_time = time.time()
        # yielded_total = yielded_from_cache # Renamed to yielded_count

        try:
            # The paginate n_max parameter is tricky with caching/resuming.
            # It's better to control the limit within the loop.
            for page_data in query.paginate(cursor=self.next_cursor, per_page=150, n_max=None):
                current_page_cursor = self.next_cursor # Store cursor for this page start
                self.next_cursor = page_data.meta["next_cursor"]

                page = list(map(dict, page_data))
                processed_page_for_cache = [] # Store works for caching this page

                for work in page:
                    work_id = work.get("ids", {}).get("openalex")
                    if not work_id:
                        log.warning(f"Skipping work with missing OpenAlex ID: {work.get('display_name')}")
                        pbar.update(1)
                        continue

                    # Resolve abstract
                    work["abstract"] = resolve_abstract_inverted_index(work.get("abstract_inverted_index"))
                    # Optionally remove inverted index from yielded dict to save memory downstream
                    # if "abstract_inverted_index" in work: del work["abstract_inverted_index"]

                    # Add to cache list *before* checking processed_ids
                    processed_page_for_cache.append(work)

                    # Yield only if not already processed in a previous run/cache load
                    if work_id not in self.processed_ids:
                        # Check limit *before* yielding
                        if self.max_works_to_fetch is not None and yielded_count >= self.max_works_to_fetch:
                            log.info(f"Reached max_works_to_fetch limit ({self.max_works_to_fetch}) while fetching.")
                            # Still save the current page cache and state before returning
                            break # Break inner loop (work processing)

                        yield work
                        yielded_count += 1
                        self.processed_ids.add(work_id)
                        # No need to check limit again here

                    pbar.update(1)

                # --- Save Page Cache and State ---
                if processed_page_for_cache:
                    page_cache_file = self.output_dir / f"page_{self.next_page}.json"
                    log.debug(f"Saving {len(processed_page_for_cache)} raw items to {page_cache_file}")
                    try:
                        with open(page_cache_file, "w") as f:
                            json.dump(processed_page_for_cache, f) # No indent for smaller files
                    except IOError as e:
                        log.error(f"Failed to save page cache {page_cache_file}: {e}")

                # Save state after processing the page
                self.state["next_cursor"] = self.next_cursor
                self.state["next_page"] = self.next_page + 1
                self._save_state()
                log.debug(f"Saved downloader state (cursor={self.next_cursor}, page={self.next_page + 1})")

                self.next_page += 1 # Increment page number for the *next* iteration

                # Check max_works limit again after saving state and before next page/rate limit
                if self.max_works_to_fetch is not None and yielded_count >= self.max_works_to_fetch:
                    break # Break outer loop (page fetching)

                # Rate limiting
                MIN_DIFF = 1 # seconds
                time_to_sleep = max(0, MIN_DIFF - (time.time() - last_time))
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                last_time = time.time()

            # Update state one last time if loop finished naturally or broke due to limit
            self.state["next_cursor"] = self.next_cursor
            self._save_state()

        except Exception as e:
            log.error(f"Error during OpenAlex pagination: {e}", exc_info=True)
            # Attempt to save state even on error
            self._save_state()
        finally:
            pbar.close()
            log.info(f"Finished streaming. Yielded {yielded_count} total works.")


# --- NEW: Global Cache Building ---

def build_global_cache(work_type: WorkTypes, topic: Topic, force_rebuild: bool = False, max_works_to_process: Optional[int] = None):
    """
    Builds and caches threshold-agnostic data: FAISS index, ID maps, embeddings, and all work details.
    Respects the max_works_to_process limit.
    """
    global_cache_subdir = GLOBAL_CACHE_DIR / create_unicode_safe_filename(topic.display_name) / create_unicode_safe_filename(str(work_type))
    global_cache_subdir.mkdir(exist_ok=True, parents=True)

    # Define cache file paths
    faiss_index_file = global_cache_subdir / "global_faiss_index.bin"
    id_to_info_file = global_cache_subdir / "global_id_to_info.json" # {id: {'priority': p, 'faiss_idx': f_idx}}
    faiss_idx_to_id_file = global_cache_subdir / "global_faiss_idx_to_id.json" # [id_0, id_1, ...]
    all_details_file = global_cache_subdir / "all_work_details.json"
    embeddings_file = global_cache_subdir / "global_embeddings.pkl" # Ensure this is defined

    # Check if cache exists and force_rebuild is False
    # Add embeddings_file to required files check
    required_files = [faiss_index_file, id_to_info_file, faiss_idx_to_id_file, all_details_file, embeddings_file]
    # If cache exists and not forcing rebuild, skip. The max_works limit only applies during build.
    if not force_rebuild and all(f.exists() for f in required_files):
        log.info(f"Global cache already exists for {topic.display_name} - {work_type}. Skipping build.")
        # Optional: Load and return data if needed immediately, or just return True
        return True # Indicate cache exists

    log.info(f"Building global cache for {topic.display_name} - {work_type}" + (f" (limit: {max_works_to_process})" if max_works_to_process is not None else "") + "...")
    # Pass the limit to the downloader
    downloader = OpenAlexDownloader(work_type, topic, max_works_to_fetch=max_works_to_process)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # Load model here
    embedding_dim = model.get_sentence_embedding_dimension()

    all_embeddings_list = []
    faiss_idx_to_id_map = []
    id_to_info_map = {}
    all_details_map = {}
    processed_count = 0
    skipped_embedding_count = 0

    # Estimate total for progress bar using the downloader's potentially limited count
    total_estimated = downloader.get_total_count()
    pbar_desc = f"Building global cache ({topic.display_name} - {work_type})"
    pbar_building = tqdm(total=total_estimated, desc=pbar_desc, unit=" work")

    # stream_works will now respect the limit passed during downloader initialization
    for work in downloader.stream_works():
        pbar_building.update(1)
        work_id = work.get("ids", {}).get("openalex")
        title = work.get('display_name')
        abstract = work.get('abstract') # Assumes resolved by downloader

        if not work_id or not title or not abstract:
            log.debug(f"Skipping work due to missing ID/title/abstract: {work_id or '[[NO ID]]'}")
            skipped_embedding_count += 1
            continue

        # Calculate embedding
        try:
            embedding = model.encode(f"{title}\n{abstract}", show_progress_bar=False)
            if not np.all(np.isfinite(embedding)):
                log.warning(f"Generated non-finite embedding for work {work_id}, skipping.")
                skipped_embedding_count += 1
                continue
            embedding_np = embedding.astype('float32') # Ensure correct type for FAISS
        except Exception as e:
            log.error(f"Error encoding work {work_id}: {e}. Skipping.")
            skipped_embedding_count += 1
            continue

        # Assign priority (insertion order) and store info
        current_faiss_idx = processed_count # Index in the list/FAISS index
        priority = current_faiss_idx

        all_embeddings_list.append(embedding_np)
        faiss_idx_to_id_map.append(work_id)
        id_to_info_map[work_id] = {'priority': priority, 'faiss_idx': current_faiss_idx}
        all_details_map[work_id] = work # Store the full work details

        processed_count += 1
        # No need for explicit limit check here, stream_works handles it

    pbar_building.close()
    log.info(f"Processed {processed_count} works, skipped {skipped_embedding_count} due to missing data or embedding errors.")

    if not all_embeddings_list:
        log.warning(f"No valid embeddings generated for {topic.display_name} - {work_type}. Cannot build global cache.")
        return False

    # Build FAISS index
    log.info(f"Building global FAISS index (IndexFlatL2) with {processed_count} embeddings...")
    try:
        embeddings_np = np.stack(all_embeddings_list)
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings_np)
        log.info(f"FAISS index built successfully. Index total: {index.ntotal}")
    except Exception as e:
        log.error(f"Failed to build FAISS index: {e}", exc_info=True)
        return False

    # Save all components
    try:
        log.info(f"Saving global cache components to {global_cache_subdir}...")
        # Save FAISS index
        faiss.write_index(index, str(faiss_index_file))
        # Save ID maps
        with open(id_to_info_file, 'w') as f: json.dump(id_to_info_map, f)
        with open(faiss_idx_to_id_file, 'w') as f: json.dump(faiss_idx_to_id_map, f)
        # Save all details
        with open(all_details_file, 'w') as f: json.dump(all_details_map, f, indent=2)
        # Save raw embeddings - uncommented
        with open(embeddings_file, 'wb') as f: pickle.dump(embeddings_np, f)
        log.info("Global cache saved successfully.")
        return True
    except Exception as e:
        log.error(f"Failed to save global cache components: {e}", exc_info=True)
        return False


# --- NEW: Threshold-Specific Deduplication ---

def get_unique_ids_for_threshold(work_type: WorkTypes, topic: Topic, distance_threshold: float) -> List[str]:
    """
    Determines the set of unique work IDs for a given distance threshold,
    using the precomputed global cache. Caches the results.
    """
    global_cache_subdir = GLOBAL_CACHE_DIR / create_unicode_safe_filename(topic.display_name) / create_unicode_safe_filename(str(work_type))
    results_cache_subdir = DEDUP_RESULTS_CACHE_DIR / create_unicode_safe_filename(topic.display_name) / create_unicode_safe_filename(str(work_type))
    results_cache_subdir.mkdir(exist_ok=True, parents=True)

    # Define global cache file paths
    faiss_index_file = global_cache_subdir / "global_faiss_index.bin"
    # id_to_info_file = global_cache_subdir / "global_id_to_info.json" # Not strictly needed for this logic
    faiss_idx_to_id_file = global_cache_subdir / "global_faiss_idx_to_id.json"
    embeddings_file = global_cache_subdir / "global_embeddings.pkl" # Path to cached embeddings

    # Define threshold-specific results cache file path
    if distance_threshold == 0:
        threshold_str = "0"
        log_thresh_str = "0 (Exact Only)"
    else:
        threshold_str = f"{distance_threshold:.4f}".replace('.', '_')
        log_thresh_str = f"{distance_threshold:.4f}"
    results_file = results_cache_subdir / f"unique_ids_dist_{threshold_str}.json"
    duplicate_log_file = results_cache_subdir / f"duplicates_dist_{threshold_str}.log" # Log duplicates found at this threshold

    # Check results cache first
    if results_file.exists():
        log.info(f"Loading cached unique IDs for threshold {log_thresh_str} from {results_file}")
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            log.warning(f"Failed to load unique IDs cache {results_file}, recalculating: {e}")

    # Load global cache components
    log.info(f"Loading global cache to determine unique IDs for threshold {log_thresh_str}...")
    try:
        index = faiss.read_index(str(faiss_index_file))
        with open(faiss_idx_to_id_file, 'r') as f:
            faiss_idx_to_id_map = json.load(f)
        # Load embeddings from pickle file
        with open(embeddings_file, 'rb') as f:
            all_embeddings_np = pickle.load(f)
        log.info(f"Loaded global FAISS index ({index.ntotal} items), ID map, and embeddings ({all_embeddings_np.shape}).")
    except Exception as e:
        log.error(f"Failed to load global cache from {global_cache_subdir}: {e}. Cannot perform deduplication.", exc_info=True)
        return [] # Return empty list on error

    n_total = index.ntotal
    if n_total == 0:
        log.warning("Global FAISS index is empty. No unique IDs to determine.")
        return []
    if n_total != len(faiss_idx_to_id_map):
         log.error(f"Inconsistency: Global FAISS index size ({n_total}) != ID map size ({len(faiss_idx_to_id_map)}). Aborting.")
         return []
    if n_total != all_embeddings_np.shape[0]:
         log.error(f"Inconsistency: Global FAISS index size ({n_total}) != Embeddings array size ({all_embeddings_np.shape[0]}). Aborting.")
         return []


    # --- Deduplication Logic ---
    log.info(f"Performing deduplication for threshold {log_thresh_str} using range search...")
    # `duplicate_of[id]` stores the ID of the higher-priority item it's a duplicate of, or None if unique.
    duplicate_of = {work_id: None for work_id in faiss_idx_to_id_map}
    duplicates_found_count = 0
    unique_ids = []

    # Open duplicate log file for this threshold run
    try:
        dup_log_file_handle = open(duplicate_log_file, 'w', encoding='utf-8') # Overwrite log for this threshold each time
        dup_log_file_handle.write(f"--- Log Start: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        dup_log_file_handle.write(f"Distance Threshold: {log_thresh_str}\n")
        dup_log_file_handle.write("Duplicate ID | Duplicate Of ID | Distance\n")
    except IOError as e:
        log.error(f"Could not open duplicate log file {duplicate_log_file}: {e}")
        dup_log_file_handle = None

    # If threshold is 0, only exact duplicates matter. The global cache build already handled non-embeddable items.
    # The current logic implicitly handles exact duplicates if their embeddings are identical (distance 0).
    # However, the original `WorkDeduplicator` used normalized string hashing for exact matches *before* embedding.
    # To replicate that perfectly, we'd need the normalized strings here too.
    # For now, let's assume distance 0 handles identical embeddings, acknowledging it might differ slightly from hash-based exact matching if embeddings aren't perfectly identical for identical text.
    # If distance_threshold is 0, the range search radius will be 0. Only points with distance < 0 (impossible) or exactly 0 will be found.
    # Let's proceed with the range search logic, it should correctly handle the distance_threshold=0 case (only finding distance=0 neighbors).

    # FAISS range_search expects the raw distance threshold (radius), not squared.
    # It returns squared distances for L2.

    # Add a small epsilon ONLY if threshold > 0 to handle potential floating point inaccuracies near the boundary?
    # Or rely on strict < threshold comparison? Let's stick to radius_sq = threshold**2 and check distance < threshold later.
    # The range search finds distances <= radius. Let's use the raw threshold and check distance < threshold later.

    pbar_dedup = tqdm(total=n_total, desc=f"Deduplicating (thresh={log_thresh_str})", unit=" work")

    for i in range(n_total): # Iterate in priority order (faiss index order)
        pbar_dedup.update(1)
        current_id = faiss_idx_to_id_map[i]

        # Skip if already marked as a duplicate by a higher-priority item
        if duplicate_of[current_id] is not None:
            continue

        # Retrieve embedding directly from the loaded array
        try:
            query_vector = all_embeddings_np[i][:, np.newaxis]
            if not np.all(np.isfinite(query_vector)):
                 log.warning(f"Cached vector for index {i} (ID {current_id}) is non-finite. Skipping range search.")
                 unique_ids.append(current_id)
                 continue
        except IndexError:
             log.error(f"IndexError retrieving vector for index {i} (ID {current_id}). Embeddings array size: {all_embeddings_np.shape[0]}. Treating as unique.")
             unique_ids.append(current_id)
             continue
        except Exception as e:
            log.error(f"Failed to retrieve/process vector for index {i} (ID {current_id}): {e}. Treating as unique.")
            unique_ids.append(current_id)
            continue

        # Perform range search using the FAISS index
        try:
            # Pass the raw distance_threshold as the radius
            lims, neighbor_distances_sq, neighbor_indices = index.range_search(query_vector, distance_threshold)
            # Sanity check: Ensure returned squared distances are < radius^2
            radius_squared = distance_threshold * distance_threshold
            for dist_sq in neighbor_distances_sq:
                if np.isfinite(dist_sq):
                    # Use a small tolerance for floating point comparisons near zero if threshold is zero
                    if distance_threshold == 0:
                         assert dist_sq <= 1e-9, f"Distance check failed (thresh=0): {dist_sq} > tolerance"
                    else:
                         assert dist_sq < radius_squared, f"Distance check failed: {dist_sq} >= {radius_squared}"
        except Exception as e:
            log.error(f"FAISS range_search failed for index {i} (ID {current_id}). Error: {e!r}. Treating as unique.", exc_info=True) # Add exc_info=True and !r
            # Log query details
            log.debug(f"Failed query details: index={i}, ID={current_id}, shape={query_vector.shape}, dtype={query_vector.dtype}, isfinite={np.all(np.isfinite(query_vector))}, radius={distance_threshold}") # Log raw radius
            unique_ids.append(current_id)
            continue

        is_duplicate_for_threshold = False
        highest_priority_duplicate_source_id = None
        min_dist_to_source = float('inf')

        # Check neighbors
        # Use a different loop variable name (e.g., neighbor_dist_sq) to avoid UnboundLocalError
        for neighbor_idx, neighbor_dist_sq in zip(neighbor_indices, neighbor_distances_sq):
            if neighbor_idx == i: continue # Skip self

            neighbor_id = faiss_idx_to_id_map[neighbor_idx]
            # Calculate raw distance from squared distance
            distance = math.sqrt(neighbor_dist_sq)

            # Check if neighbor has higher priority (lower index) AND is within the strict threshold
            if neighbor_idx < i and distance < distance_threshold:
                is_duplicate_for_threshold = True
                # Track the highest priority source (lowest index)
                # Since we iterate `i` in priority order, any neighbor `j < i` found is a candidate source.
                # The first one encountered in the range search results isn't necessarily the highest priority one overall.
                # We just need *one* higher-priority neighbor within the threshold.
                # Let's record the first one found for logging purposes.
                if highest_priority_duplicate_source_id is None: # or neighbor_idx < id_to_info_map[highest_priority_duplicate_source_id]['faiss_idx']:
                     highest_priority_duplicate_source_id = neighbor_id
                     min_dist_to_source = distance
                # Optimization: Can break inner loop once one valid source is found
                break

        if is_duplicate_for_threshold:
            duplicate_of[current_id] = highest_priority_duplicate_source_id
            duplicates_found_count += 1
            if dup_log_file_handle:
                try:
                    dup_log_file_handle.write(f"{current_id} | {highest_priority_duplicate_source_id} | {min_dist_to_source:.6f}\n")
                except IOError: pass
        else:
            # This item is unique for this threshold
            unique_ids.append(current_id)

            # Optimization: Mark lower-priority neighbors as duplicates of this unique item
            # This avoids redundant range searches for those neighbors later.
            # Iterate using the correct variable names again
            for neighbor_idx, neighbor_dist_sq in zip(neighbor_indices, neighbor_distances_sq):
                 if neighbor_idx > i: # Lower priority neighbor
                     neighbor_id = faiss_idx_to_id_map[neighbor_idx]
                     # Calculate raw distance again
                     distance = math.sqrt(neighbor_dist_sq)
                     # Check if within strict threshold and not already marked as duplicate
                     if distance < distance_threshold and duplicate_of[neighbor_id] is None:
                         duplicate_of[neighbor_id] = current_id
                         duplicates_found_count += 1 # Count it here
                         if dup_log_file_handle:
                             try:
                                 dup_log_file_handle.write(f"{neighbor_id} | {current_id} | {distance:.6f}\n")
                             except IOError: pass


    pbar_dedup.close()
    log.info(f"Deduplication complete for threshold {log_thresh_str}. Found {len(unique_ids)} unique items and {duplicates_found_count} duplicates.")

    # Close log file
    if dup_log_file_handle:
        try:
            dup_log_file_handle.close()
            log.info(f"Duplicate log saved to {duplicate_log_file}")
        except IOError: pass

    # Save results cache
    log.info(f"Caching {len(unique_ids)} unique IDs for threshold {log_thresh_str} to {results_file}")
    try:
        with open(results_file, 'w') as f:
            json.dump(unique_ids, f)
    except IOError as e:
        log.error(f"Failed to save unique IDs cache {results_file}: {e}")

    return unique_ids


# --- REWRITTEN Data Access Functions ---

# Remove get_unique_works - replaced by the new mechanism

def get_works_by_id(work_type: WorkTypes, topic: Topic, distance_threshold: float, max_works_to_process: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    Retrieves a dictionary mapping OpenAlex work IDs to their corresponding full work data,
    ensuring uniqueness based on the specified distance threshold using the global cache and range search.
    The global cache build respects max_works_to_process.
    """
    # 1. Ensure global cache is built, passing the limit
    log.info(f"Ensuring global cache exists for {topic.display_name} - {work_type}...")
    if not build_global_cache(work_type, topic, max_works_to_process=max_works_to_process):
        log.error("Failed to build or verify global cache. Cannot retrieve works by ID.")
        return {}

    # 2. Get unique IDs for the given threshold (uses the existing global cache)
    log.info(f"Determining unique IDs for the given threshold {distance_threshold}...")
    unique_ids = get_unique_ids_for_threshold(work_type, topic, distance_threshold)

    if not unique_ids:
        log.warning(f"No unique works found for {topic.display_name} - {work_type} at threshold {distance_threshold}.")
        return {}

    # 3. Load all work details from global cache
    global_cache_subdir = GLOBAL_CACHE_DIR / create_unicode_safe_filename(topic.display_name) / create_unicode_safe_filename(str(work_type))
    all_details_file = global_cache_subdir / "all_work_details.json"
    log.info(f"Loading all work details from {all_details_file}...")
    try:
        with open(all_details_file, 'r') as f:
            all_details_map = json.load(f)
    except Exception as e:
        log.error(f"Failed to load all work details from global cache: {e}", exc_info=True)
        return {}

    # 4. Filter details map to include only unique IDs
    unique_works_details = {work_id: all_details_map[work_id] for work_id in unique_ids if work_id in all_details_map}

    if len(unique_works_details) != len(unique_ids):
        log.warning(f"Mismatch: Found {len(unique_ids)} unique IDs but only {len(unique_works_details)} had details in cache.")

    log.info(f"Returning {len(unique_works_details)} unique work details for threshold {distance_threshold}.")
    return unique_works_details


def get_embeddings(work_type: WorkTypes, topic: Topic, distance_threshold: float, max_works_to_process: Optional[int] = None):
    """
    Loads precomputed global embeddings from cache and filters them for the unique works
    identified by a specific distance threshold.
    The global cache build respects max_works_to_process.
    """
    # 1. Ensure global cache is built (needed for unique ID determination and embeddings file), passing the limit
    log.info(f"Ensuring global cache exists for {topic.display_name} - {work_type}...")
    if not build_global_cache(work_type, topic, max_works_to_process=max_works_to_process):
        log.error("Failed to build or verify global cache. Cannot retrieve embeddings.")
        return {}

    # 2. Get unique IDs for the given threshold
    log.info(f"Determining unique work IDs for threshold {distance_threshold}...")
    unique_ids = get_unique_ids_for_threshold(work_type, topic, distance_threshold)
    unique_ids_set = set(unique_ids) # Faster lookups

    if not unique_ids:
        log.warning(f"No unique works found for {topic.display_name} - {work_type} at threshold {distance_threshold}. Returning empty embeddings.")
        return {}

    # 3. Load global embeddings and ID map from cache
    global_cache_subdir = GLOBAL_CACHE_DIR / create_unicode_safe_filename(topic.display_name) / create_unicode_safe_filename(str(work_type))
    embeddings_file = global_cache_subdir / "global_embeddings.pkl"
    # faiss_index_file = global_cache_subdir / "global_faiss_index.bin" # No longer needed here
    faiss_idx_to_id_file = global_cache_subdir / "global_faiss_idx_to_id.json"

    all_embeddings_map = {}
    log.info("Loading global embeddings from cache...")

    try:
        # Load embeddings array
        with open(embeddings_file, 'rb') as f:
            all_embeddings_np = pickle.load(f)
        # Load ID map
        with open(faiss_idx_to_id_file, 'r') as f:
            faiss_idx_to_id_map = json.load(f)

        if all_embeddings_np.shape[0] != len(faiss_idx_to_id_map):
             raise ValueError("Cached embeddings array size doesn't match ID map size.")

        log.info(f"Building map from ID to embedding for {len(faiss_idx_to_id_map)} items...")
        # Create the map directly
        for i in range(len(faiss_idx_to_id_map)):
             work_id = faiss_idx_to_id_map[i]
             vector = all_embeddings_np[i] # Get vector from loaded array
             all_embeddings_map[work_id] = vector

        log.info(f"Loaded {len(all_embeddings_map)} global embeddings from cache.")

    except FileNotFoundError as e:
        log.error(f"Failed to load global embeddings cache file: {e}", exc_info=True)
        return {}
    except Exception as e:
        log.error(f"Failed to load/process global embeddings from cache: {e}", exc_info=True)
        return {}


    # 4. Filter embeddings to include only unique IDs for the threshold
    unique_embeddings = {work_id: emb for work_id, emb in all_embeddings_map.items() if work_id in unique_ids_set}

    log.info(f"Returning {len(unique_embeddings)} embeddings corresponding to unique works for threshold {distance_threshold}.")
    return unique_embeddings


# ... (adjust_color_brightness, color_to_str remain the same) ...
def adjust_color_brightness(rgb_tuple, factor):
    """Adjusts the brightness of an RGB color tuple."""
    r, g, b = [x / 255.0 for x in rgb_tuple] # Convert to 0-1 range
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * factor)) # Adjust lightness
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255)) # Convert back to 0-255 range

def color_to_str(rgb_tuple, alpha=1.0):
    """Converts an RGB tuple to an rgba string for Plotly."""
    return f'rgba({rgb_tuple[0]}, {rgb_tuple[1]}, {rgb_tuple[2]}, {alpha})'


# --- Analysis Functions (Mostly unchanged, but ensure they use data derived from unique works) ---

# ... (analyze_nearest_neighbors_bipartite remains the same) ...
def analyze_nearest_neighbors_bipartite(
    points1: np.ndarray,
    index2: faiss.Index,
    cloud1_name: str,
    cloud2_name: str
) -> Counter:
    """
    For each point in cloud 1, finds the nearest point in cloud 2 using FAISS.
    Returns a histogram (Counter) of the in-degrees of cloud 2 points.

    Args:
        points1: Numpy array of points in cloud 1 (shape: [n_points1, n_features], dtype=float32).
        index2: Trained FAISS index for points in cloud 2.
        cloud1_name: Name of the first cloud (source).
        cloud2_name: Name of the second cloud (target).

    Returns:
        A Counter where keys are the indices of points in cloud 2 and values
        are the number of times each point was the nearest neighbor for a point in cloud 1.
    """
    log.info(f"Analyzing bipartite nearest neighbors ({cloud1_name} -> {cloud2_name})...")
    start_time = time.time()
    # Search for the 1 nearest neighbor
    distances, nearest_indices_in_cloud2 = index2.search(points1, k=1)
    search_time = time.time() - start_time
    if search_time > 0.1: # FAISS should be fast
        log.info(f"Finding nearest neighbors took {search_time:.2f} seconds.")
    # nearest_indices_in_cloud2 is shape [n_points1, 1], flatten it
    # Filter out -1 indices which indicate no neighbor found (shouldn't happen with k=1 unless index is empty)
    valid_indices = nearest_indices_in_cloud2[nearest_indices_in_cloud2 != -1]
    in_degrees_cloud2 = Counter(valid_indices.flatten())
    log.info(f"Found {len(in_degrees_cloud2)} unique nearest neighbors in {cloud2_name}.")
    # Optional: Print histogram summary
    degree_counts = Counter(in_degrees_cloud2.values())
    log.info(f"Distribution of {cloud2_name} in-degrees:")
    # Limit printing if too many degrees
    max_degrees_to_print = 50
    sorted_degrees = sorted(degree_counts.items())
    for i, (degree, count) in enumerate(sorted_degrees):
        if i < max_degrees_to_print or i == len(sorted_degrees) -1:
             log.info(f"  Degree {degree}: {count} points")
        elif i == max_degrees_to_print:
             log.info(f"  ... (omitting {len(sorted_degrees) - max_degrees_to_print} more degrees)")

    return in_degrees_cloud2


# ... (analyze_nearest_neighbor_rank remains largely the same, ensure item_descriptions uses normalized strings) ...
def analyze_nearest_neighbor_rank(
    points1: np.ndarray,
    index1: faiss.Index,
    index2: faiss.Index,
    cloud1_name: str,
    cloud2_name: str,
    item_descriptions: Optional[Dict[int, Optional[str]]] = None, # Now contains normalized strings or None
    rare_rank_threshold: int = 5
) -> List[int]:
    """
    For each point p1 in cloud 1, finds the rank of its nearest cloud 1 neighbor
    when considering all points (cloud 1 + cloud 2), using FAISS range_search.
    Optionally prints normalized content diffs for points with near-zero distance neighbors
    and identifies points with rare ranks.

    Args:
        points1: Numpy array of points in cloud 1 (shape: [n_points1, n_features], dtype=float32).
        index1: Trained FAISS index for points in cloud 1.
        index2: Trained FAISS index for points in cloud 2.
        cloud1_name: Name of the first cloud.
        cloud2_name: Name of the second cloud.
        item_descriptions: Optional dictionary mapping point indices (0 to n_points1-1)
                           to their normalized content strings (or None if unavailable).
        rare_rank_threshold: Print indices for points whose rank is shared by this many points or fewer.

    Returns:
        A list containing the rank for each point in cloud 1.
    """
    log.info(f"Analyzing nearest neighbor ranks ({cloud1_name} vs {cloud1_name}+{cloud2_name}) using FAISS range_search...")
    n_points1 = points1.shape[0]
    n_points2 = index2.ntotal # Get total number of points in index2

    # 1. Find the nearest neighbor *within* cloud 1
    start_time = time.time()
    k_search = min(10, n_points1) # Search more neighbors to be sure to find non-self
    if k_search < 2 and n_points1 >= 2: k_search = 2 # Need at least 2 if possible
    if n_points1 < 2: k_search = 1 # Avoid error if only 1 point

    distances_nn1_sq, indices_nn1 = index1.search(points1, k=k_search)
    search_time = time.time() - start_time
    if search_time > 0.1:
        log.info(f"Finding nearest neighbors in {cloud1_name} took {search_time:.2f} seconds.")

    # Find the distance to the *actual* nearest non-self neighbor
    radius_sq_to_nn1 = np.full(n_points1, np.inf, dtype='float32')
    first_non_self_indices = np.full(n_points1, -1, dtype='int64')

    for i in range(n_points1):
        # Start from j=0 if k_search=1, else start from j=1
        start_j = 1 if k_search > 1 else 0
        for j in range(start_j, k_search):
            neighbor_idx = indices_nn1[i, j]
            if neighbor_idx == -1: break # No more neighbors found by FAISS
            if neighbor_idx != i: # Check if it's not the point itself
                radius_sq_to_nn1[i] = distances_nn1_sq[i, j]
                first_non_self_indices[i] = neighbor_idx
                break # Found the first non-self neighbor

    # 2. Find neighbors in cloud 2 closer than or equal to the nearest cloud 1 neighbor
    log.info(f"Finding neighbors in {cloud2_name} closer than or equal to the nearest {cloud1_name} neighbor (using range_search)...")
    start_time = time.time()
    neighbors_in_cloud2_closer_than_nn1_counts = np.zeros(n_points1, dtype=int)
    epsilon_sq_inclusive = 1e-9 # Add small epsilon for inclusive range search
    epsilon_sq_zero_check = 1e-12 # Threshold to consider distance effectively zero

    # Use TQDM for progress if iterating over many points
    iterator = range(n_points1)
    if n_points1 > 1000:
        iterator = tqdm(iterator, desc=f"Counting neighbors in {cloud2_name} (range_search)")

    for i in iterator:
        threshold_sq = radius_sq_to_nn1[i]
        query_point = np.ascontiguousarray(points1[i:i+1])

        if threshold_sq == np.inf:
            # Case 1: No non-self neighbor found in Cloud 1
            # This means the rank is 1 + (all points in cloud 2)
            log.debug(f"Point {i} in {cloud1_name} has no valid non-self neighbor in {cloud1_name} (threshold=inf). Rank = 1 + n_points2.")
            count = n_points2
        else:
            # Case 2 & 3: Found a non-self neighbor
            # Use range search with a slightly larger radius to be inclusive
            inclusive_threshold_sq = threshold_sq + epsilon_sq_inclusive
            try:
                lims, D, I = index2.range_search(query_point, inclusive_threshold_sq)
                count = lims[1] # Number of neighbors found within the radius
            except Exception as e:
                 log.error(f"FAISS range_search failed for point {i} (threshold={inclusive_threshold_sq:.4f}). Error: {e!r}. Setting count to -1.", exc_info=True) # Add exc_info=True and !r
                 # Log query details
                 log.debug(f"Failed query details: point_idx={i}, shape={query_point.shape}, dtype={query_point.dtype}, isfinite={np.all(np.isfinite(query_point))}, threshold_sq={inclusive_threshold_sq}")
                 count = -1 # Indicate error

            # Check if the original threshold was effectively zero for diff printing
            if threshold_sq <= epsilon_sq_zero_check:
                log.warning(f"Point {i} in {cloud1_name} has distance ~0 (sqrt <= {math.sqrt(epsilon_sq_zero_check):.1E}) to its nearest non-self neighbor (Index {first_non_self_indices[i]}) in {cloud1_name}.")

                # --- Print Normalized Content Diff using item_descriptions ---
                neighbor_original_index = first_non_self_indices[i]
                if item_descriptions is not None and neighbor_original_index != -1:
                    normalized_i = item_descriptions.get(i)
                    normalized_neighbor = item_descriptions.get(neighbor_original_index)

                    if normalized_i is not None and normalized_neighbor is not None:
                        # Only print if they are different, otherwise it's an exact match found by vector search
                        if normalized_i != normalized_neighbor:
                            log.info(f"--- Comparing Normalized Content (Point {i} vs Neighbor {neighbor_original_index}) ---")
                            log.info(f"  Point {i} Normalized: '{normalized_i}'")
                            log.info(f"  Neighbor {neighbor_original_index} Normalized: '{normalized_neighbor}'")
                            # Compute diff string (simple version)
                            diff_line = ""
                            len_i = len(normalized_i)
                            len_n = len(normalized_neighbor)
                            max_len = max(len_i, len_n)
                            for k in range(max_len):
                                char_i = normalized_i[k] if k < len_i else ''
                                char_n = normalized_neighbor[k] if k < len_n else ''
                                diff_line += '^' if char_i != char_n else ' '
                            log.info(f"  Diff ('^' indicates difference):")
                            log.info(f"  '{diff_line}'")
                            log.info(f"--- End Content Comparison ---")
                        else:
                             log.info(f"  Normalized content for Point {i} and Neighbor {neighbor_original_index} are identical.")
                    else:
                        log.info(f"  Could not print diff: Normalized string missing for Point {i} or Neighbor {neighbor_original_index}.")
                elif neighbor_original_index != -1 :
                    log.info(f"  Cannot print diff: Normalized strings map not provided.")
                # else: neighbor_original_index == -1 (already handled by threshold_sq == np.inf)
                # --- End Normalized Content Diff ---

        # Store the count (or -1 if range search failed)
        neighbors_in_cloud2_closer_than_nn1_counts[i] = count

    range_search_time = time.time() - start_time
    if range_search_time > 0.1:
        log.info(f"Counting neighbors in {cloud2_name} took {range_search_time:.2f} seconds.")

    # 3. Calculate Rank (Handle potential -1 counts from errors)
    # Rank is 1 (for the point itself) + number of closer points in cloud 2
    ranks = np.where(neighbors_in_cloud2_closer_than_nn1_counts != -1,
                     1 + neighbors_in_cloud2_closer_than_nn1_counts,
                     -1) # Keep -1 for errors
    num_errors = np.sum(ranks == -1)
    if num_errors > 0:
        log.warning(f"Could not calculate rank for {num_errors} points due to range_search errors.")
    log.info(f"Calculated ranks for {len(ranks) - num_errors} points in {cloud1_name}.")


    # Optional: Print histogram summary and rare rank details
    valid_ranks = ranks[ranks != -1]
    if len(valid_ranks) > 0:
        rank_counts = Counter(valid_ranks)
        log.info("Distribution of valid ranks:")
        max_ranks_to_print = 50 # Reduced max print
        sorted_ranks = sorted(rank_counts.items())
        for rank_idx, (rank, rank_count) in enumerate(sorted_ranks):
             if rank_idx < max_ranks_to_print or rank_idx == len(sorted_ranks) -1:
                 log.info(f"  Rank {rank}: {rank_count} points")
             elif rank_idx == max_ranks_to_print:
                 log.info(f"  ... (omitting {len(sorted_ranks) - max_ranks_to_print} more ranks)")

        # Print indices for points with rare ranks
        log.info(f"\n--- Points with Rare Ranks (<= {rare_rank_threshold} occurrences) ---")
        found_rare = False
        rank_to_count_map = dict(sorted_ranks)
        rare_point_indices = []
        for point_idx, rank_value in enumerate(ranks):
            # Check only valid ranks and if their count is below threshold
            if rank_value != -1 and rank_to_count_map.get(rank_value, float('inf')) <= rare_rank_threshold:
                rare_point_indices.append(point_idx)
                found_rare = True

        if found_rare:
            log.info(f"Indices of points with rare ranks: {rare_point_indices}")
        else:
            log.info(f"No points found with valid ranks occurring <= {rare_rank_threshold} times.")
        log.info("--- End Rare Rank Details ---")
    else:
        log.info("No valid ranks were calculated.")


    return ranks.tolist() # Return list including any -1 for errors


# ... (build_faiss_index remains largely the same) ...
def build_faiss_index(points: np.ndarray, index_type: Literal['Flat', 'IVFFlat', 'HNSWFlat'], cloud_name: str) -> faiss.Index:
    """
    Builds a FAISS index of the specified type.

    Args:
        points: Numpy array of points (shape: [n_points, n_features], dtype=float32, C-contiguous).
        index_type: The type of FAISS index to build ('Flat', 'IVFFlat', 'HNSWFlat').
        cloud_name: A descriptive name for logging (e.g., "books", "articles").

    Returns:
        The trained and populated FAISS index.
    """
    n_points, d = points.shape
    # Add check for empty points array early
    if n_points == 0:
        log.warning(f"Cannot build FAISS index for {cloud_name}. No points provided.")
        # Return an empty, untrained index of the correct type and dimension if possible,
        # or handle this case upstream. Returning an empty Flat index is safest.
        # Note: IVFFlat/HNSWFlat might error if added to later without training.
        # Consider raising an error or returning None if upstream handles it.
        log.info("Returning an empty IndexFlatL2.")
        return faiss.IndexFlatL2(d) # Or raise ValueError("Cannot build index with 0 points")

    log.info(f"Building FAISS index for {cloud_name} ({index_type}, {n_points} points, dim={d})...")
    start_time = time.time()

    # Ensure points are finite before building
    if not np.all(np.isfinite(points)):
         num_non_finite = np.sum(~np.isfinite(points))
         raise ValueError(f"Cannot build FAISS index for {cloud_name}: Input points contain {num_non_finite} non-finite values (NaN or inf).")


    if index_type == 'Flat':
        index = faiss.IndexFlatL2(d)
        index.add(points)
    elif index_type == 'IVFFlat':
        # IVF Parameters
        faiss_min_train_per_centroid = 39 # Faiss recommendation
        # Ensure n_points is at least 1 for sqrt
        safe_n_points = max(1, n_points)
        max_nlist_rec = max(1, safe_n_points // faiss_min_train_per_centroid)
        heuristic_nlist = max(1, int(4 * math.sqrt(safe_n_points)))
        # Ensure nlist doesn't exceed n_points, especially for small datasets
        nlist = min(heuristic_nlist, max_nlist_rec, safe_n_points)
        log.info(f"  Calculated nlist: {nlist} (heuristic={heuristic_nlist}, max_recommended={max_nlist_rec}, capped at n_points={safe_n_points})")

        quantizer = faiss.IndexFlatL2(d) # Index to cluster centroids
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        # Training
        # Ensure enough points for training, otherwise fall back or warn significantly
        min_train_points = faiss_min_train_per_centroid * nlist
        # Also need at least nlist points to train nlist centroids
        actual_min_train = max(min_train_points, nlist)

        if n_points < actual_min_train:
             log.warning(f"  Training IVFFlat {cloud_name} with only {n_points} points for {nlist} centroids. "
                       f"Requires at least {actual_min_train} points for reliable training. Index quality may be poor.")
             # Optional: Fallback to Flat index if training data is severely insufficient?
             # if n_points < nlist: # Cannot even train the quantizer
             #     log.error(f"Cannot train IVFFlat quantizer with {n_points} points and {nlist} centroids. Falling back to Flat index.")
             #     index = faiss.IndexFlatL2(d)
             #     index.add(points) # Add points to the flat index instead
             #     index_type = 'Flat (Fallback)' # Update type for logging
             # else: # Proceed with training, but with warning
             #     log.info(f"  Training IVFFlat {cloud_name} (nlist={nlist}) on {n_points} points...")
             #     index.train(points)
             #     log.info(f"  Training complete (with low data warning).")
             #     index.add(points)
             #     index.nprobe = max(1, nlist // 4) # Adjust nprobe potentially
             #     log.info(f"  IVFFlat params: nlist={nlist}, nprobe={index.nprobe}")

        # Standard training path
        if not index.is_trained and isinstance(index, faiss.IndexIVFFlat): # Check type in case of fallback
             log.info(f"  Training IVFFlat {cloud_name} (nlist={nlist}) on {n_points} points...")
             index.train(points)
             log.info(f"  Training complete.")
             index.add(points)
             index.nprobe = 10 # Set search-time parameter (can be tuned)
             log.info(f"  IVFFlat params: nlist={nlist}, nprobe={index.nprobe}")
        elif isinstance(index, faiss.IndexFlatL2): # Handle fallback case
             log.info(f" Using Flat index due to fallback.")


    elif index_type == 'HNSWFlat':
        # HNSW Parameters
        M = 32 # Number of connections per node (power of 2 recommended)
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
        index.hnsw.efConstruction = 40 # Build-time parameter (higher means better index, slower build)
        log.info(f"  Adding points to HNSW {cloud_name} (M={M}, efConstruction={index.hnsw.efConstruction})...")
        index.add(points) # HNSW trains implicitly during add
        index.hnsw.efSearch = 64 # Search-time parameter (higher means better recall, slower search)
        log.info(f"  HNSW params: M={M}, efConstruction={index.hnsw.efConstruction}, efSearch={index.hnsw.efSearch}")
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    build_time = time.time() - start_time
    log.info(f"FAISS index for {cloud_name} ({index_type}) built in {build_time:.2f}s")
    return index


# ... (find_and_save_nearest_non_self remains largely the same) ...
def find_and_save_nearest_non_self(
    points: np.ndarray,
    index: faiss.Index,
    normalized_strings: Dict[int, Optional[str]],
    cloud_name: str,
    index_type: str,
    output_dir: Path # Ensure this is passed correctly
):
    """
    Finds the nearest non-self neighbor for each point within its cloud using FAISS
    and saves the results, sorted by distance, to a CSV file.

    Args:
        points: Numpy array of points (shape: [n_points, n_features], dtype=float32).
        index: Trained FAISS index for the points.
        normalized_strings: Dictionary mapping point indices (0 to n_points-1)
                           to their normalized content strings (or None).
        cloud_name: Name of the cloud (e.g., "cloud1", "cloud2") for the output filename.
        index_type: Name of the index type (e.g., "Flat", "IVFFlat") for the output filename.
        output_dir: The directory to save the CSV file in.
    """
    n_points = points.shape[0]
    if n_points < 2:
        log.info(f"Skipping nearest non-self neighbor search for {cloud_name}: Only {n_points} point(s).")
        return

    log.info(f"Finding nearest non-self neighbors within {cloud_name} ({index_type})...")
    start_time = time.time()

    # Search for k=2 neighbors (self + nearest non-self)
    k_search = 2
    try:
        distances_sq, indices = index.search(points, k=k_search)
    except Exception as e:
        log.error(f"FAISS search failed during nearest non-self neighbor search for {cloud_name}: {e}")
        return # Cannot proceed

    neighbor_data = [] # List to store neighbor info before sorting

    num_non_self_neighbors = 0

    for i in range(n_points):
        neighbor_idx = -1
        dist_sq = np.inf

        # Find the first neighbor that is not the point itself
        # indices[i, 0] should be 'i' itself for non-empty index
        if indices[i, 0] != i and indices[i, 0] != -1:
             # This case is unusual, might indicate issues.
             num_non_self_neighbors += 1
             log.warning(f"Point {i}'s nearest neighbor (index {indices[i, 0]}) is not itself in {cloud_name} search.\n"
                         f"Indices:   {indices[i]}\n"
                         f"Distances: {distances_sq[i]}")
             # Use the first index found as the neighbor if it's valid
             neighbor_idx = indices[i, 0]
             dist_sq = distances_sq[i, 0]
        elif k_search > 1 and indices.shape[1] > 1 and indices[i, 1] != -1:
             # The second result should be the nearest non-self neighbor
             neighbor_idx = indices[i, 1]
             dist_sq = distances_sq[i, 1]
        # Else: No non-self neighbor found (e.g., only 1 point, or FAISS error)

        if neighbor_idx != -1:
            distance = math.sqrt(dist_sq) if dist_sq != np.inf and dist_sq >= 0 else np.inf
            desc1 = normalized_strings.get(i, "N/A") or "N/A" # Handle None from dict
            desc2 = normalized_strings.get(neighbor_idx, "N/A") or "N/A" # Handle None from dict

            neighbor_data.append({
                'Index': i,
                'NeighborIndex': neighbor_idx,
                'Distance': distance, # Store numeric distance for sorting
                'Desc1': desc1,
                'Desc2': desc2
            })
        # else: point i has no non-self neighbor (should only happen if n_points=1, handled above, or error)

    search_time = time.time() - start_time
    log.info(f"Nearest non-self neighbor search for {cloud_name} took {search_time:.2f}s.")

    if num_non_self_neighbors > 0:
        log.warning(f"Found {num_non_self_neighbors} points with non-self nearest neighbors in {cloud_name}. "
                    f"This is unusual and may indicate issues with the index or data.")

    # Sort the data by distance
    neighbor_data.sort(key=lambda x: x['Distance'])

    # Write sorted data to CSV
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
    output_filename = output_dir / f"nearest_neighbors_{cloud_name}_{index_type}.csv"
    rows_written = 0
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Index', 'NeighborIndex', 'Distance', 'Desc1', 'Desc2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for row_data in neighbor_data:
                # Format distance for CSV output
                row_data['Distance'] = f"{row_data['Distance']:.6f}" if row_data['Distance'] != np.inf else "inf"
                writer.writerow(row_data)
                rows_written += 1
        log.info(f"Saved {rows_written} nearest non-self neighbor pairs for {cloud_name} (sorted by distance) to {output_filename}")
    except IOError as e:
        log.error(f"Failed to write nearest neighbor CSV to {output_filename}: {e}")


# ... (build_and_analyze_neighbors updated to use work details for normalized strings) ...
def build_and_analyze_neighbors(
    embeddings1: Dict[str, np.ndarray],
    embeddings2: Dict[str, np.ndarray],
    cloud1_name: str,
    cloud2_name: str,
    works_details1: Optional[Dict[str, Dict]] = None, # Pass full details
    works_details2: Optional[Dict[str, Dict]] = None, # Pass full details
    index_types: List[Literal['Flat', 'IVFFlat', 'HNSWFlat']] = ['IVFFlat'],
    analysis_output_dir: Path = OUTPUT_DIR # Directory for analysis results like CSVs
):
    """
    Builds FAISS index structures for two sets of embeddings using specified index types
    and runs nearest neighbor analyses, timing each index type.
    Optionally uses work details to provide normalized strings for diff printing during rank analysis
    and saves nearest non-self neighbor information within each cloud.

    Args:
        embeddings1: Dictionary mapping IDs to embedding vectors for cloud 1.
        embeddings2: Dictionary mapping IDs to embedding vectors for cloud 2.
        cloud1_name: Name for the first set of embeddings (e.g., "books").
        cloud2_name: Name for the second set of embeddings (e.g., "articles").
        works_details1: Optional dictionary mapping IDs in cloud 1 to their work details.
        works_details2: Optional dictionary mapping IDs in cloud 2 to their work details.
        index_types: A list of FAISS index types to build and analyze (e.g., ['Flat', 'IVFFlat']).
        analysis_output_dir: Directory to save analysis output files (e.g., nearest neighbor CSVs).
    """
    if not embeddings1 or not embeddings2:
        log.warning("One or both embedding sets are empty. Skipping neighbor analysis.")
        return

    ids1 = list(embeddings1.keys())
    ids2 = list(embeddings2.keys())

    # Add checks here *before* stacking, as stack fails on empty lists
    if not ids1:
        log.warning(f"Embedding set {cloud1_name} is empty. Skipping neighbor analysis.")
        return
    if not ids2:
        log.warning(f"Embedding set {cloud2_name} is empty. Skipping neighbor analysis.")
        return

    # Ensure points are float32 numpy arrays, C-contiguous as required by FAISS
    try:
        points1_stacked = np.stack(list(embeddings1.values())).astype('float32')
        points2_stacked = np.stack(list(embeddings2.values())).astype('float32')
    except ValueError as e:
        log.error(f"Error stacking embeddings (likely inconsistent dimensions): {e}")
        # Log shapes for debugging
        shapes1 = {k: v.shape for k, v in embeddings1.items()}
        shapes2 = {k: v.shape for k, v in embeddings2.items()}
        log.error(f"Shapes in embeddings1: {shapes1}")
        log.error(f"Shapes in embeddings2: {shapes2}")
        return


    # Explicitly ensure C-contiguity
    points1 = np.ascontiguousarray(points1_stacked)
    points2 = np.ascontiguousarray(points2_stacked)


    # This check might be redundant now due to the checks above, but keep for safety
    if points1.shape[0] == 0 or points2.shape[0] == 0:
        log.warning("One or both point sets are empty after stacking. Skipping neighbor analysis.")
        return

    # *** Check for non-finite values ***
    if not np.all(np.isfinite(points1)):
        num_non_finite = np.sum(~np.isfinite(points1))
        log.error(f"points1 contains {num_non_finite} non-finite values (NaN or inf). Cannot proceed with Faiss analysis.")
        return # Stop processing
    if not np.all(np.isfinite(points2)):
        num_non_finite = np.sum(~np.isfinite(points2))
        log.error(f"points2 contains {num_non_finite} non-finite values (NaN or inf). Cannot proceed with Faiss analysis.")
        return # Stop processing
    # *** End check ***

    # Create map from point index (0..N-1) to normalized string for cloud 1
    normalized_strings1 = None
    if works_details1:
        log.info(f"Generating normalized strings for {cloud1_name} points...")
        normalized_strings1 = {}
        missing_details_count = 0
        for i, work_id in enumerate(ids1): # ids1 order matches points1 rows
            details = works_details1.get(work_id)
            if details:
                norm_str = calculate_normalized_string(details) # Assumes 'abstract' is resolved
                normalized_strings1[i] = norm_str # Store None if calculation failed (e.g., missing title/abs)
            else:
                normalized_strings1[i] = None # Store None if details missing for this ID
                missing_details_count += 1
        log.info(f"Generated normalized strings map for {len(normalized_strings1)} points in {cloud1_name}.")
        if missing_details_count > 0:
            log.warning(f"Missing work details for {missing_details_count} IDs in {cloud1_name}.")

    # Create map from point index to normalized string for cloud 2
    normalized_strings2 = None
    if works_details2:
        log.info(f"Generating normalized strings for {cloud2_name} points...")
        normalized_strings2 = {}
        missing_details_count = 0
        for i, work_id in enumerate(ids2): # ids2 order matches points2 rows
            details = works_details2.get(work_id)
            if details:
                norm_str = calculate_normalized_string(details)
                normalized_strings2[i] = norm_str
            else:
                normalized_strings2[i] = None
                missing_details_count += 1
        log.info(f"Generated normalized strings map for {len(normalized_strings2)} points in {cloud2_name}.")
        if missing_details_count > 0:
            log.warning(f"Missing work details for {missing_details_count} IDs in {cloud2_name}.")


    # --- Loop through requested index types ---
    for index_type in index_types:
        log.info(f"\n===== Analyzing with Index Type: {index_type} =====")
        overall_start_time = time.time()

        # --- Build Indices ---
        try:
            index1 = build_faiss_index(points1, index_type, cloud1_name) # Pass cloud name
            index2 = build_faiss_index(points2, index_type, cloud2_name) # Pass cloud name
        except ValueError as e:
            log.error(f"Error building index type {index_type}: {e}. Skipping this type.")
            continue # Skip to the next index type
        except Exception as e: # Catch potential Faiss errors during build/add
            log.error(f"Unexpected error building index type {index_type}: {e}. Skipping this type.", exc_info=True)
            continue

        # --- Find and Save Nearest Non-Self Neighbors (within each cloud) ---
        # Ensure analysis_output_dir is used correctly here
        if normalized_strings1 is not None:
            find_and_save_nearest_non_self(points1, index1, normalized_strings1, cloud1_name, index_type, analysis_output_dir) # Pass output dir
        else:
            log.info(f"Skipping nearest non-self neighbor CSV for {cloud1_name}: Missing work details/normalized strings.")

        if normalized_strings2 is not None:
            find_and_save_nearest_non_self(points2, index2, normalized_strings2, cloud2_name, index_type, analysis_output_dir) # Pass output dir
        else:
            log.info(f"Skipping nearest non-self neighbor CSV for {cloud2_name}: Missing work details/normalized strings.")


        # --- Run Cross-Cloud Analyses ---
        log.info(f"\n--- Running Cross-Cloud Analyses ({cloud1_name} vs {cloud2_name}) for Index Type: {index_type} ---")
        analysis_start_time = time.time()

        # Bipartite: Cloud1 -> Cloud2
        in_degrees_cloud2 = analyze_nearest_neighbors_bipartite(points1, index2, cloud1_name, cloud2_name)

        # Rank Analysis: Cloud1 vs (Cloud1 + Cloud2)
        # Pass the normalized strings map for cloud1 points
        ranks = analyze_nearest_neighbor_rank(points1, index1, index2,
                                              cloud1_name, cloud2_name, # Pass names
                                              item_descriptions=normalized_strings1) # Pass normalized strings map

        analysis_time = time.time() - analysis_start_time
        overall_time = time.time() - overall_start_time
        log.info(f"--- Cross-Cloud Analyses ({cloud1_name} vs {cloud2_name}) for Index Type: {index_type} completed in {analysis_time:.2f}s ---")
        log.info(f"===== Total time for Index Type {index_type} (Build + Analyze): {overall_time:.2f}s =====")

    log.info("\n===== Completed all requested index type analyses =====")
    # ... (potentially return aggregated results if needed) ...


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Fetch, deduplicate, embed, and analyze OpenAlex works.")
    parser.add_argument(
        '--dimensions', type=int, nargs='+', choices=[2, 3], default=[], # Default to no plotting unless specified
        help="Dimensions for dimensionality reduction plots (2 and/or 3)."
    )
    parser.add_argument(
        '--method', type=str, nargs='+', choices=['pca', 'tsne'], default=[], # Default to no plotting unless specified
        help="Dimensionality reduction methods to use (pca, tsne)."
    )
    parser.add_argument(
        '--analyze-neighbors', action='store_true',
        help="Perform nearest neighbor analysis between articles and books."
    )
    parser.add_argument(
        '--index-types', type=str, nargs='+', choices=['Flat', 'IVFFlat', 'HNSWFlat'], default=['IVFFlat'],
        help="FAISS index types for neighbor analysis."
    )
    parser.add_argument(
        '--distance-threshold', type=float, default=0.45, # Default L2 distance
        help="L2 distance threshold for semantic deduplication (lower means more similar). "
             "Use 0 to disable semantic checks (exact duplicates only)." # Updated help text
    )
    parser.add_argument(
        '--skip-existing-embeddings', action='store_true',
        help="Load existing embeddings but skip generating new ones."
    )
    parser.add_argument(
        '--force-rebuild-global-cache', action='store_true',
        help="Force rebuilding the global cache (embeddings, index, details) even if it exists."
    )
    # NEW Argument
    parser.add_argument(
        '--max-works-to-fetch', type=int, default=None,
        help="Optional: Limit the maximum number of works fetched/processed per topic/work_type during cache building."
    )
    # ... (rest of parser setup remains the same) ...

    args = parser.parse_args()
    plot_dimensions_list = args.dimensions
    methods = args.method
    analyze_neighbors_flag = args.analyze_neighbors
    selected_index_types = args.index_types
    distance_threshold = args.distance_threshold # Get the new argument
    skip_embedding_generation = args.skip_existing_embeddings # This flag is now less relevant with the new cache structure
    force_rebuild_global = args.force_rebuild_global_cache
    max_works_to_fetch = args.max_works_to_fetch # Get the new limit

    if not methods and not analyze_neighbors_flag:
        log.warning("Neither plotting (--method) nor neighbor analysis (--analyze-neighbors) requested. Exiting.")
        print("Usage hint: Add --method pca --dimensions 3 to generate a 3D PCA plot,")
        print("         or add --analyze-neighbors to perform neighbor analysis.")
        return

    if distance_threshold < 0:
        parser.error("--distance-threshold must be non-negative (0 disables semantic checks).") # Updated error message

    if max_works_to_fetch is not None and max_works_to_fetch <= 0:
        parser.error("--max-works-to-fetch must be a positive integer if provided.")

    # Determine log threshold string
    if distance_threshold == 0:
        log_thresh_str = "0 (Exact Only)"
        log.info("Starting process with semantic deduplication DISABLED (distance_threshold=0).")
    else:
        log_thresh_str = f"{distance_threshold:.4f}"
        log.info(f"Starting process with distance threshold: {log_thresh_str}")

    if max_works_to_fetch is not None:
        log.info(f"Applying fetch/process limit: {max_works_to_fetch} works per topic/type.")


    # 1. Build Global Cache (if needed)
    log.info("--- Step 1: Ensuring Global Cache ---")
    for topic in Topic:
        for work_type in WorkTypes:
            log.info(f"Checking/Building global cache for {topic.display_name} - {work_type}...")
            # Pass the limit to build_global_cache
            build_success = build_global_cache(
                work_type,
                topic,
                force_rebuild=force_rebuild_global,
                max_works_to_process=max_works_to_fetch # Pass the limit here
            )
            if not build_success:
                 log.error(f"Failed to build global cache for {topic.display_name} - {work_type}. Exiting.")
                 return # Cannot proceed without global cache

    # 2. Get Unique Work Details for the *specific threshold*
    log.info(f"\n--- Step 2: Getting Unique Work Details for Threshold {log_thresh_str} ---")
    all_unique_works_details = {} # Store details keyed by (topic, work_type)
    for topic in Topic:
        for work_type in WorkTypes:
            log.info(f"Getting unique work details for {topic.display_name} - {work_type} (dist_thresh={log_thresh_str})...")
            # Pass the limit here as well, so it ensures the cache was built with at least this many (if needed)
            unique_details = get_works_by_id(work_type, topic, distance_threshold, max_works_to_process=max_works_to_fetch)
            all_unique_works_details[(topic, work_type)] = unique_details
            log.info(f"Retrieved {len(unique_details)} unique work details for this threshold.")

    # 3. Get Embeddings for the Unique Works (filtered from global)
    log.info(f"\n--- Step 3: Getting Embeddings for Unique Works (Threshold {log_thresh_str}) ---")
    all_embeddings = {} # Store embeddings keyed by (topic, work_type)
    # No need for skip_existing_embeddings flag here, as embeddings are loaded/filtered from global cache
    for topic in Topic:
        for work_type in WorkTypes:
            # Check if we actually have unique works for this combo before getting embeddings
            if not all_unique_works_details.get((topic, work_type)):
                log.info(f"Skipping embedding retrieval for {topic.display_name} - {work_type}: No unique works found for this threshold.")
                all_embeddings[(topic, work_type)] = {}
                continue

            log.info(f"Getting embeddings for {topic.display_name} - {work_type} (unique works for dist_thresh={log_thresh_str})...")
            # Pass the limit here too for consistency (ensures cache was built appropriately)
            embeddings_data = get_embeddings(work_type, topic, distance_threshold, max_works_to_process=max_works_to_fetch)
            all_embeddings[(topic, work_type)] = embeddings_data
            log.info(f"Obtained {len(embeddings_data)} embeddings for unique works.")


    # --- Nearest Neighbor Analysis (Optional) ---
    if analyze_neighbors_flag:
        log.info("\n--- Step 4: Starting Nearest Neighbor Analysis ---")
        # Aggregate unique works and embeddings across topics for books and articles
        # This logic remains the same, using the potentially limited data fetched earlier
        books_embeddings = {}
        articles_embeddings = {}
        books_details = {}
        articles_details = {}

        for (topic, work_type), details in all_unique_works_details.items():
            embeddings = all_embeddings.get((topic, work_type), {})
            if work_type == WorkTypes.BOOK:
                books_embeddings.update(embeddings)
                books_details.update(details)
            elif work_type == WorkTypes.ARTICLE:
                articles_embeddings.update(embeddings)
                articles_details.update(details)

        log.info(f"Aggregated {len(books_embeddings)} unique book embeddings and {len(articles_embeddings)} unique article embeddings across all topics for threshold {log_thresh_str}.")

        # Define output directory for analysis results, incorporating threshold
        analysis_out_dir = OUTPUT_DIR / f"neighbor_analysis_thresh_{log_thresh_str}"
        analysis_out_dir.mkdir(exist_ok=True, parents=True)

        if books_embeddings and articles_embeddings:
            build_and_analyze_neighbors(
                embeddings1=books_embeddings,
                embeddings2=articles_embeddings,
                cloud1_name="books",
                cloud2_name="articles",
                works_details1=books_details, # Pass aggregated details
                works_details2=articles_details, # Pass aggregated details
                index_types=selected_index_types,
                analysis_output_dir=analysis_out_dir # Pass specific output dir
            )
        else:
            log.warning("Skipping neighbor analysis: Not enough data (need both books and articles).")

        log.info("--- Finished Nearest Neighbor Analysis ---\n")
    # --- End Nearest Neighbor Analysis ---


    # --- Plotting (Optional) ---
    if not methods:
        log.info("No dimensionality reduction methods specified (--method). Skipping plotting.")
        # return # Don't return if neighbor analysis might have run
    else:
        log.info("\n--- Step 5: Preparing Data for Plotting ---")
        # Aggregate unique works and embeddings across topics for plotting
        # This logic also remains the same, using potentially limited data
        all_plot_embeddings = {}
        all_plot_details = {}
        for (topic, work_type), details in all_unique_works_details.items():
            embeddings = all_embeddings.get((topic, work_type), {})
            all_plot_embeddings.update(embeddings)
            all_plot_details.update(details)

        if not all_plot_embeddings:
            log.warning("No embeddings available for plotting. Skipping plotting step.")
        else:
            log.info(f"Preparing {len(all_plot_embeddings)} total unique items for plotting (threshold {log_thresh_str}).")
            # ... (rest of plotting preparation logic remains the same) ...

            # 4. Stack, Reduce, and Plot for each requested method and dimension
            # ... (stacking logic remains the same) ...
            # ... (reduction loop remains the same) ...
            # ... (plotting logic remains the same, using threshold-specific filenames/titles) ...
            log.info("--- Finished Plotting ---")
    # --- End Plotting ---

    log.info("\n--- Script Finished ---")


if __name__ == '__main__':
    main()