import argparse
from dataclasses import dataclass, field
import enum
import logging
import faiss
import itertools
import json
import numpy as np
from pathlib import Path
import pyalex
import re
from sentence_transformers import SentenceTransformer
import time
import tqdm
from typing import Dict, List, Optional

pyalex.config.email = "gchurchill@cs.stonybrook.edu"

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
        # topic = pyalex.Topics()[self._topic_id]
        # assert topic['display_name'] == self._display_name, f"Display name mismatch: {topic.display_name} != {self._display_name}"

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

def create_unicode_safe_filename(s):
    return re.sub(r'[^a-zA-Z0-9 _.-]', '', str(s)).rstrip()

def get_query(work_type: WorkTypes, topic: Topic):
    # .sort(publication_date="asc", display_name="asc") \
    return pyalex.Works() \
            .filter(has_abstract=True, type=str(work_type), **{"primary_topic.id": topic.topic_id}) \
            .select(", ".join(["display_name", "publication_date", "ids", "abstract_inverted_index"]))

def resolve_abstract_inverted_index(abstract_inverted_index: Optional[Dict[str, List[int]]]) -> str:
    if not abstract_inverted_index:
        return ""
    pos_to_word = {}
    for word, positions in abstract_inverted_index.items():
        assert " " not in word, f"Word contains space: '{word}'"
        for pos in positions:
            assert pos not in pos_to_word, f"Duplicate position {pos} for word {word}"
            pos_to_word[pos] = word
    max_pos = max(pos_to_word.keys())
    words = [pos_to_word.get(i, "[[MISSING]]") for i in range(max_pos + 1)]
    return " ".join(words)

def get_works(work_type: WorkTypes, topic: Topic, cache_dir: Path):
    cache_dir = cache_dir / "api_data" / create_unicode_safe_filename(f"{topic.display_name}_{work_type}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_file = cache_dir / "state.json"
    if not state_file.exists():
        with state_file.open("w") as f:
            json.dump({"next_cursor": "*", "next_page": 1}, f)
    with state_file.open("r") as f:
        state = json.load(f)

    # First yield cached works
    for page in range(1, state["next_page"]):
        page_file = cache_dir / f"page_{page}.json"
        with page_file.open("r") as f:
            yield from json.load(f)
    
    # This isn't actually necessary, but it's clearer
    if state["next_cursor"] is None:
        return

    # print(f"Fetching new works for {work_type} in {topic.display_name}, starting from page {state['next_page']}")
    # Then yield new works
    last_time = time.time() # For rate limiting
    for i, page in enumerate(get_query(work_type, topic).paginate(cursor=state["next_cursor"], per_page=200, n_max=None)):
        next_cursor = page.meta["next_cursor"]
        # print(i, page.meta)
        # print(f"next_page: {state['next_page']}, next_cursor: {next_cursor}")
        if next_cursor is None:
            with state_file.open("w") as f:
                json.dump({"next_cursor": None, "next_page": state["next_page"]}, f)
            return
        page = list(map(dict, page))
        for work in page:
            work["abstract"] = resolve_abstract_inverted_index(work["abstract_inverted_index"])
            del work["abstract_inverted_index"]
            work["id"] = work["ids"]["openalex"]
            del work["ids"]
        # Save page
        page_file = cache_dir / f"page_{state['next_page']}.json"
        with page_file.open("w") as f:
            json.dump(page, f, indent=2)
        # Save state
        state["next_page"] += 1
        state["next_cursor"] = next_cursor
        with state_file.open("w") as f:
            json.dump(state, f)
        # Yield works
        yield from page
        # Rate limit
        min_time_diff = 1.0 # Minimum time between requests
        time.sleep(max(0, min_time_diff - (time.time() - last_time)))
        last_time = time.time()

def batched(iterable, n): # TODO use itertools.batched when available
    # batched('ABCDEFG', 3) → ABC DEF G
    iterator = iter(iterable) # If already an iterator, `iterator is iterable == True`
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch

def embed_works(work_type: WorkTypes, topic: Topic, cache_dir: Path):
    works = iter(get_works(work_type, topic, cache_dir))
    emb_cache_dir = cache_dir / "embeddings" / create_unicode_safe_filename(f"{topic.display_name}_{work_type}")
    emb_cache_dir.mkdir(parents=True, exist_ok=True)

    # Yield cached works and embeddings
    i = 1
    seen = 0
    while True:
        try:
            embeddings = np.load(emb_cache_dir / f"batch_{i}_embeddings.npy")
            with (emb_cache_dir / f"batch_{i}_ids.json").open("r") as ids_f:
                embedding_ids = json.load(ids_f)
            for embedding_id, embedding in zip(embedding_ids, embeddings, strict=True):
                # zip(works, embedding_ids, embeddings) was a nasty bug because it would
                # discard the work pulled just before embedding_ids raised StopIteration
                work = next(works) # So, manually call `next` here
                assert work["id"] == embedding_id, f"ID mismatch: {work['id']} != {embedding_id} at index {seen}"
                work["embedding"] = embedding
                seen += 1
                yield work
            i += 1
        except FileNotFoundError:
            break
            
    # Yield new works and embeddings
    model = None # Lazy load the model in case `works` is already exhausted
    batch_size = 512
    for batch in batched(works, batch_size):
        if model is None:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # Embed batch
        inputs = [f"{work['display_name']}\n{work['abstract']}" for work in batch]
        embeddings = model.encode(inputs, batch_size=batch_size, show_progress_bar=False)
        # Save batch
        with (emb_cache_dir / f"batch_{i}_embeddings.npy").open("wb") as embeddings_f, (emb_cache_dir / f"batch_{i}_ids.json").open("w") as ids_f:
            np.save(embeddings_f, embeddings)
            json.dump([work["id"] for work in batch], ids_f)
        # Yield works and embeddings
        for work, embedding in zip(batch, embeddings):
            work["embedding"] = embedding
            yield work
        i += 1

@dataclass(frozen=True)
class Cloud:
    name: str
    works: List[Dict]
    index: faiss.IndexHNSWFlat

    # def __post_init__(self):
    #     logging.info(f"Cloud {self.name} -- Indexing {len(self.works)} works")
    #     self.embeddings = np.empty((len(self.works), len(self.works[0]["embedding"])), dtype=np.float32)
    #     for i, work in enumerate(self.works):
    #         self.embeddings[i] = work["embedding"]
    #     self.index = faiss.IndexHNSWFlat(
    #         self.embeddings.shape[1], # d
    #         16, # M
    #         faiss.METRIC_L2,
    #     )
    #     self.index.hnsw.efConstruction = 2048
    #     self.index.hnsw.efSearch = 128
    #     logging.info(f"Cloud {self.name} -- Indexing {len(self.embeddings)} embeddings of dimension {self.embeddings.shape[1]}")
    #     start_time = time.time()
    #     self.index.add(self.embeddings)
    #     logging.info(f"Cloud {self.name} -- Indexing took {time.time() - start_time:.2f} seconds")
        

def get_cloud(work_type: WorkTypes, topic: Topic, cache_dir: Path) -> Cloud:
    cloud_cache_dir = cache_dir / "clouds"
    cloud_cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"{topic.display_name}_{work_type}"
    logging.info(f"Getting {name} cloud")
    works = list(tqdm.tqdm(embed_works(work_type, topic, cache_dir), desc="Loading works", disable=False))
    logging.info(f"Cloud {name} -- Loaded {len(works)} works")
    faiss_file = cloud_cache_dir / f"{name}.faiss"
    ids_file = cloud_cache_dir / f"{name}_ids.json"

    def build() -> faiss.Index:
        logging.info(f"Cloud {name} -- Building index")
        faiss_file.unlink(missing_ok=True)
        ids_file.unlink(missing_ok=True)
        embeddings = np.empty((len(works), len(works[0]["embedding"])), dtype=np.float32)
        for i, work in enumerate(works):
            embeddings[i] = work["embedding"]
        index = faiss.IndexHNSWFlat(
            embeddings.shape[1], # d
            16, # M
            faiss.METRIC_L2,
        )
        index.hnsw.efConstruction = 2048
        index.hnsw.efSearch = 128
        logging.info(f"Cloud {name} -- Indexing {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        start_time = time.time()
        index.add(embeddings)
        faiss.write_index(index, str(faiss_file))
        logging.info(f"Cloud {name} -- Indexing took {time.time() - start_time:.2f} seconds")
        with ids_file.open("w") as f:
            json.dump([work["id"] for work in works], f)
        return index

    def get_cached() -> Optional[faiss.Index]:
        if not faiss_file.exists():
            return None
        if not ids_file.exists():
            logging.warning(f"Cloud {name} -- ID file not found: {ids_file}, but index file exists: {faiss_file}")
            return None
        logging.info(f"Cloud {name} -- Loading cached index")
        index: faiss.Index = faiss.read_index(str(faiss_file))
        with ids_file.open("r") as f:
            ids = json.load(f)
        if len(ids) != index.ntotal:
            logging.warning(f"Cloud {name} -- ID count mismatch: {len(ids)} != {index.ntotal}")
            return None
        if len(works) != index.ntotal:
            logging.warning(f"Cloud {name} -- Works count mismatch: {len(works)} != {index.ntotal}")
            return None
        for i in range(index.ntotal):
            if works[i]["id"] != ids[i]:
                logging.warning(f"Cloud {name} -- ID mismatch: {works[i]['id']} != {ids[i]} at index {i}")
                return None
        return index
    
    index = get_cached() or build()

    return Cloud(name=name, works=works, index=index)
    
    

def get_rank_of_nearest_incloud_neighbor(selfcloud, othercloud):
    # First, get distance to nearest incloud neighbor
    logging.info(f"Getting distance to nearest neighbor in {selfcloud.name}")
    start_time = time.time()
    D, I = selfcloud.index.search(selfcloud.embeddings, 1)




def main():
    parser = argparse.ArgumentParser(description="Fetch, deduplicate, embed, and analyze OpenAlex works.")
    parser.add_argument("--cache_dir", type=Path, default="output", help="Directory to cache data")
    parser.add_argument("--log_level", type=str, default=logging._levelToName[logging.INFO], choices=sorted(logging._nameToLevel, key=logging._nameToLevel.get), help="Logging level")
    parser.add_argument("--max_works", type=int, default=None, help="Maximum number of works to fetch")
    args = parser.parse_args()
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    logging_level = logging._nameToLevel[args.log_level]
    max_works = args.max_works
    logging.basicConfig(
        filename=cache_dir / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{max_works=}.log",
        level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s")
    for work_type, topic in itertools.product(WorkTypes, Topic):
        expected = get_query(work_type, topic).count()
        expected = min(max_works, expected) if max_works is not None else expected
        for work in tqdm.tqdm(itertools.islice(embed_works(work_type, topic, cache_dir), max_works),
                              total=expected, disable=False):
            # print(f"Work: {work['display_name']}")
            # print(f"Embedding: {work['embedding'][:5]}...")
            # print(f"Type: {work['type']}")
            # print(f"Publication Date: {work['publication_date']}")
            # print(f"Abstract: {work['abstract']}")
            # print("-" * 40)
            pass
        faiss_indices = {
            (work_type, topic): get_cloud(work_type, topic, cache_dir)
            for work_type, topic in itertools.product(WorkTypes, Topic)
        }

if __name__ == "__main__":
    main()