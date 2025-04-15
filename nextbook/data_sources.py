import abc
import json
import os
import requests
import time
from typing import Iterator, Dict, Any, Optional, List, Literal
from collections import defaultdict
from dataclasses import dataclass
import enum
from pathlib import Path
import pickle
import pyalex
from pyalex import Works, Topics
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

pyalex.config.email = "gchurchill@cs.stonybrook.edu"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = OUTPUT_DIR / "api_data"
DATA_DIR.mkdir(exist_ok=True)

KEY_DIR = OUTPUT_DIR / "api_keys"
KEY_DIR.mkdir(exist_ok=True)

EMBEDDING_DIR = OUTPUT_DIR / "embeddings"
EMBEDDING_DIR.mkdir(exist_ok=True)


# TOPICS = {
#     'Natural Language Processing Techniques': 'T10181',
#     'Robotics and Sensor-Based Localization': 'T10191'
# }

@dataclass
class Topic:
    display_name: str
    id: str

    def __post_init__(self):
        # Assert that the display_name matches the id
        topic = Topics().get(self.id)
        assert topic.display_name == self.display_name, f"Display name mismatch: {topic.display_name} != {self.display_name}"

class Topics(enum.Enum):
    NLP      = Topic("Natural Language Processing Techniques", "T10181")
    ROBOTICS = Topic("Robotics and Sensor-Based Localization", "T10191")

## To find a topic:
## https://api.openalex.org/topics?filter=display_name.search:natural%20language%20processing&select=id,display_name
## Or, programmatically:
# for page in Topics() \
#     .search_filter(display_name="natural language processing") \
#     .select("id, display_name, description, subfield, field, domain, relevance_score, works_count, cited_by_count") \
#     .paginate(per_page=200, n_max=None):
#     for topic in page:
#         for key in topic:
#             print(f"{key}: {topic[key]}")

# NLP_TOPIC = "https://openalex.org/T10181"

# From https://sqlpey.com/python/top-4-methods-to-create-safe-filenames-from-any-unsafe-string/#method-2-including-unicode-characters
def create_unicode_safe_filename(filename):
    return re.sub(r'[^a-zA-Z0-9 _.-]', '', filename).rstrip()

# def safe_fname_concat(head: Path, *tail: str) -> Path:
#     for part in tail:
#         head /= create_unicode_safe_filename(part)
#     return head

def load_api_key(service_name: str) -> str:
    key_path = KEY_DIR / f"{service_name}.key"
    assert key_path.parts[-1] == f"{service_name}.key", f"Invalid key path: {key_path}"
    with open(key_path, 'r') as f:
        return f.read()

def resolve_abstract_inverted_index(abstract_inverted_index: Dict[str, List[int]]) -> str:
    """
    Converts the abstract inverted index into a human-readable string.
    """
    pos_to_word = {}
    for word, positions in abstract_inverted_index.items():
        assert " " not in word, f"Word contains space: '{word}'"
        for pos in positions:
            assert pos not in pos_to_word, f"Duplicate position {pos} for word {word}"
            pos_to_word[pos] = word
    return " ".join(pos_to_word.get(i, "[[MISSING]]") for i in range(len(pos_to_word)))

def open_alex(type_: Literal["article", "book"], topic: Topic):
    # Retrieve next cursor from state file
    output_dir = DATA_DIR / create_unicode_safe_filename(topic.display_name) / create_unicode_safe_filename(type_)
    output_dir.mkdir(exist_ok=True, parents=True)
    state_file = output_dir / f"state.json"
    if not state_file.exists():
        with open(state_file, "w") as f:
            json.dump(dict(next_cursor="*", next_page=1), f)
    with open(state_file, 'r') as f:
        state = json.load(f)
        next_cursor = state["next_cursor"]
        next_page = state["next_page"]

    query = Works() \
        .filter(has_abstract=True, type=type_, **{"primary_topic.id": topic.id}) \
        .select(", ".join(["display_name", "type", "publication_date", "ids", "abstract_inverted_index"]))

    total_expected = query.count()
    
    # print(f"Total expected results for type {type_}: {total_expected}")

    pbar = tqdm(total=total_expected, desc=f"Processing {topic.display_name} works of type {type_}", unit="items")

    # Yield the cached pages
    for i in range(1, next_page):
        page_file = output_dir / f"page_{i}.json"
        if page_file.exists():
            with open(page_file, 'r') as f:
                # yield from json.load(f)
                for work in json.load(f):
                    pbar.update(1)
                    yield work
        else:
            raise FileNotFoundError(f"Page file {page_file} missing.")

    last_time = time.time()

    for page in query.paginate(cursor=next_cursor, per_page=200, n_max=None):
        for work in page:
            # if "abstract_inverted_index" in work:
            assert "abstract_inverted_index" in work, f"Missing abstract_inverted_index for {work['id']}"
            work["abstract"] = resolve_abstract_inverted_index(work["abstract_inverted_index"])
            del work["abstract_inverted_index"]

        for work in page:
            pbar.update(1)
            yield work
        
        # Save the page to a file
        with open(output_dir / f"page_{next_page}.json", "w") as f:
            json.dump(page, f, indent="\t")

        # Save the next cursor for resuming later
        next_cursor = page.meta["next_cursor"]
        next_page += 1

        with open(state_file, "w") as f:
            json.dump(dict(next_cursor=next_cursor, next_page=next_page), f)

        # Rate limiting
        MIN_DIFF = 1
        this_time = time.time()
        time.sleep(max(0, MIN_DIFF - (this_time - last_time)))
        last_time = this_time
    
    pbar.close()

def get_embeddings(type_: Literal["article", "book"], topic: Topic):
    # sentences = ["This is an example sentence", "Each sentence is converted"]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    cache_frequency = 1000

    # cache_file = EMBEDDING_DIR / f"embeddings_{type_}.pkl"
    # assert cache_file.parts[-1] == f"embeddings_{type_}.pkl", f"Invalid cache file path: {cache_file}"
    cache_file = EMBEDDING_DIR / create_unicode_safe_filename(topic.display_name) / f"{create_unicode_safe_filename(type_)}.pkl"
    cache_file.parent.mkdir(exist_ok=True, parents=True)
    if not cache_file.exists():
        with open(cache_file, 'wb') as f:
            pickle.dump({}, f)

    with open(cache_file, 'rb') as f:
        start_time = time.time()
        embeddings = pickle.load(f)
        if (diff := time.time() - start_time) > 1:
            print(f"Loading took {diff:.2f} seconds.")
        print(f"Loaded {len(embeddings)} embeddings from cache.")

    newly_embedded = 0
    for work in open_alex(type_, topic):
        # print(f"work:\n{work}")
        assert "ids" in work and "openalex" in work["ids"], f"Missing openalex id for work:\n{work}"
        work_id = work["ids"]["openalex"]
        if work_id in embeddings:
            continue
        embedding = model.encode(work["abstract"])
        embeddings[work_id] = embedding
        newly_embedded += 1
        if newly_embedded != 0 and newly_embedded % cache_frequency == 0:
            # print(f"Saving {len(embeddings)} embeddings to cache ({newly_embedded=}).") # TODO consider paging
            with open(cache_file, 'wb') as f:
                start_time = time.time()
                pickle.dump(embeddings, f)
                if (diff := time.time() - start_time) > 1:
                    print(f"Saving {len(embeddings)} embeddings took {diff:.2f} seconds.")
        

def main():
    get_embeddings("book", Topics.NLP.value)
    get_embeddings("article", Topics.NLP.value)
    get_embeddings("book", Topics.ROBOTICS.value)
    get_embeddings("article", Topics.ROBOTICS.value)

if __name__ == '__main__':
    # types = ["article", "book"]
    # iterators = [(t, open_alex(t)) for t in types]
    # type_counts = defaultdict(lambda: defaultdict(int))
    # item_counter = 0

    # while iterators:
    #     for i, (name, it) in enumerate(list(iterators)):
    #         try:
    #             item = next(it)
    #             item_type = item.get('type', 'unknown')
    #             type_counts[name][item_type] += 1
    #             item_counter += 1
    #             if item_counter % 10000 == 0:
    #                 print(f"--- Retrieved {item_counter} items ---")
    #                 for src, counts in type_counts.items():
    #                     print(f"{src}:")
    #                     for t, c in counts.items():
    #                         print(f"  {t}: {c}")
    #         except StopIteration:
    #             print(f"--- Finished retrieving items from {name} ---")
    #             iterators.pop(i)
    main()