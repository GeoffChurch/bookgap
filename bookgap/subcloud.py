
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import faiss
import numpy as np
from numpy.typing import NDArray

from bookgap.cloud import Cloud
from bookgap.index import build_index
from bookgap.util import MEMORY, Filter, WorkType


@dataclass(frozen=True)
class SubCloud:
    parent: Cloud
    post_filter: Filter

    def works(self) -> list[dict[str, Any]]:
        parent_works = self.parent.works()
        retained_idxs = get_subcloud_retained_idxs(self)
        return [parent_works[i] for i in retained_idxs]        
    
    def work_ids(self) -> list[str]:
        return get_subcloud_work_ids(self)
    
    def work_references(self) -> list[list[str]]:
        return get_subcloud_work_references(self)
    
    def index(self) -> faiss.Index:
        return get_subcloud_index(self)
    
    def embeddings(self):
        return get_subcloud_embeddings(self)
    
    @property
    def normalize_embeddings(self) -> bool:
        return self.parent.normalize_embeddings

@lru_cache(maxsize=2)
@MEMORY.cache
def get_subcloud_retained_idxs(subcloud: SubCloud) -> NDArray[np.uint64]:
    return np.array([i for i, work in enumerate(subcloud.parent.works()) if subcloud.post_filter.check_work(work)], dtype=np.uint64)

@lru_cache(maxsize=2)
@MEMORY.cache
def get_subcloud_work_ids(subcloud: SubCloud) -> list[str]:
    parent_work_ids = subcloud.parent.work_ids()
    retained_idxs = get_subcloud_retained_idxs(subcloud)
    return [parent_work_ids[i] for i in retained_idxs]

@lru_cache(maxsize=2)
@MEMORY.cache
def get_subcloud_work_references(subcloud: SubCloud) -> list[list[str]]:
    parent_work_references = subcloud.parent.work_references()
    retained_idxs = get_subcloud_retained_idxs(subcloud)
    return [parent_work_references[i] for i in retained_idxs]

@lru_cache(maxsize=2)
def get_subcloud_index(subcloud: SubCloud) -> faiss.Index:
    return build_index(subcloud.parent.index_info, get_subcloud_embeddings(subcloud))

@lru_cache(maxsize=2)
def get_subcloud_embeddings(subcloud: SubCloud):
    parent_embeddings = subcloud.parent.embeddings()
    retained_idxs = get_subcloud_retained_idxs(subcloud)
    return parent_embeddings[retained_idxs]

def get_subclouds_for_topic(cloud: Cloud, topic_id: Optional[str]) -> tuple:
    """Return (books, articles) SubClouds for the given topic_id."""
    books = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_types=(WorkType.BOOK.value,)))
    articles = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_types=(WorkType.ARTICLE.value,)))
    return books, articles
