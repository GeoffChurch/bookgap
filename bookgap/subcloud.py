from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from bookgap.cloud import Cloud
from bookgap.index import build_index
from bookgap.util import Filter, MEMORY


@dataclass(frozen=True)
class SubCloud:
    """Lightweight view over a parent cloud filtered by a post_filter."""

    parent: Cloud
    post_filter: Filter

    @property
    def normalize_embeddings(self) -> bool:
        return self.parent.normalize_embeddings

    def works(self) -> list[dict[str, Any]]:
        parent_works = self.parent.works()
        retained_idxs = get_subcloud_retained_idxs(self)
        return [parent_works[i] for i in retained_idxs]

    def work_ids(self) -> list[str]:
        parent_work_ids = self.parent.work_ids()
        retained_idxs = get_subcloud_retained_idxs(self)
        return [parent_work_ids[i] for i in retained_idxs]

    def work_references(self) -> list[list[str]]:
        parent_work_references = self.parent.work_references()
        retained_idxs = get_subcloud_retained_idxs(self)
        return [parent_work_references[i] for i in retained_idxs]

    def work_titles(self) -> list[str]:
        parent_work_titles = self.parent.work_titles()
        retained_idxs = get_subcloud_retained_idxs(self)
        return [parent_work_titles[i] for i in retained_idxs]

    def work_abstracts(self) -> list[str]:
        parent_work_abstracts = self.parent.work_abstracts()
        retained_idxs = get_subcloud_retained_idxs(self)
        return [parent_work_abstracts[i] for i in retained_idxs]

    def embeddings(self) -> NDArray[np.float32]:
        parent_embeddings = self.parent.embeddings()
        retained_idxs = get_subcloud_retained_idxs(self)
        return parent_embeddings[retained_idxs]

    def index(self):
        return build_index(self.parent.index_info, self.embeddings())


@lru_cache(maxsize=8)
@MEMORY.cache
def get_subcloud_retained_idxs(subcloud: SubCloud) -> NDArray[np.uint64]:
    return np.array(
        [i for i, work in enumerate(subcloud.parent.works()) if subcloud.post_filter.check_work(work)],
        dtype=np.uint64,
    )


def get_subclouds_for_topic(cloud: Cloud, topic_id: Optional[str]) -> tuple[SubCloud, SubCloud]:
    """Return (books, articles) slices for the given topic using the parent cloud cache."""
    books = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_types=("book",)))
    articles = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_types=("article",)))
    return books, articles
