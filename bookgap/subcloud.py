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

    def name(self) -> str:
        return f"openalex[{self.post_filter}]"

    @property
    def embedder_fingerprint(self) -> str:
        from bookgap.corpus import embedder_fingerprint
        return embedder_fingerprint(self.parent.embedder)

    @property
    def identity(self):
        return self  # frozen dataclass: hashable identity = parent fingerprint + post_filter


@lru_cache(maxsize=8)
@MEMORY.cache
def get_subcloud_retained_idxs(subcloud: SubCloud) -> NDArray[np.uint64]:
    # Fast path: if the post_filter implements a vectorised columnar mask,
    # skip the ~6-10 GB load of every work dict and just do numpy ops.
    mask = subcloud.post_filter.compute_columnar_mask(subcloud.parent)
    if mask is not None:
        expected_len = len(subcloud.parent.work_ids())
        assert mask.shape == (expected_len,), (
            f"compute_columnar_mask returned shape {mask.shape}, "
            f"expected ({expected_len},); the mask must be 1-D and aligned "
            f"with cloud.works() row positions."
        )
        return np.where(np.asarray(mask, dtype=bool))[0].astype(np.uint64)
    return np.array(
        [i for i, work in enumerate(subcloud.parent.works()) if subcloud.post_filter.check_work(work)],
        dtype=np.uint64,
    )


def get_subclouds_for_topic(
    cloud: Cloud,
    topic_id: Optional[str],
    ambient_work_type: str = "article",
    landmark_work_type: str = "book",
) -> tuple[SubCloud, SubCloud]:
    """Return (landmark, ambient) slices for the topic.

    Defaults reproduce the historical (books, articles) split byte-for-byte.
    """
    landmark = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_types=(landmark_work_type,)))
    ambient = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_types=(ambient_work_type,)))
    return landmark, ambient
