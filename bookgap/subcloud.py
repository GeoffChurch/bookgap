from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from bookgap.cloud import (
    Cloud,
    cloud_embeddings_parquet_positions,
    denoise,
    read_embeddings_rows,
)
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
        # Read ONLY this subcloud's rows from the embeddings parquet rather
        # than materialising the entire parent cloud (58+ GiB at domain
        # scope) just to fancy-index a tiny subset out of it.
        #
        # The parent maps its own row i to parquet row parent_positions[i]
        # (post-dedup, post-filter; ascending). get_subcloud_retained_idxs
        # indexes into the parent's retained set, so this subcloud's parquet
        # positions are parent_positions[retained_idxs]. The parent's
        # denoising is applied row-wise (see cloud.denoise: a per-row affine
        # projection + optional per-row normalize), so reading the raw rows
        # and denoising them reproduces parent.embeddings()[retained_idxs]
        # byte-for-byte.
        emb_path, parent_positions = cloud_embeddings_parquet_positions(self.parent)
        retained_idxs = get_subcloud_retained_idxs(self)
        sub_positions = parent_positions[retained_idxs]

        # read_embeddings_rows requires ascending positions. parent_positions
        # is ascending and retained_idxs is ascending, so sub_positions is
        # normally already ascending and no permutation is needed. Handle the
        # general case anyway: sort, read, then invert to restore row order.
        n = sub_positions.shape[0]
        if n > 1 and not np.all(sub_positions[:-1] <= sub_positions[1:]):
            order = np.argsort(sub_positions, kind="stable")
            raw_sorted = read_embeddings_rows(emb_path, sub_positions[order])
            inverse = np.empty(n, dtype=np.int64)
            inverse[order] = np.arange(n, dtype=np.int64)
            raw = raw_sorted[inverse]
        else:
            raw = read_embeddings_rows(emb_path, sub_positions)

        return denoise(self.parent, raw)

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
