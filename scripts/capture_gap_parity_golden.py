# scripts/capture_gap_parity_golden.py
"""Capture the deterministic gap-parity golden for a 2-topic AI-subfield subset.

Calls ``detect_gaps`` directly (uncached) so it always executes the refactored
source-agnostic path rather than reading the joblib cache populated by the
pre-refactor ``_compute_gaps_for_topic``. ``faiss.omp_set_num_threads(1)`` is
pinned because multithreaded faiss makes gap membership non-deterministic
(consecutive runs flip a few articles); single-threaded faiss is fully
reproducible, so the fixture is byte-stable across runs.

The same ``compose_ai_cfg`` / ``golden_entry_for_topic`` helpers are imported by
``tests/test_gap_parity.py`` so the capture and the assertion compute gaps the
exact same way.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from bookgap.configs.config_store import register_configs
from bookgap.gaps import detect_gaps
from bookgap.subcloud import get_subclouds_for_topic

# Two AI-subfield topics with small, fully-cached clouds.
GOLDEN_TOPICS = ("T13629", "T12761")

GOLDEN_PATH = Path("tests/fixtures/gap_parity_golden.json")

# Cloud overrides that pin the cached AI-subfield (1702) production cloud, so
# the capture runs offline against on-disk embeddings and reproduces the
# production gap output. MUST match tests/conftest.py::_CLOUD_OVERRIDES.
_CLOUD_OVERRIDES = [
    "cloud.filter_pair.pre.domain=null",
    "cloud.filter_pair.pre.subfield='1702'",
    "cloud.filter_pair.pre.work_types=[book,article]",
    "cloud.denoiser=LPP",
    "cloud.drop_k=10",
    "gap.radius_fraction=0.75",
]


def pin_faiss_single_thread() -> None:
    """Force faiss to a single OMP thread for deterministic gap membership."""
    import faiss

    faiss.omp_set_num_threads(1)


def compose_ai_cfg():
    """Compose the ``standard`` config with the pinned AI-subfield cloud overrides."""
    register_configs()
    with initialize_config_module(version_base=None, config_module="bookgap.configs"):
        return compose(config_name="standard", overrides=_CLOUD_OVERRIDES)


def golden_entries_for_topic(cloud, topic_id: str, cfg) -> dict[str, dict[str, Any]]:
    """Compute golden entries for one topic via a direct (uncached) detect_gaps call.

    Uses exactly the arguments ``compute_gaps_for_config`` / ``_compute_gaps_for_topic``
    would pass, so the capture and the parity test share one code path.
    Returns a dict keyed by ``"{topic_id}:{gap_idx}"``.
    """
    landmark, ambient = get_subclouds_for_topic(cloud, topic_id)
    gaps = detect_gaps(
        ambient=ambient,
        landmark=landmark,
        query=ambient,
        cell_type=cfg.gap.cell_type,
        num_gaps=cfg.batch.num_gaps,
        radius_fraction=cfg.gap.radius_fraction,
        deflation_strategy=cfg.gap.deflation_strategy,
        size_metric=cfg.gap.size_metric,
        candidate_pool_size=cfg.gap.candidate_pool_size,
        label_id=topic_id,
        label_name=topic_id,
    )
    entries: dict[str, dict[str, Any]] = {}
    for g in gaps:
        ambient_ids = ambient.work_ids()
        entries[f"{topic_id}:{g.gap_idx}"] = {
            "in_gap_article_ids": sorted(
                ambient_ids[int(i)] for i in g.gap.get_dists_idxs().idxs
            ),
            "gap_size": int(g.gap_size),
            "gap_size_percentile": round(float(g.gap_size_percentile), 9),
        }
    return entries


def main() -> None:
    pin_faiss_single_thread()
    cfg = compose_ai_cfg()
    cloud = instantiate(cfg.cloud)
    golden: dict[str, dict[str, Any]] = {}
    for topic_id in GOLDEN_TOPICS:
        golden.update(golden_entries_for_topic(cloud, topic_id, cfg))
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2, sort_keys=True))
    print(f"Wrote {GOLDEN_PATH} with {len(golden)} (topic,gap) entries")


if __name__ == "__main__":
    main()
