"""Parity gate: detect_gaps reproduces the captured single-threaded golden.

The real proof the Task-5 refactor preserved behaviour is that ``detect_gaps``
is a rename-only lift of the old ``_compute_gaps_for_topic`` body. This test is
the determinism/regression guard on top of that: it recomputes gaps via the
same direct (uncached) ``detect_gaps`` call the capture script uses and asserts
byte-equality against ``tests/fixtures/gap_parity_golden.json``.

faiss is pinned to one thread because multithreaded faiss makes gap membership
non-deterministic; the golden was captured single-threaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra.utils import instantiate

from scripts.capture_gap_parity_golden import (
    GOLDEN_PATH,
    GOLDEN_TOPICS,
    compose_ai_cfg,
    golden_entries_for_topic,
    pin_faiss_single_thread,
)


@pytest.mark.integration
def test_detect_gaps_matches_golden() -> None:
    golden = json.loads(Path(GOLDEN_PATH).read_text())
    assert golden, "golden fixture is empty; run scripts/capture_gap_parity_golden.py"

    pin_faiss_single_thread()
    cfg = compose_ai_cfg()
    cloud = instantiate(cfg.cloud)

    recomputed: dict[str, dict] = {}
    for topic_id in GOLDEN_TOPICS:
        recomputed.update(golden_entries_for_topic(cloud, topic_id, cfg))

    assert set(recomputed) == set(golden), (
        f"(topic,gap) keys diverged: only-recomputed={set(recomputed) - set(golden)}, "
        f"only-golden={set(golden) - set(recomputed)}"
    )
    for key in golden:
        exp, got = golden[key], recomputed[key]
        assert got["gap_size"] == exp["gap_size"], f"{key}: gap_size {got['gap_size']} != {exp['gap_size']}"
        assert got["gap_size_percentile"] == exp["gap_size_percentile"], (
            f"{key}: gap_size_percentile {got['gap_size_percentile']} != {exp['gap_size_percentile']}"
        )
        assert got["in_gap_article_ids"] == exp["in_gap_article_ids"], (
            f"{key}: in_gap_article_ids diverged"
        )
