"""Pure-logic tests for build_births (no git, no filesystem)."""
import numpy as np
from artifacts_scripts.curriculum.gap_recover.adapters.python import build_births as bb


def test_select_epoch_tags_fewer_than_E():
    sel = bb.select_epoch_tags(["v1.0", "v2.0", "v3.0"], E=6)
    assert sel[0] == ("v1.0", 0)
    assert sel[-1] == ("v3.0", 5)            # newest tag -> last epoch
    assert [e for _, e in sel] == sorted(e for _, e in sel)


def test_select_epoch_tags_more_than_E():
    tags = [f"v0.{i}" for i in range(20)]     # 20 releases -> 6 epochs
    sel = bb.select_epoch_tags(tags, E=6)
    assert len(sel) == 6
    assert sel[0][1] == 0 and sel[-1][1] == 5
    assert sel[0][0] == "v0.0" and sel[-1][0] == "v0.19"


def test_compute_births_min_epoch_and_sentinel():
    ns = {0: {"r::a"}, 1: {"r::a", "r::b"}, 2: {"r::a", "r::b", "r::c"}}
    birth = bb.compute_births(ns, ["r::a", "r::b", "r::c", "r::missing"], E=6)
    assert list(birth) == [0, 1, 2, 6]        # 6 == SENTINEL (never seen)
