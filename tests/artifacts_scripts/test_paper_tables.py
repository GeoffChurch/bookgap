"""The paper's radius-sweep table and denoiser-difference intervals, on toy inputs."""
from __future__ import annotations

import math

from scipy import stats

from artifacts_scripts.paper.denoiser_difference_cis import paired_difference
from artifacts_scripts.paper.radius_sweep_table import Row, render


def test_paired_difference_uses_shared_topics_and_a_t_interval():
    a = {"T1": 10.0, "T2": 20.0, "T3": 30.0}
    b = {"T1": 8.0, "T2": 15.0, "T3": 31.0, "T4": 99.0}   # T4 is not shared
    d = paired_difference(a, b)                           # differences 2, 5, -1
    assert d.n == 3
    assert math.isclose(d.mean, 2.0)
    assert math.isclose(d.half_width, stats.t.ppf(0.975, 2) * 3.0 / math.sqrt(3))
    assert not d.excludes_zero


def test_paired_difference_excludes_zero_when_the_interval_does():
    a = {f"T{i}": 50.0 + i for i in range(20)}
    b = {f"T{i}": 40.0 + i + (0.5 if i % 2 else -0.5) for i in range(20)}
    d = paired_difference(a, b)
    assert math.isclose(d.mean, 10.0) and d.excludes_zero


def test_radius_table_orders_by_p_and_bolds_every_best_value():
    rows = [
        Row(1.0, (51.5, 57.0, 59.0, 74.3), 1040),
        Row(0.5, (80.6, 100.0, 65.5, 97.7), 80),
        Row(0.5625, (81.4, 100.0, 68.1, 97.8), 120),
    ]
    lines = render(rows).splitlines()
    body = lines[4:7]
    assert [line.split(" & ")[0] for line in body] == ["$1/2$", "$9/16$", "$1$"]
    assert body[0] == r"$1/2$ & 80.6 & \textbf{100.0} & 65.5 & 97.7 & 80 \\"
    assert body[1] == r"$9/16$ & \textbf{81.4} & \textbf{100.0} & \textbf{68.1} & \textbf{97.8} & 120 \\"
    assert "textbf" not in body[2]
