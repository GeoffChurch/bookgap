"""The paper's generated tables and reports, on toy inputs."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from artifacts_scripts.paper.book_article_separability import topic_matched_sample
from artifacts_scripts.paper.denoiser_difference_cis import paired_difference
from artifacts_scripts.paper.generate_latex_tables import generate_combined_best_denoiser_table
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


def _denoiser_runs(rows):
    return pd.DataFrame(rows, columns=["denoiser", "drop_k", "gap_coverage_fraction_mean",
                                       "proposal_gap_size_percentile_mean"])


def _combined_table(tmp_path, gist_rows, gemma_rows) -> str:
    out = tmp_path / "combined.tex"
    generate_combined_best_denoiser_table(
        {"GIST": _denoiser_runs(gist_rows), "Gemma": _denoiser_runs(gemma_rows)}, out, tabular_only=True)
    return out.read_text()


def test_denoiser_table_omits_k_when_each_denoiser_has_one_run(tmp_path):
    text = _combined_table(
        tmp_path,
        [("NONE", 0, 0.80, 0.70), ("LPP", 10, 0.86, 0.75)],
        [("NONE", 0, 0.84, 0.28), ("LPP", 394, 0.83, 0.88)],
    )
    assert r"\begin{tabular}{l|rr|rr}" in text
    assert "Denoiser & GIST & Gemma & GIST & Gemma" in text
    assert "$k$" not in text
    assert r"LPP & \textbf{86.00} & 83.00 & \textbf{75.00} & \textbf{88.00} \\" in text


def test_denoiser_table_reports_the_winning_k_when_a_denoiser_has_several_runs(tmp_path):
    text = _combined_table(
        tmp_path,
        [("NONE", 0, 0.80, 0.70), ("LPP", 10, 0.86, 0.75), ("LPP", 20, 0.84, 0.79)],
        [("NONE", 0, 0.84, 0.28), ("LPP", 394, 0.83, 0.88)],
    )
    assert r"\begin{tabular}{l|rr|rr|rr|rr}" in text
    # GIST's LPP coverage came from k=10 and its percentile from k=20
    assert r"LPP & 10 & \textbf{86.00} & 394 & 83.00 & 20 & \textbf{79.00} & 394 & \textbf{88.00} \\" in text


def test_topic_matched_sample_pairs_each_topics_books_with_as_many_articles():
    types = np.array(["book", "article", "article", "article", "book", "book", "article", "book"], dtype=object)
    topics = np.array(["T1", "T1", "T1", "T1", "T2", "T2", "T2", "T3"], dtype=object)
    pick = topic_matched_sample(types, topics, rng=np.random.default_rng(0))
    for topic in ("T1", "T2", "T3"):
        mine = pick[topics[pick] == topic]
        assert (types[mine] == "book").sum() == (types[mine] == "article").sum()
    assert (types[pick][topics[pick] == "T2"] == "book").sum() == 1  # T2 has one article for its two books
    assert not (topics[pick] == "T3").any()  # T3 has no articles
