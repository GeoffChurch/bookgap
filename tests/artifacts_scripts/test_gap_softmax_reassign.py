import json
import csv
from pathlib import Path

import numpy as np
import pytest

from artifacts_scripts.gap_softmax_reassign import load_enriched_csv


def test_load_enriched_csv_builds_M0(tmp_path: Path):
    """Enriched CSV with member_indices_json + member_weights_json
    round-trips into a dense (N_articles, N_basins) M0 matrix."""
    csv_path = tmp_path / "enriched.csv"
    rows = [
        {
            "rank": "1",
            "seed_article": "100",
            "n_books": "0",
            "n_articles": "2",
            "book_anchored": "0",
            "member_indices_json": json.dumps([100, 101]),
            "member_weights_json": json.dumps(
                [{"idx": 100, "type": "article", "weight": 1.0},
                 {"idx": 101, "type": "article", "weight": 0.7}]
            ),
        },
        {
            "rank": "2",
            "seed_article": "200",
            "n_books": "1",
            "n_articles": "1",
            "book_anchored": "1",
            "member_indices_json": json.dumps([200, 300]),
            "member_weights_json": json.dumps(
                [{"idx": 200, "type": "article", "weight": 0.9},
                 {"idx": 300, "type": "book", "weight": 1.0}]
            ),
        },
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    is_article = np.zeros(400, dtype=bool)
    is_article[[100, 101, 200]] = True

    M0, basin_meta = load_enriched_csv(csv_path, is_article)
    # M0 shape: (N_articles_in_topic, N_basins). Article rows are
    # the global article indices; only article-side weights are used.
    assert M0.shape == (3, 2)
    assert M0[0, 0] == pytest.approx(1.0)   # article 100 in basin 1
    assert M0[1, 0] == pytest.approx(0.7)   # article 101 in basin 1
    assert M0[2, 1] == pytest.approx(0.9)   # article 200 in basin 2
    assert M0[0, 1] == 0.0                   # article 100 not in basin 2
    assert basin_meta[1]["book_anchored"] == 1
