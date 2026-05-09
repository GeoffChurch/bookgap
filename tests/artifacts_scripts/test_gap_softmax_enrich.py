import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Soft-EM CSV member_weights_json fields can exceed default 131072 bytes
csv.field_size_limit(sys.maxsize)


def write_soft_em_csv(path: Path, basins: list[dict]) -> None:
    """Helper: write a synthetic soft-EM raw CSV with the columns the
    Phase 2 driver reads."""
    fields = [
        "rank", "seed_article", "first_seed_type",
        "n_articles", "n_books", "book_anchored",
        "member_indices_json", "member_weights_json",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k, b in enumerate(basins):
            w.writerow({
                "rank": k + 1,
                "seed_article": b.get("seed_article", -1),
                "first_seed_type": b.get("first_seed_type", "article"),
                "n_articles": b.get("n_articles", 0),
                "n_books": b.get("n_books", 0),
                "book_anchored": b.get("book_anchored", 0),
                "member_indices_json": json.dumps(b.get("member_indices", [])),
                "member_weights_json": json.dumps(b.get("member_weights", [])),
            })


def test_load_soft_em_csv_returns_M0_and_meta(tmp_path: Path):
    """The driver reuses gap_softmax_reassign.load_enriched_csv for parsing,
    so this test just confirms the integration: synthetic CSV roundtrips
    into a (N_articles, N_basins) M0 matrix."""
    from artifacts_scripts.gap_softmax_enrich import load_soft_em_basins

    csv_path = tmp_path / "soft_em.csv"
    write_soft_em_csv(csv_path, [
        {
            "seed_article": 100,
            "n_articles": 2, "n_books": 0, "book_anchored": 0,
            "member_indices": [{"idx": 100, "type": "article"},
                               {"idx": 101, "type": "article"}],
            "member_weights": [{"idx": 100, "type": "article", "weight": 1.0},
                               {"idx": 101, "type": "article", "weight": 0.7}],
        },
        {
            "seed_article": 200,
            "n_articles": 1, "n_books": 1, "book_anchored": 1,
            "member_indices": [{"idx": 200, "type": "article"},
                               {"idx": 300, "type": "book"}],
            "member_weights": [{"idx": 200, "type": "article", "weight": 0.9},
                               {"idx": 300, "type": "book", "weight": 1.0}],
        },
    ])

    is_article = np.zeros(400, dtype=bool)
    is_article[[100, 101, 200]] = True

    M0, basin_meta = load_soft_em_basins(csv_path, is_article)
    assert M0.shape == (3, 2)
    assert M0[0, 0] == pytest.approx(1.0)
    assert basin_meta[1]["book_anchored"] == 1
    assert basin_meta[1]["n_books"] == 1
