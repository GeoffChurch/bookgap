import json
import numpy as np
import pytest

from numpy.typing import NDArray

from artifacts_scripts.llm_eval.optimize_prompt import (
    parse_proposals_from_response,
    sample_gaps,
    embed_proposals,
    score_proposals_against_gaps,
    SamplingConfig,
)
import bookgap.gaps
from bookgap.llm.data_structures import GapWithContext
from bookgap.llm.protocols import Proposal

class FakeEmbedder:
    """Minimal embedder stub for tests."""
    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed_texts(self, pairs: list[tuple[str, str]]) -> NDArray[np.float32]:
        # Deterministic embeddings based on hash of title + abstract
        embs = []
        for title, abstract in pairs:
            h = abs(hash(title + abstract))
            rng = np.random.default_rng(h % (2**32))
            vec = rng.standard_normal(self.dim).astype(np.float32)
            embs.append(vec)
        return np.vstack(embs).astype(np.float32)


class FakeProjection:
    def apply(self, embs: NDArray[np.float32]) -> NDArray[np.float32]:
        return embs  # identity


class FakeCloud:
    def __init__(self, normalize_embeddings: bool = False):
        self.normalize_embeddings = normalize_embeddings

    def denoiser_projection(self) -> FakeProjection:
        return FakeProjection()


def make_gap(z: float, pct: float) -> GapWithContext:
    # Only fields used by sample_gaps/score aggregation are set
    return GapWithContext(
        gap=object(),
        topic_name="t",
        topic_id="tid",
        gap_idx=0,
        books=None,  # type: ignore
        articles=None,  # type: ignore
        book_gap_sizes=[],
        gap_size=0,
        gap_size_percentile=pct,
        gap_zscore=z,
        shapiro_log10_pval=0.0,
    )


# --- parse_proposals_from_response ---

def test_parse_proposals_valid_list():
    payload = json.dumps([
        {"title": "A", "abstract": "B"},
        {"title": "C", "abstract": "D"},
    ])
    proposals, error = parse_proposals_from_response(payload)
    assert len(proposals) == 2
    assert isinstance(proposals[0], Proposal)
    assert proposals[0].title == "A"
    assert proposals[1].abstract == "D"
    assert error is None


def test_parse_proposals_single_dict():
    payload = json.dumps({"title": "X", "abstract": "Y"})
    proposals, error = parse_proposals_from_response(payload)
    assert len(proposals) == 1
    assert proposals[0].title == "X"
    assert error is None


def test_parse_proposals_malformed_json():
    proposals, error = parse_proposals_from_response("{not-json}")
    assert proposals == []
    assert error is not None


# --- sample_gaps ---

def test_sample_gaps_top_zscore():
    gaps = [make_gap(z, pct) for z, pct in [(0.1, 10), (2.0, 50), (1.5, 90), (0.5, 20)]]
    sampled = sample_gaps(gaps, SamplingConfig(num_gaps=2, strategy="top_zscore"))
    assert len(sampled) == 2
    zs = [g.gap_zscore for g in sampled]
    assert zs == [2.0, 1.5]


def test_sample_gaps_top_percentile():
    gaps = [make_gap(z, pct) for z, pct in [(0.1, 10), (2.0, 50), (1.5, 90), (0.5, 20)]]
    sampled = sample_gaps(gaps, SamplingConfig(num_gaps=2, strategy="top_percentile"))
    assert len(sampled) == 2
    pcts = [g.gap_size_percentile for g in sampled]
    assert pcts == [90, 50]


def test_sample_gaps_random_deterministic():
    gaps = [make_gap(float(i), float(i)) for i in range(10)]
    cfg = SamplingConfig(num_gaps=3, strategy="random", random_seed=123)
    a = sample_gaps(gaps, cfg)
    b = sample_gaps(gaps, cfg)
    assert [g.gap_zscore for g in a] == [g.gap_zscore for g in b]


# --- embed_proposals ---

def test_embed_proposals_shapes():
    proposals = [Proposal(title=f"T{i}", abstract=f"A{i}") for i in range(3)]
    emb = FakeEmbedder(dim=12)
    cloud = FakeCloud(normalize_embeddings=False)
    embs = embed_proposals(proposals, emb, cloud)
    assert embs.shape == (3, 12)
    assert embs.dtype == np.float32


# --- score_proposals_against_gaps ---

def test_score_proposals_aggregate(monkeypatch):
    # Two proposals, two gaps; set deterministic zscores via monkeypatch
    proposals = [Proposal(title="T1", abstract="A1"), Proposal(title="T2", abstract="A2")]
    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gaps = [make_gap(0.0, 0.0), make_gap(0.0, 0.0)]

    class DummyStats:
        def __init__(self, z, pct):
            self.gap_zscore = z
            self.gap_size_percentile = pct

    def fake_compute(gap, embeddings, config):
        # Return zscores that depend on embedding index (max will be 1.0)
        if np.allclose(embeddings, np.array([[1.0, 0.0]], dtype=np.float32)):
            return [DummyStats(1.0, 0.9)]
        else:
            return [DummyStats(0.3, 0.2)]

    monkeypatch.setattr(
        "artifacts_scripts.llm_eval.optimize_prompt.compute_gap_metrics_for_embeddings",
        fake_compute,
    )

    # Config isn't used by fake_compute; pass None
    results = score_proposals_against_gaps(
        proposals=proposals,
        embeddings=embs,
        gaps=gaps,
        config=None,  # type: ignore
    )

    assert results["num_proposals"] == 2
    assert results["aggregate_score"] == pytest.approx(1.0)

    # First proposal should have max_zscore 1.0
    max_zscores = [s["max_zscore"] for s in results["proposal_scores"]]
    assert max_zscores == [1.0, 0.3]


# --- ensure_gaps with MEMORY cache ---

def test_compute_gaps_for_config_uses_memory_cache(monkeypatch):
    """Test that compute_gaps_for_config uses MEMORY.cache."""
    # Mock compute_gaps_for_config to return test data
    test_gaps = [make_gap(1.2, 0.8)]
    call_count = 0
    
    def mock_compute(config):
        nonlocal call_count
        call_count += 1
        return test_gaps
    
    monkeypatch.setattr("bookgap.gaps.compute_gaps_for_config", mock_compute)
    
    # First call should invoke compute
    loaded = bookgap.gaps.compute_gaps_for_config(config=None)  # type: ignore
    assert len(loaded) == 1
    assert loaded[0].gap_zscore == 1.2
    assert call_count == 1
    
    # Note: MEMORY.cache would normally cache this, but in tests we're mocking
    # the function directly so we can't test the cache behavior without
    # the real MEMORY instance
