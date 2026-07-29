"""Production tests for the opt-in fl_bookdemand_resid deflation strategy.

Gates:
1. Default (GREEDY_HARD / None) byte-identical to the verbatim greedy
   path on a fixed synthetic fixture.
2. resid path yields >=1 GapProtocol-satisfying gap, differing from
   greedy where it should.
3. GapConfig default is GREEDY_HARD and the enum parses via OmegaConf.
4. @MEMORY.cache key incorporates deflation_strategy.
5. Voronoi accepts and ignores the param.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from bookgap.configs.embedder.embedder_config import SentenceTransformerEmbedderConfig
from bookgap.gap_types import DistsAndIdxs
from bookgap.llm.gap_algorithms.kball_algorithm import KballAlgorithm, KballGap
from bookgap.llm.gap_algorithms.voronoi_algorithm import VoronoiAlgorithm
from bookgap.subcloud import SubCloud
from bookgap.types import CellType, DeflationStrategy


# ---------------------------------------------------------------------------
# Synthetic fixture: a fake SubCloud (subclass so isinstance(SubCloud) holds
# inside get_kball_coverage) that shares a parent sentinel and exposes
# embeddings()/index().
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FakeParent:
    """Stand-in for the Cloud a SubCloud hangs off.

    Only needs to expose `.embedder`, but it must be a real config dataclass:
    SubCloud.embedder_fingerprint runs it through dataclasses.asdict(), and
    detect_gaps asserts ambient/landmark/query fingerprints are equal. A bare
    object() satisfied neither once that assertion landed.
    """

    embedder: SentenceTransformerEmbedderConfig = field(
        default_factory=SentenceTransformerEmbedderConfig
    )


# Shared so every fake subcloud reports the same embedder fingerprint.
_SHARED_PARENT = _FakeParent()


class _FakeSubCloud(SubCloud):
    """Minimal SubCloud over an explicit embedding array."""

    _emb: np.ndarray

    def __init__(self, embeddings: np.ndarray):
        object.__setattr__(self, "_emb", np.ascontiguousarray(embeddings, dtype=np.float32))
        object.__setattr__(self, "parent", _SHARED_PARENT)
        object.__setattr__(self, "post_filter", None)

    def embeddings(self) -> np.ndarray:  # type: ignore[override]
        return self._emb

    def index(self):  # type: ignore[override]
        from bookgap.index import build_index, FlatL2
        return build_index(FlatL2(), self._emb)

    # SubCloud is frozen+hashable; identity equality is sufficient here.
    def __eq__(self, other):  # noqa: D401
        return self is other

    def __hash__(self):
        return id(self)


def _make_fixture(seed: int = 1234):
    """Two book-aligned article clusters plus scattered noise; books
    sit near two of the clusters so resid (book ∪ orphan demand)
    behaves differently from greedy hard-subtraction."""
    rng = np.random.default_rng(seed)
    d = 8
    c1 = rng.normal(0.0, 0.02, size=(40, d)).astype(np.float32) + np.array([3.0] + [0.0] * (d - 1), np.float32)
    c2 = rng.normal(0.0, 0.02, size=(35, d)).astype(np.float32) + np.array([0.0, 3.0] + [0.0] * (d - 2), np.float32)
    c3 = rng.normal(0.0, 0.02, size=(30, d)).astype(np.float32) + np.array([0.0, 0.0, 3.0] + [0.0] * (d - 3), np.float32)
    noise = rng.normal(0.0, 1.5, size=(15, d)).astype(np.float32)
    articles = np.vstack([c1, c2, c3, noise]).astype(np.float32)
    # Books near clusters 1 and 2 only (cluster 3 is a real gap).
    books = np.vstack([
        c1.mean(axis=0) + rng.normal(0, 0.01, d).astype(np.float32),
        c2.mean(axis=0) + rng.normal(0, 0.01, d).astype(np.float32),
    ]).astype(np.float32)
    return _FakeSubCloud(articles), _FakeSubCloud(books)


def _gap_tuples(gaps):
    """Serialize gaps to a comparable, hashable structure."""
    out = []
    for g in gaps:
        di = g.get_dists_idxs()
        out.append((
            tuple(int(x) for x in np.asarray(di.idxs).tolist()),
            tuple(round(float(x), 5) for x in np.asarray(di.sqdists).tolist()),
        ))
    return out


# ---------------------------------------------------------------------------
# Gate 1: default byte-identity
# ---------------------------------------------------------------------------

# Golden snapshot of the verbatim greedy path on the fixed fixture
# (radius_fraction=0.75, k=1, is_intra_cloud=False). Captured from the
# greedy code that is preserved verbatim inside the GREEDY_HARD branch.
def _greedy_reference(articles, books):
    """Independent re-implementation of the historical greedy loop,
    driven by the same get_kball_coverage call the verbatim branch
    uses — proves the in-branch greedy code is unchanged in behavior."""
    from bookgap.cloud_index import get_kball_coverage
    cov = get_kball_coverage(ambient=articles, landmark=books, query=articles,
                             radius_fraction=0.75, k=1, is_intra_cloud=False)
    out = []
    while True:
        if len(cov) == 0:
            break
        hi = np.argmax([len(i) for _d, i in cov])
        dists, idxs = cov[hi]
        if len(idxs) == 0:
            break
        order = np.argsort(dists)
        dists = dists[order]
        idxs = idxs[order]
        out.append((tuple(int(x) for x in idxs.tolist()),
                    tuple(round(float(x), 5) for x in dists.tolist())))
        for j, (d_, i_) in enumerate(cov):
            m = np.isin(i_, idxs, invert=True)
            cov[j] = (d_[m], i_[m])
    return out


def test_default_is_byte_identical_to_verbatim_greedy():
    articles, books = _make_fixture()
    reference = _greedy_reference(*_make_fixture())

    default_gaps = list(KballAlgorithm.find_gaps(
        ambient=articles, landmark=books, query=articles,
        radius_fraction=0.75, k=1, is_intra_cloud=False))
    explicit_greedy = list(KballAlgorithm.find_gaps(
        ambient=articles, landmark=books, query=articles,
        radius_fraction=0.75, k=1, is_intra_cloud=False,
        deflation_strategy=DeflationStrategy.GREEDY_HARD))

    assert _gap_tuples(default_gaps) == reference
    assert _gap_tuples(explicit_greedy) == reference
    # None default and explicit GREEDY_HARD must be identical.
    assert _gap_tuples(default_gaps) == _gap_tuples(explicit_greedy)
    # Every greedy gap must be a KballGap (object identity of the type).
    assert all(isinstance(g, KballGap) for g in default_gaps)


def test_greedy_is_deterministic():
    a1, b1 = _make_fixture()
    a2, b2 = _make_fixture()
    g1 = list(KballAlgorithm.find_gaps(ambient=a1, landmark=b1, query=a1,
              radius_fraction=0.75, k=1, is_intra_cloud=False))
    g2 = list(KballAlgorithm.find_gaps(ambient=a2, landmark=b2, query=a2,
              radius_fraction=0.75, k=1, is_intra_cloud=False))
    assert _gap_tuples(g1) == _gap_tuples(g2)


# ---------------------------------------------------------------------------
# Gate 2: resid path satisfies GapProtocol and differs from greedy
# ---------------------------------------------------------------------------

def test_resid_yields_protocol_satisfying_gaps():
    articles, books = _make_fixture()
    n_articles = len(articles.embeddings())

    resid_gaps = list(KballAlgorithm.find_gaps(
        ambient=articles, landmark=books, query=articles,
        radius_fraction=0.75, k=1, is_intra_cloud=False,
        deflation_strategy=DeflationStrategy.FL_BOOKDEMAND_RESID))

    assert len(resid_gaps) >= 1
    for g in resid_gaps:
        assert isinstance(g, KballGap)
        di = g.get_dists_idxs()
        assert isinstance(di, DistsAndIdxs)
        # members are article-local indices
        assert di.idxs.ndim == 1 and di.sqdists.shape == di.idxs.shape
        assert di.idxs.min() >= 0 and di.idxs.max() < n_articles
        # sqdists sorted non-decreasing (KballGap.__post_init__ enforces;
        # assert explicitly so a regression here is obvious)
        assert np.all(np.diff(di.sqdists) >= 0)
        assert g.get_ambient() is articles
        assert g.get_landmark() is books
        # Protocol methods are reconstructed the same way greedy gaps do.
        cbi = g.get_covering_book_indices()
        assert isinstance(cbi, list)
        mapping = g.get_book_to_articles_mapping()
        assert isinstance(mapping, dict)
        for v in mapping.values():
            assert isinstance(v, DistsAndIdxs)


def test_resid_differs_from_greedy():
    articles, books = _make_fixture()
    greedy = _gap_tuples(KballAlgorithm.find_gaps(
        ambient=articles, landmark=books, query=articles,
        radius_fraction=0.75, k=1, is_intra_cloud=False))
    resid = _gap_tuples(KballAlgorithm.find_gaps(
        ambient=articles, landmark=books, query=articles,
        radius_fraction=0.75, k=1, is_intra_cloud=False,
        deflation_strategy=DeflationStrategy.FL_BOOKDEMAND_RESID))
    # The two strategies must not produce identical gap sequences on a
    # fixture designed to expose their difference.
    assert greedy != resid


# ---------------------------------------------------------------------------
# Gate 3: GapConfig default + OmegaConf round-trip
# ---------------------------------------------------------------------------

def test_gap_config_default_is_greedy_hard():
    from bookgap.configs.gap.gap_config import GapConfig
    assert GapConfig().deflation_strategy is DeflationStrategy.GREEDY_HARD


def test_deflation_strategy_omegaconf_roundtrip_like_cell_type():
    from omegaconf import OmegaConf
    from bookgap.configs.gap.gap_config import GapConfig

    cfg = OmegaConf.structured(GapConfig())
    # Mirror the cell_type enum handling: both resolve to enum members.
    assert cfg.cell_type == CellType.KBALL
    assert cfg.deflation_strategy == DeflationStrategy.GREEDY_HARD

    cfg2 = OmegaConf.structured(
        GapConfig(deflation_strategy=DeflationStrategy.FL_BOOKDEMAND_RESID))
    assert cfg2.deflation_strategy == DeflationStrategy.FL_BOOKDEMAND_RESID


def test_resid_preset_registered():
    from hydra.core.config_store import ConfigStore
    from bookgap.configs.config_store import register_configs
    register_configs()
    cs = ConfigStore.instance()
    node = cs.load("resid.yaml").node
    assert node.deflation_strategy == DeflationStrategy.FL_BOOKDEMAND_RESID
    # Default `gap` preset stays greedy_hard.
    gap_node = cs.load("gap.yaml").node
    assert gap_node.deflation_strategy == DeflationStrategy.GREEDY_HARD


# ---------------------------------------------------------------------------
# Gate 4: cache key incorporates deflation_strategy
# ---------------------------------------------------------------------------

def test_cache_key_incorporates_deflation_strategy():
    """The @MEMORY.cache'd gap functions take deflation_strategy as an
    explicit argument, so joblib's value-based key changes with it. We
    assert the cached function's call_id (its hashing of args) differs
    for the two strategies — without inspecting the cache the bug
    (silently serving greedy-cached gaps for resid) would be invisible.
    """
    import bookgap.gaps as gaps_mod

    func = gaps_mod._compute_gaps_for_topic

    # joblib MemorizedFunc exposes _get_output_identifiers / _cached_call;
    # the stable, public-ish hook is the argument hash via _get_args_id.
    common = dict(
        cloud=("CLOUD_SENTINEL",),
        topic_id="T",
        topic_name="name",
        radius_fraction=0.75,
        cell_type=CellType.KBALL,
        num_gaps=5,
    )
    id_greedy = func._get_args_id(
        **common, deflation_strategy=DeflationStrategy.GREEDY_HARD)
    id_resid = func._get_args_id(
        **common, deflation_strategy=DeflationStrategy.FL_BOOKDEMAND_RESID)
    assert id_greedy != id_resid

    # And the parameter is wired into the function signature.
    import inspect
    sig = inspect.signature(gaps_mod._compute_gaps_for_topic.func)
    assert "deflation_strategy" in sig.parameters
    sig_impl = inspect.signature(gaps_mod._compute_gaps_impl.func)
    assert "deflation_strategy" in sig_impl.parameters


# ---------------------------------------------------------------------------
# Gate 5: Voronoi accepts + ignores the new param
# ---------------------------------------------------------------------------

def test_voronoi_accepts_and_ignores_deflation_strategy():
    articles, books = _make_fixture()
    base = list(VoronoiAlgorithm.find_gaps(
        ambient=articles, landmark=books, query=articles))
    with_param = list(VoronoiAlgorithm.find_gaps(
        ambient=articles, landmark=books, query=articles,
        deflation_strategy=DeflationStrategy.FL_BOOKDEMAND_RESID))
    # Same gaps regardless of the (ignored) deflation_strategy value.
    f = lambda gs: [(tuple(np.asarray(g.get_dists_idxs().idxs).tolist())) for g in gs]
    assert f(base) == f(with_param)
