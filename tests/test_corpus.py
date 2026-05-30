# tests/test_corpus.py
import numpy as np
from bookgap.corpus import Corpus, UnionCorpus, embedder_fingerprint


def test_subcloud_satisfies_corpus_protocol(tiny_subcloud):
    # tiny_subcloud: an OpenAlex SubCloud over a cached cloud (fixture below).
    c = tiny_subcloud
    assert isinstance(c, Corpus)                      # runtime_checkable protocol
    assert c.embeddings().ndim == 2
    assert len(c.work_ids()) == c.embeddings().shape[0]
    assert isinstance(c.name(), str)
    assert isinstance(c.embedder_fingerprint, str) and len(c.embedder_fingerprint) > 0


def test_embedder_fingerprint_is_stable_and_config_sensitive(gist_embedder_cfg, gemma_embedder_cfg):
    fp1 = embedder_fingerprint(gist_embedder_cfg)
    assert fp1 == embedder_fingerprint(gist_embedder_cfg)        # stable
    assert fp1 != embedder_fingerprint(gemma_embedder_cfg)       # config-sensitive


def test_union_merge_index_matches_bruteforce(tiny_subcloud, _ai_cloud):
    from bookgap.subcloud import SubCloud
    from bookgap.util import Filter
    a = tiny_subcloud
    b = SubCloud(parent=_ai_cloud, post_filter=Filter(topic="T12761", work_types=("article",)))
    union = UnionCorpus(children=(a, b))

    # embeddings concatenate in child order
    assert union.embeddings().shape[0] == a.embeddings().shape[0] + b.embeddings().shape[0]
    assert union.work_ids()[: len(a.work_ids())][0].startswith("openalex:")  # namespaced

    # merge index returns the SAME nearest neighbor as a brute-force concat index
    concat = np.vstack([a.embeddings(), b.embeddings()]).astype(np.float32)
    import faiss
    flat = faiss.IndexFlatL2(concat.shape[1]); flat.add(concat)
    q = concat[:20]
    _, i_ref = flat.search(q, 5)
    _, i_got = union.index().search(q, 5)
    assert np.array_equal(i_ref, i_got)


def test_detect_gaps_runs_on_union_corpus_ambient(_ai_cloud):
    """SP3: gap algorithms accept any Corpus, so detect_gaps runs with a
    UnionCorpus ambient/query (no shared .parent) and returns gaps."""
    from bookgap.gaps import detect_gaps
    from bookgap.subcloud import SubCloud
    from bookgap.types import CellType
    from bookgap.util import Filter
    from scripts.capture_gap_parity_golden import pin_faiss_single_thread

    pin_faiss_single_thread()
    landmark = SubCloud(parent=_ai_cloud, post_filter=Filter(topic="T13629", work_types=("book",)))
    a1 = SubCloud(parent=_ai_cloud, post_filter=Filter(topic="T13629", work_types=("article",)))
    a2 = SubCloud(parent=_ai_cloud, post_filter=Filter(topic="T12761", work_types=("article",)))
    union = UnionCorpus(children=(a1, a2))

    # The union ambient has no .parent; the old isinstance(SubCloud)/shared-parent
    # assert in get_kball_coverage would have rejected it. detect_gaps must now run.
    gaps = detect_gaps(
        ambient=union, landmark=landmark, query=union,
        cell_type=CellType.KBALL, num_gaps=3, radius_fraction=0.75,
        label_id="union-smoke", label_name="union-smoke",
    )
    assert len(gaps) > 0
    for gwc in gaps:
        members = gwc.gap.get_dists_idxs().idxs
        assert members.ndim == 1 and len(members) >= 1
        # ambient-local ids index into the union concatenation (0..N_union-1)
        assert int(members.max()) < union.embeddings().shape[0]


def test_get_subclouds_for_topic_respects_work_types(_ai_cloud):
    from bookgap.subcloud import get_subclouds_for_topic
    landmark, ambient = get_subclouds_for_topic(
        _ai_cloud, "T13629", ambient_work_type="article", landmark_work_type="book"
    )
    assert landmark.post_filter.work_types == ("book",)
    assert ambient.post_filter.work_types == ("article",)
    # default args reproduce historical (book, article) split
    l2, a2 = get_subclouds_for_topic(_ai_cloud, "T13629")
    assert (l2.post_filter.work_types, a2.post_filter.work_types) == (("book",), ("article",))


def test_union_rejects_mismatched_embedders(tiny_subcloud):
    import pytest
    # Construct a child whose embedder_fingerprint differs; expect ValueError.
    with pytest.raises(ValueError):
        UnionCorpus(children=(tiny_subcloud, _FakeCorpus(fp="different")))


class _FakeCorpus:
    def __init__(self, fp): self._fp = fp
    def embeddings(self): import numpy as np; return np.zeros((1, 384), dtype="float32")
    def index(self): import faiss; i = faiss.IndexFlatL2(384); i.add(self.embeddings()); return i
    def work_ids(self): return ["x:1"]
    def work_titles(self): return ["t"]
    def work_abstracts(self): return ["a"]
    def name(self): return "fake"
    @property
    def embedder_fingerprint(self): return self._fp
    @property
    def identity(self): return ("fake", self._fp)
