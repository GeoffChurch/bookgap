# tests/wikipedia/test_wikipedia_corpus.py
"""Tests for the Wikipedia leaf corpus: cirrus ingest, embed store, and the
``WikipediaCorpus`` ``Corpus``-protocol implementation (incl. union with an
OpenAlex ``SubCloud``).

These tests read/build a small store under ``cache/wiki/store/wiki_test/`` from
the (possibly partial) CirrusSearch dump. If the dump is absent they skip, so
the suite stays green on machines without the dump.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bookgap.util import CACHE_DIR

DUMP = CACHE_DIR / "wiki" / "dumps" / "enwiki-20251229-cirrussearch-content.json.gz"
TEST_STORE = CACHE_DIR / "wiki" / "store" / "wiki_test"
N_TEST = 500


# integration: also uses the `_ai_cloud` fixture (cached production corpus,
# gitignored). Both conditions must hold for these to run.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DUMP.exists(), reason="cirrus dump not present"),
]


@pytest.fixture(scope="module")
def wiki_store() -> Path:
    """Build (once) a tiny Wikipedia embedding store and return its dir."""
    import json

    from bookgap.configs.embedder.embedder_config import SentenceTransformerEmbedderConfig
    from bookgap.corpus import embedder_fingerprint
    from scripts.embed_wikipedia import (
        embed_articles,
        write_embeddings_parquet,
    )
    from bookgap.wikipedia.ingest import write_articles_parquet

    TEST_STORE.mkdir(parents=True, exist_ok=True)
    articles = TEST_STORE / "articles.parquet"
    embeddings = TEST_STORE / "embeddings.parquet"
    manifest = TEST_STORE / "manifest.json"
    if not (articles.exists() and embeddings.exists() and manifest.exists()):
        n = write_articles_parquet(DUMP, articles, limit=N_TEST)
        emb = embed_articles(articles, batch_size=128)
        write_embeddings_parquet(emb, embeddings)
        cfg = SentenceTransformerEmbedderConfig(model_name="avsolatorio/GIST-small-Embedding-v0")
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "n_articles": n,
                    "embedding_dim": int(emb.shape[1]),
                    "embedder": {
                        "model_name": cfg.model_name,
                        "embedder_fingerprint": embedder_fingerprint(cfg),
                    },
                }
            )
        )
    return TEST_STORE


def test_ingest_sample_triples():
    """First articles parse to sensible (page_id, title, abstract) triples."""
    from bookgap.wikipedia.ingest import iter_articles

    arts = list(iter_articles(DUMP, limit=200))
    assert len(arts) == 200
    assert all(isinstance(a["page_id"], int) and a["page_id"] > 0 for a in arts)
    assert all(isinstance(a["title"], str) and a["title"] for a in arts)
    assert all(isinstance(a["abstract"], str) and a["abstract"].strip() for a in arts)
    assert len({a["page_id"] for a in arts}) == 200  # ids unique


def test_wikipedia_corpus_satisfies_protocol(wiki_store):
    from bookgap.corpus import Corpus
    from bookgap.wikipedia.corpus import WikipediaCorpus

    c = WikipediaCorpus(store_dir=wiki_store)
    assert isinstance(c, Corpus)
    emb = c.embeddings()
    assert emb.dtype == np.float32
    assert emb.ndim == 2 and emb.shape[1] == 384
    assert emb.shape[0] == N_TEST
    assert len(c.work_ids()) == N_TEST
    assert len(c.work_titles()) == N_TEST
    assert len(c.work_abstracts()) == N_TEST
    # native (non-namespaced) page ids as str
    assert all(wid.isdigit() for wid in c.work_ids())
    assert c.name() == "wikipedia"
    assert len(c.embedder_fingerprint) > 0


def test_wikipedia_index_search(wiki_store):
    from bookgap.wikipedia.corpus import WikipediaCorpus

    c = WikipediaCorpus(store_dir=wiki_store)
    idx = c.index()
    assert idx.ntotal == N_TEST
    q = c.embeddings()[:3]
    d, i = idx.search(q, 5)
    # each query's nearest neighbor is itself (exact FlatL2)
    assert np.array_equal(i[:, 0], np.arange(3))
    assert np.allclose(d[:, 0], 0.0, atol=1e-4)


@pytest.fixture(scope="module")
def raw_openalex_subcloud():
    """An OpenAlex SubCloud in the RAW (un-denoised) 384-D GIST space.

    The default ``_ai_cloud`` fixture pins ``denoiser=LPP`` / ``drop_k=10``,
    which projects embeddings to 374-D — incompatible with the raw 384-D
    Wikipedia store. WikipediaCorpus is built in the raw GIST space, so the
    union counterpart must also be un-denoised (denoiser=NONE, drop_k=0).
    """
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate

    from bookgap.configs.config_store import register_configs
    from bookgap.subcloud import SubCloud
    from bookgap.util import Filter

    register_configs()
    overrides = [
        "cloud.filter_pair.pre.domain=null",
        "cloud.filter_pair.pre.subfield='1702'",
        "cloud.filter_pair.pre.work_types=[book,article]",
        "cloud.denoiser=NONE",
        "cloud.drop_k=0",
    ]
    with initialize_config_module(version_base=None, config_module="bookgap.configs"):
        cfg = compose(config_name="standard", overrides=overrides)
    cloud = instantiate(cfg.cloud)
    return SubCloud(parent=cloud, post_filter=Filter(topic="T13629", work_types=("article",)))


def test_union_with_openalex_subcloud(wiki_store, raw_openalex_subcloud):
    """WikipediaCorpus unions with an OpenAlex SubCloud (shared fingerprint)
    and its merge index returns merged neighbors matching brute force."""
    import faiss

    from bookgap.corpus import UnionCorpus
    from bookgap.wikipedia.corpus import WikipediaCorpus

    wiki = WikipediaCorpus(store_dir=wiki_store)
    oa = raw_openalex_subcloud
    assert wiki.embedder_fingerprint == oa.embedder_fingerprint
    assert wiki.embeddings().shape[1] == oa.embeddings().shape[1] == 384

    union = UnionCorpus(children=(wiki, oa))  # must construct successfully
    assert union.embeddings().shape[0] == wiki.embeddings().shape[0] + oa.embeddings().shape[0]
    # union namespaces the wikipedia child's ids
    assert union.work_ids()[0].startswith("wikipedia:")

    concat = np.vstack([wiki.embeddings(), oa.embeddings()]).astype(np.float32)
    flat = faiss.IndexFlatL2(concat.shape[1])
    flat.add(concat)
    q = concat[:20]
    _, i_ref = flat.search(q, 5)
    _, i_got = union.index().search(q, 5)
    assert np.array_equal(i_ref, i_got)
