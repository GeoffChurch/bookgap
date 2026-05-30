# tests/test_corpus.py
import numpy as np
from bookgap.corpus import Corpus, embedder_fingerprint


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
