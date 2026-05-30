#!/usr/bin/env python
# artifacts_scripts/wiki/_prep_oa_nlp_cache.py
"""One-time prep: materialize the OpenAlex NLP slice (T10181 articles) as a small
npz so the joint-graph experiment can iterate without re-loading the cloud.

Caches raw GIST-384 embeddings, titles, ids, AND per-work reference id lists
(needed to build the intra-slice OA citation graph) to
``/tmp/oa_nlp_slice.npz`` + ``/tmp/oa_nlp_refs.json``.

This reuses ``shared_space_analysis.load_openalex_nlp`` (the validated raw-GIST
NLP loader used by all the cross-corpus work) plus a SubCloud ``work_references``
read on the same slice — done ONCE here so the main script never touches the
60 GB cloud machinery again.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_NPZ = Path("/tmp/oa_nlp_slice.npz")
OUT_REFS = Path("/tmp/oa_nlp_refs.json")


def main() -> None:
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate

    from bookgap.configs.config_store import register_configs
    from bookgap.corpus import embedder_fingerprint
    from bookgap.subcloud import SubCloud
    from bookgap.util import Filter
    from artifacts_scripts.wiki.shared_space_analysis import (
        OA_OVERRIDES,
        NLP_TOPIC,
    )

    register_configs()
    with initialize_config_module(version_base=None, config_module="bookgap.configs"):
        cfg = compose(config_name="standard", overrides=OA_OVERRIDES)
    cloud = instantiate(cfg.cloud)
    fp = embedder_fingerprint(cloud.embedder)
    nlp = SubCloud(parent=cloud, post_filter=Filter(topic=NLP_TOPIC, work_types=("article",)))

    emb = np.ascontiguousarray(nlp.embeddings(), dtype=np.float32)
    titles = nlp.work_titles()
    ids = nlp.work_ids()
    print(f"OA NLP slice: emb={emb.shape} fp={fp} n_titles={len(titles)}", flush=True)

    refs = nlp.work_references()  # list[list[str]] aligned with ids
    print(f"references loaded: {len(refs)} works, "
          f"mean refs/work={np.mean([len(r or []) for r in refs]):.1f}", flush=True)

    np.savez_compressed(
        OUT_NPZ,
        emb=emb,
        titles=np.asarray(titles, dtype=object),
        ids=np.asarray(ids, dtype=object),
        fp=fp,
    )
    with open(OUT_REFS, "w") as f:
        json.dump(refs, f)
    print(f"wrote {OUT_NPZ} ({OUT_NPZ.stat().st_size/1e6:.1f} MB) and {OUT_REFS}", flush=True)


if __name__ == "__main__":
    main()
