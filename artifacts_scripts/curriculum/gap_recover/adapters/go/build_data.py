"""Ingest the Go extractor TSVs into the generic data.npz contract — ONE dataset per
call-graph resolution (``cha`` = over-approximate/noisy, ``rta`` = precise/clean). The node
set and embeddings are shared across resolutions; only the edge set (hence
structural_users/operational_users + subgraph) differs. That difference IS the cleanness
dial: run plant-recover on ``data_cha`` vs ``data_rta`` over the same corpus.

    # produce the TSVs first (Go extractor):
    cd artifacts_scripts/curriculum/gap_recover/adapters/go/extractor
    PATH=~/go-sdk/bin:$PATH GOPATH=~/go go run . -out cache/code_gap/go/std -pkgs std
    # then:
    python -m artifacts_scripts.curriculum.gap_recover.adapters.go.build_data \
        --tsv cache/code_gap/go/std --out cache/code_gap/go
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import scipy.sparse as sp


def read_nodes(tsv_dir: Path):
    names, kinds, pkgs, is_test, concept_pref, texts = [], [], [], [], [], []
    with open(tsv_dir / "nodes.tsv", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            p = line.rstrip("\n").split("\t")
            while len(p) < 7:
                p.append("")
            names.append(p[1]); kinds.append(p[2]); pkgs.append(p[3])
            is_test.append(p[4] == "1"); concept_pref.append(p[5] == "1")
            texts.append(p[6])
    return (names, kinds, pkgs, np.asarray(is_test, bool),
            np.asarray(concept_pref, bool), texts)


def read_edges(tsv_path: Path, n: int):
    src, dst = [], []
    with open(tsv_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, _, b = line.partition("\t")
            try:
                s, d = int(a), int(b)
            except ValueError:
                continue
            if 0 <= s < n and 0 <= d < n and s != d:
                src.append(s); dst.append(d)
    return np.asarray(src, np.int64), np.asarray(dst, np.int64)


def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    print(f"[embed] encoding {len(texts)} node texts on CPU…", flush=True)
    return model.encode([t[:512] for t in texts], batch_size=256,
                        normalize_embeddings=True, show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)


def write_dataset(out: Path, names, kinds, pkgs, is_F, is_B, concept_pref, emb,
                  src, dst) -> int:
    n = len(names)
    struct = np.zeros(n, np.int64); oper = np.zeros(n, np.int64)
    np.add.at(struct, dst, is_B[src].astype(np.int64))
    np.add.at(oper, dst, is_F[src].astype(np.int64))
    sub = sp.csr_matrix((np.ones(src.size, np.int8), (src, dst)), shape=(n, n))
    sub.sum_duplicates(); sub.data[:] = 1
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "data.npz", emb=emb, is_F=is_F, is_B=is_B,
             kinds=np.array(kinds, object), mods=np.array(pkgs, object),
             names=np.array(names, object), structural_users=struct,
             operational_users=oper, concept_pref=concept_pref)
    sp.save_npz(out / "subgraph.npz", sub)
    (out / "meta.json").write_text(json.dumps(
        {"adapter": "go", "n_nodes": n, "n_B": int(is_B.sum()), "n_F": int(is_F.sum()),
         "n_edges": int(src.size), "embedder": "BAAI/bge-small-en-v1.5"}, indent=2))
    return int(src.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="dir with nodes.tsv + edges_{cha,rta}.tsv")
    ap.add_argument("--out", default="cache/code_gap/go")
    args = ap.parse_args()
    tsv = Path(args.tsv)
    names, kinds, pkgs, is_test, concept_pref, texts = read_nodes(tsv)
    n = len(names)
    kinds_arr = np.array(kinds, object)
    is_B = kinds_arr == "type"
    is_F = ~is_B
    print(f"[go] {n} nodes: B(type)={int(is_B.sum())} "
          f"F(func/method/test)={int(is_F.sum())} tests={int(is_test.sum())}", flush=True)
    emb = embed(texts)  # shared across resolutions
    for res in ("cha", "rta"):
        src, dst = read_edges(tsv / f"edges_{res}.tsv", n)
        ne = write_dataset(Path(args.out) / f"data_{res}", names, kinds, pkgs,
                           is_F, is_B, concept_pref, emb, src, dst)
        print(f"[go:{res}] wrote {args.out}/data_{res}  edges={ne}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
