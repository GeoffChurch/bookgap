"""Build the generic data.npz from Bazel target graphs (the clean+coarse substrate).

For each corpus repo: ``bazel query 'kind("cc_(library|binary|test) rule", <scope>)'
--output=xml`` (cached to ``cache/code_gap/bazel/<name>.xml``), parse targets + their
target-valued rule-inputs (= build deps). B = cc_library, F = cc_test/cc_binary. Embed
the target label tokens (``//absl/container:flat_hash_map`` -> "absl container
flat_hash_map") with bge-small. Nodes namespaced by repo.

    python -m artifacts_scripts.gap_recover.adapters.bazel.build_data --out cache/code_gap/bazel/data
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import scipy.sparse as sp

from .corpus import BAZELISK, PACKAGES, REPO_ROOT

KIND_B = {"cc_library"}
KIND_F = {"cc_test", "cc_binary"}


def query_xml(name: str, scope: str) -> str | None:
    cache = REPO_ROOT / f"{name}.xml"
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_text()
    repo = REPO_ROOT / name
    if not repo.exists():
        print(f"[skip] {name}: repo not cloned at {repo}", flush=True)
        return None
    try:
        r = subprocess.run(
            [BAZELISK, "query", f'kind("cc_(library|binary|test) rule", {scope})',
             "--output=xml", "--noimplicit_deps"],
            cwd=str(repo), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print(f"[skip] {name}: bazel query TIMEOUT", flush=True)
        return None
    if r.returncode != 0:
        print(f"[skip] {name}: bazel query rc={r.returncode}: {r.stderr.strip()[-200:]}", flush=True)
        return None
    cache.write_text(r.stdout)
    return r.stdout


def parse_repo(xml_text: str):
    root = ET.fromstring(xml_text)
    kind = {r.get("name"): r.get("class") for r in root.findall("rule")}
    edges = []
    for r in root.findall("rule"):
        s = r.get("name")
        for ri in r.findall("rule-input"):
            d = ri.get("name")
            if d in kind and d != s:
                edges.append((s, d))
    return kind, edges


def label_text(label: str) -> str:
    return label.lstrip("/").replace("/", " ").replace(":", " ").replace("_", " ").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cache/code_gap/bazel/data")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    names, kinds, mods, texts, is_test = [], [], [], [], []
    edges_src, edges_dst = [], []
    idof: dict[str, int] = {}
    for name, _url, scope in PACKAGES:
        if args.only and name != args.only:
            continue
        xml = query_xml(name, scope)
        if xml is None:
            continue
        kind, edges = parse_repo(xml)
        base = len(names)
        local = {}
        for label, cls in kind.items():
            gname = f"{name}::{label}"
            local[label] = len(names)
            idof[gname] = len(names)
            names.append(gname); kinds.append(cls)
            mods.append(f"{name}::{label.split(':')[0]}")     # package = label dir
            texts.append(label_text(label))
            is_test.append(cls == "cc_test")
        for s, d in edges:
            edges_src.append(local[s]); edges_dst.append(local[d])
        print(f"[bazel] {name}: {len(kind)} targets "
              f"(lib={sum(c=='cc_library' for c in kind.values())} "
              f"test={sum(c=='cc_test' for c in kind.values())} "
              f"bin={sum(c=='cc_binary' for c in kind.values())}), {len(edges)} edges", flush=True)

    n = len(names)
    if n == 0:
        print("[bazel] no targets — nothing built", flush=True)
        return
    kinds_arr = np.array(kinds, object)
    is_B = np.isin(kinds_arr, list(KIND_B))
    is_F = np.isin(kinds_arr, list(KIND_F))
    is_test = np.asarray(is_test, bool)
    concept_pref = is_B & ~is_test          # all cc_library are "preferred concepts"
    src = np.asarray(edges_src, np.int64); dst = np.asarray(edges_dst, np.int64)
    struct = np.zeros(n, np.int64); oper = np.zeros(n, np.int64)
    np.add.at(struct, dst, is_B[src].astype(np.int64))
    np.add.at(oper, dst, is_F[src].astype(np.int64))
    sub = sp.csr_matrix((np.ones(src.size, np.int8), (src, dst)), shape=(n, n))
    sub.sum_duplicates(); sub.data[:] = 1

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    print(f"[embed] {n} target labels…", flush=True)
    emb = model.encode([t[:512] for t in texts], batch_size=256, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True).astype(np.float32)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "data.npz", emb=emb, is_F=is_F, is_B=is_B,
             kinds=kinds_arr, mods=np.array(mods, object), names=np.array(names, object),
             structural_users=struct, operational_users=oper, concept_pref=concept_pref)
    sp.save_npz(out / "subgraph.npz", sub)
    (out / "meta.json").write_text(json.dumps(
        {"adapter": "bazel", "n_nodes": n, "n_B": int(is_B.sum()), "n_F": int(is_F.sum()),
         "n_edges": int(src.size), "embedder": "BAAI/bge-small-en-v1.5"}, indent=2))
    print(f"[bazel] wrote {out}  nodes={n} B={int(is_B.sum())} F={int(is_F.sum())} edges={src.size}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
