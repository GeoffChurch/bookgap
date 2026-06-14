"""Build the math-core-gated Lean decl set, F/B split, and bge-small embeddings.

CPU ONLY (a gemma build owns the GPU). Caches everything to /tmp/lean_gap so the
plant-and-recover runner is cheap to re-run.

Gating (the "habitability" analog): module under ``Mathlib.*`` minus
``.Tactic/.Util/.Meta``; drop typeclass-instance decls (``inst[A-Z]…`` / ``.inst``)
and kernel auto-generated decls (``.casesOn/.recOn/.rec/.below/.noConfusion…/.injEq/
.sizeOf/…``); require a non-empty ``type_str`` (needed to embed).

  F = theorems/lemmas      (foreground = the dense literature)
  B = def + inductive      (background = the objects/scaffolding; "structure" is
                            represented as a ``def`` of its projections plus an
                            ``inductive``/``ctor`` family — both land in B by kind)

Emits ``/tmp/lean_gap/data.npz`` with:
  emb            float32 [N,384]  L2-normalized bge-small statement embeddings
  is_F, is_B     bool   [N]
  kinds, mods    object [N]
  names          object [N]
  def_users, thm_users  int [N]   # citation in-degree split by citer kind (for M1)
and ``/tmp/lean_gap/meta.json``.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a gemma build; CPU only.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[3]
GRAPH = ROOT / "cache" / "lean" / "decl_graph"
OUT = Path("/tmp/lean_gap")
OUT.mkdir(parents=True, exist_ok=True)

EXCLUDE_MOD = re.compile(r"\.Tactic|\.Util|\.Meta")
AUTO = re.compile(
    r"(?:^|\.)(?:casesOn|recOn|rec|below|brecOn|ibelow|binductionOn|noConfusion|"
    r"noConfusionType|injEq|sizeOf|toCtorIdx|ofNat|mk\.inj|fold|proof_\d+|eq_\d+|"
    r"match_\d+|_proof_\d+|_eq_\d+|_cstage\d+|_sunfold|_unsafe_rec)"
)


def is_instance(name: str) -> bool:
    last = name.split(".")[-1]
    if last.startswith("inst") and len(last) > 4 and last[4].isupper():
        return True
    return ".inst" in name


def load_nodes() -> dict:
    names: list[str] = []
    kinds: list[str] = []
    mods: list[str] = []
    stmts: list[str] = []
    with gzip.open(GRAPH / "nodes.tsv.gz", "rt", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            while len(p) < 6:
                p.append("")
            names.append(p[1])
            kinds.append(p[2])
            mods.append(p[3])
            stmts.append(p[4])
    return {"names": names, "kinds": kinds, "mods": mods, "stmts": stmts}


def gated(name: str, kind: str, mod: str, stmt: str) -> bool:
    if not mod.startswith("Mathlib."):
        return False
    if EXCLUDE_MOD.search(mod):
        return False
    if not stmt:
        return False
    if AUTO.search(name):
        return False
    if is_instance(name):
        return False
    return True


def main() -> None:
    t0 = time.time()
    nd = load_nodes()
    names, kinds, mods, stmts = nd["names"], nd["kinds"], nd["mods"], nd["stmts"]
    n = len(names)
    print(f"[data] {n} total nodes loaded in {time.time()-t0:.1f}s", flush=True)

    F_KINDS = {"theorem"}
    B_KINDS = {"def", "inductive"}

    keep = np.zeros(n, dtype=bool)
    is_F = np.zeros(n, dtype=bool)
    is_B = np.zeros(n, dtype=bool)
    for i in range(n):
        if not gated(names[i], kinds[i], mods[i], stmts[i]):
            continue
        if kinds[i] in F_KINDS:
            keep[i] = True
            is_F[i] = True
        elif kinds[i] in B_KINDS:
            keep[i] = True
            is_B[i] = True
    keep_idx = np.where(keep)[0]
    print(f"[data] gated F={int(is_F.sum())} B={int(is_B.sum())} total={keep_idx.size}", flush=True)

    # ---- citation in-degree split by citer kind (for M1). Edges: src depends on dst.
    # users(dst) = in-edges; def_users = #citers that are defs, thm_users = #theorems.
    t1 = time.time()
    rows = []
    cols = []
    with gzip.open(GRAPH / "edges.tsv.gz", "rt", encoding="utf-8") as f:
        for line in f:
            s, _, d = line.partition("\t")
            rows.append(int(s))
            cols.append(int(d))
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    citer_is_def = np.zeros(n, dtype=bool)
    citer_is_thm = np.zeros(n, dtype=bool)
    for i in range(n):
        if kinds[i] == "def":
            citer_is_def[i] = True
        elif kinds[i] == "theorem":
            citer_is_thm[i] = True
    def_users = np.zeros(n, dtype=np.int64)
    thm_users = np.zeros(n, dtype=np.int64)
    # count, per dst, how many citers (src) are defs / theorems
    np.add.at(def_users, cols, citer_is_def[rows].astype(np.int64))
    np.add.at(thm_users, cols, citer_is_thm[rows].astype(np.int64))
    print(f"[data] citation in-degree split in {time.time()-t1:.1f}s "
          f"(edges={rows.size})", flush=True)

    # ---- gated->gated citation subgraph (for concept plants: which gated theorems
    #      cite which gated concept). Keep edges where BOTH endpoints are gated, then
    #      remap to compact [0..n_gated) indexing aligned to keep_idx.
    remap = -np.ones(n, dtype=np.int64)
    remap[keep_idx] = np.arange(keep_idx.size)
    em = keep[rows] & keep[cols]
    gr = remap[rows[em]]
    gc = remap[cols[em]]
    # CSR with src->dst (src depends on dst); transpose gives "users of dst".
    sub = sp.csr_matrix(
        (np.ones(gr.size, dtype=np.int8), (gr, gc)),
        shape=(keep_idx.size, keep_idx.size),
    )
    sub.sum_duplicates()
    sub.data[:] = 1
    sp.save_npz(OUT / "subgraph.npz", sub)
    print(f"[data] gated subgraph: {sub.nnz} edges among {keep_idx.size} decls", flush=True)

    # ---- embed gated decls' type_str with bge-small (CPU)
    t2 = time.time()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    dim = model.get_sentence_embedding_dimension()
    texts = [stmts[i][:512] for i in keep_idx]
    print(f"[embed] encoding {len(texts)} type_strs (dim {dim}) on CPU…", flush=True)
    emb_keep = model.encode(
        texts, batch_size=256, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)
    print(f"[embed] done in {time.time()-t2:.1f}s "
          f"({len(texts)/(time.time()-t2):.0f}/s)", flush=True)

    # scatter back to full-N arrays (only gated rows carry embeddings)
    emb = np.zeros((n, dim), dtype=np.float32)
    emb[keep_idx] = emb_keep

    np.savez(
        OUT / "data.npz",
        keep_idx=keep_idx,
        emb=emb_keep,                       # compact: gated rows only, aligned to keep_idx
        is_F=is_F[keep_idx],
        is_B=is_B[keep_idx],
        kinds=np.array([kinds[i] for i in keep_idx], dtype=object),
        mods=np.array([mods[i] for i in keep_idx], dtype=object),
        names=np.array([names[i] for i in keep_idx], dtype=object),
        def_users=def_users[keep_idx],
        thm_users=thm_users[keep_idx],
    )
    meta = {
        "n_total_nodes": int(n),
        "n_gated": int(keep_idx.size),
        "n_F": int(is_F.sum()),
        "n_B": int(is_B.sum()),
        "dim": int(dim),
        "embedder": "BAAI/bge-small-en-v1.5",
        "gating": "Mathlib.* minus .Tactic/.Util/.Meta; drop instances + auto-generated; "
                  "non-empty type_str; F=theorem, B=def+inductive",
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print("[data] wrote", OUT / "data.npz", "meta:", json.dumps(meta), flush=True)


if __name__ == "__main__":
    sys.exit(main())
