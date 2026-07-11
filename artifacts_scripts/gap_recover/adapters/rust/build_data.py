"""Build the Rust corpus -> generic gap-recover data contract (4th-language check).

Reads from REPO_ROOT (cache/code_gap/rust/repos), runs the tree-sitter
``extract_repo`` on each crate in ``corpus.PACKAGES`` (skipping any not yet
cloned), concatenates the per-repo graphs (node names are ALREADY prefixed with
``{repo}::`` by ``extract_repo``, so only edge indices need offsetting), gates on
non-empty text, embeds with bge-small-en-v1.5 on CPU, and writes
data.npz / subgraph.npz / meta.json to the output dir
(default: cache/code_gap/rust/data/).

CLI:
    python -m artifacts_scripts.gap_recover.adapters.rust.build_data
    python -m artifacts_scripts.gap_recover.adapters.rust.build_data \\
        --only bytes --out /tmp/rust_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

# GPU must be disabled before any torch / sentence-transformers import.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[5]  # .../src/bookgap
sys.path.insert(0, str(ROOT))

from artifacts_scripts.gap_recover.adapters.rust.corpus import (
    PACKAGES,
    REPO_ROOT,
)
from artifacts_scripts.gap_recover.adapters.rust.extract import (
    ExtractResult,
    extract_repo,
)


# --------------------------------------------------------------------------- #
# per-repo extraction
# --------------------------------------------------------------------------- #
class RepoData(NamedTuple):
    name: str
    result: ExtractResult
    n_nodes: int


def build_one_repo(name: str, repo_dir: Path) -> RepoData | None:
    """Run ``extract_repo`` on one crate. Returns None to skip (parse-empty etc.)."""
    print(f"[extract] {name}: scanning {repo_dir.relative_to(ROOT)}", flush=True)
    try:
        r = extract_repo(repo_dir, name)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[skip] {name}: extract_repo failed: {exc}", flush=True)
        return None

    n = len(r.names)
    if n == 0:
        print(f"[skip] {name}: 0 nodes extracted", flush=True)
        return None
    print(
        f"[extract] {name}: {n} nodes, "
        f"B={int(r.is_B.sum())}, F={int(r.is_F.sum())}, "
        f"test={int(r.is_test.sum())}, edges={r.edges_src.size}",
        flush=True,
    )
    return RepoData(name=name, result=r, n_nodes=n)


# --------------------------------------------------------------------------- #
# global assembly
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Rust corpus -> gap-recover data contract"
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "cache" / "code_gap" / "rust" / "data"),
        help="Output directory (default: cache/code_gap/rust/data)",
    )
    parser.add_argument(
        "--only",
        metavar="CRATE_NAME",
        default=None,
        help="Build from a single crate only (for smoke tests, e.g. --only bytes)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- select repos -------------------------------------------------------
    if args.only:
        names = [name for name, _, _ in PACKAGES]
        if args.only not in names:
            print(f"ERROR: --only {args.only!r} not found in PACKAGES", flush=True)
            print(f"Available: {names}", flush=True)
            return 1
        repos_to_process = [(args.only, REPO_ROOT / args.only)]
    else:
        repos_to_process = [(name, REPO_ROOT / name) for name, _, _ in PACKAGES]

    # ---- per-repo extraction ------------------------------------------------
    t0 = time.time()
    repo_results: list[RepoData] = []
    for name, repo_dir in repos_to_process:
        if not repo_dir.is_dir():
            print(f"[skip] {name}: repo dir not found at {repo_dir}", flush=True)
            continue
        rd = build_one_repo(name, repo_dir)
        if rd is not None:
            repo_results.append(rd)

    if not repo_results:
        print("ERROR: no crates successfully extracted", flush=True)
        return 1

    # ---- global concatenation (names already carry the {repo}:: prefix) ------
    all_names: list[str] = []
    all_kinds: list[str] = []
    all_mods: list[str] = []
    all_texts: list[str] = []
    all_is_F: list[np.ndarray] = []
    all_is_B: list[np.ndarray] = []
    all_is_test: list[np.ndarray] = []
    all_concept_pref: list[np.ndarray] = []
    all_edges_src: list[np.ndarray] = []
    all_edges_dst: list[np.ndarray] = []

    base_idx = 0
    repo_meta: list[dict] = []
    for rd in repo_results:
        r = rd.result
        n = rd.n_nodes

        all_names.extend(r.names)          # already "{repo}::module::Item"
        all_kinds.extend(r.kinds)
        all_mods.extend(r.mods)            # already "{repo}::module"
        all_texts.extend(r.texts)
        all_is_F.append(r.is_F)
        all_is_B.append(r.is_B)
        all_is_test.append(r.is_test)
        all_concept_pref.append(r.concept_pref)

        if r.edges_src.size:               # edges stay within-repo: offset only
            all_edges_src.append(r.edges_src + base_idx)
            all_edges_dst.append(r.edges_dst + base_idx)

        repo_meta.append({
            "name": rd.name,
            "n_nodes": n,
            "n_B": int(r.is_B.sum()),
            "n_F": int(r.is_F.sum()),
            "n_test": int(r.is_test.sum()),
            "n_edges": int(r.edges_src.size),
        })
        base_idx += n

    names_all = np.array(all_names, dtype=object)
    kinds_all = np.array(all_kinds, dtype=object)
    mods_all = np.array(all_mods, dtype=object)
    texts_all = all_texts

    is_F_all = np.concatenate(all_is_F) if all_is_F else np.empty(0, dtype=bool)
    is_B_all = np.concatenate(all_is_B) if all_is_B else np.empty(0, dtype=bool)
    is_test_all = np.concatenate(all_is_test) if all_is_test else np.empty(0, dtype=bool)
    concept_pref_all = (
        np.concatenate(all_concept_pref) if all_concept_pref else np.empty(0, dtype=bool)
    )
    edges_src_all = (
        np.concatenate(all_edges_src, dtype=np.int64)
        if all_edges_src
        else np.empty(0, dtype=np.int64)
    )
    edges_dst_all = (
        np.concatenate(all_edges_dst, dtype=np.int64)
        if all_edges_dst
        else np.empty(0, dtype=np.int64)
    )

    N_total = len(names_all)
    print(
        f"\n[assemble] {N_total} total nodes across {len(repo_results)} crates; "
        f"B={int(is_B_all.sum())}, F={int(is_F_all.sum())}, "
        f"edges={edges_src_all.size}",
        flush=True,
    )

    # ---- gate on non-empty text ---------------------------------------------
    has_text = np.array([bool(t) for t in texts_all], dtype=bool)
    n_empty = int((~has_text).sum())
    if n_empty:
        print(f"[gate] dropping {n_empty} nodes with empty text", flush=True)

    keep_idx = np.where(has_text)[0]
    Ng = keep_idx.size
    print(f"[gate] kept {Ng} nodes (gated on non-empty text)", flush=True)

    # compact remap
    remap = -np.ones(N_total, dtype=np.int64)
    remap[keep_idx] = np.arange(Ng)

    # ---- gated subgraph (remap edges, drop cut ones) ------------------------
    if edges_src_all.size:
        valid = has_text[edges_src_all] & has_text[edges_dst_all]
        gr = remap[edges_src_all[valid]]
        gc = remap[edges_dst_all[valid]]
    else:
        gr = np.empty(0, dtype=np.int64)
        gc = np.empty(0, dtype=np.int64)

    sub = sp.csr_matrix(
        (np.ones(gr.size, dtype=np.int8), (gr, gc)),
        shape=(Ng, Ng),
    )
    sub.sum_duplicates()
    sub.data[:] = 1
    sp.save_npz(out_dir / "subgraph.npz", sub)
    print(f"[subgraph] {sub.nnz} edges among {Ng} nodes", flush=True)

    # ---- structural / operational user counts (over the gated graph) --------
    is_B_g = is_B_all[keep_idx]
    is_F_g = is_F_all[keep_idx]

    structural_users = np.zeros(Ng, dtype=np.int64)
    operational_users = np.zeros(Ng, dtype=np.int64)
    if gr.size:
        np.add.at(structural_users, gc, is_B_g[gr].astype(np.int64))
        np.add.at(operational_users, gc, is_F_g[gr].astype(np.int64))

    # ---- embed gated texts --------------------------------------------------
    from sentence_transformers import SentenceTransformer

    texts_g = [texts_all[i][:512] for i in keep_idx]
    print(
        f"[embed] encoding {len(texts_g)} texts with bge-small-en-v1.5 on CPU…",
        flush=True,
    )
    t_emb = time.time()
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    dim = model.get_sentence_embedding_dimension() or 384
    emb = model.encode(
        texts_g,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    t_emb_elapsed = time.time() - t_emb
    print(
        f"[embed] done in {t_emb_elapsed:.1f}s "
        f"({len(texts_g) / max(t_emb_elapsed, 1e-9):.0f} nodes/s)",
        flush=True,
    )

    # ---- write data.npz -----------------------------------------------------
    np.savez(
        out_dir / "data.npz",
        emb=emb,
        is_F=is_F_g,
        is_B=is_B_g,
        kinds=kinds_all[keep_idx],
        mods=mods_all[keep_idx],
        names=names_all[keep_idx],
        structural_users=structural_users,
        operational_users=operational_users,
        concept_pref=concept_pref_all[keep_idx],
    )
    print(f"[out] wrote {out_dir / 'data.npz'}", flush=True)

    # ---- write meta.json ----------------------------------------------------
    elapsed_total = time.time() - t0
    provenance = (
        "python -m artifacts_scripts.gap_recover.adapters.rust.build_data "
        + " ".join(sys.argv[1:])
    )
    meta = {
        "adapter": "rust",
        "dim": int(dim),
        "n_nodes": int(Ng),
        "n_B": int(is_B_g.sum()),
        "n_F": int(is_F_g.sum()),
        "n_test": int(is_test_all[keep_idx].sum()),
        "n_edges": int(sub.nnz),
        "n_repos": len(repo_results),
        "repos": repo_meta,
        "embedder": "BAAI/bge-small-en-v1.5",
        "embed_time_s": round(t_emb_elapsed, 1),
        "elapsed_s": round(elapsed_total, 1),
        "provenance": provenance,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[out] wrote {out_dir / 'meta.json'}", flush=True)

    # ---- summary + top-Δ sanity check ---------------------------------------
    print(
        f"\n{'=' * 60}",
        f"Summary",
        f"  repos:   {len(repo_results)}",
        f"  nodes:   {Ng}",
        f"  B (type):    {int(is_B_g.sum())}",
        f"  F (func):    {int(is_F_g.sum())}",
        f"  test nodes:  {int(is_test_all[keep_idx].sum())}",
        f"  edges:   {sub.nnz}",
        f"  emb dim: {dim}",
        f"  emb finite: {bool(np.all(np.isfinite(emb)))}",
        f"  embed time: {t_emb_elapsed:.1f}s",
        f"  total time: {elapsed_total:.1f}s",
        sep="\n",
        flush=True,
    )

    # Top-12 B-nodes by Δ = log(structural_users+1) - log(operational_users+1)
    b_mask = is_B_g
    b_positions = np.where(b_mask)[0]
    if b_positions.size:
        su_b = structural_users[b_positions].astype(float)
        ou_b = operational_users[b_positions].astype(float)
        delta = np.log1p(su_b) - np.log1p(ou_b)
        top12_pos = np.argsort(delta)[::-1][:12]
        names_g = names_all[keep_idx]
        print(
            f"\nTop-12 B-nodes by Δ=log(struct+1)−log(oper+1) "
            f"(structurally embedded but under-exercised):",
            flush=True,
        )
        for rank, pos in enumerate(top12_pos, 1):
            global_pos = b_positions[pos]
            name = names_g[global_pos]
            su = int(su_b[pos])
            ou = int(ou_b[pos])
            print(f"  {rank:2}. Δ={delta[pos]:.2f}  su={su}  ou={ou}  {name}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
