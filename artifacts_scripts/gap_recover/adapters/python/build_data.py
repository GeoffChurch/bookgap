"""Build the Python corpus -> generic gap-recover data contract.

Reads from REPO_ROOT (cache/code_gap/python/repos), discovers the primary
package inside each repo, runs extract_repo(), prefixes names/mods with the
GitHub repo name to avoid inter-repo collisions, gates on non-empty text,
embeds with bge-small-en-v1.5 on CPU, and writes data.npz / subgraph.npz /
meta.json to the output directory (default: cache/code_gap/python/data/).

CLI:
    python -m artifacts_scripts.gap_recover.adapters.python.build_data
    python -m artifacts_scripts.gap_recover.adapters.python.build_data \\
        --only click --out /tmp/py_gap_smoke
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

from artifacts_scripts.gap_recover.adapters.python.corpus import (
    PACKAGES,
    REPO_ROOT,
)
from artifacts_scripts.gap_recover.adapters.python.extract import (
    ExtractResult,
    extract_repo,
)

_SKIP_DIRS = frozenset({
    "tests", "test", "testing",
    "docs", "doc",
    "examples", "example",
    "benchmarks", "bench",
    "scripts", "tools",
    "ci", ".github",
})


# --------------------------------------------------------------------------- #
# package discovery
# --------------------------------------------------------------------------- #
def _count_py_files(d: Path) -> int:
    return sum(1 for _ in d.rglob("*.py"))


def discover_package(repo_dir: Path) -> tuple[str, Path] | None:
    """Return (import_name, pkg_dir) for the primary Python package in repo_dir.

    Searches repo_dir/ and repo_dir/src/ for subdirectories that contain an
    __init__.py and are NOT excluded utility dirs. Among candidates picks the one
    with the most .py files (the primary package). Returns None if nothing found.
    """
    candidates: list[tuple[int, str, Path]] = []

    for search_root in (repo_dir, repo_dir / "src"):
        if not search_root.is_dir():
            continue
        for child in sorted(search_root.iterdir()):
            if not child.is_dir():
                continue
            if child.name.lower() in _SKIP_DIRS:
                continue
            if child.name.startswith("."):
                continue
            if not (child / "__init__.py").exists():
                continue
            n_py = _count_py_files(child)
            candidates.append((n_py, child.name, child))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, import_name, pkg_dir = candidates[0]
    return import_name, pkg_dir


# --------------------------------------------------------------------------- #
# per-repo extraction + namespacing
# --------------------------------------------------------------------------- #
class RepoData(NamedTuple):
    github: str
    import_name: str
    result: ExtractResult
    n_nodes: int


def build_one_repo(github: str, repo_dir: Path) -> RepoData | None:
    """Discover, extract, and return a RepoData for one repo. Returns None to skip."""
    found = discover_package(repo_dir)
    if found is None:
        print(
            f"[skip] {github}: no package dir with __init__.py found "
            f"(excluding {sorted(_SKIP_DIRS)})",
            flush=True,
        )
        return None

    import_name, pkg_dir = found
    print(
        f"[extract] {github}: package={import_name}  pkg_dir={pkg_dir.relative_to(ROOT)}",
        flush=True,
    )
    try:
        r = extract_repo(repo_dir, import_name)
    except Exception as exc:
        print(f"[skip] {github}: extract_repo failed: {exc}", flush=True)
        return None

    n = len(r.names)
    print(
        f"[extract] {github}: {n} nodes, "
        f"B={int(r.is_B.sum())}, F={int(r.is_F.sum())}, "
        f"edges={r.edges_src.size}",
        flush=True,
    )
    return RepoData(github=github, import_name=import_name, result=r, n_nodes=n)


# --------------------------------------------------------------------------- #
# global assembly
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Python corpus -> gap-recover data contract"
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "cache" / "code_gap" / "python" / "data"),
        help="Output directory (default: cache/code_gap/python/data)",
    )
    parser.add_argument(
        "--only",
        metavar="GITHUB_NAME",
        default=None,
        help="Build from a single repo only (for smoke tests, e.g. --only click)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- select repos -------------------------------------------------------
    if args.only:
        # Find matching entry from PACKAGES list
        pkg_map = {github: (url, ntags) for github, url, ntags in PACKAGES}
        if args.only not in pkg_map:
            # Also allow matching just the last part of github_name (e.g. "click" -> "click")
            matches = [g for g in pkg_map if g == args.only or g.endswith("/" + args.only)]
            if not matches:
                print(f"ERROR: --only {args.only!r} not found in PACKAGES", flush=True)
                print(f"Available: {[g for g, _, _ in PACKAGES]}", flush=True)
                return 1
            selected_github = matches[0]
        else:
            selected_github = args.only
        repos_to_process = [(selected_github, REPO_ROOT / selected_github)]
    else:
        repos_to_process = [(github, REPO_ROOT / github) for github, _, _ in PACKAGES]

    # ---- per-repo extraction ------------------------------------------------
    t0 = time.time()
    repo_results: list[RepoData] = []
    for github, repo_dir in repos_to_process:
        if not repo_dir.is_dir():
            print(f"[skip] {github}: repo dir not found at {repo_dir}", flush=True)
            continue
        rd = build_one_repo(github, repo_dir)
        if rd is not None:
            repo_results.append(rd)

    if not repo_results:
        print("ERROR: no repos successfully extracted", flush=True)
        return 1

    # ---- global concatenation with per-repo namespacing --------------------
    # Prefix names and mods with "{github}::" to avoid cross-repo collisions.
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
        prefix = rd.github + "::"

        all_names.extend(prefix + name for name in r.names)
        all_kinds.extend(r.kinds)
        all_mods.extend(prefix + mod for mod in r.mods)
        all_texts.extend(r.texts)
        all_is_F.append(r.is_F)
        all_is_B.append(r.is_B)
        all_is_test.append(r.is_test)
        all_concept_pref.append(r.concept_pref)

        # Offset edges by base_idx (edges stay within-repo)
        if r.edges_src.size:
            all_edges_src.append(r.edges_src + base_idx)
            all_edges_dst.append(r.edges_dst + base_idx)

        repo_meta.append({
            "github": rd.github,
            "import_name": rd.import_name,
            "n_nodes": n,
            "n_B": int(r.is_B.sum()),
            "n_F": int(r.is_F.sum()),
            "n_test": int(r.is_test.sum()),
            "n_edges": int(r.edges_src.size),
        })
        base_idx += n

    # Convert to arrays
    names_all = np.array(all_names, dtype=object)
    kinds_all = np.array(all_kinds, dtype=object)
    mods_all = np.array(all_mods, dtype=object)
    texts_all = all_texts  # keep as list for now

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
        f"\n[assemble] {N_total} total nodes across {len(repo_results)} repos; "
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

    # ---- structural / operational user counts (recomputed over kept graph) --
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
    dim = model.get_sentence_embedding_dimension()
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
        f"({len(texts_g)/t_emb_elapsed:.0f} nodes/s)",
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
        "python -m artifacts_scripts.gap_recover.adapters.python.build_data "
        + " ".join(sys.argv[1:])
    )
    meta = {
        "adapter": "python",
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
        f"\n{'='*60}",
        f"Summary",
        f"  repos:   {len(repo_results)}",
        f"  nodes:   {Ng}",
        f"  B (class):   {int(is_B_g.sum())}",
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
