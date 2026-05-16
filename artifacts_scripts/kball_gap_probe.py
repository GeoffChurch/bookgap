"""Stripped-down kball / Voronoi gap-detection probe.

Runs the existing ball-based or Voronoi-based gap algorithms over a
single scope, prints the resulting gaps per topic, and writes a CSV of
the per-gap statistics (gap_size, percentile, z-score, Shapiro-Wilk
log10 p-value) for direct comparison with the HDBSCAN probe's CSV
output. Skips the LLM proposal-generation pipeline that
``batch_generator.py`` would do next.

Usage::

    conda run -p .conda python -m artifacts_scripts.kball_gap_probe \\
        --scope 17/1702/T10181 \\
        --algorithm kball \\
        --radius-fraction 0.75 \\
        --num-gaps 5

    # With denoiser bypass (scope-restricted, no full-corpus load):
    conda run -p .conda python -m artifacts_scripts.kball_gap_probe \\
        --scope 17/1702/T10181 \\
        --denoiser pca --drop-k 5 \\
        --radius-fraction 0.75 --num-gaps 5

When ``--denoiser none`` (default), the script builds a Cloud at the
project's standard defaults (SimHash dedup, GIST embedder, NONE
denoiser, etc.) and runs gap detection via
``bookgap.gaps._compute_gaps_for_topic`` (joblib-cached).

When ``--denoiser <name>`` is anything other than ``none``, the script
uses the **denoiser bypass path**: loads embeddings directly from the
per-scope caches (same as ``hdbscan_gap_probe.py``), applies the chosen
denoiser, L2-normalises, then runs K-ball gap-finding inline per topic
without going through the Cloud abstraction.  This avoids corpus-scope
denoising on 40 M works.

Note on asymmetry from HDBSCAN: K-ball gaps are by construction
book-free spheres (the radius stops at the nearest book), so the
binomial-on-occupancy depletion test would always score them as extreme
and is degenerate here.  We use the existing percentile/z-score/Shapiro
framework against the book-as-query null distribution instead.

CSV output for the Cloud path::

    artifacts/kball_gap_probe/{scope_safe}__{algorithm}__rf{rf}__num{n}.csv

CSV output for the denoiser bypass path::

    artifacts/kball_gap_probe/{scope_safe}__{denoiser}__drop{drop_k}.csv

Columns (bypass path adds ``denoiser`` and ``drop_k``)::

    scope, algorithm, denoiser, drop_k, class_axis, radius_fraction,
    num_gaps_per_topic, topic_id, topic_name,
    gap_idx, gap_size, gap_size_percentile, gap_zscore, shapiro_log10_pval
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from bookgap.cloud import Cloud
from bookgap.configs.cloud.cloud_config import CloudConfig
from bookgap.configs.util.filter_config import FilterConfig, FilterPairConfig
from bookgap.gaps import _compute_gaps_for_topic
from bookgap.gap_statistics import compute_gap_statistics
from bookgap.index import build_index, kballs_with, HNSW, FlatL2
from bookgap.openalex_scopes import (
    get_field_topics, get_subfield_topics,
)
from bookgap.snapshot_ingest import load_taxonomy
from bookgap.types import CellType
from bookgap.util import WorkType
from hydra.utils import instantiate

# ---------------------------------------------------------------------------
# Denoiser helpers — imported from hdbscan_gap_probe to avoid duplication.
# The HDBSCAN probe owns these implementations; we reuse them here verbatim.
# ---------------------------------------------------------------------------
from artifacts_scripts.hdbscan_gap_probe import (
    load_scope,
    _build_in_scope_citation_csr,
    _ProbeDenoiserContext,
    _project_via_basis,
    _drop_top_k_pca,
    _drop_top_k_multi_lda,
    _denoise_lpp,
    _denoise_lip,
    _denoise_lip_eigen_only,
    _denoise_lip_diff,
    _denoise_cclip,
    _denoise_wc_cclip,
    _denoise_hierarchical,
    _denoise_criteria_union,
    _denoise_rnojd,
    _class_labels,
)

# ---------------------------------------------------------------------------
# Cloud-path helpers (unchanged from original probe)
# ---------------------------------------------------------------------------

def build_cloud(scope_key: str, snapshot_date: str) -> Cloud:
    """Build a Cloud whose pre-filter is the given scope, post-filter empty.

    The gap-detection code creates per-topic SubClouds under this parent
    via ``get_subclouds_for_topic`` (post-filter on work_type), so the
    parent's pre-filter just needs to cover the universe — all 4 work
    types so books and articles are both in scope.
    """
    parts = scope_key.split("/")
    work_types = tuple(wt.value for wt in WorkType)
    if len(parts) == 1:
        pre = FilterConfig(field=parts[0], work_types=work_types)
    elif len(parts) == 2:
        pre = FilterConfig(field=parts[0], subfield=parts[1], work_types=work_types)
    elif len(parts) == 3:
        pre = FilterConfig(field=parts[0], subfield=parts[1], topic=parts[2],
                           work_types=work_types)
    else:
        raise ValueError(f"scope_key must be F, F/SF, or F/SF/T, got {scope_key!r}")

    fp_cfg = FilterPairConfig(pre=pre, post=FilterConfig())
    # Build a default CloudConfig and instantiate via Hydra utility so we
    # pick up the project defaults (SimHash dedup, GIST embedder, etc.).
    cloud_cfg = CloudConfig(filter_pair=fp_cfg, snapshot_date=snapshot_date)
    return instantiate(cloud_cfg)


def topics_in_scope(scope_key: str, snapshot_date: str) -> List[str]:
    parts = scope_key.split("/")
    if len(parts) == 3:
        return [parts[2]]
    if len(parts) == 2:
        return get_subfield_topics(parts[1], snapshot_date=snapshot_date)
    return get_field_topics(parts[0], snapshot_date=snapshot_date)


# ---------------------------------------------------------------------------
# Denoiser bypass path
# ---------------------------------------------------------------------------

def _apply_denoiser(
    emb: np.ndarray,
    ids: List[str],
    types: List[str],
    topic_ids: List[str],
    refs: Optional[List],
    denoiser: str,
    drop_k: int,
    drop_k_graph: int,
    class_axis: str,
    cclip_alpha: Optional[float],
) -> np.ndarray:
    """Apply the named denoiser and return projected embeddings (float32).

    Near-copy of ``hdbscan_gap_probe.run`` lines 640-711, extracted so
    ``run_with_denoiser`` stays readable.  Graph-based denoisers require
    refs to be non-None; callers are responsible for loading with
    ``with_refs=True``.
    """
    n, d = emb.shape

    if denoiser == "pca":
        if drop_k > 0:
            print(f"Denoiser pca: projecting out top {drop_k} PCA components...")
            emb = _drop_top_k_pca(emb, drop_k)
    elif denoiser == "multi_lda":
        if drop_k > 0:
            labels = _class_labels(topic_ids, types, class_axis)
            n_classes = len(set(labels))
            print(f"Denoiser multi_lda[{class_axis}]: projecting out top "
                  f"{drop_k} between-class directions ({n_classes} classes)...")
            emb = _drop_top_k_multi_lda(emb, labels, drop_k)
    elif denoiser in {"lpp", "lip", "lip_eigen", "lip_diff",
                      "cclip", "wc_cclip", "hierarchical", "criteria_union", "rnojd"}:
        labels = _class_labels(topic_ids, types, class_axis)
        n_classes = len(set(labels))
        print(f"Building in-scope citation graph for {n:,} works...")
        assert refs is not None, (
            f"denoiser={denoiser!r} requires citation refs but refs is None; "
            "ensure load_scope was called with with_refs=True"
        )
        A = _build_in_scope_citation_csr(ids, refs)
        print(f"  citation graph: nnz={A.nnz:,}, density={A.nnz/(n*n):.2e}")
        ctx = _ProbeDenoiserContext(emb, labels, A)
        n_keep = max(1, d - drop_k)
        if denoiser == "lpp":
            print(f"Denoiser lpp: keeping {n_keep} smallest eigvecs of Xc^T L Xc / Xc^T D Xc...")
            basis = _denoise_lpp(ctx, n_keep)
            emb = _project_via_basis(emb, basis)
        elif denoiser == "lip":
            print(f"Denoiser lip: keeping {n_keep} smallest eigvecs of S_L / S_T...")
            basis = _denoise_lip(ctx, n_keep)
            emb = _project_via_basis(emb, basis)
        elif denoiser == "lip_eigen":
            print(f"Denoiser lip_eigen: keeping {n_keep} smallest eigvecs of S_L alone (no GEP)...")
            basis = _denoise_lip_eigen_only(ctx, n_keep)
            emb = _project_via_basis(emb, basis)
        elif denoiser == "lip_diff":
            print(f"Denoiser lip_diff: keeping {n_keep} smallest eigvecs of S_L / (S_T − S_L)...")
            basis = _denoise_lip_diff(ctx, n_keep)
            emb = _project_via_basis(emb, basis)
        elif denoiser == "cclip":
            alpha_used = 0.5 if cclip_alpha is None else cclip_alpha
            print(f"Denoiser cclip[{class_axis}]: α={alpha_used:.3f}, "
                  f"keeping {n_keep} smallest eigvecs of "
                  f"((1-α) Ŝ_B + α Ŝ_L) / S_T, pre-trace-normalised "
                  f"({n_classes} classes)...")
            basis, _ = _denoise_cclip(ctx, n_keep, alpha_used)
            emb = _project_via_basis(emb, basis)
        elif denoiser == "wc_cclip":
            alpha_used = 0.5 if cclip_alpha is None else cclip_alpha
            print(f"Denoiser wc_cclip[{class_axis}]: α={alpha_used:.3f}, "
                  f"keeping {n_keep} smallest eigvecs of "
                  f"((1-α) Ŝ_B + α Ŝ_L) / S_W, pre-trace-normalised "
                  f"({n_classes} classes)...")
            basis, _ = _denoise_wc_cclip(ctx, n_keep, alpha_used)
            emb = _project_via_basis(emb, basis)
        elif denoiser == "hierarchical":
            print(f"Denoiser hierarchical[{class_axis}]: project out top {drop_k} class-disc, "
                  f"then top {drop_k_graph} citation-incoherent in residual...")
            basis = _denoise_hierarchical(ctx, drop_k_class=drop_k, drop_k_graph=drop_k_graph)
            print(f"  final basis: {basis.shape[1]} kept dims out of {d}")
            emb = _project_via_basis(emb, basis)
        elif denoiser == "criteria_union":
            print(f"Denoiser criteria_union[{class_axis}]: project out union of top "
                  f"{drop_k} eigvecs of (S_B, S_T) and top {drop_k_graph} of (S_L, S_T)...")
            basis = _denoise_criteria_union(ctx, drop_k_class=drop_k, drop_k_graph=drop_k_graph)
            emb = _project_via_basis(emb, basis)
        elif denoiser == "rnojd":
            print(f"Denoiser rnojd[{class_axis}]: jointly diagonalise {{S_B, S_L, S_T}}, "
                  f"drop top {drop_k} scoring axes...")
            basis = _denoise_rnojd(ctx, n_drop=drop_k)
            emb = _project_via_basis(emb, basis)
    else:
        raise ValueError(f"unknown denoiser {denoiser!r}")

    return emb.astype(np.float32, copy=False)


def _build_topic_index(topic_emb: np.ndarray):
    """Build a FAISS index for topic_emb.  HNSW for n >= 100, FlatL2 otherwise.

    If env BOOKGAP_EXACT_INDEX is set to "1", always use exact FlatL2
    regardless of n — used by the denoiser bake-off for fully
    deterministic, approximation-free K-ball seeds. Default behavior
    (env unset or any value other than "1") is unchanged for all other callers.
    """
    n = len(topic_emb)
    if os.environ.get("BOOKGAP_EXACT_INDEX", "0") == "1":
        return build_index(FlatL2(), topic_emb)
    if n >= 100:
        return build_index(HNSW(hnsw_m=32, hnsw_efc=200, hnsw_efs=128), topic_emb)
    else:
        return build_index(FlatL2(), topic_emb)


def _compute_book_gap_sizes_inline(
    book_emb: np.ndarray,
    article_emb: np.ndarray,
    radius_fraction: float,
    k: int,
) -> List[int]:
    """Null distribution: each book's gap size = #articles in its k-ball.

    Mirrors ``KballAlgorithm.compute_gap_sizes_for_embeddings`` with
    ``is_intra_cloud=True``: effective_k = k + 1 so the book's self-
    neighbor (distance 0 in the book index) doesn't dominate the radius.

    Uses ``kballs_with`` from ``bookgap.index`` directly, bypassing
    SubCloud; this is the same primitive the canonical code calls under
    the hood.
    """
    book_index = _build_topic_index(book_emb)
    article_index = _build_topic_index(article_emb)
    effective_k = k + 1  # +1 to skip self-match when book queries book index
    counts: List[int] = [-1] * len(book_emb)

    def _kernel(query_idx: int, _dists, _ids) -> None:
        counts[query_idx] = len(_ids)

    kballs_with(
        ambient=article_index,
        landmark=book_index,
        query=book_emb,
        radius_fraction=radius_fraction,
        k=effective_k,
        kernel_fn=_kernel,
        show_progress=False,
    )
    return counts


def _greedy_kball_gaps_inline(
    book_emb: np.ndarray,
    article_emb: np.ndarray,
    radius_fraction: float,
    k: int,
    num_gaps: int,
    min_gap_size: int = 0,
) -> List[Tuple[int, int, np.ndarray]]:
    """Greedy gap-finding on articles mirroring ``KballAlgorithm.kball_gaps``.

    Returns a list of (gap_idx, gap_size, member_ids) tuples, at most
    ``num_gaps`` long. ``member_ids`` is the ``article_emb``-local index
    array of articles in the gap (the seed itself is included).

    Algorithm (near-copy of kball_algorithm.py:46-73):
    1. For each article, compute k-ball coverage = articles within
       radius_fraction * dist_to_kth_book.  is_intra_cloud=False so no +1.
    2. Greedy: pick highest-coverage article → yield gap; prune its
       article ids from every other article's coverage set; repeat.
    3. Stop once the largest remaining ball has size < ``min_gap_size``
       (skip the cutoff with ``min_gap_size <= 0``).
    """
    book_index = _build_topic_index(book_emb)
    article_index = _build_topic_index(article_emb)

    # coverage[i] = (dists_array, ids_array) of articles in article i's k-ball
    coverage: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * len(article_emb)

    def _kernel(query_idx: int, dists: np.ndarray, ids: np.ndarray) -> None:
        coverage[query_idx] = (dists.copy(), ids.copy())

    kballs_with(
        ambient=article_index,
        landmark=book_index,
        query=article_emb,
        radius_fraction=radius_fraction,
        k=k,
        kernel_fn=_kernel,
        show_progress=False,
    )

    # Convert to mutable list-of-pair for greedy pruning
    covered: List[Tuple[np.ndarray, np.ndarray]] = [
        (np.array([], dtype=np.float32), np.array([], dtype=np.int64))
        if c is None else c
        for c in coverage
    ]

    gaps: List[Tuple[int, int, np.ndarray]] = []
    for gap_idx in range(num_gaps):
        if not any(len(ids) > 0 for _dists, ids in covered):
            break
        best = int(np.argmax([len(ids) for _dists, ids in covered]))
        best_dists, best_ids = covered[best]
        if len(best_ids) == 0:
            break
        gap_size = int(len(best_ids))
        if min_gap_size > 0 and gap_size < min_gap_size:
            break
        gaps.append((gap_idx, gap_size, best_ids.copy()))
        # Prune the gap's article ids from all coverages
        pruned_set = set(best_ids.tolist())
        for i, (d, ids) in enumerate(covered):
            mask = np.array([x not in pruned_set for x in ids.tolist()], dtype=bool)
            covered[i] = (d[mask], ids[mask])

    return gaps


def run_with_denoiser(
    scope: str,
    denoiser: str,
    drop_k: int,
    radius_fraction: float,
    num_gaps: int,
    class_axis: str = "topic_x_type",
    cclip_alpha: Optional[float] = None,
    drop_k_graph: int = 2,
    algorithm: str = "kball",
    snapshot_date: str = "2025-11-06",
    min_gap_size: int = 0,
    article_types: tuple[str, ...] = ("article",),
    book_types: tuple[str, ...] = ("book",),
) -> None:
    """K-ball gap probe with scope-restricted denoising (bypass path).

    Loads embeddings from per-scope caches (same as ``hdbscan_gap_probe``),
    applies the chosen denoiser, L2-normalises, then runs K-ball gap-
    finding per topic inline without the Cloud abstraction.  This avoids
    corpus-scope denoising on 40 M works.

    K-ball gaps are by construction book-free spheres so binomial-on-
    occupancy is degenerate; we use the existing percentile/zscore/Shapiro
    framework against the book-as-query null distribution.

    Args:
        scope:          F, F/SF, or F/SF/T scope key.
        denoiser:       Denoiser name (must not be "none").
        drop_k:         Dimensions to project out (meaning depends on denoiser).
        radius_fraction: Fraction of dist-to-kth-book used as k-ball radius.
        num_gaps:       Max gaps per topic.
        class_axis:     Class partition for LDA/CCLIP/hierarchical/rnojd.
        cclip_alpha:    Alpha for CCLIP/WC-CCLIP (None → 0.5).
        drop_k_graph:   Laplacian-step drop_k for hierarchical/criteria_union.
        algorithm:      "kball" only (voronoi not implemented in bypass path).
        snapshot_date:  Pinned taxonomy date for topic name lookup.
    """
    if algorithm != "kball":
        raise NotImplementedError(
            f"Denoiser bypass only implements kball; got {algorithm!r}"
        )

    k = 1  # K-ball uses k=1 nearest book by default

    needs_graph = denoiser in {"lpp", "lip", "lip_eigen", "lip_diff",
                               "cclip", "wc_cclip", "hierarchical",
                               "criteria_union", "rnojd"}
    print(f"Loading scope {scope} (with_refs={needs_graph})...")
    emb, ids, types, titles, topic_ids, refs = load_scope(scope, with_refs=needs_graph)
    n, d = emb.shape
    print(f"  {n:,} works, {d}-d embeddings")

    # Apply chosen denoiser (modifies emb in scope; returns float32)
    emb = _apply_denoiser(
        emb=emb,
        ids=ids,
        types=types,
        topic_ids=topic_ids,
        refs=refs,
        denoiser=denoiser,
        drop_k=drop_k,
        drop_k_graph=drop_k_graph,
        class_axis=class_axis,
        cclip_alpha=cclip_alpha,
    )

    # L2-normalise so Euclidean on unit sphere ≈ cosine (matches HDBSCAN probe)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norms > 0, norms, 1.0)
    emb = emb.astype(np.float32, copy=False)

    topic_ids_arr = np.array(topic_ids)
    types_arr = np.array(types)
    is_book_arr = np.isin(types_arr, np.asarray(book_types))
    is_article_arr = np.isin(types_arr, np.asarray(article_types))
    print(
        f"  work-type partition: is_article={int(is_article_arr.sum())} "
        f"(types={list(article_types)}), is_book={int(is_book_arr.sum())} "
        f"(types={list(book_types)}), other={int((~is_book_arr & ~is_article_arr).sum())}"
    )

    unique_topics = sorted(set(topic_ids))
    print(f"Topics in scope: {len(unique_topics)}")

    # Optionally look up topic display names
    try:
        pinned_tax = load_taxonomy(snapshot_date)
    except Exception:
        pinned_tax = None

    csv_rows: List[dict] = []
    total_gaps = 0

    for i, topic in enumerate(unique_topics, 1):
        topic_mask = topic_ids_arr == topic
        book_mask = topic_mask & is_book_arr
        article_mask = topic_mask & is_article_arr

        book_emb = emb[book_mask].astype(np.float32, copy=False)
        article_emb = emb[article_mask].astype(np.float32, copy=False)
        # Map from article_emb-local index → scope-global (load_scope) index.
        # Used to record gap membership in the same indexing as the
        # max-bookless extractor, so set-overlap analyses are direct.
        article_local_to_scope = np.where(article_mask)[0]

        topic_name: str = topic
        if pinned_tax is not None:
            rec = pinned_tax.topics.get(topic, {})
            topic_name = rec.get("display_name", topic)

        n_books = len(book_emb)
        n_articles = len(article_emb)
        print(f"\n[{i}/{len(unique_topics)}] Topic {topic} — {topic_name} "
              f"({n_books} books, {n_articles} articles)")

        # Skip degenerate topics
        if n_books == 0:
            print(f"  (skipped: no books)")
            continue
        if n_articles <= k:
            print(f"  (skipped: n_articles={n_articles} <= k={k})")
            continue

        # --- Null distribution: each book's gap size ---
        book_gap_sizes = _compute_book_gap_sizes_inline(
            book_emb=book_emb,
            article_emb=article_emb,
            radius_fraction=radius_fraction,
            k=k,
        )
        print(f"  book null distribution: {len(book_gap_sizes)} books, "
              f"mean={np.mean(book_gap_sizes):.1f}, "
              f"max={max(book_gap_sizes)}")

        # --- Greedy gap-finding on articles ---
        gaps = _greedy_kball_gaps_inline(
            book_emb=book_emb,
            article_emb=article_emb,
            radius_fraction=radius_fraction,
            k=k,
            num_gaps=num_gaps,
            min_gap_size=min_gap_size,
        )

        if not gaps:
            print(f"  (no gaps found)")
            continue

        total_gaps += len(gaps)
        for gap_idx, gap_size, member_local_ids in gaps:
            pct, z, shapiro = compute_gap_statistics(gap_size, book_gap_sizes)
            print(f"  Gap {gap_idx}: size={gap_size}, "
                  f"pctile={pct:.3f}, "
                  f"z={z:.2f}, "
                  f"shapiro_log10p={shapiro:.2f}")
            scope_ids = [int(article_local_to_scope[j]) for j in member_local_ids]
            csv_rows.append({
                "scope": scope,
                "algorithm": algorithm,
                "denoiser": denoiser,
                "drop_k": drop_k,
                "class_axis": class_axis,
                "radius_fraction": radius_fraction,
                "num_gaps_per_topic": num_gaps,
                "topic_id": topic,
                "topic_name": topic_name,
                "gap_idx": int(gap_idx),
                "gap_size": int(gap_size),
                "gap_size_percentile": float(pct),
                "gap_zscore": float(z),
                "shapiro_log10_pval": float(shapiro),
                "article_indices_json": json.dumps(scope_ids),
            })

    print(f"\n=== Total gaps: {total_gaps} across {len(unique_topics)} topic(s) ===")

    csv_dir = Path("artifacts/kball_gap_probe")
    csv_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = scope.replace("/", "-")
    # Include work-type suffix only when non-default, to avoid breaking
    # existing book-only paths.
    if article_types == ("article",) and book_types == ("book",):
        wt_suffix = ""
    else:
        wt_suffix = (
            "__a-" + "+".join(article_types).replace("-", "")
            + "__b-" + "+".join(book_types).replace("-", "")
        )
    csv_path = csv_dir / f"{safe_scope}__{denoiser}__drop{drop_k}{wt_suffix}.csv"
    fieldnames = [
        "scope", "algorithm", "denoiser", "drop_k", "class_axis",
        "radius_fraction", "num_gaps_per_topic",
        "topic_id", "topic_name",
        "gap_idx", "gap_size",
        "gap_size_percentile", "gap_zscore", "shapiro_log10_pval",
        "article_indices_json",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in csv_rows:
            w.writerow(row)
    print(f"CSV: {csv_path} ({len(csv_rows)} rows)")


# ---------------------------------------------------------------------------
# Cloud path (original, unchanged except for denoiser-flag routing in main)
# ---------------------------------------------------------------------------

def _run_cloud_path(args) -> None:
    """Original Cloud-based gap probe (no denoising)."""
    cell_type = CellType.KBALL if args.algorithm == "kball" else CellType.VORONOI
    print(f"Cell type:        {cell_type}")
    print(f"Radius fraction:  {args.radius_fraction}")
    print(f"Gaps per topic:   {args.num_gaps}")

    cloud = build_cloud(args.scope, args.snapshot_date)
    topics = topics_in_scope(args.scope, args.snapshot_date)
    print(f"Cloud:            {cloud.name()}")
    print(f"Topics in scope:  {len(topics)}")

    pinned_tax = load_taxonomy(args.snapshot_date)

    all_gaps = []
    csv_rows: List[dict] = []
    for i, topic_id in enumerate(topics, 1):
        topic_rec = pinned_tax.topics.get(topic_id, {})
        topic_name = topic_rec.get("display_name", topic_id)
        print(f"\n[{i}/{len(topics)}] Topic {topic_id} — {topic_name}")
        gaps = _compute_gaps_for_topic(
            cloud=cloud,
            topic_id=topic_id,
            topic_name=topic_name,
            radius_fraction=args.radius_fraction,
            cell_type=cell_type,
            num_gaps=args.num_gaps,
        )
        all_gaps.extend(gaps)
        if not gaps:
            print(f"  (no gaps found)")
            continue
        for g in gaps:
            print(f"  Gap {g.gap_idx}: size={g.gap_size}, "
                  f"pctile={g.gap_size_percentile:.1f}, "
                  f"z={g.gap_zscore:.2f}, "
                  f"shapiro_log10p={g.shapiro_log10_pval:.2f}")
            idxs = list(g.gap.get_dists_idxs().idxs[:args.top_titles])
            article_titles = g.articles.work_titles()
            for ii in idxs:
                t = article_titles[int(ii)]
                print(f"    - {(t or '(no title)')[:90]}")
            csv_rows.append({
                "scope": args.scope,
                "algorithm": args.algorithm,
                "radius_fraction": args.radius_fraction,
                "num_gaps_per_topic": args.num_gaps,
                "topic_id": topic_id,
                "topic_name": topic_name,
                "gap_idx": int(g.gap_idx),
                "gap_size": int(g.gap_size),
                "gap_size_percentile": float(g.gap_size_percentile),
                "gap_zscore": float(g.gap_zscore),
                "shapiro_log10_pval": float(g.shapiro_log10_pval),
            })

    print(f"\n=== Total gaps: {len(all_gaps)} across {len(topics)} topic(s) ===")

    csv_dir = Path("artifacts/kball_gap_probe")
    csv_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = args.scope.replace("/", "-")
    csv_path = (
        csv_dir
        / f"{safe_scope}__{args.algorithm}__rf{args.radius_fraction}__num{args.num_gaps}.csv"
    )
    fieldnames = [
        "scope", "algorithm", "radius_fraction", "num_gaps_per_topic",
        "topic_id", "topic_name",
        "gap_idx", "gap_size",
        "gap_size_percentile", "gap_zscore", "shapiro_log10_pval",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in csv_rows:
            w.writerow(row)
    print(f"CSV: {csv_path} ({len(csv_rows)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, help="F, F/SF, or F/SF/T")
    parser.add_argument("--algorithm", choices=["kball", "voronoi"], default="kball")
    parser.add_argument("--radius-fraction", type=float, default=0.75)
    parser.add_argument("--num-gaps", type=int, default=5)
    parser.add_argument("--snapshot-date", default="2025-11-06")
    parser.add_argument("--top-titles", type=int, default=5,
                        help="number of sample titles to print per gap (Cloud path only)")
    # Denoiser bypass flags — same choices as hdbscan_gap_probe.py
    parser.add_argument(
        "--denoiser",
        choices=["none", "pca", "multi_lda", "lpp", "lip", "lip_eigen",
                 "lip_diff", "cclip", "wc_cclip", "hierarchical",
                 "criteria_union", "rnojd"],
        default="none",
        help=(
            "none=use Cloud path (default); pca/multi_lda/lpp/lip/lip_eigen/"
            "lip_diff/cclip/wc_cclip/hierarchical/criteria_union/rnojd=denoiser "
            "bypass path (scope-restricted, no full-corpus load)"
        ),
    )
    parser.add_argument(
        "--drop-k", type=int, default=0,
        help="dimensions to project out (meaning depends on denoiser)",
    )
    parser.add_argument(
        "--class-axis",
        choices=["topic", "type", "topic_x_type"],
        default="topic_x_type",
        help="class partition for multi_lda/cclip/hierarchical/rnojd",
    )
    parser.add_argument(
        "--cclip-alpha", type=float, default=None,
        help="alpha for CCLIP/WC-CCLIP (default 0.5)",
    )
    parser.add_argument(
        "--drop-k-graph", type=int, default=2,
        help="Laplacian-step drop_k for hierarchical/criteria_union",
    )
    parser.add_argument(
        "--min-gap-size", type=int, default=0,
        help="Stop greedy K-ball once the largest remaining ball is smaller "
             "than this. 0 disables the cutoff (default).",
    )
    parser.add_argument(
        "--article-types", default="article",
        help="Comma-separated work types treated as ambient/query set "
             "(default: article). Example: 'article,preprint'.",
    )
    parser.add_argument(
        "--book-types", default="book",
        help="Comma-separated work types treated as landmark set "
             "(default: book). Example: 'book,book-chapter' or "
             "'book-chapter' to use only chapters.",
    )
    args = parser.parse_args()

    article_types = tuple(t.strip() for t in args.article_types.split(",") if t.strip())
    book_types = tuple(t.strip() for t in args.book_types.split(",") if t.strip())

    if args.denoiser == "none":
        _run_cloud_path(args)
    else:
        run_with_denoiser(
            scope=args.scope,
            denoiser=args.denoiser,
            drop_k=args.drop_k,
            radius_fraction=args.radius_fraction,
            num_gaps=args.num_gaps,
            class_axis=args.class_axis,
            cclip_alpha=args.cclip_alpha,
            drop_k_graph=args.drop_k_graph,
            algorithm=args.algorithm,
            snapshot_date=args.snapshot_date,
            min_gap_size=args.min_gap_size,
            article_types=article_types,
            book_types=book_types,
        )


if __name__ == "__main__":
    main()
