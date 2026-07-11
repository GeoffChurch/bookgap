#!/usr/bin/env python3
"""Capture the deterministic curriculum-parity goldens (Phase 3 T7).

This is THE gate that authorizes deleting the legacy OpenAlex curriculum
pipeline (``artifacts_scripts/curriculum/``) in favor of the standalone
``curriculum`` engine. It runs the LEGACY pipeline (``run_curriculum_building``
+ ``run_keyword_extraction``, called as library functions -- NOT the full
``main()`` with plotting) over a small, real OpenAlex cloud under two pinned
configs, and emits small JSON goldens that ``curriculum/tests/test_parity_legacy.py``
replays through the new engine's ``build_curriculum`` for an exact comparison
(selection set, cluster membership up to a label permutation, topic order,
top-8 keywords per topic -- NO numeric tolerances).

Mirrors ``scripts/capture_gap_parity_golden.py``'s pattern: shared helpers,
pinned overrides/seeds, deterministic capture, small committed golden JSON.

Cloud scope
-----------
Re-instantiates ``artifacts/curriculum/cl_A1_avg_cyberbullying``'s cloud: the
smallest real OpenAlex-subfield cloud on disk (its ``citation_graph.npz`` is
19.9MB). Per ``bookgap/configs/util/filter_config.py``'s ``FilterPairConfig``
defaults, ``domain=3`` (``Domain.PHYSICAL_SCIENCES``) and
``work_types=[book,article,book-chapter,preprint]`` are ALREADY the
"curriculum"/"standard" config defaults -- verified against that artifact's own
``pipeline.log`` ``Cloud(...)`` repr -- so the only real override is
``subfield='1702'``. The 341 ``target_work_ids`` are read directly from that
artifact's ``metadata.json`` (not transcribed) so they stay tied to their
source-of-truth.

Config pair (the brief's exact axes; translation to both surfaces is
documented inline and re-emitted into each golden's ``"mapping"`` block as
T8's mapping-table seed):

(a) PageRank + top_n(=300 by raw PR mass) + bisecting mass-frontier
    (unweighted ``bisecting_kmeans``) + MASS ordering + keywords ON.
(b) Bidirectional PageRank + background-normalized selection +
    citation-avg-linkage (binary Jaccard) + FRONTIER-ADJACENCY ordering +
    keywords ON.

Both configs pin ``pagerank.{tol,max_iter}`` to the SAME explicit values on
both surfaces (legacy defaults: tol=1e-10, max_iter=500; the engine's own
defaults are looser, 1e-9/200) -- otherwise two independently-converged power
iterations could land a boundary work on opposite sides of a top-k/mass-
threshold cut despite being numerically indistinguishable, which would show up
as a spurious selection/clustering mismatch having nothing to do with real
behavioral drift.

``keywords.doc_weighting`` is pinned to ``EQUAL`` on the legacy side (its
default is ``RELEVANCE``) because the engine's ``keywords_for_topics`` has NO
``doc_weighting`` knob at all -- it hardcodes ``DocWeighting.EQUAL`` (see
``curriculum/layers/keywords.py``). Leaving legacy's default would silently
compare two different keyword-weighting recipes.

Bidirectional PageRank needs at least one ``source_work_id`` to exercise its
reverse-seed pass at all (with none, it gracefully degrades to forward-only
PR -- see ``build_curriculum.py``'s docstring); two "known" works already
present in this cloud (drawn from the cyberbullying artifact's own
``topic_work_ids.json``) are used as config (b)'s sources.

Order + representative-id computation deliberately imports the ENGINE's own
``curriculum.core.order`` functions (bookgap -> curriculum is the allowed
import direction, already established by T6's bridge) rather than
re-implementing them, so the golden's "order" field is computed by the EXACT
SAME code the engine will run at assertion time -- the same "capture and
check share one code path" discipline as ``capture_gap_parity_golden.py``.

Artifacts
---------
- Goldens: ``curriculum/tests/fixtures/parity/config_{a,b}.json`` (small,
  committed).
- Bridged GraphSource npz: ``curriculum/cache/parity/cyberbullying_ai1702.npz``
  (shared by both configs -- same cloud, only ranking/selection/clustering/
  ordering differ). Expected to be several GB (full-corpus 1.9M dense
  embeddings dominate, not the 19.9MB citation graph alone) -- gitignored, not
  committed; the parity test skips-if-absent.
"""
from __future__ import annotations

import os

# Pin BLAS/OpenMP threading BEFORE numpy/scipy/sklearn are imported anywhere
# in the process, so KMeans (config a) and the PageRank power iteration
# (both configs) are bit-reproducible across runs -- the same determinism
# discipline as capture_gap_parity_golden.py's faiss.omp_set_num_threads(1)
# pinning, applied to the BLAS backends sklearn/numpy/scipy actually use here.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize
from hydra.utils import instantiate

from artifacts_scripts.curriculum.build_curriculum import run_curriculum_building
from artifacts_scripts.curriculum.main import run_keyword_extraction
from artifacts_scripts.export_graph_source import (
    assemble_arrays,
    build_meta,
    compose_cloud_cfg,
)
from curriculum.core.order import linearization_order_from_adjacency, order_by_mass
from curriculum.sources.graph_npz import save_graph_npz

BOOKGAP_ROOT = Path(__file__).resolve().parent.parent
CURRICULUM_ROOT = BOOKGAP_ROOT.parent / "curriculum"

_CYBERBULLYING_METADATA = (
    BOOKGAP_ROOT / "artifacts/curriculum/cl_A1_avg_cyberbullying/metadata.json"
)

# Only the subfield needs overriding for the "curriculum" config surface
# (CurriculumConfig.experiment.cloud); domain/work_types/denoiser/drop_k are
# already the FilterPairConfig/CloudConfig defaults. See module docstring.
_CLOUD_OVERRIDES_CURRICULUM = ["experiment.cloud.filter_pair.pre.subfield='1702'"]
# Same cloud, composed via the flatter "standard"/export_graph_source surface
# (top-level `cloud`, not `experiment.cloud` -- see T6's report).
_CLOUD_OVERRIDES_STANDARD = ["cloud.filter_pair.pre.subfield='1702'"]

# Two "known" works (present in this cloud) used only by config (b) as
# source_work_ids, to actually exercise BIDIRECTIONAL_PAGE_RANK's reverse
# pass. Drawn from cl_A1_avg_cyberbullying's own topic_work_ids.json (topic 3,
# the C4.5 singleton; topic 0's top member) -- guaranteed present, no semantic
# claim about "what a learner already knows" is intended, this is a
# parity-mechanism smoke, not a curriculum-quality choice.
_SOURCE_WORK_IDS = [
    "https://openalex.org/W2125055259",
    "https://openalex.org/W3146306708",
]

GOLDEN_DIR = CURRICULUM_ROOT / "curriculum/tests/fixtures/parity"
NPZ_PATH = CURRICULUM_ROOT / "cache/parity/cyberbullying_ai1702.npz"

# Shared PageRank convergence knobs, pinned identically on the legacy AND
# engine config surfaces (see module docstring for why this must be pinned
# rather than left at each side's own default).
_PAGERANK_ALPHA = 0.85
_PAGERANK_TOL = 1e-10
_PAGERANK_MAX_ITER = 500

_DESIRED_N_TOPICS = 8
_TOP_N = 300
_KEYWORDS_TOP_K = 12
_KEYWORDS_SAMPLES_PER_TOPIC = 80


def load_target_work_ids() -> list[str]:
    """The 341 target work ids from cl_A1_avg_cyberbullying's metadata.json.

    Read at capture time (not transcribed) so the golden's seed set stays
    tied to its source-of-truth artifact.
    """
    meta = json.loads(_CYBERBULLYING_METADATA.read_text())
    ids = meta["target_work_ids"]
    if len(ids) != meta["n_target_works"]:
        raise ValueError(
            f"{_CYBERBULLYING_METADATA}: target_work_ids length {len(ids)} != "
            f"n_target_works {meta['n_target_works']}"
        )
    return list(ids)


def compose_curriculum_cfg(overrides: list[str]) -> Any:
    """Compose the "curriculum" ``CurriculumConfig`` under ``overrides``.

    "curriculum" is registered into the process-global ``ConfigStore`` by
    importing ``build_curriculum`` (which calls ``register_configs()`` +
    ``cs.store(name="curriculum", ...)`` at module import time, mirroring
    ``main.py``'s own registration) -- it has no filesystem config path
    (``main.py`` itself uses ``@hydra.main(config_path=None, ...)``), so
    composing needs only a path-less ``initialize``, not
    ``initialize_config_module``.
    """
    with initialize(version_base=None, config_path=None):
        return compose(config_name="curriculum", overrides=list(overrides))


def build_config_a() -> Any:
    """PageRank + top_n + bisecting mass-frontier + mass ordering + keywords ON."""
    cfg = compose_curriculum_cfg(
        [
            *_CLOUD_OVERRIDES_CURRICULUM,
            f"desired_n_topics={_DESIRED_N_TOPICS}",
            "method=PAGE_RANK",
            "rank_before_clustering=true",
            f"ranking_top_k={_TOP_N}",
            "linkage_method=bisecting_kmeans",
            "min_cluster_size=1",
            "background_normalize=false",
            "boundary_selection=false",
            "semantic_rerank=false",
            f"pagerank.alpha={_PAGERANK_ALPHA}",
            f"pagerank.tol={_PAGERANK_TOL}",
            f"pagerank.max_iter={_PAGERANK_MAX_ITER}",
            "keywords.enabled=true",
            "keywords.method=LOG_ODDS",
            f"keywords.top_k={_KEYWORDS_TOP_K}",
            f"keywords.samples_per_topic={_KEYWORDS_SAMPLES_PER_TOPIC}",
            "keywords.doc_weighting=EQUAL",
            "keywords.topic_weighting=EQUAL",
            "keywords.background_mode=MARGINAL",
            "keywords.weight_by_freq=true",
            "keywords.dynamic_topk_floor=0.0",
            "keywords.soft_topics=false",
        ]
    )
    cfg.target_work_ids = load_target_work_ids()
    cfg.source_work_ids = []
    return cfg


def build_config_b() -> Any:
    """Bidirectional PR + background-normalized + citation-avg-linkage (binary
    jaccard) + frontier(-adjacency) ordering + keywords ON."""
    cfg = compose_curriculum_cfg(
        [
            *_CLOUD_OVERRIDES_CURRICULUM,
            f"desired_n_topics={_DESIRED_N_TOPICS}",
            "method=BIDIRECTIONAL_PAGE_RANK",
            "rank_before_clustering=true",
            f"ranking_top_k={_TOP_N}",
            "linkage_method=citation_avg_linkage",
            "citation_jaccard_mode=binary",
            "background_normalize=true",
            "background_exponent=0.5",
            "background_eps=1e-12",
            "boundary_selection=false",
            "semantic_rerank=false",
            f"pagerank.alpha={_PAGERANK_ALPHA}",
            f"pagerank.tol={_PAGERANK_TOL}",
            f"pagerank.max_iter={_PAGERANK_MAX_ITER}",
            "keywords.enabled=true",
            "keywords.method=LOG_ODDS",
            f"keywords.top_k={_KEYWORDS_TOP_K}",
            f"keywords.samples_per_topic={_KEYWORDS_SAMPLES_PER_TOPIC}",
            "keywords.doc_weighting=EQUAL",
            "keywords.topic_weighting=EQUAL",
            "keywords.background_mode=MARGINAL",
            "keywords.weight_by_freq=true",
            "keywords.dynamic_topk_floor=0.0",
            "keywords.soft_topics=false",
        ]
    )
    cfg.target_work_ids = load_target_work_ids()
    cfg.source_work_ids = list(_SOURCE_WORK_IDS)
    return cfg


def export_npz(out_path: Path) -> dict[str, Any]:
    """Bridge the cyberbullying-scope cloud to a GraphSource npz via T6."""
    cfg = compose_cloud_cfg(_CLOUD_OVERRIDES_STANDARD, config_name="standard")
    cloud = instantiate(cfg.cloud)
    ids, adj, titles, abstracts, emb = assemble_arrays(cloud)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_graph_npz(out_path, ids=ids, adj=adj, titles=titles, abstracts=abstracts, emb=emb)

    meta = build_meta(
        cloud,
        cfg,
        ids=ids,
        adj=adj,
        target_work_ids=load_target_work_ids(),
        config_name="standard",
        overrides=_CLOUD_OVERRIDES_STANDARD,
    )
    meta_path = out_path.parent / f"{out_path.name}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
    return meta


def _topic_representative_ids(
    point_to_cluster: np.ndarray, point_masses: np.ndarray, ids: list[str], n_topics: int
) -> dict[int, str]:
    """The top-raw-mass member id of each cluster label (the golden's stable
    per-topic identity -- label INTEGERS are implementation artifacts that
    need not agree between legacy and engine, but a topic's top-mass member
    id does, once the clusters themselves match)."""
    rep: dict[int, str] = {}
    for label in range(n_topics):
        members = np.where(point_to_cluster == label)[0]
        if len(members) == 0:
            continue
        top_member = members[np.argmax(point_masses[members])]
        rep[label] = ids[int(top_member)]
    return rep


def run_and_capture(name: str, cfg: Any, output_dir: Path) -> dict[str, Any]:
    """Run the legacy pipeline for one config and assemble its golden dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cloud, _emb, filtered_indices = run_curriculum_building(cfg, output_dir=output_dir)
    topic_keywords = run_keyword_extraction(cfg.keywords, output_dir, cloud, filtered_indices)
    assert topic_keywords is not None, "keywords.enabled=true but got no keywords back"

    filtered_ids: list[str] = json.loads((output_dir / "filtered_work_ids.json").read_text())
    point_to_cluster = np.load(output_dir / "point_to_cluster.npy")
    point_masses = np.load(output_dir / "point_masses.npy")
    adj_frontier = np.load(output_dir / "adj_frontier.npy")

    n = len(filtered_ids)
    if point_to_cluster.shape != (n,) or point_masses.shape != (n,):
        raise ValueError(
            f"{name}: array/id length mismatch: n_ids={n} "
            f"point_to_cluster={point_to_cluster.shape} point_masses={point_masses.shape}"
        )
    n_topics = int(point_to_cluster.max()) + 1 if n else 0

    if cfg.linkage_method == "bisecting_kmeans":
        # config (a): MASS ordering -- the engine's own order_by_mass, run
        # directly over the captured (point_to_cluster, point_masses).
        order_labels, _topic_mass = order_by_mass(point_to_cluster, point_masses, n_topics)
    else:
        # config (b): FRONTIER_ADJACENCY ordering -- the engine's own
        # linearization_order_from_adjacency, run over the captured
        # adj_frontier (already reordered into legacy's own canonical
        # linearization -- see build_curriculum.py's "Relabel topics..." step
        # -- so this recomputation is expected to be a no-op permutation;
        # asserted below as a self-consistency check on that expectation,
        # not a load-bearing step for the golden itself).
        order_labels = linearization_order_from_adjacency(adj_frontier)
        assert list(order_labels) == list(range(n_topics)), (
            f"{name}: legacy's own canonical relabeling IS frontier-adjacency "
            "order already; recomputing it independently should be a no-op "
            f"permutation, got {list(order_labels)}"
        )

    rep_id_by_label = _topic_representative_ids(point_to_cluster, point_masses, filtered_ids, n_topics)
    order_representative_ids = [rep_id_by_label[int(label)] for label in order_labels]

    topic_keywords_top8 = {
        rep_id_by_label[label]: [w for w, _ in topic_keywords.get(label, [])][:8]
        for label in range(n_topics)
        if label in rep_id_by_label
    }

    id_to_cluster = {filtered_ids[i]: int(point_to_cluster[i]) for i in range(n)}

    return {
        "config_name": name,
        "cloud_overrides": _CLOUD_OVERRIDES_CURRICULUM,
        "target_work_ids": list(cfg.target_work_ids),
        "source_work_ids": list(cfg.source_work_ids),
        "n_topics": n_topics,
        "selection_ids": sorted(filtered_ids),
        "id_to_cluster": id_to_cluster,
        "order_representative_ids": order_representative_ids,
        "topic_keywords_top8": topic_keywords_top8,
        "mapping": {
            "legacy": {
                "method": str(cfg.method),
                "rank_before_clustering": bool(cfg.rank_before_clustering),
                "ranking_top_k": int(cfg.ranking_top_k),
                "linkage_method": str(cfg.linkage_method),
                "citation_jaccard_mode": str(cfg.citation_jaccard_mode),
                "background_normalize": bool(cfg.background_normalize),
                "pagerank.alpha": float(cfg.pagerank.alpha),
                "pagerank.tol": float(cfg.pagerank.tol),
                "pagerank.max_iter": int(cfg.pagerank.max_iter),
                "keywords.doc_weighting": str(cfg.keywords.doc_weighting),
                "keywords.top_k": int(cfg.keywords.top_k),
                "keywords.samples_per_topic": int(cfg.keywords.samples_per_topic),
            },
            "engine": {
                "method": (
                    "RankMethod.BIDIRECTIONAL_PAGE_RANK"
                    if "BIDIRECTIONAL" in str(cfg.method)
                    else "RankMethod.PAGE_RANK"
                ),
                "selection.mode": (
                    "SelectionMode.BACKGROUND_NORMALIZED"
                    if cfg.background_normalize
                    else "SelectionMode.TOP_N"
                ),
                "selection.top_n": int(cfg.ranking_top_k),
                "embeddings_mode": (
                    "EmbeddingsMode.CITATION_LINKAGE"
                    if cfg.linkage_method == "citation_avg_linkage"
                    else "EmbeddingsMode.PROVIDED"
                ),
                "clustering.linkage_method": "average",
                "clustering.citation_jaccard_mode": str(cfg.citation_jaccard_mode),
                "clustering.weighted": False,
                "ordering": (
                    "OrderingMode.MASS"
                    if cfg.linkage_method == "bisecting_kmeans"
                    else "OrderingMode.FRONTIER_ADJACENCY"
                ),
                "keywords.top_k": int(cfg.keywords.top_k),
                "keywords.samples_per_topic": int(cfg.keywords.samples_per_topic),
                "random_state": 42,
            },
        },
    }


def main() -> None:
    print(f"Exporting GraphSource npz to {NPZ_PATH} ...")
    meta = export_npz(NPZ_PATH)
    size_mb = NPZ_PATH.stat().st_size / 1e6
    print(
        f"Wrote {NPZ_PATH} ({meta['n_nodes']} nodes, {meta['n_edges']} edges, "
        f"{size_mb:.1f} MB)"
    )

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    configs = {"a": build_config_a(), "b": build_config_b()}
    for name, cfg in configs.items():
        output_dir = BOOKGAP_ROOT / f"artifacts/curriculum/parity_{name}"
        print(f"\n=== Running legacy pipeline for config {name!r} -> {output_dir} ===")
        golden = run_and_capture(name, cfg, output_dir)
        golden_path = GOLDEN_DIR / f"config_{name}.json"
        golden_path.write_text(json.dumps(golden, indent=2, sort_keys=True))
        print(
            f"Wrote {golden_path}: {golden['n_topics']} topics, "
            f"{len(golden['selection_ids'])} selected ids"
        )


if __name__ == "__main__":
    main()
