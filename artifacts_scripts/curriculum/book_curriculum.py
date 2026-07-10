"""Track A of the real-book ToC reconstruction eval: build OUR curriculum from a
book's OWN gap.

Reads a ``book_own_gap.py`` output JSON (the in-ball article ids = the book's own
gap, computed by treating the book as an article and finding its nearest non-self
book), seeds the curriculum pipeline's ``target_work_ids`` with them, and runs the
build at the SAME scope the gap was computed at. The result is "our curriculum"
for that book's gap, to be compared against the book's real ToC (Track B).

Mirrors ``seed_from_gap.build_curriculum_for_gap``'s build body, but the seed is a
book's own gap rather than a detector-chosen gap. Run ONE per process.

NOTE: the reference book is NOT yet held out of the cloud (a refinement). The seed
is articles, not the book, and the book is one node among the scope's works, so the
leakage is minor for validation; flagged for the clean run.

Example::

    uv run python -m artifacts_scripts.curriculum.book_curriculum \\
        --gap-json artifacts/book_gaps/W2898513995_cyberbullying_ai.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from bookgap.util import LOGGER, setup_logging

from .build_curriculum import run_curriculum_building
from .config import CurriculumConfig, get_curriculum_output_dir
from .main import run_keyword_extraction, run_plotting
from .output_summary import generate_curriculum_summary


def build_curriculum_from_book_gap(
    gap_json: Path,
    *,
    output_dir: Optional[str] = None,
    subfield: Optional[str] = None,
    field: Optional[str] = None,
    desired_n_topics: int = 15,
    boundary: bool = False,
    semantic_rerank: bool = False,
    semantic_bandwidth: Optional[float] = None,
    similarity: bool = False,
    similarity_mix_weight: Optional[float] = None,
    similarity_knn_k: Optional[int] = None,
    keywords_top_k: int = 30,
    dynamic_topk_floor: float = 0.0,
    anchor_mode: Optional[str] = None,
    soft_topics: bool = False,
    pagerank_alpha: Optional[float] = None,
    decouple_selection: bool = False,
    paratext_weights: Optional[Path] = None,
    plot_enabled: bool = True,
    linkage_method: Optional[str] = None,
    denoiser: Optional[str] = None,
    drop_k: Optional[int] = None,
    ranking_top_k: Optional[int] = None,
) -> Path:
    rec = json.loads(Path(gap_json).read_text())
    bk = rec.get("book", rec)  # single-book mode nests under "book"; fall back to flat
    ids = bk["gap_article_ids"]
    if not ids:
        raise ValueError(f"empty gap_article_ids in {gap_json}")
    weights = bk.get("gap_article_weights")  # keyword-concept soft target weights (optional)
    if paratext_weights is not None:
        # PIPELINE-WIDE paratext mitigation: multiply each work's target weight by (1 - p_w), so paratext
        # works are down-weighted in the PageRank personalization AND the per-chapter exemplar selection —
        # they never become a chapter's representative work or a central node, regardless of how the
        # clustering resolves. Soft, no boolean; a paratext work is never a good exemplar, so this is
        # do-no-harm even when p_w is imperfect.
        pw = json.loads(Path(paratext_weights).read_text())
        weights = list(weights) if weights else [1.0] * len(ids)
        weights = [w * (1.0 - float(pw.get(wid, 0.0))) for w, wid in zip(weights, ids)]
        nz = sum(1 for wid in ids if float(pw.get(wid, 0.0)) > 0.5)
        LOGGER.info("paratext-weights: down-weighted %d/%d works (p_w>0.5) in target_work_weights", nz, len(ids))
    sf = subfield or rec.get("scope_subfield")
    fld = field or rec.get("scope_field")
    book_id = bk.get("book_id", "unknown")
    book_title = bk.get("book", book_id)
    name = output_dir or f"book_curriculum_{book_id.rsplit('/', 1)[-1]}"

    cfg = OmegaConf.structured(CurriculumConfig())
    cfg.target_work_ids = ids
    if weights:
        cfg.target_work_weights = list(weights)      # concept-weighted personalization
    if pagerank_alpha is not None:
        cfg.pagerank.alpha = float(pagerank_alpha)   # α sweep (low α anchors to the concept)
    if decouple_selection:
        cfg.decouple_selection = True                # concept selects; citations only order
    cfg.plot.enabled = plot_enabled                  # skip the slow field-background plot in sweeps
    cfg.output_dir = name
    cfg.desired_n_topics = desired_n_topics
    cfg.rank_before_clustering = True
    cfg.ranking_top_k = int(ranking_top_k) if ranking_top_k is not None else 5000
    cfg.keywords.top_k = keywords_top_k
    cfg.keywords.dynamic_topk_floor = dynamic_topk_floor
    if linkage_method:
        # Clustering geometry lever: override the config default ('average', scipy
        # agglomerative average-linkage on the top-k subset). E.g. pass
        # 'weighted_bisecting_kmeans' to A/B against it. anchor_mode (facility/dpp)
        # below still wins if both are passed (it is the more specific selector).
        cfg.linkage_method = linkage_method
    if denoiser is not None:
        # Embedding-SPACE lever: e.g. 'PCA' with drop_k to reduce GIST-small 384d to
        # (384-drop_k) dims, fit unsupervised on the scope corpus. Applied to the
        # clustered works here AND, by sharing the same cloud config, to anything the
        # recon embeds (book_toc_reconstruction projects chapters through the same
        # denoiser_projection so the chapter-assignment space matches).
        cfg.experiment.cloud.denoiser = denoiser
    if drop_k is not None:
        cfg.experiment.cloud.drop_k = int(drop_k)
    if anchor_mode:
        # FL/DPP diverse-anchor clustering (de-duplicates topics) instead of the
        # dendrogram cut. Pairs with soft_topics for clean per-chapter keywords.
        cfg.linkage_method = anchor_mode
    if soft_topics:
        cfg.keywords.soft_topics = True
    if sf:
        cfg.experiment.cloud.filter_pair.pre.subfield = str(sf)
    if fld:
        cfg.experiment.cloud.filter_pair.pre.field = str(fld)
    if boundary:
        cfg.boundary_selection = True
        cfg.background_exponent = 0.5
    if semantic_rerank:
        # Product-of-experts selection: log(p/g_local) + log f_hat(seed KDE).
        # Takes priority over boundary_selection; demotes citation-central but
        # semantically-distant foundations (the universal specificity defect).
        cfg.semantic_rerank = True
        if semantic_bandwidth is not None:
            cfg.semantic_bandwidth = semantic_bandwidth
    if similarity:
        # Mix an embedding-kNN graph into the citation graph before PageRank so the
        # random walk prefers semantically-similar neighbours (demotes citation-central
        # but semantically-distant foundations -> targets the specificity defect).
        # Scope-proportional self-kNN: fine at field/subfield, OOMs only at domain.
        cfg.similarity.enabled = True
        if similarity_mix_weight is not None:
            cfg.similarity.mix_weight = similarity_mix_weight
        if similarity_knn_k is not None:
            cfg.similarity.knn_k = similarity_knn_k

    out = get_curriculum_output_dir(name, auto_derived=False)
    setup_logging(log_name="pipeline", log_level="DEBUG", include_time=False,
                  log_path=out / "pipeline.log", explicit_args=None)
    LOGGER.info("=" * 60)
    LOGGER.info("BOOK-GAP CURRICULUM (Track A)")
    LOGGER.info("=" * 60)
    # A parseable blind title (eval_book_judge._parse_topic_name reads "Topic Tx (name)")
    # so the pairwise judge shows the same neutral book title for Track A and Track B.
    LOGGER.info("Topic T0000 (%s) gap 0: book-gap curriculum (Track A)", book_title)
    LOGGER.info("Book %s: seeding %d in-gap articles; subfield=%s; field=%s; boundary=%s; similarity=%s",
                book_id, len(ids), sf, fld, boundary, similarity)

    cloud, embeddings_for_clustering, filtered_indices = run_curriculum_building(cfg, output_dir=out)
    topic_keywords = run_keyword_extraction(cfg.keywords, out, cloud, filtered_indices)
    generate_curriculum_summary(out, cloud, filtered_indices, topic_keywords)
    if cfg.plot.enabled:
        run_plotting(cfg.plot, out, embeddings_for_clustering, topic_keywords)
    LOGGER.info("Curriculum complete for %s -> %s", book_id, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Track A: curriculum from a book's own gap.")
    ap.add_argument("--gap-json", required=True, type=Path,
                    help="book_own_gap.py output JSON (single-book mode, has gap_article_ids)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--subfield", default=None, help="override scope subfield (default: from gap JSON)")
    ap.add_argument("--field", default=None, help="override scope field (default: from gap JSON)")
    ap.add_argument("--desired-n-topics", type=int, default=15)
    ap.add_argument("--boundary", action="store_true", help="use the boundary-background selector")
    ap.add_argument("--semantic-rerank", action="store_true",
                    help="product-of-experts selection: log(p/g_local) + log f_hat(seed KDE); "
                         "demotes citation-central but semantically-distant foundations")
    ap.add_argument("--semantic-bandwidth", type=float, default=None,
                    help="KDE bandwidth h^2 (squared-dist units); default = median heuristic over seeds")
    ap.add_argument("--similarity", action="store_true",
                    help="mix an embedding-kNN similarity graph into the citation graph before "
                         "PageRank (scope-proportional; fine at field/subfield, OOMs at domain)")
    ap.add_argument("--similarity-mix-weight", type=float, default=None,
                    help="convex blend weight for the similarity graph (default cfg 0.1)")
    ap.add_argument("--similarity-knn-k", type=int, default=None,
                    help="kNN neighbors per node for the similarity graph (default cfg 5)")
    ap.add_argument("--keywords-top-k", type=int, default=30,
                    help="keywords per chapter (default 30, to match the pairwise eval)")
    ap.add_argument("--dynamic-topk-floor", type=float, default=0.0,
                    help="dynamic top-k: keep keywords scoring >= this fraction of the "
                         "topic's top score (0=off; ~0.05 trims ~0-score padding on thin chapters)")
    ap.add_argument("--anchor-mode", choices=["facility", "dpp"], default=None,
                    help="diverse-anchor clustering instead of the dendrogram cut (de-duplicates topics)")
    ap.add_argument("--soft-topics", action="store_true",
                    help="fuzzy keyword extraction (works contribute to all topics by mass*r); "
                         "~5x slower but dissolves degenerate chapters")
    ap.add_argument("--pagerank-alpha", type=float, default=None,
                    help="PageRank link-follow prob (default cfg 0.85). Lower = more teleport to the "
                         "(concept-weighted) seeds = recency-anchored; sweep for the de-frontiering diagnostic.")
    ap.add_argument("--decouple-selection", action="store_true",
                    help="use the concept personalization DIRECTLY as the selection mass (citation PR "
                         "only orders chapters) — recency-faithful; the α→0 limit for selection.")
    ap.add_argument("--paratext-weights", type=Path, default=None,
                    help="paratext_work_weights.json (work_id -> p_w in [0,1]); multiplies target_work_weights "
                         "by (1-p_w) so paratext works fade from selection + PageRank, pipeline-wide.")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the curriculum plot (the field-background projection is the ~20min tail).")
    ap.add_argument("--linkage-method", default=None,
                    help="clustering geometry: e.g. 'average' (scipy agglomerative average-linkage on "
                         "the top-k subset) vs default 'weighted_bisecting_kmeans'. See config.py "
                         "CurriculumConfig.linkage_method for the full set.")
    ap.add_argument("--denoiser", default=None,
                    help="embedding-space denoiser (WorkDenoisers name, e.g. 'PCA' or 'NONE'); applied to "
                         "the cloud, so clustering and recon chapter-assignment share the same space.")
    ap.add_argument("--drop-k", type=int, default=None,
                    help="dims to drop for the denoiser (PCA keeps 384-drop_k; e.g. 334 => PCA-50).")
    ap.add_argument("--ranking-top-k", type=int, default=None,
                    help="top-k works by relevance to cluster (rank_before_clustering); default 5000.")
    args = ap.parse_args()
    out = build_curriculum_from_book_gap(
        args.gap_json, output_dir=args.output_dir, subfield=args.subfield, field=args.field,
        desired_n_topics=args.desired_n_topics, boundary=args.boundary,
        semantic_rerank=args.semantic_rerank, semantic_bandwidth=args.semantic_bandwidth,
        similarity=args.similarity, similarity_mix_weight=args.similarity_mix_weight,
        similarity_knn_k=args.similarity_knn_k, keywords_top_k=args.keywords_top_k,
        dynamic_topk_floor=args.dynamic_topk_floor,
        anchor_mode=args.anchor_mode, soft_topics=args.soft_topics,
        pagerank_alpha=args.pagerank_alpha, decouple_selection=args.decouple_selection,
        paratext_weights=args.paratext_weights,
        plot_enabled=not args.no_plot,
        linkage_method=args.linkage_method, denoiser=args.denoiser, drop_k=args.drop_k,
        ranking_top_k=args.ranking_top_k,
    )
    print(f"DONE -> {out}")


if __name__ == "__main__":
    main()
