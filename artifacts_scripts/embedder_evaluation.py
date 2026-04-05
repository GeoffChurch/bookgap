"""Embedder evaluation: citation recall@k, MAP, and scatter metrics.

Usage (Hydra entry point):

    conda run -p .conda python -m artifacts_scripts.embedder_evaluation \\
        embedders='[{_target_: bookgap.embedder.SentenceTransformerEmbedder, model_name: "avsolatorio/GIST-small-Embedding-v0"}]' \\
        denoisers='[NONE]'

    conda run -p .conda python -m artifacts_scripts.embedder_evaluation --help
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp


def citation_recall_at_k(
    embeddings: NDArray[np.float32],
    citation_graph: sp.spmatrix,
    k: int = 20,
) -> float:
    """Mean fraction of actual citations found among k nearest embedding neighbors.

    Documents with no citations (zero rows in citation_graph) are excluded.
    """
    N, d = embeddings.shape
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    # k+1 because nearest neighbor of a point is itself
    _, I = index.search(embeddings, k + 1)

    recalls = []
    for i in range(N):
        true_citations = set(citation_graph[i].nonzero()[1])
        if not true_citations:
            continue
        # Exclude self from neighbor list
        neighbors = set(int(j) for j in I[i] if j != i)
        # Take only k neighbors (excluding self may have given us k from k+1)
        if len(neighbors) > k:
            neighbors = set(list(neighbors)[:k])
        recall = len(true_citations & neighbors) / len(true_citations)
        recalls.append(recall)

    return float(np.mean(recalls)) if recalls else 0.0


def citation_map(
    embeddings: NDArray[np.float32],
    citation_graph: sp.spmatrix,
) -> float:
    """Mean Average Precision: how well does embedding distance rank actual citations?"""
    N, d = embeddings.shape
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    _, I = index.search(embeddings, N)  # full ranking

    aps = []
    for i in range(N):
        true_citations = set(citation_graph[i].nonzero()[1])
        if not true_citations:
            continue
        ranking = [int(j) for j in I[i] if j != i]
        hits = 0
        precision_sum = 0.0
        for rank, j in enumerate(ranking, 1):
            if j in true_citations:
                hits += 1
                precision_sum += hits / rank
        ap = precision_sum / len(true_citations)
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def scatter_metrics(
    embeddings: NDArray,
    citation_graph: sp.spmatrix,
    book_mask: Optional[NDArray] = None,
) -> dict[str, float]:
    """Compute scatter-based embedding quality metrics."""
    N, d = embeddings.shape
    X_c = embeddings - embeddings.mean(axis=0, keepdims=True)

    S_T_trace = float(np.sum(X_c ** 2) / N)

    # Citation error
    cited_means = citation_graph @ embeddings
    D_error = embeddings - cited_means
    D_error_c = D_error - D_error.mean(axis=0, keepdims=True)
    S_C_trace = float(np.sum(D_error_c ** 2) / N)

    # Laplacian
    A_sym = citation_graph + citation_graph.T
    degrees = np.asarray(A_sym.sum(axis=1)).ravel()
    DX = degrees[:, None] * X_c
    S_D_trace = float(np.sum(X_c * DX) / N)
    AX = A_sym @ X_c
    S_A_trace = float(np.sum(X_c * AX) / N)
    S_L_trace = S_D_trace - S_A_trace

    result = {
        "variance_retention": S_T_trace,
        "citation_error_ratio": S_C_trace / S_T_trace if S_T_trace > 0 else 0.0,
        "laplacian_roughness": S_L_trace / S_T_trace if S_T_trace > 0 else 0.0,
    }

    if book_mask is not None:
        article_mask = ~book_mask
        n_b, n_a = int(book_mask.sum()), int(article_mask.sum())
        if n_b > 0 and n_a > 0:
            mu_b = X_c[book_mask].mean(axis=0)
            mu_a = X_c[article_mask].mean(axis=0)
            mu_diff = mu_b - mu_a
            S_B_trace = float((n_b * n_a / N) * np.dot(mu_diff, mu_diff))
            S_W_trace = float(np.sum(X_c[book_mask] ** 2) / N + np.sum(X_c[article_mask] ** 2) / N)
            result["class_separability"] = S_B_trace / S_W_trace if S_W_trace > 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """Hydra config for the embedder evaluation harness."""

    # List of embedder configs. Each entry is a dict with `_target_` and model
    # params, e.g. {_target_: bookgap.embedder.SentenceTransformerEmbedder,
    # model_name: "avsolatorio/GIST-small-Embedding-v0"}.
    embedders: List[Any] = field(default_factory=list)

    # List of denoiser names from WorkDenoisers enum. Use "NONE" for no denoising.
    denoisers: List[str] = field(default_factory=lambda: ["NONE"])

    # Number of dimensions to drop when applying denoisers (0 = keep all).
    drop_k: int = 0

    # k values for recall@k metrics.
    ks: List[int] = field(default_factory=lambda: [10, 20, 50])

    # OpenAlex subfield ID used to filter works (1702 = AI/ML).
    subfield: int = 1702

    # Directory where the output CSV is written.
    output_dir: str = "artifacts/embedder_evaluation"


def _embedder_cfg_from_dict(embedder_dict: Any) -> Any:
    """Convert a Hydra DictConfig/dict to a concrete embedder config dataclass.

    Inspects the ``_target_`` field to determine which config class to create.
    Supports ``SentenceTransformerEmbedderConfig`` and ``TfGepEmbedderConfig``.
    """
    from omegaconf import DictConfig, OmegaConf

    from bookgap.configs.embedder.embedder_config import (
        SentenceTransformerEmbedderConfig,
        TfGepEmbedderConfig,
    )

    if isinstance(embedder_dict, DictConfig):
        d: Dict[str, Any] = OmegaConf.to_container(embedder_dict, resolve=True)  # type: ignore[assignment]
    else:
        d = dict(embedder_dict)

    target = d.get("_target_", "")

    if "SentenceTransformerEmbedder" in target:
        return SentenceTransformerEmbedderConfig(
            model_name=d.get("model_name", SentenceTransformerEmbedderConfig.__dataclass_fields__["model_name"].default),
        )
    elif "TfGepEmbedder" in target:
        defaults = TfGepEmbedderConfig()
        return TfGepEmbedderConfig(
            n_components=d.get("n_components", defaults.n_components),
            gep_type=d.get("gep_type", defaults.gep_type),
            ridge_fraction=d.get("ridge_fraction", defaults.ridge_fraction),
            min_df=d.get("min_df", defaults.min_df),
            max_df=d.get("max_df", defaults.max_df),
            add_self_loops=d.get("add_self_loops", defaults.add_self_loops),
            normalize_adjacency=d.get("normalize_adjacency", defaults.normalize_adjacency),
        )
    else:
        raise ValueError(
            f"Unknown embedder target '{target}'. "
            "Expected a _target_ containing 'SentenceTransformerEmbedder' or 'TfGepEmbedder'."
        )


def _embedder_label(embedder_cfg: Any) -> str:
    """Return a short human-readable label for the embedder config."""
    from bookgap.configs.embedder.embedder_config import (
        SentenceTransformerEmbedderConfig,
        TfGepEmbedderConfig,
    )

    if isinstance(embedder_cfg, SentenceTransformerEmbedderConfig):
        # Use the last path component of the model name for brevity
        return embedder_cfg.model_name.split("/")[-1]
    elif isinstance(embedder_cfg, TfGepEmbedderConfig):
        return f"TfGep({embedder_cfg.gep_type},d={embedder_cfg.n_components})"
    return repr(embedder_cfg)


def _build_eval_citation_graph(
    work_ids: List[str],
    work_references: List[List[str]],
) -> sp.csr_matrix:
    """Build a symmetric, binary citation graph with no self-loops for evaluation.

    No self-loops ensures that a document cannot trivially retrieve itself.
    Symmetrized so that a work is reachable via both citing and cited-by edges.
    """
    from bookgap.util import build_citation_adjacency_from_ids

    adj, _ = build_citation_adjacency_from_ids(
        work_ids, work_references, add_self_loops=False, normalize=False
    )
    # Symmetrize: A_sym[i,j] = 1 if i->j or j->i
    adj_sym = adj + adj.T
    # Binarize (entries may be 1 or 2 after addition)
    adj_sym.data[:] = 1.0
    adj_sym.eliminate_zeros()
    return adj_sym.tocsr()


def _build_scatter_citation_graph(
    work_ids: List[str],
    work_references: List[List[str]],
) -> sp.csr_matrix:
    """Build a normalized citation graph (with self-loops) for scatter metrics.

    scatter_metrics() uses this as a row-stochastic transition matrix to compute
    the 'cited_mean' of each document's neighbourhood.
    """
    from bookgap.util import build_citation_adjacency_from_ids

    adj, _ = build_citation_adjacency_from_ids(
        work_ids, work_references, add_self_loops=True, normalize=True
    )
    return adj.tocsr()


import hydra
from hydra.core.config_store import ConfigStore

from bookgap.configs.config_store import register_configs
from bookgap.util import setup_logging

# Register bookgap configs (needed for any sub-components that rely on the store)
register_configs()

# Register our own config
_cs = ConfigStore.instance()
_cs.store(name="embedder_eval", node=EvalConfig())

setup_logging(log_name="embedder_evaluation", log_level="INFO", include_time=True)


@hydra.main(config_name="embedder_eval", version_base=None)
def main(cfg: EvalConfig) -> None:
    """Iterate over embedder/denoiser combinations and write a comparison CSV."""
    import logging
    from pathlib import Path

    import pandas as pd

    from bookgap.cloud import load_works
    from bookgap.cloud_denoiser import WorkDenoisers
    from bookgap.util import Filter, WorkType, LOGGER

    log = logging.getLogger(__name__)

    if not cfg.embedders:
        raise ValueError(
            "No embedders specified. Pass e.g. "
            "embedders='[{_target_: bookgap.embedder.SentenceTransformerEmbedder, "
            'model_name: "avsolatorio/GIST-small-Embedding-v0"}]\''
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the pre-filter using the configured subfield
    pre_filter = Filter(
        subfield=str(cfg.subfield),
        work_types=(
            WorkType.BOOK.value,
            WorkType.ARTICLE.value,
            WorkType.BOOK_CHAPTER.value,
            WorkType.PREPRINT.value,
        ),
    )

    all_rows: List[Dict[str, Any]] = []

    for raw_embedder_cfg in cfg.embedders:
        embedder_cfg = _embedder_cfg_from_dict(raw_embedder_cfg)
        emb_label = _embedder_label(embedder_cfg)
        log.info(f"=== Embedder: {emb_label} ===")

        # Load works and raw embeddings (cached on disk)
        store = load_works(pre_filter, None, embedder_cfg)

        # Retain all works (no deduplication in evaluation harness for simplicity)
        all_indices = list(range(store.length))

        raw_embeddings: NDArray[np.float32] = store.embeddings(indices=all_indices)
        works: List[Dict[str, Any]] = store.to_dicts(indices=all_indices)
        work_ids: List[str] = [w["id"] for w in works]
        work_references: List[List[str]] = [w.get("referenced_works", []) for w in works]

        log.info(
            f"Loaded {len(works)} works, embeddings shape={raw_embeddings.shape}"
        )

        # Build citation graphs once per embedder (they depend only on the works, not the denoiser)
        eval_citation_graph = _build_eval_citation_graph(work_ids, work_references)
        scatter_citation_graph = _build_scatter_citation_graph(work_ids, work_references)

        log.info(
            f"Eval citation graph: shape={eval_citation_graph.shape}, "
            f"nnz={eval_citation_graph.nnz}"
        )

        # Build book mask for class-separability metric
        work_types_list = store.to_list("type", indices=all_indices)
        book_mask: NDArray[np.bool_] = np.array(
            [wt == WorkType.BOOK.value for wt in work_types_list], dtype=bool
        )

        for denoiser_name in cfg.denoisers:
            log.info(f"  Denoiser: {denoiser_name}, drop_k={cfg.drop_k}")

            try:
                denoiser_impl = WorkDenoisers[denoiser_name].value
                proj = denoiser_impl.compute_projection_artifact(
                    cfg.drop_k, works, raw_embeddings
                )
                embeddings: NDArray[np.float32] = proj.apply(raw_embeddings).astype(
                    np.float32
                )
            except Exception as exc:
                log.error(
                    f"  Failed to apply denoiser '{denoiser_name}': {exc}"
                )
                all_rows.append(
                    {
                        "embedder": emb_label,
                        "denoiser": denoiser_name,
                        "drop_k": cfg.drop_k,
                        "error": str(exc),
                    }
                )
                continue

            row: Dict[str, Any] = {
                "embedder": emb_label,
                "denoiser": denoiser_name,
                "drop_k": cfg.drop_k,
                "n_works": len(works),
                "embedding_dim": embeddings.shape[1],
            }

            # Recall@k
            for k in cfg.ks:
                try:
                    recall = citation_recall_at_k(embeddings, eval_citation_graph, k=k)
                    row[f"recall@{k}"] = recall
                    log.info(f"    recall@{k} = {recall:.4f}")
                except Exception as exc:
                    log.error(f"    recall@{k} failed: {exc}")
                    row[f"recall@{k}"] = float("nan")

            # MAP
            try:
                map_score = citation_map(embeddings, eval_citation_graph)
                row["MAP"] = map_score
                log.info(f"    MAP = {map_score:.4f}")
            except Exception as exc:
                log.error(f"    MAP failed: {exc}")
                row["MAP"] = float("nan")

            # Scatter metrics
            try:
                scatter = scatter_metrics(
                    embeddings,
                    scatter_citation_graph,
                    book_mask=book_mask,
                )
                row.update(scatter)
                log.info(
                    f"    variance_retention={scatter.get('variance_retention', float('nan')):.4f}, "
                    f"citation_error_ratio={scatter.get('citation_error_ratio', float('nan')):.4f}, "
                    f"laplacian_roughness={scatter.get('laplacian_roughness', float('nan')):.4f}"
                )
            except Exception as exc:
                log.error(f"    scatter_metrics failed: {exc}")

            all_rows.append(row)

    # Write CSV
    df = pd.DataFrame(all_rows)
    csv_path = output_dir / "embedder_comparison.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"\nResults written to {csv_path}")

    # Print summary table
    metric_cols = [c for c in df.columns if c not in {"error"}]
    print("\n" + "=" * 72)
    print("EMBEDDER EVALUATION SUMMARY")
    print("=" * 72)
    print(df[metric_cols].to_string(index=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
