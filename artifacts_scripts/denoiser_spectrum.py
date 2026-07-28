"""Denoiser GEP eigenvalue-spectrum diagnostic.

For each (denoiser, embedder) pair: loads the real bookgap AI cloud, calls
the denoiser's projector at full n_components, and reports the eigenvalue
spectrum. Two questions:

1. **Rank-1 hypothesis confirmation** — for ANTI_LDA and CCP (rank-1 S_B),
   we expect exactly 1 eigenvalue with substantial magnitude and (d-1)
   numerical-noise eigenvalues.

2. **Empirical low-rank check** — for CITATION_ERROR_LDA, JAD_EQUAL, and
   CCLIP_01 (rank-1-dominated by α=0.01·S_L perturbation), we expect a
   skewed spectrum (a few outliers, then a wall). Quantified via
   participation ratio and 90%-mass elbow.

Outputs:
- ``artifacts/denoiser_spectra/spectra_<embedder>.csv`` — full spectrum per arm
- ``artifacts/denoiser_spectra/summary_<embedder>.csv`` — per-arm stats
- ``artifacts/denoiser_spectra/spectrum_<embedder>.png`` — log-spectrum plot

Usage:
    python -m artifacts_scripts.denoiser_spectrum --embedder gist
    python -m artifacts_scripts.denoiser_spectrum --embedder gemma
    python -m artifacts_scripts.denoiser_spectrum --embedder both
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from bookgap.cloud import (
    Cloud, _embeddings_store_path, _read_corpus_dicts,
    corpus_dataset, ensure_cloud_data, read_embeddings_store,
)
from bookgap.cloud_denoiser import CloudDenoiserContext, ProjectiveDenoiser, WorkDenoisers
from bookgap.configs.embedder.embedder_config import SentenceTransformerEmbedderConfig
from bookgap.embedder import Embedder
from bookgap.index import FlatL2
from bookgap.util import LOGGER, Filter, FilterPair, SubField, WorkType


ARMS = (
    "LPP", "ANTI_LDA", "ANTI_CSP", "AUG_LPP", "CITATION_ERROR_LDA",
    "SYM_NORM_LPP", "CCP", "CCLIP", "CCLIP_01", "CCLIP_1", "LIP",
    "JAD_EQUAL", "PCA",
)


def compute_spectrum(denoiser_name: str, context: CloudDenoiserContext) -> Optional[np.ndarray]:
    """Get the full eigenvalue spectrum from a denoiser's GEP."""
    d = context.X().shape[1]
    impl = WorkDenoisers[denoiser_name].value
    if not isinstance(impl, ProjectiveDenoiser):
        return None
    try:
        _P, eigvals = impl.denoiser_impl.projector(context, n_components=d)
    except Exception:
        LOGGER.error(f"{denoiser_name}: projector failed at n_components={d}\n{traceback.format_exc()}")
        return None
    if eigvals is None:
        return None
    return np.asarray(eigvals)


def summarize(name: str, eigvals: np.ndarray) -> dict:
    """Compute the rank-1 / low-rank diagnostic stats for one spectrum."""
    abs_ev = np.abs(eigvals)
    sorted_desc = np.sort(abs_ev)[::-1]
    total = float(sorted_desc.sum())
    sumsq = float((sorted_desc ** 2).sum())

    participation_ratio = (total ** 2 / sumsq) if sumsq > 0 else 0.0
    cumsum = np.cumsum(sorted_desc)
    elbow_90 = int(np.searchsorted(cumsum, 0.9 * total) + 1) if total > 0 else 0
    eff_rank_eps = int((abs_ev > 1e-6 * abs_ev.max()).sum()) if abs_ev.max() > 0 else 0

    return {
        "denoiser": name,
        "n_eigvals": len(eigvals),
        "max_abs": float(abs_ev.max()),
        "min_abs": float(abs_ev.min()),
        "dynamic_range": float(abs_ev.max() / abs_ev.min()) if abs_ev.min() > 0 else float("inf"),
        "participation_ratio": round(participation_ratio, 2),
        "elbow_90pct_mass": elbow_90,
        "eff_rank_eps_1e-6": eff_rank_eps,
        # First 5 absolute values, descending — for visual inspection
        "top5_abs": ";".join(f"{v:.4e}" for v in sorted_desc[:5]),
    }


def run_for_embedder(emb_name: str, model_id: str, out_dir: Path,
                     max_works: int) -> None:
    LOGGER.info(f"=== Embedder: {emb_name} ({model_id}) ===")

    ai_filter = Filter(
        subfield=SubField.AI.value,
        work_types=(WorkType.BOOK.value, WorkType.ARTICLE.value),
    )
    embedder_cfg = SentenceTransformerEmbedderConfig(model_name=model_id)

    # Build a minimal Cloud just to access the cached corpus + embeddings.
    # Mirrors the pattern in artifacts_scripts/embedder_evaluation.py.
    cloud = Cloud(
        filter_pair=FilterPair(pre=ai_filter, post=Filter()),
        index_info=FlatL2(),
        max_works=None if max_works <= 0 else max_works,
        deduplicator=None,
        normalize_embeddings=False,
        denoiser="NONE",
        drop_k=0,
        embedder=embedder_cfg,
    )
    LOGGER.info("ensure_cloud_data (may take a moment on first run)...")
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(ai_filter, cloud.snapshot_date)
    positions = sorted_idxs[retained_positions]

    emb_path = _embeddings_store_path(corpus_fp, embedder_cfg)
    all_emb = read_embeddings_store(emb_path)
    embeddings = all_emb[retained_positions]
    embeddings = normalize(embeddings, norm="l2", axis=1)
    works = _read_corpus_dicts(corpus, positions)
    LOGGER.info(f"Loaded {len(works)} works, embeddings shape {embeddings.shape}")

    book_f = Filter(work_types=(WorkType.BOOK.value,))
    art_f = Filter(work_types=(WorkType.ARTICLE.value,))
    context = CloudDenoiserContext(works, embeddings, book_f, art_f)
    # Eagerly build the citation graph so we don't pay repeatedly per projector.
    _ = context.A_sym()

    summary_rows = []
    spectra: dict[str, np.ndarray] = {}
    for arm in ARMS:
        LOGGER.info(f"Computing spectrum for {arm}...")
        eigvals = compute_spectrum(arm, context)
        if eigvals is None:
            LOGGER.warning(f"  -> SKIPPED ({arm}: no spectrum)")
            continue
        spectra[arm] = eigvals
        summary_rows.append(summarize(arm, eigvals))
        LOGGER.info(f"  -> n={len(eigvals)}, max={eigvals.max():.4e}, min={eigvals.min():.4e}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-arm full spectrum dump
    spectra_df = pd.DataFrame({arm: pd.Series(np.sort(np.abs(v))[::-1]) for arm, v in spectra.items()})
    spectra_df.to_csv(out_dir / f"spectra_{emb_name}.csv", index=False)
    LOGGER.info(f"Wrote {out_dir / f'spectra_{emb_name}.csv'}")

    # Summary stats
    pd.DataFrame(summary_rows).to_csv(out_dir / f"summary_{emb_name}.csv", index=False)
    LOGGER.info(f"Wrote {out_dir / f'summary_{emb_name}.csv'}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for arm, eigvals in spectra.items():
        sorted_desc = np.sort(np.abs(eigvals))[::-1]
        ax.semilogy(np.arange(1, len(sorted_desc) + 1), sorted_desc + 1e-15,
                    label=arm, alpha=0.8)
    ax.set_xlabel("Eigenvalue rank (sorted by |λ| descending)")
    ax.set_ylabel("|λ| (log scale)")
    ax.set_title(f"Denoiser GEP eigenvalue spectra ({emb_name}, AI subfield)")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / f"spectrum_{emb_name}.png", dpi=120)
    plt.close(fig)
    LOGGER.info(f"Wrote {out_dir / f'spectrum_{emb_name}.png'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", choices=("gist", "gemma", "both"), default="both")
    parser.add_argument("--max-works", type=int, default=10000,
                        help="Cap on works loaded (None = no cap; default 10k for speed)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("artifacts/denoiser_spectra"))
    args = parser.parse_args()

    embedders = []
    if args.embedder in ("gist", "both"):
        embedders.append(("gist", Embedder.ModelName.GIST_SMALL.value))
    if args.embedder in ("gemma", "both"):
        embedders.append(("gemma", Embedder.ModelName.GEMMA_300M.value))

    for emb_name, model_id in embedders:
        run_for_embedder(emb_name, model_id, args.out_dir, args.max_works)

    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
