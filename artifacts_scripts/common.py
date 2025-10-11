from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)

PROPOSAL_BOOK_RANK_FRACTION_COLUMN = "proposal_book_rank_fraction"
GAP_ZSCORE_COLUMN = "gap_zscore"


@dataclass
class RunMetadata:
    denoiser: Optional[str]
    drop_k: Optional[int]
    embedder_model: Optional[str]
    num_articles: Optional[int]
    num_books: Optional[int]


def find_run_directories(outputs_dir: Path) -> Iterable[Path]:
    """Yield candidate run directories containing a Hydra config."""
    if not outputs_dir.exists():
        LOGGER.warning("Outputs directory %s does not exist", outputs_dir)
        return []

    config_paths = sorted(outputs_dir.rglob("config.yaml"))
    for config_path in config_paths:
        if config_path.parent.name != ".hydra":
            continue
        run_dir = config_path.parent.parent
        if run_dir.is_dir():
            yield run_dir


def load_run_metadata(run_dir: Path) -> Optional[RunMetadata]:
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        LOGGER.warning("Missing config.yaml in %s", run_dir)
        return None

    config = OmegaConf.load(config_path)

    denoiser = OmegaConf.select(config, "cloud.denoiser", default=None)
    drop_k = OmegaConf.select(config, "cloud.drop_k", default=None)
    embedder_model = OmegaConf.select(config, "cloud.embedder_model", default=None)
    num_articles = OmegaConf.select(config, "llm.num_articles", default=None)
    num_books = OmegaConf.select(config, "llm.num_books", default=None)

    return RunMetadata(
        denoiser=str(denoiser) if denoiser is not None else None,
        drop_k=int(drop_k) if drop_k is not None else None,
        embedder_model=str(embedder_model) if embedder_model is not None else None,
        num_articles=int(num_articles) if num_articles is not None else None,
        num_books=int(num_books) if num_books is not None else None,
    )


def load_results_dataframe(run_dir: Path) -> Optional[pd.DataFrame]:
    csv_path = run_dir / "gaps_analysis_results.csv"
    if not csv_path.exists():
        LOGGER.warning("Missing gaps_analysis_results.csv in %s", run_dir)
        return None

    try:
        return pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - unexpected IO errors
        LOGGER.warning("Failed to read %s: %s", csv_path, exc)
        return None


def valid_rank_fraction_mask(df: pd.DataFrame) -> pd.Series:
    if PROPOSAL_BOOK_RANK_FRACTION_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{PROPOSAL_BOOK_RANK_FRACTION_COLUMN}' missing from dataframe"
        )
    column = df[PROPOSAL_BOOK_RANK_FRACTION_COLUMN]
    return column.notna() & (column != -1)


def relative_run_path(run_dir: Path, outputs_dir: Path) -> Path:
    try:
        return run_dir.relative_to(outputs_dir)
    except ValueError:
        return run_dir
