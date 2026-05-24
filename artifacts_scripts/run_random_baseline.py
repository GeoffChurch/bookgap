"""Random-context baseline for the gap-targeted proposal pipeline.

Tests whether the gap-detection pipeline is doing real work by running
the same Ollama LLM under the same metrics, but with randomly-sampled
articles and books from each topic instead of the gap-targeted set
chosen by ``build_contexts_for_gaps``.

Output:
- ``outputs/random_baseline/ollama_results.jsonl`` (one line per
  successful proposal, custom_id = ``gap_0_topic_<id>``) — written
  incrementally for crash-resume.
- ``outputs/random_baseline/results.csv`` (one row per topic with the
  two headline metrics, column names matching production's
  ``gaps_analysis_results.csv``).

Usage::

    conda run -p .conda python -m artifacts_scripts.run_random_baseline
    # optional overrides:
    conda run -p .conda python -m artifacts_scripts.run_random_baseline \\
        --run-dir outputs/gemma4_sweep/2026-05-22_11-11-43_gist_a10_b5_dropk10 \\
        --output-dir outputs/random_baseline
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import enum
import hashlib
import json
import random
import typing
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

# Reuse production helpers; do not duplicate logic.
from bookgap.configs.config import ExperimentConfig
from bookgap.configs.data.data_config import (
    SimHashDeduplicatorConfig,
    SqDistDeduplicatorConfig,
)
from bookgap.configs.embedder.embedder_config import (
    SentenceTransformerEmbedderConfig,
    TfGepEmbedderConfig,
)
from bookgap.gap_metrics import compute_gap_metrics_for_embeddings
from bookgap.gaps import compute_gaps_for_config
from bookgap.index import range_search_radii_with
from bookgap.llm.data_structures import (
    BatchResults, GapWithContext, ProposalOutrankCounts,
)
from bookgap.llm.generators.ollama_local import OllamaProposalGenerator
from bookgap.llm.generators.workflow_functions import (
    _compute_proposal_embeddings,
    PendingProposal,
)
from bookgap.llm.protocols import (
    Article, Book, LLMBackend, ParsedProposal, Proposal, PromptPair,
    ProposalContext, ProposalCounts,
)
from bookgap.util import LOGGER


# ---------------------------------------------------------------------------
# Hard-coded constants (per spec)
# ---------------------------------------------------------------------------

RUN_DIR = Path("outputs/gemma4_sweep/2026-05-22_11-11-43_gist_a10_b5_dropk10")
OUTPUT_DIR = Path("outputs/random_baseline")
SAMPLING_SEED = 42  # master seed; per-topic seeds derived from (SEED, topic_id)
N_ARTICLES = 10
N_BOOKS = 5
SKIP_TOPIC_IDS = {"T13567"}  # no books in this topic


# Ollama JSON-only instruction; identical to production. Imported below at
# runtime to keep this module's import surface obvious.
from bookgap.llm.prompt_render import _OLLAMA_JSON_INSTRUCTION


# Prompt template — locked verbatim per spec. Articles/books blocks
# rendered separately so we keep formatting (tab before *Abstract*) exact.
_PROMPT_HEADER_TEMPLATE = (
    "You are helping an academic publishing company brainstorm new "
    "textbook ideas.\n\n"
    "Propose a title and abstract for a textbook in the field of "
    "{topic_name}. The proposed textbook should fill a gap: a topic that "
    "has many research articles but no dedicated textbook yet.\n\n"
    "Below is a random sample of articles and books from the field, to "
    "give a sense of its topical and stylistic norms.\n\n"
)


# ---------------------------------------------------------------------------
# Config loading (mirrors the deleted artifacts_scripts/_oneoff_backfill_pca
# pattern; the CloudConfig.embedder / .deduplicator fields are typed `Any`
# so we need to resolve the concrete config class via _target_).
# ---------------------------------------------------------------------------

_TARGET_TO_CLS = {
    "bookgap.embedder.SentenceTransformerEmbedder": SentenceTransformerEmbedderConfig,
    "bookgap.tf_gep_embedder.TfGepEmbedder": TfGepEmbedderConfig,
    "bookgap.deduplicators.SimHashDeduplicator": SimHashDeduplicatorConfig,
    "bookgap.deduplicators.SqDistDeduplicator": SqDistDeduplicatorConfig,
}


def _build_dataclass(cls, value):
    """Construct a frozen dataclass from a plain dict.

    OmegaConf merge refuses frozen dataclasses with init=False fields,
    so we walk the type tree manually.
    """
    if value is None:
        return value
    if isinstance(cls, type) and issubclass(cls, enum.Enum) and isinstance(value, str):
        return cls[value] if value in cls.__members__ else cls(value)
    if cls is typing.Any and isinstance(value, dict) and "_target_" in value:
        resolved = _TARGET_TO_CLS.get(value["_target_"])
        if resolved is not None:
            cls = resolved
    if not dataclasses.is_dataclass(cls):
        return value
    if not isinstance(value, dict):
        return value
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if not f.init or f.name not in value:
            continue
        ftype = hints.get(f.name, f.type)
        origin = typing.get_origin(ftype)
        args = typing.get_args(ftype)
        if origin is typing.Union:
            non_none = [a for a in args if a is not type(None)]
            ftype = non_none[0] if non_none else ftype
        kwargs[f.name] = _build_dataclass(ftype, value[f.name])
    return cls(**kwargs)


def load_experiment_config(run_dir: Path) -> ExperimentConfig:
    loaded = OmegaConf.to_container(
        OmegaConf.load(run_dir / "experiment_config.yaml"),
        resolve=True,
    )
    assert isinstance(loaded, dict)
    return _build_dataclass(ExperimentConfig, loaded)


# ---------------------------------------------------------------------------
# Sampling + prompt rendering
# ---------------------------------------------------------------------------


def _topic_rng(topic_id: str) -> random.Random:
    h = hashlib.md5(f"{SAMPLING_SEED}_{topic_id}".encode()).hexdigest()
    return random.Random(int(h[:8], 16))


def _work_field(work: Any, name: str) -> Any:
    if isinstance(work, dict):
        return work.get(name)
    return getattr(work, name, None)


def _work_to_article(work: Any) -> Article | None:
    display_name = _work_field(work, "display_name")
    abstract = _work_field(work, "abstract")
    openalex_id = _work_field(work, "id")
    if not display_name or not abstract:
        return None
    return Article(
        display_name=display_name,
        abstract=abstract,
        openalex_id=str(openalex_id or ""),
    )


def _work_to_book(work: Any) -> Book | None:
    display_name = _work_field(work, "display_name")
    abstract = _work_field(work, "abstract")
    openalex_id = _work_field(work, "id")
    if not display_name or not abstract:
        return None
    return Book(
        display_name=display_name,
        abstract=abstract,
        openalex_id=str(openalex_id or ""),
    )


def _sample_from_works(
    works: list[Any],
    n: int,
    rng: random.Random,
    converter,
) -> list:
    """Sample up to ``n`` works *without replacement*, filtering out any
    that lack display_name or abstract. Resamples once if filtering drops
    below ``n``: scan the full pool sequentially in shuffled order until
    we have ``n`` valid entries or run out.
    """
    if not works:
        return []
    indices = list(range(len(works)))
    rng.shuffle(indices)
    out = []
    for i in indices:
        item = converter(works[i])
        if item is None:
            continue
        out.append(item)
        if len(out) >= n:
            break
    return out


def render_random_baseline_prompt(
    *,
    topic_name: str,
    articles: list[Article],
    books: list[Book],
) -> str:
    parts: list[str] = [_PROMPT_HEADER_TEMPLATE.format(topic_name=topic_name)]
    parts.append("## Articles\n\n")
    for i, a in enumerate(articles, 1):
        parts.append(f"**Article {i}**: {a.display_name}\n")
        parts.append(f"\t*Abstract*: {a.abstract}\n\n")
    if books:
        parts.append("## Books\n\n")
        for i, b in enumerate(books, 1):
            parts.append(f"**Book {i}**: {b.display_name}\n")
            parts.append(f"\t*Abstract*: {b.abstract}\n\n")
    user_prompt = "".join(parts) + _OLLAMA_JSON_INSTRUCTION
    return user_prompt


# ---------------------------------------------------------------------------
# Per-topic metric computation (lifted from workflow_functions._compute_proposal_outrank_counts
# and _compute_proposal_gap_stats, restricted to one proposal per gap).
# ---------------------------------------------------------------------------


def _compute_topic_metrics(
    *,
    gap: GapWithContext,
    proposal_emb: np.ndarray,
    config: ExperimentConfig,
    epsilon: float = float(np.finfo(np.float32).eps * 8),
) -> tuple[float | None, float | None]:
    """Return (gap_coverage_fraction, proposal_gap_size_percentile) for
    one (gap, proposal) pair. Either may be None on degenerate inputs.

    Coverage: lifts the per-proposal subset of
    :func:`_compute_proposal_outrank_counts` — an article in the
    pipeline-detected gap is "covered" by the proposal iff no existing
    topic book sits strictly closer to it than the proposal does.

    Percentile: lifts :func:`_compute_proposal_gap_stats` via
    :func:`compute_gap_metrics_for_embeddings`.
    """
    proposal_vec = np.asarray(proposal_emb, dtype=np.float32)

    # --- Coverage ---
    gap_article_indices = gap.gap.get_dists_idxs().idxs
    coverage: float | None
    if len(gap_article_indices) == 0:
        coverage = None
    else:
        article_embeddings_all = gap.articles.embeddings()
        try:
            gap_article_vecs = np.ascontiguousarray(
                article_embeddings_all[np.asarray(gap_article_indices, dtype=int)],
                dtype=np.float32,
            )
        except IndexError:
            LOGGER.warning(
                "Gap %s/%d has invalid article indices; "
                "coverage = None", gap.topic_id, gap.gap_idx,
            )
            gap_article_vecs = np.empty((0, 0), dtype=np.float32)

        if gap_article_vecs.size == 0:
            coverage = None
        else:
            try:
                book_index = gap.books.index()
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning(
                    "Failed to build book index for gap %s/%d: %s",
                    gap.topic_id, gap.gap_idx, exc,
                )
                coverage = None
            else:
                diffs = gap_article_vecs - proposal_vec
                sq_dists = np.einsum("ij,ij->i", diffs, diffs, dtype=np.float32)
                counts = np.zeros(len(gap_article_vecs), dtype=np.int32)

                def _kernel(article_idx: int, neighbor_dists, _neighbor_ids):
                    threshold = float(sq_dists[article_idx]) - epsilon
                    if threshold <= 0.0:
                        counts[article_idx] = 0
                    else:
                        counts[article_idx] = int(
                            np.count_nonzero(neighbor_dists < threshold)
                        )

                range_search_radii_with(
                    book_index,
                    query=gap_article_vecs,
                    radii=sq_dists,
                    kernel_fn=_kernel,
                    show_progress=False,
                )
                coverage = float(
                    ProposalOutrankCounts(counts=counts).coverage_fraction()
                )

    # --- Gap-size percentile ---
    try:
        gap_stats_list = compute_gap_metrics_for_embeddings(
            gap=gap,
            embeddings=proposal_vec[np.newaxis, :].astype(np.float32),
            config=config,
        )
        percentile = float(gap_stats_list[0].gap_size_percentile)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning(
            "Failed to compute gap percentile for %s/%d: %s",
            gap.topic_id, gap.gap_idx, exc,
        )
        percentile = None

    return coverage, percentile


# ---------------------------------------------------------------------------
# Streaming JSONL persistence (same shape as ollama_results.jsonl in
# production runs, so the parser pipeline is unchanged).
# ---------------------------------------------------------------------------


def _read_completed_custom_ids(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    done: set[str] = set()
    with results_path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = entry.get("custom_id")
            if isinstance(cid, str) and cid:
                done.add(cid)
    return done


def _read_completed_records(
    results_path: Path,
) -> dict[str, dict[str, Any]]:
    """Return {custom_id -> raw JSONL entry} for resume. Skips entries
    whose proposal payload is None (parse failures from a prior run)."""
    out: dict[str, dict[str, Any]] = {}
    if not results_path.exists():
        return out
    with results_path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = entry.get("custom_id")
            if not isinstance(cid, str) or not cid:
                continue
            out[cid] = entry
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading experiment config from %s", run_dir)
    config = load_experiment_config(run_dir)

    LOGGER.info("Replaying gap detection (joblib cache expected to hit)…")
    gaps = compute_gaps_for_config(config)
    LOGGER.info("Recovered %d gaps", len(gaps))

    # Build (topic_id -> GapWithContext) keeping just one gap per topic.
    # The production headline run is num_gaps=1, so this is a 1:1 map.
    gap_by_topic: dict[str, GapWithContext] = {}
    for g in gaps:
        if g.topic_id in SKIP_TOPIC_IDS:
            continue
        gap_by_topic.setdefault(g.topic_id, g)
    LOGGER.info(
        "Will run baseline on %d topics (skipping %s)",
        len(gap_by_topic), sorted(SKIP_TOPIC_IDS),
    )

    results_path = output_dir / "ollama_results.jsonl"
    already_done = _read_completed_custom_ids(results_path)
    if already_done:
        LOGGER.info(
            "Resuming: %d topics already generated; will skip those",
            len(already_done),
        )

    generator = OllamaProposalGenerator(
        model=str(config.llm.ollama_model),
        temperature=float(config.llm.temperature),
        num_ctx=int(config.llm.ollama_num_ctx),
        num_predict=int(config.llm.ollama_num_predict),
    )

    # Generation phase: stream into JSONL, one line per topic.
    sorted_topic_ids = sorted(gap_by_topic.keys())
    for i, topic_id in enumerate(sorted_topic_ids, 1):
        custom_id = f"gap_0_topic_{topic_id}"
        if custom_id in already_done:
            continue

        gap = gap_by_topic[topic_id]
        rng = _topic_rng(topic_id)

        article_works = gap.articles.works()
        book_works = gap.books.works()

        articles = _sample_from_works(article_works, N_ARTICLES, rng, _work_to_article)
        books = _sample_from_works(book_works, N_BOOKS, rng, _work_to_book)

        if not articles:
            LOGGER.warning(
                "[%d/%d] %s: no usable articles; skipping",
                i, len(sorted_topic_ids), topic_id,
            )
            continue
        if not books:
            LOGGER.warning(
                "[%d/%d] %s: no usable books; skipping (this topic "
                "should be in SKIP_TOPIC_IDS)",
                i, len(sorted_topic_ids), topic_id,
            )
            continue

        user_prompt = render_random_baseline_prompt(
            topic_name=gap.topic_name, articles=articles, books=books,
        )
        prompts = PromptPair(system_prompt="", user_prompt=user_prompt)

        # ProposalContext is only echoed back on the returned record; the
        # OllamaProposalGenerator only consults `prompts` for the chat call.
        context = ProposalContext(
            topic_name=gap.topic_name,
            topic_id=gap.topic_id,
            gap_idx=gap.gap_idx,
            requested_num_prompt_articles=N_ARTICLES,
            requested_num_prompt_books=N_BOOKS,
            sampled_articles=articles,
            covering_books=books,
            counts=ProposalCounts(
                num_articles_for_llm=len(articles),
                num_books_for_llm=len(books),
            ),
        )

        LOGGER.info(
            "[%d/%d] %s (%s): calling ollama (%d articles, %d books)",
            i, len(sorted_topic_ids), topic_id, gap.topic_name,
            len(articles), len(books),
        )
        record = generator.generate(context, prompts)

        proposal_dict = (
            record.proposal.model_dump() if record.proposal is not None else None
        )
        line = {
            "custom_id": custom_id,
            "proposal": proposal_dict,
            "raw_content": record.raw_content,
            "thinking": record.thinking,
            "metadata": record.metadata,
        }
        with results_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(line) + "\n")
        already_done.add(custom_id)

        if record.proposal is None:
            LOGGER.warning(
                "[%d/%d] %s: proposal parse failed; no row will be "
                "written for this topic in results.csv",
                i, len(sorted_topic_ids), topic_id,
            )

    # ----- Embedding + metric pass -----
    all_records = _read_completed_records(results_path)
    LOGGER.info("Loaded %d on-disk records", len(all_records))

    pending: list[PendingProposal] = []
    cid_to_record: dict[str, dict[str, Any]] = {}
    for cid, entry in all_records.items():
        proposal_dict = entry.get("proposal")
        if proposal_dict is None:
            continue
        try:
            proposal = Proposal.model_validate(proposal_dict)
        except Exception as exc:
            LOGGER.warning("Skipping %s: invalid proposal payload (%s)", cid, exc)
            continue
        pending.append(PendingProposal(proposal=proposal, custom_id=cid))
        cid_to_record[cid] = entry

    LOGGER.info("Embedding %d proposals via _compute_proposal_embeddings", len(pending))
    embeddings = _compute_proposal_embeddings(pending, config=config)
    parsed_proposals = [
        ParsedProposal(
            proposal=p.proposal,
            custom_id=p.custom_id,
            embedding=emb,
        )
        for p, emb in zip(pending, embeddings, strict=True)
    ]
    proposal_by_topic: dict[str, ParsedProposal] = {}
    for pp in parsed_proposals:
        # custom_id format: gap_0_topic_T<id>
        prefix = "gap_0_topic_"
        if pp.custom_id.startswith(prefix):
            topic_id = pp.custom_id[len(prefix):]
            proposal_by_topic[topic_id] = pp

    # ----- Per-topic metrics + CSV write -----
    csv_path = output_dir / "results.csv"
    fieldnames = [
        "topic_id", "topic_name",
        "gap_coverage_fraction", "proposal_gap_size_percentile",
        "proposal_title", "proposal_abstract",
    ]
    rows_written = 0
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for topic_id in sorted_topic_ids:
            pp = proposal_by_topic.get(topic_id)
            if pp is None:
                LOGGER.warning(
                    "Skipping %s in results.csv: no parsed proposal", topic_id,
                )
                continue
            gap = gap_by_topic[topic_id]
            coverage, percentile = _compute_topic_metrics(
                gap=gap, proposal_emb=pp.embedding, config=config,
            )
            writer.writerow({
                "topic_id": topic_id,
                "topic_name": gap.topic_name,
                "gap_coverage_fraction": (
                    "" if coverage is None else f"{coverage:.15g}"
                ),
                "proposal_gap_size_percentile": (
                    "" if percentile is None else f"{percentile:.15g}"
                ),
                "proposal_title": pp.proposal.title,
                "proposal_abstract": pp.proposal.abstract,
            })
            rows_written += 1
    LOGGER.info("Wrote %d rows to %s", rows_written, csv_path)


if __name__ == "__main__":
    main()
