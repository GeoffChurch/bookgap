"""bg-norm LexRank vs plain LexRank: extractive summarization eval on Wikipedia.

Hypothesis (falsifiable): Fightin'-Words-corrected sentence centrality
(``p^0.5 * log((p+eps)/(g+eps))`` on the two-step sentence-word Gramian walk)
beats plain LexRank at extractive summarization.

Data: Wikipedia articles where the LEAD section (text before the first heading)
is the human reference summary and the BODY is the input document. Source
corpus: bookgen's scraped corpus (``~/src/bookgen/data/corpus/*.txt``,
partially-cleaned wikitext with surviving ``==`` headings), finished-cleaned by
``textproc``.

Methods (all budget-matched to the reference lead's token length, except the
classic lead-3):
- ``lead3``      first 3 body sentences
- ``random``     seeded random sentences to budget
- ``lexrank``    top sentences by uniform-teleport PageRank ``p`` on the Gramian
- ``bgnorm_g1``  selection by ``p^0.5 * log(p/g1)``, g1 = normalized mean global
                 word frequency per sentence (corpus unigram model, all bodies)
- ``bgnorm_g2``  same with g2 = PageRank on the global-frequency-weighted Gramian
                 (expected centrality under generic language)
- ``degree``     tf-idf cosine degree baseline
- ``posthoc_bgnorm_on_degree``  POST-HOC (added after the designed variants both
                 lost, clearly labeled): bg-norm re-rank with p = normalized
                 degree centrality, g = g1. Isolates the Fightin'-Words
                 correction from the raw walk's length artifact (the designed
                 foreground p is ~a sentence-length ranking, Spearman ~0.93);
                 tests whether bg-norm adds value on a non-broken base.

Scoring: vendored ROUGE-1/2/L F1 vs the lead (no stemmer; see ``rouge.py``).
Stats: per-article paired deltas, mean +/- bootstrap 95% CI (seeded), win/tie/loss.

Run (CPU-only, ~2 min):
    PYTHONPATH=. uv run python -m artifacts_scripts.lexrank_bgnorm.run_eval
    # data-funnel probe only:
    PYTHONPATH=. uv run python -m artifacts_scripts.lexrank_bgnorm.run_eval --probe
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from artifacts_scripts.lexrank_bgnorm import engine, rouge, textproc

_MODULE = "artifacts_scripts.lexrank_bgnorm.run_eval"

# bookgen is a separate sibling checkout; override with --corpus-dir.
DEFAULT_CORPUS = str(Path.home() / "src/bookgen/data/corpus")

METHODS = ("lead3", "random", "lexrank", "bgnorm_g1", "bgnorm_g2", "degree",
           "posthoc_bgnorm_on_degree")
PAIRS = (
    ("bgnorm_g1", "lexrank"),
    ("bgnorm_g2", "lexrank"),
    ("degree", "lexrank"),
    ("lexrank", "lead3"),
    ("bgnorm_g1", "lead3"),
    ("bgnorm_g2", "lead3"),
    ("posthoc_bgnorm_on_degree", "degree"),
)
METRICS = ("rouge1", "rouge2", "rougeL")
TIE_EPS = 1e-12


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #

@dataclass
class Article:
    page_id: int
    title: str
    lead_sentences: list[str]
    body_sentences: list[str]


def prepare_dataset(
    corpus_dir: Path,
    lead_min: int,
    lead_max: int,
    min_body_sents: int,
    cache_path: Path,
    refresh: bool,
) -> tuple[list[Article], dict[str, int]]:
    """Clean + split + filter the corpus; cache the prepared dataset as JSON."""
    prep_cfg = {"lead_min": lead_min, "lead_max": lead_max,
                "min_body_sents": min_body_sents, "min_sent_tokens": textproc.MIN_SENT_TOKENS}
    if cache_path.exists() and not refresh:
        blob = json.loads(cache_path.read_text(encoding="utf-8"))
        if blob.get("prep_cfg") == prep_cfg and blob.get("corpus_dir") == str(corpus_dir):
            arts = [Article(**a) for a in blob["articles"]]
            return arts, blob["funnel"]

    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    funnel = Counter()
    articles: list[Article] = []
    for entry in manifest:
        funnel["manifest_entries"] += 1
        if entry.get("status") != "ok":
            funnel["scrape_failed"] += 1
            continue
        path = corpus_dir / entry["filename"]
        if not path.exists():
            funnel["file_missing"] += 1
            continue
        raw = path.read_text(encoding="utf-8")
        cleaned = textproc.finish_clean_wikitext(raw)
        lead_text, body_text = textproc.split_lead_body(cleaned)
        if not body_text.strip():
            funnel["no_body_sections"] += 1
            continue
        lead = textproc.split_sentences(lead_text)
        body = textproc.split_sentences(body_text)
        if not (lead_min <= len(lead) <= lead_max):
            funnel["lead_out_of_range"] += 1
            continue
        if len(body) < min_body_sents:
            funnel["body_too_short"] += 1
            continue
        articles.append(Article(int(entry["page_id"]), entry["title"], lead, body))
        funnel["passed"] += 1

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "corpus_dir": str(corpus_dir),
        "prep_cfg": prep_cfg,
        "funnel": dict(funnel),
        "generated_by": _provenance_cmd(),
        "articles": [asdict(a) for a in articles],
    }), encoding="utf-8")
    return articles, dict(funnel)


def build_global_freq(articles: list[Article]) -> dict[str, float]:
    """Corpus-level unigram model over ALL articles' bodies (content tokens).

    Leads are deliberately excluded (they are the references). The scored
    article's own body is included — with n in the hundreds the self-term is
    negligible and this matches the 'vs corpus background' design.
    """
    counts: Counter[str] = Counter()
    for art in articles:
        for s in art.body_sentences:
            counts.update(textproc.content_tokens(s))
    total = sum(counts.values())
    return {t: c / total for t, c in counts.items()}


# --------------------------------------------------------------------------- #
# per-article methods + scoring
# --------------------------------------------------------------------------- #

def run_article(
    art: Article,
    global_freq: dict[str, float],
    seed: int,
) -> dict:
    body = art.body_sentences
    sent_ctoks = [textproc.content_tokens(s) for s in body]
    sent_rtoks = [textproc.word_tokens(s) for s in body]
    lens = [len(t) for t in sent_rtoks]
    ref_tokens = textproc.word_tokens(" ".join(art.lead_sentences))
    budget = len(ref_tokens)

    B, vocab = engine.build_sentence_word_matrix(sent_ctoks)
    p = engine.centrality(engine.gramian_walk(B))
    mask = engine.rankable_mask(sent_ctoks)
    g1 = engine.g1_global_frequency(sent_ctoks, global_freq)
    g2 = engine.g2_generic_walk(B, vocab, global_freq)

    deg = engine.tfidf_degree(B)
    deg_total = deg.sum()
    p_deg = deg / deg_total if deg_total > 0 else np.full(len(deg), 1.0 / max(len(deg), 1))
    rank_scores = {
        "lexrank": p,
        "bgnorm_g1": engine.bgnorm_score(p, g1),
        "bgnorm_g2": engine.bgnorm_score(p, g2),
        "degree": deg,
        # POST-HOC variant (see module docstring): bg-norm on the de-length-biased base.
        "posthoc_bgnorm_on_degree": engine.bgnorm_score(p_deg, g1),
    }

    selections: dict[str, list[int]] = {"lead3": list(range(min(3, len(body))))}
    rng = np.random.default_rng([seed, art.page_id])
    perm = [int(i) for i in rng.permutation(len(body))]
    selections["random"] = engine.select_by_budget(perm, lens, budget)
    for name, sc in rank_scores.items():
        ranking = engine.ranking_from_scores(sc, mask)
        selections[name] = engine.select_by_budget(ranking, lens, budget)

    scores: dict[str, dict[str, float]] = {}
    for name, sel in selections.items():
        hyp = [tok for i in sel for tok in sent_rtoks[i]]
        scores[name] = rouge.score_all(ref_tokens, hyp)

    return {
        "page_id": art.page_id,
        "title": art.title,
        "n_lead_sents": len(art.lead_sentences),
        "n_body_sents": len(body),
        "budget_tokens": budget,
        "selections": selections,
        "scores": scores,
    }


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

def bootstrap_ci(
    deltas: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def aggregate(results: list[dict], n_boot: int, seed: int) -> dict:
    agg: dict = {"n_articles": len(results), "mean_f1": {}, "paired": {}}
    for m in METHODS:
        agg["mean_f1"][m] = {
            met: float(np.mean([r["scores"][m][met] for r in results])) for met in METRICS
        }
    rng = np.random.default_rng(seed + 1)
    for a, b in PAIRS:
        key = f"{a}_vs_{b}"
        agg["paired"][key] = {}
        for met in METRICS:
            deltas = np.array([r["scores"][a][met] - r["scores"][b][met] for r in results])
            lo, hi = bootstrap_ci(deltas, n_boot, rng)
            agg["paired"][key][met] = {
                "mean_delta": float(deltas.mean()),
                "ci95": [lo, hi],
                "win": int((deltas > TIE_EPS).sum()),
                "tie": int((np.abs(deltas) <= TIE_EPS).sum()),
                "loss": int((deltas < -TIE_EPS).sum()),
            }
    return agg


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def _provenance_cmd() -> str:
    return f"python -m {_MODULE} " + " ".join(shlex.quote(a) for a in sys.argv[1:])


def _provenance() -> str:
    return ("\n\n---\n"
            f"*Generated by `{_provenance_cmd()}`*  \n"
            "*Regenerate with the command above; do not edit by hand.*\n")


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.2f}"


def _qualitative_examples(
    results: list[dict], articles: dict[int, Article], variant: str, k: int = 2
) -> str:
    """Best and worst R1-delta articles for ``variant`` vs lexrank, side by side."""
    cands = [r for r in results if r["selections"][variant] != r["selections"]["lexrank"]]
    if not cands:
        return "_No articles where the selections differ._\n"
    cands.sort(key=lambda r: r["scores"][variant]["rouge1"] - r["scores"]["lexrank"]["rouge1"])
    picks = []
    if k >= 1:
        picks.append(("largest WIN", cands[-1]))
    if k >= 2 and len(cands) > 1:
        picks.append(("largest LOSS", cands[0]))

    out: list[str] = []
    for label, r in picks:
        art = articles[r["page_id"]]
        d = r["scores"][variant]["rouge1"] - r["scores"]["lexrank"]["rouge1"]
        out.append(f"### {label} for `{variant}`: {art.title} "
                   f"(ΔR1 = {100 * d:+.2f}, budget {r['budget_tokens']} tokens)\n")
        sel_v = set(r["selections"][variant])
        sel_l = set(r["selections"]["lexrank"])

        def _render(idx_set: set[int], tag: str) -> None:
            out.append(f"**{tag}**\n")
            for i in sorted(idx_set):
                s = art.body_sentences[i]
                s = s if len(s) <= 220 else s[:217] + "..."
                out.append(f"- [{i}] {s}")
            out.append("")

        _render(sel_l, "plain LexRank picked")
        _render(sel_v, f"{variant} picked")
        demoted = sel_l - sel_v
        promoted = sel_v - sel_l
        if demoted:
            out.append(f"**demoted by {variant}** (in LexRank's pick, dropped): "
                       + ", ".join(f"[{i}]" for i in sorted(demoted)))
        if promoted:
            out.append(f"**promoted by {variant}** (new in pick): "
                       + ", ".join(f"[{i}]" for i in sorted(promoted)))
        out.append("")
    return "\n".join(out)


def write_report(
    out_dir: Path,
    cfg: argparse.Namespace,
    funnel: dict[str, int],
    agg: dict,
    results: list[dict],
    articles: list[Article],
    elapsed_s: float,
) -> None:
    by_id = {a.page_id: a for a in articles}
    n = agg["n_articles"]
    lines: list[str] = []
    lines.append("# bg-norm LexRank vs plain LexRank — extractive summarization eval\n")
    lines.append(
        "Falsifiable hypothesis under test: Fightin'-Words-corrected sentence centrality "
        "(`p^0.5 * log((p+eps)/(g+eps))` on the two-step sentence-word Gramian walk) beats "
        "plain LexRank at extractive summarization, scored by ROUGE F1 against Wikipedia "
        "lead sections (budget-matched to each lead's token length).\n")

    lines.append("## Method × ROUGE F1 (mean over articles, %)\n")
    lines.append("| method | ROUGE-1 | ROUGE-2 | ROUGE-L |")
    lines.append("|---|---:|---:|---:|")
    for m in METHODS:
        f = agg["mean_f1"][m]
        lines.append(f"| {m} | {_fmt_pct(f['rouge1'])} | {_fmt_pct(f['rouge2'])} "
                     f"| {_fmt_pct(f['rougeL'])} |")
    lines.append("")
    lines.append("`posthoc_bgnorm_on_degree` is a POST-HOC variant added after both designed "
                 "g variants lost: bg-norm re-rank (g1 background) on p = normalized tf-idf "
                 "degree centrality, isolating the Fightin'-Words correction from the raw "
                 "walk's length bias (Spearman(p, sentence length) ≈ 0.93 across articles; "
                 "degree ≈ 0.26). All other rows are as pre-designed.\n")

    lines.append("## Paired per-article deltas (mean Δ F1 pp, bootstrap 95% CI, win/tie/loss)\n")
    lines.append(f"Bootstrap: {cfg.n_boot} resamples, seed {cfg.seed}. "
                 f"Tie = |Δ| ≤ {TIE_EPS:g} (identical selections).\n")
    lines.append("| pair | metric | Δ mean (pp) | 95% CI (pp) | win | tie | loss |")
    lines.append("|---|---|---:|---|---:|---:|---:|")
    for key, by_met in agg["paired"].items():
        for met in METRICS:
            s = by_met[met]
            lo, hi = s["ci95"]
            lines.append(
                f"| {key.replace('_vs_', ' vs ')} | {met} | {100 * s['mean_delta']:+.3f} "
                f"| [{100 * lo:+.3f}, {100 * hi:+.3f}] | {s['win']} | {s['tie']} | {s['loss']} |")
    lines.append("")

    lines.append("## Qualitative examples (bgnorm_g1 vs lexrank)\n")
    lines.append(_qualitative_examples(results, by_id, "bgnorm_g1"))

    lines.append("## Provenance / filters\n")
    lines.append(f"- corpus: `{cfg.corpus_dir}` (bookgen scraped Wikipedia corpus; "
                 "partially-cleaned wikitext, finishing cleaner = "
                 "`artifacts_scripts/lexrank_bgnorm/textproc.py`)")
    lines.append(f"- data funnel: {json.dumps(funnel, sort_keys=True)}")
    lines.append(f"- n = {n} articles; filters: lead {cfg.lead_min}-{cfg.lead_max} sentences, "
                 f"body ≥ {cfg.min_body_sents} sentences, sentence ≥ "
                 f"{textproc.MIN_SENT_TOKENS} word-tokens")
    lines.append("- reference = lead section; document = body (References/External-links-type "
                 "sections and list/table lines dropped)")
    lines.append(f"- budget control: each method (except lead3) extracts sentences until "
                 f"reaching the reference lead's token count; crossing sentence included")
    lines.append(f"- centrality: `curriculum.core.pagerank.compute_pagerank`, uniform teleport, "
                 f"alpha={engine.PR_ALPHA}, tol={engine.PR_TOL}, max_iter={engine.PR_MAX_ITER}; "
                 f"Gramian kept sparse (never densified)")
    lines.append(f"- bg-norm: exponent a={engine.BG_EXPONENT}, eps={engine.BG_EPS:g} "
                 "(canonical values from `curriculum/config.py`); g1 = normalized mean global "
                 "word frequency; g2 = PageRank on global-frequency-weighted Gramian")
    lines.append("- ROUGE: vendored (no Porter stemmer; flat-sequence ROUGE-L), see "
                 "`artifacts_scripts/lexrank_bgnorm/rouge.py`")
    lines.append(f"- seed {cfg.seed}; wall time {elapsed_s:.0f}s; CPU only")
    lines.append(_provenance())

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", default=DEFAULT_CORPUS)
    ap.add_argument("--out-dir", default="artifacts/lexrank_bgnorm_eval")
    ap.add_argument("--cache-dir", default="cache/lexrank_eval")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--probe", action="store_true", help="data funnel + samples, then exit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--lead-min", type=int, default=2)
    ap.add_argument("--lead-max", type=int, default=10)
    ap.add_argument("--min-body-sents", type=int, default=25)
    ap.add_argument("--max-articles", type=int, default=0, help="0 = no cap")
    cfg = ap.parse_args()

    engine.silence_pagerank_progress()
    t0 = time.time()

    articles, funnel = prepare_dataset(
        Path(cfg.corpus_dir), cfg.lead_min, cfg.lead_max, cfg.min_body_sents,
        Path(cfg.cache_dir) / "dataset.json", cfg.refresh_cache,
    )
    print(f"[data] funnel: {json.dumps(funnel, sort_keys=True)}")
    print(f"[data] qualifying articles: {len(articles)}")

    if cfg.probe:
        for art in articles[:3]:
            print(f"\n--- {art.title} (lead {len(art.lead_sentences)}, "
                  f"body {len(art.body_sentences)}) ---")
            print("LEAD[0]:", art.lead_sentences[0][:200])
            print("BODY[0]:", art.body_sentences[0][:200])
            print("BODY[1]:", art.body_sentences[1][:200])
        return

    if cfg.max_articles and len(articles) > cfg.max_articles:
        rng = np.random.default_rng(cfg.seed)
        keep = sorted(rng.choice(len(articles), size=cfg.max_articles, replace=False))
        articles = [articles[i] for i in keep]
        print(f"[data] capped to {len(articles)} articles (seeded subsample)")

    global_freq = build_global_freq(articles)
    print(f"[freq] corpus unigram model: {len(global_freq)} types")

    results = []
    for i, art in enumerate(articles):
        results.append(run_article(art, global_freq, cfg.seed))
        if (i + 1) % 50 == 0:
            print(f"[run] {i + 1}/{len(articles)} articles")

    agg = aggregate(results, cfg.n_boot, cfg.seed)
    elapsed = time.time() - t0

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({
        "config": vars(cfg),
        "funnel": funnel,
        "aggregate": agg,
        "per_article": results,
        "generated_by": _provenance_cmd(),
    }, indent=1), encoding="utf-8")
    write_report(out_dir, cfg, funnel, agg, results, articles, elapsed)

    print(f"\n[done] n={agg['n_articles']} elapsed={elapsed:.0f}s")
    for m in METHODS:
        f = agg["mean_f1"][m]
        print(f"  {m:10s} R1={_fmt_pct(f['rouge1'])} R2={_fmt_pct(f['rouge2'])} "
              f"RL={_fmt_pct(f['rougeL'])}")
    for key in ("bgnorm_g1_vs_lexrank", "bgnorm_g2_vs_lexrank"):
        s = agg["paired"][key]["rouge1"]
        print(f"  {key}: ΔR1={100 * s['mean_delta']:+.3f}pp "
              f"CI[{100 * s['ci95'][0]:+.3f},{100 * s['ci95'][1]:+.3f}] "
              f"W/T/L={s['win']}/{s['tie']}/{s['loss']}")
    print(f"  report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
