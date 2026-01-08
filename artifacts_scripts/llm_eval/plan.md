# Prompt-optimization script plan (end-to-end)

## Goal
Implement a script in `artifacts_scripts/llm_eval/` that optimizes an LLM prompt to maximize proposal gap quality (prefer gap z-score, fallback percentile) using cached gaps and the new reusable gap-metric helper.

## Inputs
- Hydra config extending project defaults (ExperimentConfig) for:
  - cloud/embedder/denoiser/drop_k
  - gap cell_type/radius_fraction
  - llm model and decoding params
  - batch sizing (for generations per iteration)
- Cached gaps (`gaps.joblib`) produced by `find_gaps_for_config` (reuse if present).

## Core steps
1) **Load config & gaps**
   - Resolve Hydra config (possibly via a local `config.py` override) to build an `ExperimentConfig`.
   - Call `ensure_gaps(config)` which uses `@MEMORY.cache` decorated `compute_gaps_for_config` for automatic caching.

2) **Sample gap contexts for scoring**
   - Select a subset of gaps for optimization (e.g., top-N by gap_size_percentile or random seedable sample).
   - For each selected gap, build minimal context: topic info + `GapWithContext` for later embedding alignment.

3) **Generation loop (per iteration)**
   - Produce K candidate title+abstract pairs from the current prompt template (could be 1–n per iteration).
   - Embed candidates with `Embedder(model_name=config.cloud.embedder_model)` and project/normalize via cloud.

4) **Scoring**
   - For each candidate and each chosen gap, compute stats with `compute_gap_metrics_for_embeddings(gap, embeddings, config)`.
   - Aggregate to a scalar objective per candidate (e.g., max or mean of gap_zscore across sampled gaps).
   - Track secondary metrics (percentile, raw gap_size) for logging.

5) **Prompt update strategy** (choose one)
   - **Simple hill-climb:** keep best candidate prompt seen; perturb via a templated rewrite prompt to the LLM.
   - **TSGD-M style loop:** maintain momentum buffer of high-scoring prompts; sample/perturb using LLM with constrained format; accept/reject based on score.

6) **Logging & persistence**
   - Save per-iteration JSONL: prompt text, candidate texts, per-gap scores, aggregate objective.
   - Optionally store best prompt so far and reproducible RNG seeds.

## Script structure (proposed `optimize_prompt.py`)
- `load_experiment_config()` – builds ExperimentConfig from Hydra override.
- `ensure_gaps(config)` – calls `compute_gaps_for_config` (automatically cached via `@MEMORY.cache`).
- `sample_gaps(gaps, k, strategy)` – returns a working set for scoring.
- `generate_candidates(prompt_state, k)` – calls OpenAI client per config.
- `embed_candidates(candidates, config)` – uses Embedder + cloud projection.
- `score_candidates(candidates, embeddings, gaps, config)` – uses `compute_gap_metrics_for_embeddings` and aggregates.
- `update_prompt(prompt_state, scored_candidates, strategy)` – hill-climb or TSGD-M flavored update.
- `main()` – argument parsing, loop orchestration, logging to disk.

## AdalFlow integration (TSGD‑M + Gumbel)
- Trainer: prefer `TrainerGumbel` to enable exploration via Gumbel-Top‑k and textual gradient momentum.
- Key params (reasonable starters):
  - `history_update_strategy`: "always" (or "improvement_only" for stricter updates)
  - `exploration_epsilon`: 0.2 (ε‑greedy for off-policy exploration)
  - `confidence_threshold`: 0.1 (for confidence-based updates if used)
  - `tsgd_m_enabled`: true (enable momentum)
  - `tsgd_m_momentum_window`: 5 (window of retained high scorers)

### Minimal component/loss sketch
```python
import adalflow as adal
from adalflow.core import Component, Generator, Parameter, ParameterType
from adalflow.optim.trainer.trainer import TrainerGumbel

from bookgap.gap_metrics import compute_gap_metrics_for_embeddings
from bookgap.cloud import Embedder, Cloud

class ProposalGenerator(Component):
   def __init__(self, model_name: str):
      super().__init__()
      self.system_prompt = Parameter(
         name="system_prompt",
         data="You are a commissioning editor...",
         role_desc="Instructions for textbook proposal generation",
         param_type=ParameterType.PROMPT,
         requires_opt=True,
      )
      template = """
<SYSTEM>
{{system_prompt}}
</SYSTEM>
<USER>
Generate N proposal(s) with strict JSON schema ...
</USER>
"""
      self.generator = Generator(
         model_client=adal.OpenAIClient(),
         model_kwargs={"model": model_name, "temperature": 0.7},
         template=template,
      )

   def call(self, n: int):
      return self.generator(prompt_kwargs={"system_prompt": self.system_prompt.data, "n": n})


class GapScoreLoss(adal.optim.loss_component.LossComponent):
   def __init__(self, config, gaps, cloud: Cloud):
      super().__init__()
      self.config = config
      self.gaps = gaps
      self.cloud = cloud

   def _embed(self, title_abs_pairs):
      with Embedder(model_name=self.config.cloud.embedder_model) as emb:
         embs = emb.embed_texts(title_abs_pairs)
      proj = self.cloud.denoiser_projection()
      embs = proj.apply(embs)
      if self.cloud.normalize_embeddings and embs.shape[0] > 0:
         embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
      return embs.astype(np.float32)

   def forward(self, pred: adal.GeneratorOutput, target=None):
      # Parse proposals (ensure JSON schema in prompt)
      proposals = parse_json_proposals(pred.data)
      pairs = [(p["title"], p["abstract"]) for p in proposals]
      embs = self._embed(pairs)
      # Score per gap; aggregate with max z-score
      best_scores = []
      for i in range(embs.shape[0]):
         zscores = []
         for gap in self.gaps:
            stats = compute_gap_metrics_for_embeddings(gap, embs[i:i+1], self.config)[0]
            zscores.append(stats.gap_zscore)
         best_scores.append(max(zscores))
      # Loss = 1 - normalized z (or negative z); keep simple:
      # Use negative mean z to maximize z-score
      loss = -float(np.mean(best_scores))
      return loss


def build_trainer(config, gaps):
   task = ProposalGenerator(model_name=config.llm.model.model_id)
   cloud = adal.utils.instantiate(config.cloud)  # or bookgap.instantiate
   loss = GapScoreLoss(config, gaps, cloud)
   trainer = TrainerGumbel(
      adaltask=task,
      loss_fn=loss,
      history_update_strategy="always",
      exploration_epsilon=0.2,
      confidence_threshold=0.1,
      tsgd_m_enabled=True,
      tsgd_m_momentum_window=5,
   )
   return trainer
```

Notes:
- Ensure the prompt enforces strict JSON output so parsing is reliable in the loss.
- You can pass a small, fixed subset of `gaps` to speed up each iteration.
- If your objective saturates on percentile, the negative mean z‑score loss keeps improving with larger gaps.

## Metrics and objective
- Primary: maximize `gap_zscore`; if ties or NaNs, fall back to `gap_size_percentile`.
- Aggregation: start with `max_zscore` across sampled gaps per candidate; optionally experiment with `mean_zscore`.

## Next steps
- Confirm preferred update strategy (hill-climb vs TSGD-M-like sampling).
- Decide gap sampling policy and iteration budget (K candidates per iter, I iters).
- Implement `optimize_prompt.py` following this outline and wire to OpenAI client + Embedder.

## Practical settings
- Start with: K=3–5 candidates/iter, 20–50 iters, 8–16 sampled gaps.
- Use `history_update_strategy="improvement_only"` if you see drift; otherwise keep `"always"` for faster progress.
- Log per‑gap z-scores and the final selected proposal text each iteration.
