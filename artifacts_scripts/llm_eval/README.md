# Prompt Optimization for Gap Analysis

This directory contains tools for optimizing LLM prompts to maximize textbook proposal gap quality using TSGD-M (Textual Stochastic Gradient Descent with Momentum) and Gumbel exploration.

## Overview

The optimization script (`optimize_prompt.py`) iteratively improves a system prompt that generates textbook proposals. The objective is to maximize the gap z-score of generated proposals when evaluated against cached gaps from the main analysis pipeline.

## Architecture

### Core Components

1. **`optimize_prompt.py`** - Main optimization script with two modes:
   - Simple hill-climbing (no dependencies on AdalFlow)
   - AdalFlow TSGD-M optimization (requires `adalflow` package)
   - Uses Hydra's ConfigStore (no YAML files needed)

### Key Features

- **Pure ConfigStore**: No YAML files required - all config via CLI overrides or programmatic defaults
- **Embedder Efficiency**: Single embedder instance reused throughout optimization (no repeated initialization)
- **Gap Sampling**: Configurable strategies (top z-score, top percentile, random)
- **Flexible Scoring**: Primary metric is gap z-score with percentile fallback
- **Robust Parsing**: Error handling for malformed LLM JSON responses
- **Comprehensive Logging**: Per-iteration metrics and best prompt tracking

## Installation

### Basic Requirements (Hill-Climbing Mode)

Already satisfied by main `bookgap` dependencies:
```bash
# No additional installation needed for hill-climbing mode
```

### AdalFlow Requirements (TSGD-M Mode)

```bash
pip install adalflow
```

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
optimization = [
    "adalflow",
]
```

## Usage

### Basic Usage (Hill-Climbing)

```bash
# Run with defaults
python -m artifacts_scripts.llm_eval.optimize_prompt

# Override specific settings
python -m artifacts_scripts.llm_eval.optimize_prompt \
    output_dir=outputs/opt_run_1 \
    trainer.iterations=50 \
    trainer.candidates_per_iter=3 \
    sampling.num_gaps=20

# Change experiment settings
python -m artifacts_scripts.llm_eval.optimize_prompt \
    experiment.batch.num_gaps=100 \
    experiment.cloud.embedder_model=GEMMA_300M \
    sampling.strategy=top_percentile
```

### View Current Configuration

```bash
# See all default values and available overrides
python -m artifacts_scripts.llm_eval.optimize_prompt --help

# Print config without running
python -m artifacts_scripts.llm_eval.optimize_prompt --cfg job
```

## Configuration

All configuration is done via Hydra CLI overrides or programmatic defaults (no YAML files needed).

### Experiment Config

The `experiment` section inherits from the main bookgap experiment configuration:
- `cloud`: Embedder model, denoiser, normalization
- `gap`: Cell type (k-ball/Voronoi), radius fraction
- `llm`: Model, temperature, max tokens
- `batch`: Batch size, num_gaps to generate
- `output`: Output formatting options

### Generator Config

```bash
# Override generator settings
python -m artifacts_scripts.llm_eval.optimize_prompt \
    generator.model_name=gpt-4o-mini \
    generator.temperature=0.9 \
    generator.max_tokens=1024
```

Default values:

Default values:
- `model_name`: gpt-4o
- `temperature`: 0.7
- `max_tokens`: 2048
- `initial_system_prompt`: Expert commissioning editor prompt

### Sampling Config

```bash
# Override sampling settings
python -m artifacts_scripts.llm_eval.optimize_prompt \
    sampling.num_gaps=32 \
    sampling.strategy=top_percentile \
    sampling.random_seed=123
```

Default values:
- `num_gaps`: 16
- `strategy`: top_zscore (options: top_zscore, top_percentile, random)
- `random_seed`: 42

### Trainer Config

```bash
# Override trainer settings
python -m artifacts_scripts.llm_eval.optimize_prompt \
    trainer.iterations=100 \
    trainer.candidates_per_iter=10 \
    trainer.history_update_strategy=improvement_only
```

Default values:
- `iterations`: 30
- `candidates_per_iter`: 5
- `history_update_strategy`: always (options: always, improvement_only)
- `exploration_epsilon`: 0.2
- `confidence_threshold`: 0.1
- `tsgd_m_enabled`: true
- `tsgd_m_momentum_window`: 5
- `random_seed`: 42

## Workflow

### 1. Load or Compute Gaps

```python
gaps = ensure_gaps(
    config=config.experiment,
    output_dir=config.output_dir,
)
```

Loads cached `gaps.joblib` or computes gaps via `compute_gaps_for_config()`.

### 2. Sample Gaps for Optimization

```python
sampled_gaps = sample_gaps(gaps, config.sampling)
```

Selects subset of gaps based on strategy (top z-score, percentile, or random).

### 3. Initialize Components

```python
cloud = instantiate(config.experiment.cloud)
with Embedder(model_name=config.experiment.cloud.embedder_model) as embedder:
    # Embedder stays initialized for entire optimization
    ...
```

### 4. Optimization Loop

For each iteration:
1. **Generate**: Create N candidate proposals with current prompt
2. **Embed**: Embed proposals using shared embedder instance
3. **Score**: Compute gap metrics for each proposal against sampled gaps
4. **Update**: Select best prompt (hill-climb) or apply TSGD-M update (AdalFlow)
5. **Log**: Save iteration results to JSON

### 5. Output

Each iteration produces:
- `iteration_XXX.json` - Proposals, scores, and metrics
- `best_prompt.txt` - Best prompt discovered so far

## Metrics

### Per-Proposal Metrics
- **max_zscore**: Maximum z-score across all sampled gaps
- **mean_zscore**: Average z-score across all sampled gaps
- **max_percentile**: Maximum gap size percentile
- **gap_zscores**: List of z-scores for each gap

### Aggregate Metrics
- **aggregate_score**: Best max_zscore among all proposals in iteration
- Used as optimization objective (maximize)

## Implementation Notes

### Addressed Concerns

✅ **Embedder Efficiency**: Single embedder context manager, no repeated initialization  
✅ **Gap Sampling**: Configurable strategies with clear semantics  
✅ **JSON Parsing**: Error handling with logging for malformed responses  
✅ **Checkpointing**: Per-iteration saves and best prompt tracking  

### Remaining TODOs

⚠️ **Prompt Mutation** (line ~515): Hill-climbing mode needs mutation strategy
```python
# TODO: Implement prompt mutation/perturbation
# Real implementation would:
# 1. Generate textual gradient/feedback based on scores
# 2. Use LLM to mutate prompt based on feedback
# 3. Decide whether to accept new prompt
```

⚠️ **AdalFlow Components** (lines ~253-382): Commented out pending installation
```python
# TODO: Uncomment and test once adalflow is installed
# class ProposalGeneratorComponent(Component): ...
# class GapScoreLoss(LossComponent): ...
```

⚠️ **Config Loading** (line ~664): Hydra integration incomplete
```python
# TODO: Load OptimizationConfig from Hydra
# config = OmegaConf.to_object(...)
```

⚠️ **Validation Set** (not implemented): Consider holding out gaps for overfitting detection

⚠️ **Z-Score Clipping** (not implemented): May need outlier handling if z-scores dominate

⚠️ **Aggregation Strategy** (line ~210): Currently uses max; consider mean or top-k mean

### Error Handling

- **Unparseable JSON**: Returns empty list with error log, heavy loss penalty
- **Malformed responses**: Logged with truncated preview
- **Missing gaps cache**: Automatically computes and persists gaps
- **Generation failures**: Skip iteration with error log, continue optimization

## Testing

### Unit Tests (Not Yet Implemented)

Recommended test coverage:
- `test_sample_gaps()` - All sampling strategies
- `test_parse_proposals_from_response()` - JSON parsing edge cases
- `test_embed_proposals()` - Embedding and projection correctness
- `test_score_proposals_against_gaps()` - Metric computation
- Integration test with mocked OpenAI client

### Manual Testing

```bash
# Dry-run check (will fail on config loading, but tests imports)
python artifacts_scripts/llm_eval/optimize_prompt.py --help

# Once config loading is implemented:
python artifacts_scripts/llm_eval/optimize_prompt.py \
    --config-name=prompt_opt_example \
    trainer.iterations=1 \
    trainer.candidates_per_iter=1 \
    sampling.num_gaps=2
```

## Comparison with Plan

The implementation follows `plan.md` with these differences:

### Implemented
- ✅ Core optimization loop structure
- ✅ Gap loading and sampling
- ✅ Embedding with cloud projection
- ✅ Gap metric computation via `compute_gap_metrics_for_embeddings`
- ✅ Iteration logging
- ✅ Simple hill-climbing option
- ✅ AdalFlow component stubs

### Deviations
- **Embedder pattern**: Using context manager instead of repeated instantiation (efficiency improvement)
- **Config structure**: Separated into generator/sampling/trainer sub-configs (better organization)
- **Two modes**: Implemented both hill-climbing and AdalFlow paths (flexibility)

### Not Yet Complete
- ⚠️ Prompt mutation logic in hill-climbing mode
- ⚠️ Full Hydra config integration
- ⚠️ AdalFlow mode activation (waiting on package)
- ⚠️ Validation set / overfitting detection
- ⚠️ Warm-start / resume from checkpoint

## Next Steps

1. **Complete Hydra Integration**: Implement config loading in `main()`
2. **Implement Prompt Mutation**: Add LLM-based prompt perturbation for hill-climbing
3. **Install AdalFlow**: Uncomment TSGD-M components and test
4. **Add Tests**: Unit tests for core functions
5. **Validation Set**: Hold out gaps for overfitting detection
6. **Resume Logic**: Support warm-start from previous runs
7. **Hyperparameter Sweep**: Grid search over trainer parameters
