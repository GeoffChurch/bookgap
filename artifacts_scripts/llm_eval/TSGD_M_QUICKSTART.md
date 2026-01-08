# TSGD-M Quick Start Guide

## What is TSGD-M?

Text Stochastic Gradient Descent with Momentum (TSGD-M) is an optimizer from AdalFlow that uses textual gradients to iteratively improve LLM prompts. Instead of traditional gradient-based optimization with numerical parameters, it directly optimizes the prompt text.

## Quick Start

### Run with TSGD-M (Recommended for optimization)
```bash
python -m artifacts_scripts.llm_eval.optimize_prompt trainer.tsgd_m_enabled=true trainer.iterations=20
```

### Run with Simple Hill-Climbing (Faster baseline)
```bash
python -m artifacts_scripts.llm_eval.optimize_prompt \
    trainer.tsgd_m_enabled=false \
    trainer.iterations=30
```

## Key Configuration Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `trainer.tsgd_m_enabled` | `true` | `true`/`false` | Enable/disable TSGD-M optimizer |
| `trainer.tsgd_m_momentum_window` | `5` | 1-20 | History size for momentum (more = smoother) |
| `trainer.iterations` | `30` | 1-∞ | Number of optimization iterations |
| `sampling.num_gaps` | `16` | 1-∞ | Number of gaps sampled for training |
| `sampling.strategy` | `top_zscore` | `top_zscore`, `top_percentile`, `random` | Which gaps to focus on |

## Common Use Cases

### Quick Test Run (5 minutes)
```bash
python -m artifacts_scripts.llm_eval.optimize_prompt \
    trainer.tsgd_m_enabled=true \
    trainer.iterations=3 \
    sampling.num_gaps=5 \
    output_dir=outputs/test_run
```

### Production Run (1-2 hours)
```bash
python -m artifacts_scripts.llm_eval.optimize_prompt \
    trainer.tsgd_m_enabled=true \
    trainer.iterations=50 \
    sampling.num_gaps=32 \
    sampling.strategy=top_zscore \
    output_dir=outputs/production_run
```

### Compare Methods
```bash
# Run TSGD-M
python -m artifacts_scripts.llm_eval.optimize_prompt \
    trainer.tsgd_m_enabled=true \
    trainer.iterations=20 \
    output_dir=outputs/tsgd_m_run

# Run Simple Mode
python -m artifacts_scripts.llm_eval.optimize_prompt \
    trainer.tsgd_m_enabled=false \
    trainer.iterations=20 \
    output_dir=outputs/simple_run

# Compare best_prompt.txt in both directories
```

## Output Files

After running, check these files in `output_dir`:

- **best_prompt.txt** - The optimized system prompt (main output)
- **iteration_001.json, iteration_002.json, ...** - Per-iteration logs with scores
- **logs/** - Hydra logs

## Troubleshooting

### Script fails with API key error
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."
python -m artifacts_scripts.llm_eval.optimize_prompt trainer.tsgd_m_enabled=true
```

### Out of memory on GPU
```bash
# Use simpler model or fewer gaps
python -m artifacts_scripts.llm_eval.optimize_prompt \
    trainer.tsgd_m_enabled=true \
    sampling.num_gaps=8
```

### Optimization not converging
```bash
# Try faster decay (smaller momentum window)
python -m artifacts_scripts.llm_eval.optimize_prompt \
    trainer.tsgd_m_enabled=true \
    trainer.tsgd_m_momentum_window=2

# Or switch to simple mode for more iterations
python -m artifacts_scripts.llm_eval.optimize_prompt \
    trainer.tsgd_m_enabled=false \
    trainer.iterations=100
```

## Performance Expectations

### TSGD-M Mode
- **Convergence**: Usually converges in 10-30 iterations
- **Speed**: 10-30 seconds per iteration (depends on gap size)
- **Quality**: Typically finds better prompts than simple mode with same budget

### Simple Hill-Climbing Mode  
- **Convergence**: May need 50-100+ iterations (or not converge)
- **Speed**: 3-10 seconds per iteration (no gradient computation)
- **Quality**: Good baseline, but limited exploration

## Monitoring Progress

Watch the optimization in real-time:
```bash
# In another terminal, tail the iteration logs
tail -f outputs/prompt_opt/iteration_*.json | jq '.results.aggregate_score'
```

## Interpreting Results

The aggregate score is the maximum z-score achieved:
- **< 0**: Prompt performs worse than baseline
- **0-2**: Prompt performs slightly better
- **2-5**: Good improvement  
- **> 5**: Excellent improvement

Check `best_prompt.txt` to see what the optimizer learned.

## Advanced: Custom Momentum Window

Momentum window controls how much past history influences updates:
- **Small (1-2)**: Fast adaptation, but may oscillate
- **Medium (3-5)**: Good balance (default)
- **Large (8-20)**: Smooth updates, may miss fast improvements

```bash
# Fast adaptation for quick changing landscape
python -m artifacts_scripts.llm_eval.optimize_prompt \
    trainer.tsgd_m_momentum_window=2

# Smooth optimization for stable improvements
python -m artifacts_scripts.llm_eval.optimize_prompt \
    trainer.tsgd_m_momentum_window=10
```

## Next Steps

1. Try both TSGD-M and simple mode on your dataset
2. Compare convergence speed and final prompt quality
3. Experiment with different momentum windows
4. Use the best prompt from optimization in your production pipeline

For detailed implementation info, see `TSGD_M_IMPLEMENTATION.md`.
