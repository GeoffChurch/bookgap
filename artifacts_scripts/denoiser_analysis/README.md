# Denoiser Analysis

This module provides comprehensive comparison of denoising strategies for the book-article embedding space.

## Usage

Run the analysis with default settings:
```bash
python -m artifacts_scripts.denoiser_analysis.denoiser_analysis
```

Analyze specific denoisers:
```bash
python -m artifacts_scripts.denoiser_analysis.denoiser_analysis denoisers='[PCA,CCLIP,WC_CCLIP,ANTI_LDA]'
```

Change drop_k parameter:
```bash
python -m artifacts_scripts.denoiser_analysis.denoiser_analysis drop_k=200
```

Limit works for faster testing:
```bash
python -m artifacts_scripts.denoiser_analysis.denoiser_analysis max_works=10000
```

## Key Metrics

### SNR Score (Signal-to-Noise Ratio)
- **Signal**: Within-class variance (S_W) - semantic variance preserved within each class
- **Noise**: Between-class scatter (S_B) + Laplacian scatter (S_L)
- Higher is better: preserves meaningful variance while reducing class separation and graph inconsistency

### Variance Retention
- Fraction of total variance preserved by the projection
- Higher is better (up to a point - too high may retain noise)

### Scatter Matrix Decomposition
- **S_T**: Total scatter (overall variance)
- **S_B**: Between-class scatter (how much classes are separated)
- **S_W**: Within-class scatter (variance within each class)
- **S_L**: Laplacian scatter (graph inconsistency)

## Denoisers Analyzed

### Standard CCLIP Family (S_T constraint)
- `CCP`: Pure class-conflating (α=0)
- `CCLIP_01`: α=0.01
- `CCLIP_1`: α=0.1
- `CCLIP`: α=0.5
- `LIP`: Pure locality-inducing (α=1)

### Within-Class CCLIP Family (S_W constraint)
- `WC_CCP`: Pure Anti-LDA with S_W constraint (α=0)
- `WC_CCLIP_01`: α=0.01
- `WC_CCLIP_1`: α=0.1
- `WC_CCLIP`: α=0.5
- `WC_LIP`: Pure LPP with S_W constraint (α=1)

The key difference is the constraint matrix in the denominator:
- Standard CCLIP uses S_T (total scatter) which implicitly includes S_B
- WC_CCLIP uses S_W (within-class scatter) avoiding the conflict where maximizing S_T rewards class separation

## Output

Results are written to `artifacts_scripts/denoiser_analysis/results/`:
- `denoiser_analysis_drop{K}_{embedder}.csv`: Full metrics table
- `snr_comparison_drop{K}.png`: SNR score comparison
- `scatter_decomposition_drop{K}.png`: Scatter matrix decomposition
- `variance_vs_snr_drop{K}.png`: Trade-off visualization
- `class_scatter_balance_drop{K}.png`: Book vs article scatter balance
- `metrics_heatmap_drop{K}.png`: Normalized metrics heatmap
- `cclip_family_comparison_drop{K}.png`: CCLIP vs WC_CCLIP family comparison

---

## Eigenbasis Analysis

A separate script analyzes the full eigenbasis of each denoiser (no truncation) to understand
how each denoiser's principal directions relate to the scatter matrices.

### Usage

```bash
python -m artifacts_scripts.denoiser_analysis.eigenbasis_analysis
python -m artifacts_scripts.denoiser_analysis.eigenbasis_analysis denoisers='[PCA,CCLIP,WC_CCLIP]'
```

### What it computes

For each denoiser's eigenbasis V = [v_1, v_2, ..., v_d]:
- **% Variance Explained**: `v_i.T @ S @ v_i / Tr(S)` for each scatter matrix S
- This shows how each eigenvector captures variance in S_T, S_B, S_W, S_L

### Key insights

- Does PCA (which maximizes S_T variance) also capture S_B variance? (bad for conflation)
- Does CCLIP successfully push S_B variance into dropped dimensions?
- Are S_B and S_L variance concentrated in the same eigenvectors?

### Output

Results are written to `artifacts_scripts/denoiser_analysis/results/eigenbasis/`:
- `eigenbasis_variance_{embedder}.csv`: Full variance data per component
- `variance_by_matrix.png`: 2x2 plot showing cumulative variance for all denoisers (one panel per matrix)
- `eigenbasis_{denoiser}.png`: Individual plots per denoiser (one line per matrix)
- `components_to_threshold.png`: How many components needed to reach 50/90/95/99% variance
- `signal_noise_separation.png`: Signal (S_W) vs Noise (S_B, S_L) in top-k components
