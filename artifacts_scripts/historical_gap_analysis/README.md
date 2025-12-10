# Historical Gap Analysis

Validates book gap definitions by analyzing whether historically published books tend to fill large gaps in the embedding space.

## Purpose

For a given year Y:
1. Build a "landmark set" of books published up to year Y
2. Identify books and articles published in year Y+1 (the "test sets")
3. Compute gap sizes for each test book/article with respect to the landmark books
4. Output statistics showing whether books fill larger gaps than articles

If books historically fill larger gaps than articles, this validates our p-ball gap definition as a meaningful signal for identifying research opportunities.

## Usage


### Basic usage (Hydra, single run)

```bash
python -m artifacts_scripts.historical_gap_analysis.historical_gap_analysis year=2015
```

### Background run with `nohup` (logs to file)

```bash
nohup python -m artifacts_scripts.historical_gap_analysis.historical_gap_analysis year=2024 > historical_analysis_2024.log 2>&1 &
```

Tail the log as it runs:

```bash
tail -f historical_analysis_2012.log
```

### Sweep multiple years (Hydra multirun)

```bash
python -m artifacts_scripts.historical_gap_analysis.historical_gap_analysis -m year=2010,2012,2014,2016,2018,2020
```

### Override gap radius_fraction (p-value)

```bash
python -m artifacts_scripts.historical_gap_analysis.historical_gap_analysis year=2015 experiment.gap.radius_fraction=0.8
```

### Sweep radius_fraction values

```bash
python -m artifacts_scripts.historical_gap_analysis.historical_gap_analysis -m year=2015 experiment.gap.radius_fraction=0.5,0.6,0.7,0.8,0.9
```

### Sweep both year and radius_fraction

```bash
python -m artifacts_scripts.historical_gap_analysis.historical_gap_analysis -m year=2010,2015,2020 experiment.gap.radius_fraction=0.5,0.75,0.9
```

### Change embedder/denoiser

```bash
python -m artifacts_scripts.historical_gap_analysis.historical_gap_analysis year=2015 experiment.cloud.embedder_model=GIST_LARGE experiment.cloud.denoiser=NONE
```

## Output

### Per-run outputs (in `artifacts_scripts/historical_gap_analysis/results/`)

Each run generates two files:

#### 1. Summary CSV
`historical_gap_analysis_summary_<params>.csv` contains aggregated statistics:
- `year`: Landmark year (books up to this year)
- `p`: Radius fraction for p-ball definition
- `embedder_model`: Embedding model used
- `denoiser`: Denoiser applied
- `drop_k`: Number of components dropped
- `num_landmark_books`: Number of books in landmark set
- `num_test_books`: Number of books in test set (year+1)
- `num_test_articles`: Number of articles in test set (year+1)
- Mean/median/std of gap sizes for books
- Mean/median/std of gap sizes for articles
- Mean/median/std of gap sizes for landmark books

#### 2. Diagnostic plot PNG
`historical_gap_distributions_<params>.png` shows gap size distributions for books vs articles.

### Combining and plotting results

After running multiple experiments, use these scripts to synthesize and visualize:

#### Synthesize results
```bash
python artifacts_scripts/historical_gap_analysis/synthesize_results.py
```

Combines all summary CSVs from `artifacts_scripts/historical_gap_analysis/results/` into a single file:
- Output: `artifacts/historical_gap_analysis/results/historical_gap_analysis_combined.csv`

#### Plot combined results
```bash
python artifacts_scripts/historical_gap_analysis/plot_results.py
```

Generates plots from the combined CSV showing gap size evolution over years:
- Input: `artifacts/historical_gap_analysis/results/historical_gap_analysis_combined.csv`
- Output: Plots saved to `artifacts/historical_gap_analysis/results/`

## Caching

The script uses `@MEMORY.cache` decorators to cache:
- Temporal subclouds (per year, work type, cloud config)
- Gap size computations

This makes sweeps much more efficient as indices are reused across different p-values.

## Configuration

All configuration comes from dataclass defaults in `config.py` and the bookgap config system. The script uses `HistoricalAnalysisConfig` which wraps `ExperimentConfig`.

Default settings:
- `year`: Default is `2010` (override via Hydra: `year=YYYY`)
- `experiment.gap.radius_fraction`: 0.75 (from `GapConfig`)
- `experiment.cloud.embedder_model`: GIST_SMALL (from `CloudConfig`)
- `experiment.cloud.denoiser`: NONE (from `CloudConfig`)
- `experiment.cloud.filter_pair`: AI subfield, books+articles (from `FilterPairConfig`)
- `min_books_per_year`: 10
- `min_articles_per_year`: 10
- `output_dir`: `artifacts_scripts/historical_gap_analysis/results/` (default, can be overridden)

Override any parameter via Hydra CLI syntax: `experiment.gap.radius_fraction=0.8` or `output_dir=/path/to/output`
