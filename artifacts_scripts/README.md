# Artifacts Scripts

This folder contains helper scripts for working with BookGap experiment outputs. All commands below assume you run them from the repository root with your project environment activated.

## `do_sweep.py`

Launch a batch of `bookgap.batch_generator` jobs based on override patches defined in JSON or JSONL files.

```bash
python -m artifacts_scripts.do_sweep --config-name anti_csp_drop1 --patch-file artifacts_scripts/do_sweep_patches.jsonl
```

### Typical options
- `--override key=value`: add Hydra override(s) applied to every run (repeatable).
- `--patch-file path/to/patches.jsonl`: custom patch file. Defaults to `artifacts_scripts/do_sweep_patches.jsonl`.
- `--dry-run`: print the commands without launching jobs.
- `--stop-on-error`: halt subsequent runs if any job fails.

## `output_run_summary.py`

Aggregate per-run statistics into `artifacts/run_gap_summary.csv`.

```bash
python -m artifacts_scripts.output_run_summary
```

### Useful flags
- `--outputs-dir path/to/outputs`: root directory containing run subfolders.
- `--output-file path/to/summary.csv`: where to write the summary.
- `--overwrite/--no-overwrite`: control whether an existing summary is replaced.

## `paper/generate_latex_tables.py`

Create LaTeX tables for each `(embedder_model, llm_num_articles, llm_num_books)` group using the summary CSV and per-run gap analysis files. Each group receives its own directory under `artifacts/latex_tables/` containing `run_gap_summary.tex` and `best_run_gaps_analysis.tex`.

```bash
python -m artifacts_scripts.paper.generate_latex_tables --overwrite
```

### Handy options
- `--run-summary-file path/to/summary.csv`: defaults to `artifacts/run_gap_summary.csv`.
- `--outputs-dir path/to/outputs`: root directory with individual run folders.
- `--output-root path/to/latex_tables`: destination for generated LaTeX files.
- `--max-gap-rows N`: limit the number of rows in each best-run table (omit to include all rows).
- `--max-title-length 0`: truncate long proposal titles (set to 0 to omit the title column entirely).
- `--topic-name-length 15`: truncate topic display names (set negative to leave them untrimmed).
- `--overwrite`: regenerate tables even if the files already exist.

## `output_plots.py`

Produce multiple plots.

```bash
python -m artifacts_scripts.output_plots
```

### Customization
- `--outputs-dir path`: choose the source runs.
- `--output-file path`: pick a destination PNG file.
- `--llm-counts-output-file path`: write the LLM count scatter plot somewhere else.
- `--dpi 150`: adjust image resolution (default 150).
