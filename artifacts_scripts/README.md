# Artifacts Scripts

This folder contains helper scripts for working with BookGap experiment outputs. All commands below assume you run them from the repository root with your project environment activated.

## `clean_up_runs.py`

Remove incomplete runs under `outputs/` (missing `gaps_analysis_results.csv`) and optionally delete empty date folders.

```bash
python -m artifacts_scripts.clean_up_runs
```

### Notable flags
- `--no-delete-missing`: report missing runs without deleting them.
- `--delete-grace-period-hours 4`: wait at least 4 hours from the last modification before deleting.
- `--dry-run`: show what would be removed without making changes.
- `--no-cleanup-empty-date-dirs`: keep empty `outputs/<date>/` directories.

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
