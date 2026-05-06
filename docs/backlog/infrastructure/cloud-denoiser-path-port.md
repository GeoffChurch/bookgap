# Port HDBSCAN probe + K-ball denoiser bypass to the Cloud denoiser path

(Migrated from `artifacts_scripts/curriculum/TODO.md` "Architectural debt" item. *Defer until probe sweeps identify production-worthy denoisers.*)

`hdbscan_gap_probe.py` and `kball_gap_probe.py --denoiser X` currently use a `load_scope`+numpy bypass instead of `Cloud(filter_pair=scope, denoiser_config=...)`.

The bypass was chosen for software-engineering reasons:
- Per-call flexibility for `--cclip-alpha` / `--drop-k-graph` / `--class-axis`
- Reusing one canonical denoiser dispatch implementation
- Less Cloud-cache pressure during a 10×8 sweep

But it creates real technical debt:

- Two parallel "load embeddings for scope X" code paths (Cloud vs `load_scope`)
- Two storage formats kept in sync (legacy per-shard `cache/embeddings/{model}/.../batch_*.npy` vs the new embeddings-parquet store)
- `denoiser_projection()` caching on Cloud isn't reused — proposal embeddings generated downstream may not match the probe's projection (potential vector-space-consistency issue per CLAUDE.md's note about why the projection is cached on Cloud in the first place)

**When to port**: after the probe sweep identifies which (denoiser, drop_k, α) combos are worth productionising. Bake those into `bookgap/cloud_denoiser.py` with proper class-aware config (`class_axis = topic | type | topic_x_type`), extend `CloudConfig.denoiser` to a structured `denoiser_config` field, refactor probes to construct Clouds via `Cloud(filter_pair=scope, denoiser_config=...)`, then delete the per-probe denoiser dispatch code (`_drop_top_k_pca`, `_denoise_lpp`, etc.) so a single canonical implementation lives in `bookgap.cloud_denoiser`.

**Empirical results first, abstraction second.**
