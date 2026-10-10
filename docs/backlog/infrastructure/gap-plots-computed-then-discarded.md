# Gap plots are computed, then discarded

`generate_gap_plots_for_results` (`bookgap/llm/artifacts/gap_plots.py`) builds each gap's PCA plot and then asks `HydraConfig.get()` for the directory to save it in.
Under anything but `batch_generator.py` (`artifacts_scripts/paper/regen_sweep.py`, the tests) Hydra is not set, so every plot fails after the work is done: "Failed to generate plot for gap 0: HydraConfig was not set", once per topic, and "Generated 0/77 gap plots successfully".
That is about 1.5 s a topic, two minutes of each 77-topic cell, while the GPU waits.

Two more faults surface once it saves:

- It ignores the `output_dir` it is passed, which is where the plots belong.
- It groups proposals by `gap_idx` alone, and every topic's first gap is index 0, so each topic's plot would draw every topic's proposals, and the files would overwrite one another under `plots/gap_0/`.

## Fix

Save under `output_dir`, group by (topic, gap), and decide whether plots are made at all where the run is configured, so a sweep that does not want them does not compute them.
Do it between sweeps: adding the option to `ExperimentConfig` changes every run's resolved configuration, and so the `run_identity` that regen_sweep resumes by.
