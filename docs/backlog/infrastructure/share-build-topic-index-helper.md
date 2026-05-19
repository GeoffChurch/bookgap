# Share `_build_topic_index` between production and research

## Idea

`_build_topic_index` (HNSW efc=512/efs=2000 for n≥100, FlatL2 otherwise;
`BOOKGAP_EXACT_INDEX=1` → always FlatL2) now exists **verbatim in two
places**: `artifacts_scripts/kball_gap_probe.py` and (copied, to honor
the no-`artifacts_scripts`-import constraint) the new production
`bookgap/llm/gap_algorithms/deflation_strategies.py`. The bake-off
harness also relies on the original. Two copies of a recall-critical
index-config helper will drift.

## Work

Extract the canonical implementation into a shared `bookgap/` home
(e.g. a helper in `bookgap/index.py`), have both the production
deflation module and the research scripts import it. `bookgap.index`
has no equivalent today (it exposes `build_index`/`FlatL2`/
`kballs_with` but not the n-threshold HNSW-vs-FlatL2 +
`BOOKGAP_EXACT_INDEX` selection logic).

## Caveat

Do NOT fold this into the resid-promotion commit — the verbatim-port
byte-equivalence gate (production resid logic == validated bake-off
logic) must not be perturbed by an index-helper refactor in the same
change. Do it as an isolated refactor with the byte-identity test
(`test_default_is_byte_identical_to_verbatim_greedy`) + the bake-off
suite as the regression guard.
