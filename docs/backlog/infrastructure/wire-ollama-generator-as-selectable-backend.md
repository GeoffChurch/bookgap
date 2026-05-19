# Wire the existing Ollama generator as a selectable proposal backend

## Idea

Make local-Ollama proposal generation a first-class, config-selectable
backend (to save $$ vs the OpenAI Batch API for routine runs and for
the deferred LLM-coverage check).

**It already exists, it just isn't wired in.**
`bookgap/llm/generators/ollama_local.py` has a working
`OllamaProposalGenerator` / `OllamaProposalRecord` (synchronous
`ollama.chat`, e.g. `gemma4:31b` / `qwen3:30b`, strips `thinking`
blocks, JSON-schema-as-instruction since Ollama lacks structured
output). But `bookgap/llm/generators/__init__.py` exports only
`OpenAIBatchGenerator`, and the pipeline
(`workflow_functions.py`) hard-wires the OpenAI batch path — Ollama
is not reachable from config.

## Work

Mirror the just-landed `deflation_strategy` opt-in pattern: a
config-selectable generator backend (enum/`LLMConfig` field +
dispatch) defaulting to OpenAI (unchanged), with `ollama_local`
selectable. Reconcile the interface gap: `OpenAIBatchGenerator`
subclasses `BatchProposalGenerator` (batch lifecycle, `run_state`
resumability, strict JSON schema); `OllamaProposalGenerator` is
synchronous per-context with no batch/run_state. Decide whether to
adapt Ollama to the batch ABC (a trivial "batch" = sequential loop
with the same run_state/artifact shape) or add a parallel
non-batch generation path. Parity to check: structured-output
fidelity (schema-as-instruction vs strict schema), artifact/record
shape so downstream parsing + the proposal artifacts are identical,
resumability.

## Why it matters

Routine + iteration runs become ~free; unblocks the deferred
pre-paper LLM-target-gap-coverage confirmation on a local model
(see memory `project_llm_coverage_ollama.md` — that check is
explicitly slated for a local Ollama adapter, not OpenAI).
