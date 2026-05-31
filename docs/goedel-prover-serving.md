# Goedel-Prover-V2 serving setup (the stronger-prover lever)

`docs/lean-curriculum-experiment.md` and the 2026-05-31 handoff both name the
**single binding constraint** on the Phase-4 curriculum-proving result: the local
general models (`gemma4:31b` / `qwen3.5:35b` at `think=False`) are weak Lean
provers. They close the easy targets monolithically but cannot close the hard
sub-lemmas, so the headline win — *decomposition rescues a target the monolith
misses* — never materialises. The verifier's per-goal failure split (52.5%
type_error, 23.5% **pure syntax** parse_error, 17.9% unsolved_goals, 6.2%
success) says a large slice is reclaimable with a better prover, *not* new
harness infrastructure.

**Goedel-Prover-V2** is that lever: a Lean-tuned, whole-proof prover that is SOTA
on miniF2F. This note stands it up as a loadable, documented backend. It is
**not** wired into a run yet — that is a model swap behind the existing
`provider(goal) -> proof` contract.

## 1. Models, IDs, availability

| | HuggingFace ID (canonical) | Base | ollama (first-party) | GGUF (community) |
|---|---|---|---|---|
| 8B  | `Goedel-LM/Goedel-Prover-V2-8B`  | Qwen3-8B  | none | `mradermacher/Goedel-Prover-V2-8B-GGUF`, `DevQuasar/...` |
| 32B | `Goedel-LM/Goedel-Prover-V2-32B` | Qwen3-32B | none | `mradermacher/Goedel-Prover-V2-32B-GGUF`, `DevQuasar/...` |

- **No first-party ollama library entry.** A couple of low-pull community
  re-uploads exist (`kdavis/Goedel-Prover-V2`, `alessandrorumampuk/...`) but are
  unvetted. We use the **mradermacher** GGUFs (a trusted, widely-used quantizer),
  pulled directly into ollama via the `hf.co/<repo>:<quant>` form — no manual
  llama.cpp conversion needed.
- Both models are **Qwen3** (`Qwen3ForCausalLM`, standard ChatML
  `<|im_start|>…<|im_end|>` template, `enable_thinking` honored).

## 2. Recommended inference + exact prompt format (from the model card)

- **Recommended engine:** the card shows **HuggingFace Transformers** only
  (`AutoModelForCausalLM`, `device_map="auto"`, `torch_dtype=bfloat16`,
  `trust_remote_code=True`). No vLLM/sglang example is given (though both work
  with Qwen3). We do not use Transformers directly — see the serving decision.
- **Mode:** **whole-proof generation** (complete the `sorry`-ed theorem in one
  shot), *not* tactic-mode. The prompt explicitly asks for a **proof plan first**,
  then the full Lean code — that reasoning step is load-bearing for quality.
- **Self-correction:** the card describes a *self-correction mode* (generate a
  candidate → feed Lean compiler feedback → revise, **two rounds**; 88.0% → 90.4%
  pass@32 on miniF2F-32B). **No code is shipped** for it — it is a described
  pipeline, not a packaged API. Our experiment's bounded-retry loop is a weaker
  analogue; a faithful self-correction arm is future work.

**Exact prompt (verbatim from the card):**

```python
prompt = """
Complete the following Lean 4 code:

```lean4
{}```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

chat = [{"role": "user", "content": prompt.format(formal_statement)}]
inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=32768)
```

Where `formal_statement` is the **full theorem** — imports, `set_option`, `open`s,
and `theorem … := by sorry` — e.g. (card example):

```lean4
import Mathlib
import Aesop
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat
theorem square_equation_solution {x y : ℝ} (h : x^2 + y^2 = 2*x - 4*y - 5) : x + y = -1 := by
  sorry
```

There is **no system prompt**: a single user turn + the chat template. Note
`max_new_tokens=32768` — the plan+proof can be long, so the token budget must be
generous (the opposite of the experiment's weak-model `num_predict=256`).

## 3. Serving decision (under the single ~48 GB GPU)

The box has one **RTX 6000 Ada, 49 GB**. fp16 footprints: 8B ~16 GB (fits), 32B
~64 GB (**does not fit**) — the 32B *must* be 4-bit (~18–20 GB).

**Chosen: ollama-via-GGUF (mradermacher).** Rationale:

- **(a) GPU fit.** 8B @ **Q8_0** (~8.7 GB, near-lossless) and 32B @ **Q4_K_M**
  (~19.8 GB, mradermacher's "recommended" 4-bit) both fit, the 32B with ~28 GB to
  spare for KV cache at a long (16k) context. Q4_K_M is the standard GPU-fit quant
  for a 32B on a single 48 GB card.
- **(b) prompt format preserved.** The Qwen3 ChatML template ships inside the
  GGUF; the Goedel "proof plan first" wrapper is applied per-goal by the Python
  client (it is the full theorem statement, so it cannot live in a static
  Modelfile TEMPLATE). Format is byte-identical to the card.
- **(c) cleanest integration.** The experiment *already* drives provers through
  `ollama.chat` (`curriculum_lean/experiment.py::OllamaProver`). Reusing that
  plumbing means the Goedel backend is a sibling class, not new serving infra +
  a new client. vLLM would give better throughput and fp16/AWQ fidelity but needs
  a separate server process and an OpenAI-style client — a larger change for no
  benefit at this experiment's scale (serial, one model resident at a time).

Cost accepted: Q4 quantization may shave a few points of prover precision vs
fp16. If the 32B Q4 underperforms its reported numbers materially, the fallback
is **vLLM with an AWQ/GPTQ 32B** (or fp16 8B) — documented here, not built.

## 4. Download status / sizes

Pulled into `~/.ollama/models` (outside the repo — **never committed**) via
`scripts/setup_goedel_prover.sh`:

| Tag | Source GGUF | On-disk |
|---|---|---|
| `goedel-prover-v2:8b`  | `hf.co/mradermacher/Goedel-Prover-V2-8B-GGUF:Q8_0`   | ~8.7 GB  |
| `goedel-prover-v2:32b` | `hf.co/mradermacher/Goedel-Prover-V2-32B-GGUF:Q4_K_M` | ~19.8 GB |

(Other quants available upstream if needed: 8B Q4_K_M 5.0 GB … f16 16.4 GB; 32B
Q3_K_M 16.0 GB … Q8_0 34.8 GB. The 32B Q8_0 would NOT fit the GPU.) Disk before
download: ~191 GB free of 1.9 TB; the two GGUFs add ~28 GB.

## 5. How it plugs into the experiment

`curriculum_lean/goedel_prover.py::GoedelProver` is **API-compatible with
`OllamaProver`**: same `make_provider(context_builder, stats) -> (Goal -> str)`
contract, so `curriculum_lean/experiment.py`'s `run_target_for_model` /
`_score_decomposition` / scorer grade it unchanged. The differences are all
prover-correctness:

- **Goedel's own prompt**, built from the **full theorem statement**. This is the
  flagged subtlety: Goedel's whole-proof format wants the *statement*, not a bare
  goal — and our `Goal` carries exactly that. `format_full_statement(goal)`
  renders `theorem {goal.fresh_name} : {goal.type_str} := by sorry` under the
  standard mathlib preamble, which is precisely the card's `formal_statement`. No
  data is missing; the mapping is direct.
- **thinking ON** + **large `num_predict` (8192)** — the proof plan is the model's
  reasoning. (Contrast the weak-model arms, which must run `think=False` /
  `num_predict=256` for feasibility.)
- **no system prompt** (single user turn, per the card).
- **extraction reuses `experiment.extract_proof`**: Goedel returns a fenced
  ```lean4 block with the *full* `theorem … := <proof>`; the existing extractor
  already takes the last lean fence, strips the `theorem … :=` header to
  `<proof>`, and `by`-wraps a bare tactic sequence — exactly what Goedel needs.

For the **one-shot (control) arm** the provider is literally
`provider(goal) -> proof` with Goedel's template and an empty context. For **T1**
the curriculum-selected prerequisite statements are prepended to Goedel's prompt
as an extra context block. (The decomposition arms `t2_*` work unchanged because
they call the same `_one_attempt` per sub-goal.)

To wire it into a run (not done here), pass a `GoedelProver` where the experiment
constructs an `OllamaProver` (`experiment.run_experiment`'s per-model loop) — e.g.
add `goedel-prover-v2:8b` / `:32b` to `--models` and branch the prover
construction on the tag. That is the only code change; everything downstream
(arms, scoring, dashboard) is untouched.

## 6. Setup + smoke test

```bash
bash scripts/setup_goedel_prover.sh            # pulls 8B Q8_0 + 32B Q4_K_M, tags both
conda run -p .conda python -m curriculum_lean.goedel_prover --model goedel-prover-v2:8b
conda run -p .conda python -m curriculum_lean.goedel_prover --model goedel-prover-v2:32b
```

The smoke test loads the model, sends Goedel's prompt for a trivial theorem, and
confirms a fenced Lean proof is emitted + extractable (it does **not** invoke the
Lean verifier — that is exercised by the experiment proper).

## Files

- `scripts/setup_goedel_prover.sh` — pulls the GGUFs into ollama, creates the
  `goedel-prover-v2:{8b,32b}` tags. Idempotent; weights stay in `~/.ollama`.
- `curriculum_lean/goedel_prover.py` — `GoedelProver` (the `OllamaProver`-compatible
  backend), the verbatim Goedel prompt + full-statement renderer, the
  proof extractor, and the standalone smoke test.

## Sources

- https://huggingface.co/Goedel-LM/Goedel-Prover-V2-8B
- https://huggingface.co/Goedel-LM/Goedel-Prover-V2-32B
- https://huggingface.co/mradermacher/Goedel-Prover-V2-8B-GGUF
- https://huggingface.co/mradermacher/Goedel-Prover-V2-32B-GGUF
- Paper: *Goedel-Prover-V2: Scaling Formal Theorem Proving with Scaffolded Data
  Synthesis and Self-Correction* (arXiv:2508.03613)
