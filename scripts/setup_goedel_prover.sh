#!/usr/bin/env bash
# Stand up Goedel-Prover-V2 (8B + 32B) as ollama-served Lean provers for the
# curriculum_lean Phase-4 experiment. See docs/goedel-prover-serving.md.
#
# Serving path: ollama-via-GGUF (mradermacher quants), reusing the experiment's
# existing `ollama.chat` plumbing. Fits the single ~48 GB GPU:
#   - 8B  @ Q8_0   ~8.7 GB  (near-lossless; fp16 ~16 GB also fits but Q8_0 ample)
#   - 32B @ Q4_K_M ~19.8 GB (the ONLY way the 32B fits; fp16 ~64 GB does not)
#
# Weights live in ~/.ollama/models (OUTSIDE the repo) — never committed.
#
# Idempotent: re-running re-pulls only missing blobs and recreates the tags.
#
# Usage:
#   bash scripts/setup_goedel_prover.sh            # both models
#   bash scripts/setup_goedel_prover.sh 8b         # 8B only
#   bash scripts/setup_goedel_prover.sh 32b        # 32B only
set -euo pipefail

# Upstream GGUF repos (mradermacher — a trusted, widely-used quantizer). ollama
# can pull these directly from HuggingFace via the hf.co/<repo>:<quant> form.
HF_8B="hf.co/mradermacher/Goedel-Prover-V2-8B-GGUF:Q8_0"
HF_32B="hf.co/mradermacher/Goedel-Prover-V2-32B-GGUF:Q4_K_M"

# Short local tags the Python backend (curriculum_lean.goedel_prover) references.
TAG_8B="goedel-prover-v2:8b"
TAG_32B="goedel-prover-v2:32b"

WHICH="${1:-both}"

command -v ollama >/dev/null || { echo "ollama not on PATH" >&2; exit 1; }

# Modelfile defaults tuned for Goedel's whole-proof generation. Goedel emits a
# detailed proof PLAN (its chain-of-thought) before the proof, so the budget and
# context must be generous — the opposite of the weak-model arms' tight
# `num_predict`. The Qwen3 chat template ships inside the GGUF; we only set
# generation params + a long context here. The PROMPT itself (the "complete this
# Lean 4 code … proof plan first" wrapper) is applied by the Python client
# (curriculum_lean.goedel_prover.render_goedel_prompt), NOT here, because it is
# per-goal (the full theorem statement) — a static Modelfile TEMPLATE cannot
# inject it.
make_tag () {
  local from="$1" tag="$2" ctx="$3"
  echo ">>> Creating ollama tag '$tag' from '$from' (num_ctx=$ctx) ..."
  local mf
  mf="$(mktemp)"
  cat > "$mf" <<EOF
FROM $from
PARAMETER temperature 1.0
PARAMETER num_ctx $ctx
PARAMETER num_predict 8192
EOF
  ollama create "$tag" -f "$mf"
  rm -f "$mf"
}

pull_8b () {
  echo "=== 8B: pulling $HF_8B ==="
  ollama pull "$HF_8B"
  make_tag "$HF_8B" "$TAG_8B" 16384
}

pull_32b () {
  echo "=== 32B: pulling $HF_32B (~19.8 GB; Q4_K_M is the GPU-fit quant) ==="
  ollama pull "$HF_32B"
  # 32B at Q4 leaves ~28 GB of the 48 GB GPU for KV cache; 16k ctx is safe.
  make_tag "$HF_32B" "$TAG_32B" 16384
}

case "$WHICH" in
  8b)   pull_8b ;;
  32b)  pull_32b ;;
  both) pull_8b; pull_32b ;;
  *) echo "usage: $0 [8b|32b|both]" >&2; exit 2 ;;
esac

echo
echo "=== Registered Goedel tags ==="
ollama list | grep -iE "goedel" || true
echo
echo "Smoke test (loads model, checks it speaks Goedel's whole-proof format):"
echo "  uv run python -m curriculum_lean.goedel_prover --model $TAG_8B"
