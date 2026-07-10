#!/usr/bin/env bash
# Close-the-loop ablation driver (CPU-only; the Markov GPU build is untouched).
# Per book: A0 random-region seeds -> A1/A0 curriculum builds -> Track-B materialize
# -> co-primary metrics. Pilot one book, then fan out.
#   bash artifacts_scripts/curriculum/run_close_loop.sh run_book \
#        cyberbullying artifacts/book_gaps/W2898513995_cyberbullying_ai.json \
#        artifacts/book_gaps/W2898513995_toc.json 8
set -uo pipefail
cd /home/gchurchill/src/bookgap
PY="uv run python"
export CUDA_VISIBLE_DEVICES=""        # never contend for the GPU
export TOKENIZERS_PARALLELISM=false

run_book () {
  local slug="$1" gap="$2" toc="$3" nchap="$4"
  local A1="artifacts/curriculum/cl_A1_${slug}"
  local B="artifacts/curriculum/cl_B_${slug}"
  echo "=== [$slug] n_chap=$nchap gap=$(basename "$gap") $(date -u +%H:%M:%SZ) ==="

  echo "--- [$slug] (1/5) A0 random-region seeds"
  $PY -m artifacts_scripts.curriculum.random_region_gap \
      --like-gap "$gap" --n-seeds 3 --seed 0 \
      --out-prefix "artifacts/book_gaps/cl_A0/${slug}_rand" || { echo "FAIL seeds"; return 1; }

  echo "--- [$slug] (2/5) A1 build (detected gap)  $(date -u +%H:%M:%SZ)"
  $PY -m artifacts_scripts.curriculum.book_curriculum \
      --gap-json "$gap" --desired-n-topics "$nchap" \
      --output-dir "cl_A1_${slug}" --no-plot || { echo "FAIL A1"; return 1; }

  echo "--- [$slug] (3/5) A0 builds (random regions)  $(date -u +%H:%M:%SZ)"
  for s in 0 1 2; do
    $PY -m artifacts_scripts.curriculum.book_curriculum \
        --gap-json "artifacts/book_gaps/cl_A0/${slug}_rand_${s}.json" \
        --desired-n-topics "$nchap" --output-dir "cl_A0_${slug}_${s}" --no-plot \
        || { echo "FAIL A0_$s"; return 1; }
  done

  echo "--- [$slug] (4/5) Track-B materialize + A1 absolute reconstruction  $(date -u +%H:%M:%SZ)"
  $PY -m artifacts_scripts.curriculum.book_toc_reconstruction \
      --gap-json "$gap" --toc-json "$toc" \
      --curriculum-dir "$A1" --materialize-dir "$B" \
      --out "artifacts/book_gaps/cl_${slug}_recon.json" || { echo "FAIL trackB"; return 1; }

  echo "--- [$slug] (5/5) co-primary metrics  $(date -u +%H:%M:%SZ)"
  $PY -m artifacts_scripts.curriculum.close_loop_metrics \
      --gap-json "$gap" --toc-json "$toc" \
      --run-a1 "$A1" --run-b "$B" \
      --runs-a0 "artifacts/curriculum/cl_A0_${slug}_0" \
                "artifacts/curriculum/cl_A0_${slug}_1" \
                "artifacts/curriculum/cl_A0_${slug}_2" \
      --out "artifacts/book_gaps/cl_${slug}_metrics.json" || { echo "FAIL metrics"; return 1; }

  echo "=== [$slug] DONE $(date -u +%H:%M:%SZ) ==="
}

"$@"
