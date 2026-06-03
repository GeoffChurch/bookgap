#!/bin/bash
# Generality sweep for keyword-concept seeding + clustering (de-incant experimentation, 2026-06-03).
# Per field-17 book w/ a ToC: concept-seed -> cluster via CCA-hybrid AND keyword-TF-IDF.
# Then score genericness (specificity) + foreign-fragments to see if CCA's advantage generalizes.
# Launch: .conda/bin/python -m bookgap.jobs run --prefix concept_generality -- bash artifacts_scripts/curriculum/experiment_concept_generality.sh
set -u
cd /home/gchurchill/src/bookgap
export PYTHONPATH=/home/gchurchill/src/bookgap OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=.conda/bin/python

# name  toc-stem  n_chapters
BOOKS=( "neuromorphic W4247813589 13" "color_image W4211017872 24" "mcs W4235013513 10" )

for entry in "${BOOKS[@]}"; do
  set -- $entry; name=$1; toc=$2; nch=$3
  echo "===== $name ($nch chapters) ====="
  $PY -m artifacts_scripts.curriculum.gap_keyword_concept \
    --toc-json artifacts/book_gaps/${toc}_toc.json --field 17 --backend full --per-chapter --iters 3 \
    --out artifacts/book_gaps/${name}_concept 2>&1 | grep -iE "focused|book_lift" | tail -2
  $PY -m artifacts_scripts.curriculum.cluster_concept_cca \
    --concept-gap artifacts/book_gaps/${name}_concept/concept_gap.json --field 17 --n $nch \
    --out artifacts/curriculum/${name}_cca 2>&1 | grep -iE "CCA k=|keeping k=" | tail -2
  $PY -m artifacts_scripts.curriculum.cluster_concept_by_keywords \
    --concept-gap artifacts/book_gaps/${name}_concept/concept_gap.json --field 17 --n $nch \
    --out artifacts/curriculum/${name}_kwcluster 2>&1 | grep -iE "clustered" | tail -1
done

echo; echo "===== SCORING (genericness + foreign-fragments) ====="
for entry in "${BOOKS[@]}"; do
  set -- $entry; name=$1
  for variant in cca kwcluster; do
    echo "--- $name / $variant ---"
    $PY -m artifacts_scripts.curriculum.topic_genericness \
      --run-dir artifacts/curriculum/${name}_${variant} --field 17 --top-k 15 2>&1 | tail -1
    $PY -c "
import json,numpy as np
twi=json.load(open('artifacts/curriculum/${name}_${variant}/topic_work_ids.json'))
kw=json.load(open('artifacts/curriculum/${name}_${variant}/topic_keywords.json'))
s=sorted((len(v) for v in twi.values()),reverse=True)
foreign=sum(1 for t in kw if any(w in ' '.join(k for k,_ in kw[t][:6]) for w in ['aprendizaje','resultats','akurasi','profundo','les ','dan ','utilizando','con ','dengan']))
print(f'   sizes max={max(s)} min={min(s)} n_tiny(<20)={sum(1 for x in s if x<20)} foreign={foreign}')
"
  done
done
echo "DONE generality sweep"
