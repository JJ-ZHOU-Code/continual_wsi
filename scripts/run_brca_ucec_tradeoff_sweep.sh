#!/usr/bin/env bash
set -euo pipefail

cd ~/code/continual_wsi

ROOT=/data_2_4T/data_zjj/continual_wsi/conch_text_concepts
INDEX=/data_2_4T/data_zjj/continual_wsi/indices/multicancer_conch_s1024.csv
TEXT_CACHE=$ROOT/conch_text_concept_embeddings.pt
OUT_ROOT=$ROOT/brca_ucec_tradeoff_sweep_20260519
PY=/home/zjj/miniconda3/envs/clam/bin/python
export PYTHONPATH=/home/zjj/code/VLSA/model
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUT_ROOT"

for l2 in 0.5 1 2; do
  for subspace in 0.5 1 2; do
    for anti in 0.25 1; do
      for seed in 7 11; do
        out="$OUT_ROOT/l2${l2}_sub${subspace}_anti${anti}_seed${seed}"
        echo "RUN l2=$l2 subspace=$subspace anti=$anti seed=$seed"
        "$PY" scripts/conch_text_concept_drift.py \
          --index-csv "$INDEX" \
          --positive-label 1 \
          --negative-label 7 \
          --max-per-label 120 \
          --n-major-per-cell 35 \
          --n-minor-per-cell 15 \
          --n-test-per-cell 10 \
          --epochs-task1 60 \
          --epochs-task2 60 \
          --anti-threshold 0.01 \
          --score-power 2 \
          --relevance-power 3 \
          --fallback-strength 0.2 \
          --fallback-tau 0.05 \
          --l2-lambda "$l2" \
          --anti-penalty "$anti" \
          --subspace-lambda "$subspace" \
          --seed "$seed" \
          --out-dir "$out" \
          --text-cache "$TEXT_CACHE" \
          >"$out.log" 2>&1
      done
    done
  done
done
