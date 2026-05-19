#!/usr/bin/env bash
set -euo pipefail

cd ~/code/continual_wsi

ROOT=/data_2_4T/data_zjj/continual_wsi/conch_text_concepts
INDEX=/data_2_4T/data_zjj/continual_wsi/indices/multicancer_conch_s1024.csv
TEXT_CACHE=$ROOT/conch_text_concept_embeddings.pt
PY=/home/zjj/miniconda3/envs/clam/bin/python
export PYTHONPATH=/home/zjj/code/VLSA/model
export CUDA_VISIBLE_DEVICES=0

run_variant() {
  local name="$1"
  local fallback_strength="$2"
  local out_root="$ROOT/$name"
  mkdir -p "$out_root"
  for seed in 7 11 13 17 19; do
    local out="$out_root/seed${seed}"
    echo "RUN variant=$name seed=$seed fallback=$fallback_strength"
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
      --fallback-strength "$fallback_strength" \
      --fallback-tau 0.05 \
      --l2-lambda 2 \
      --anti-penalty 1 \
      --seed "$seed" \
      --out-dir "$out" \
      --text-cache "$TEXT_CACHE" \
      >"$out.log" 2>&1
  done
}

run_variant brca_ucec_no_fallback_5seed_20260519 0
run_variant brca_ucec_fallback_5seed_20260519 0.2
