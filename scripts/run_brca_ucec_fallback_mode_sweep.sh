#!/usr/bin/env bash
set -euo pipefail

cd ~/code/continual_wsi

ROOT=/data_2_4T/data_zjj/continual_wsi/conch_text_concepts
INDEX=/data_2_4T/data_zjj/continual_wsi/indices/multicancer_conch_s1024.csv
TEXT_CACHE=$ROOT/conch_text_concept_embeddings.pt
OUT_ROOT=$ROOT/brca_ucec_fallback_mode_sweep_20260519
PY=/home/zjj/miniconda3/envs/clam/bin/python
export PYTHONPATH=/home/zjj/code/VLSA/model
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUT_ROOT"

for mode in task1 task2 max shift; do
  for strength in 0.1 0.2 0.4; do
    for seed in 7 11; do
      out="$OUT_ROOT/mode${mode}_str${strength}_seed${seed}"
      echo "RUN mode=$mode strength=$strength seed=$seed"
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
        --fallback-mode "$mode" \
        --fallback-strength "$strength" \
        --fallback-tau 0.05 \
        --l2-lambda 2 \
        --anti-penalty 1 \
        --subspace-lambda 2 \
        --seed "$seed" \
        --out-dir "$out" \
        --text-cache "$TEXT_CACHE" \
        >"$out.log" 2>&1
    done
  done
done
