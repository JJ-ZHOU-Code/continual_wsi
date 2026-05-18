# Server Validation

Date: 2026-05-18

## Paths

- Code entry: /home/zjj/code/continual_wsi
- Actual code storage: /data_2_4T/data_zjj/continual_wsi/code/continual_wsi
- Data: /data_1_16T/data_tcga
- Outputs: /data_2_4T/data_zjj/continual_wsi

/home is full, so ~/code/continual_wsi is a symlink to the data-disk code directory.

## Data Feasibility

The data root contains enough pre-extracted features for basic experiments.

- .pt feature-like files: 29,789
- .h5 files: 13,128
- raw .svs slides: 8,614

Built indices:

- /data_2_4T/data_zjj/continual_wsi/indices/multicancer_conch_s1024.csv
- /data_2_4T/data_zjj/continual_wsi/indices/rcc_subtype_conch_s1024.csv

The multicancer CONCH index has 6,846 slides across 8 cancer labels.

## Smoke Test

Command:

`ash
cd /home/zjj/code/continual_wsi
/home/zjj/miniconda3/envs/clam/bin/python scripts/train_multicancer_smoke.py \
  --max-per-class 60 \
  --epochs 60 \
  --batch-size 64 \
  --out-dir /data_2_4T/data_zjj/continual_wsi/smoke_multicancer/max60_seed7
`

Result:

- Slides: 480
- Classes: 8
- Train/test split: 384/96
- Best test accuracy: 0.8125 at epoch 20
- Final test accuracy: 0.75
- Output: /data_2_4T/data_zjj/continual_wsi/smoke_multicancer/max60_seed7/result.json

## Notes

- Feature tensor example: (5020, 512) float32 patch-level CONCH bag.
- The smoke test ran on CPU because PyTorch reported CUDA initialization failure in the SSH session, despite 
vidia-smi showing A6000 GPUs. This should be debugged before heavy training.
- RCC subtype labels match features but only have 30 labeled slides, so they are suitable for smoke tests, not main experiments.

## Next Experiment

Implement controlled shortcut reversal on cached slide embeddings:

1. Create synthetic shortcut dimensions correlated with labels in task 1.
2. Reverse/remove shortcut correlation in task 2.
3. Compare fine-tuning, EWC/LwF-style regularization, uniform distillation, and causal-selective regularization.
