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

```bash
cd /home/zjj/code/continual_wsi
/home/zjj/miniconda3/envs/clam/bin/python scripts/train_multicancer_smoke.py \
  --max-per-class 60 \
  --epochs 60 \
  --batch-size 64 \
  --out-dir /data_2_4T/data_zjj/continual_wsi/smoke_multicancer/max60_seed7
```

Result:

- Slides: 480
- Classes: 8
- Train/test split: 384/96
- Best test accuracy: 0.8125 at epoch 20
- Final test accuracy: 0.75
- Output: /data_2_4T/data_zjj/continual_wsi/smoke_multicancer/max60_seed7/result.json

## Notes

- Feature tensor example: (5020, 512) float32 patch-level CONCH bag.
- The smoke test ran on CPU because PyTorch reported CUDA initialization failure in the SSH session, despite nvidia-smi showing A6000 GPUs. This should be debugged before heavy training.
- RCC subtype labels match features but only have 30 labeled slides, so they are suitable for smoke tests, not main experiments.

## Shortcut Reversal Smoke Test

Command:

```bash
cd /home/zjj/code/continual_wsi
/home/zjj/miniconda3/envs/clam/bin/python scripts/shortcut_reversal_smoke.py
```

Result file:

- /data_2_4T/data_zjj/continual_wsi/shortcut_reversal/max60_seed7/result.json

Setup:

- 480 cached slide-level CONCH mean embeddings from the multicancer smoke set.
- Binary label formed from the 8 cancer classes: labels 0-3 vs. labels 4-7.
- Three balanced splits: 160 task-1 slides, 160 task-2 slides, 160 held-out slides.
- A synthetic 2D shortcut is appended to the embedding. It is label-correlated in task 1 and reversed in task 2.

Key observations:

- Task-1 only model overfits the shortcut: old-correlation test accuracy is 0.9625, but reversed-correlation accuracy drops to 0.5875.
- Fine-tuning flips toward the task-2 shortcut: reversed accuracy rises to 0.9000, while old-correlation accuracy falls to 0.7437.
- Uniform L2 preservation does not solve shortcut carryover in this toy setting; it still favors the latest shortcut direction and has lower random-shortcut accuracy.
- The current CSR augmentation variant improves neutral/random shortcut robustness modestly over uniform L2: neutral 0.8625 vs. 0.8375, random 0.8062 vs. 0.7688.

Interpretation:

This is not yet a final method result. It is useful because it cleanly validates the proposed failure mode: continual objectives can preserve or overwrite spurious associations depending on environment order, and held-out shortcut interventions expose the problem better than ordinary IID accuracy. The next step is to replace the hand-coded synthetic shortcut mask with a data-driven causal/concept score and run multi-seed variants.

## Next Experiment

1. Run multi-seed shortcut reversal with varied shortcut strength and L2 coefficients.
2. Add a metric for shortcut reliance, such as accuracy gap between old-correlation and neutral/reversed interventions.
3. Implement data-driven shortcut/concept scoring so CSR does not rely on knowing the appended shortcut dimensions.
