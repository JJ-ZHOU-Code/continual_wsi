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

## Automated Shortcut Sweep

Command:

```bash
cd /home/zjj/code/continual_wsi
scripts/launch_shortcut_sweep.sh auto_20260518_round1
```

Output:

- /data_2_4T/data_zjj/continual_wsi/shortcut_sweeps/auto_20260518_round1
- /data_2_4T/data_zjj/continual_wsi/shortcut_sweeps/auto_20260518_round1/summary.csv
- /data_2_4T/data_zjj/continual_wsi/shortcut_sweeps/auto_20260518_round1/aggregate.json

Grid:

- Seeds: 7, 11, 13, 17, 19
- Shortcut strengths: 2, 4, 6, 8
- L2 coefficients: 20, 80
- Shortcut penalties: 0.0, 0.1
- Completed cells: 80 / 80

Aggregate observations:

- `task1_only` has a large old-vs-reversed shortcut gap: old correlation 0.9278 vs. reversed 0.6041, gap +0.3237.
- `finetune` adapts to the reversed environment but does not preserve old shortcut behavior: old correlation 0.7725 vs. reversed 0.8284.
- `l2_all` and `selective_l2` favor the latest reversed shortcut and do not improve neutral/random robustness over fine-tuning.
- `csr_aug` slightly improves robustness over uniform L2, but not over fine-tuning: neutral/random mean 0.8000 vs. L2 0.7947 and fine-tune 0.8045.

Interpretation:

The automated sweep supports the failure-mode claim more strongly than the method claim. The next method iteration should not spend time tuning the toy shortcut penalty. It should implement streaming environment-invariant concept scoring, then test whether selective consolidation based on estimated scores improves intervention robustness beyond fine-tuning and uniform consolidation.

## Shortcut Carryover Trajectory

Command pattern:

```bash
cd /home/zjj/code/continual_wsi
/home/zjj/miniconda3/envs/clam/bin/python scripts/shortcut_trajectory_smoke.py \
  --seed 7 \
  --out-dir /data_2_4T/data_zjj/continual_wsi/shortcut_trajectory/seed7_str6_l280_pen0p1
```

Multi-seed outputs:

- /data_2_4T/data_zjj/continual_wsi/shortcut_trajectory/seed7_str6_l280_pen0p1
- /data_2_4T/data_zjj/continual_wsi/shortcut_trajectory/seed11_str6_l280_pen0p1
- /data_2_4T/data_zjj/continual_wsi/shortcut_trajectory/seed13_str6_l280_pen0p1
- /data_2_4T/data_zjj/continual_wsi/shortcut_trajectory/seed17_str6_l280_pen0p1
- /data_2_4T/data_zjj/continual_wsi/shortcut_trajectory/seed19_str6_l280_pen0p1

Five-seed trajectory means at shortcut strength 6, L2 80, penalty 0.1:

| Model | Task-2 epoch | Old corr acc | Reversed acc | Neutral/random acc | Old shortcut sensitivity | Reversed shortcut sensitivity |
|---|---:|---:|---:|---:|---:|---:|
| finetune | 0 | 0.9500 | 0.5563 | 0.7913 | 0.1391 | 0.2221 |
| finetune | 100 | 0.8363 | 0.7813 | 0.8119 | 0.0289 | 0.0310 |
| finetune | 300 | 0.7612 | 0.8300 | 0.8012 | 0.0407 | 0.0343 |
| l2_all | 0 | 0.9500 | 0.5563 | 0.7913 | 0.1391 | 0.2221 |
| l2_all | 100 | 0.8387 | 0.7687 | 0.8031 | 0.0339 | 0.0370 |
| l2_all | 300 | 0.6350 | 0.9262 | 0.7869 | 0.1564 | 0.1139 |
| csr_aug | 0 | 0.9500 | 0.5563 | 0.7913 | 0.1391 | 0.2221 |
| csr_aug | 100 | 0.8413 | 0.7675 | 0.8031 | 0.0339 | 0.0369 |
| csr_aug | 300 | 0.6625 | 0.9225 | 0.7956 | 0.1350 | 0.1010 |

Interpretation:

The trajectory reveals a non-monotonic effect hidden by endpoint tables. Around epoch 100, all methods reduce shortcut sensitivity and reach better neutral/random robustness. By epoch 300, uniform L2 and the current CSR augmentation absorb the reversed shortcut strongly, while fine-tuning has lower shortcut sensitivity. This suggests that the current toy CSR objective is not yet the right method, but the trajectory metric is the correct diagnostic for the paper's Figure 1.

## Streaming Stability Score Smoke Test

Command pattern:

```bash
cd /home/zjj/code/continual_wsi
/home/zjj/miniconda3/envs/clam/bin/python scripts/streaming_stability_smoke.py \
  --seed 7 \
  --score-power 4 \
  --shortcut-penalty 50 \
  --out-dir /data_2_4T/data_zjj/continual_wsi/streaming_stability/final_p4_pen50_seed7
```

Purpose:

This variant makes the streaming constraint explicit. Task 1 contains proxy environment labels. The shortcut is generated from environment/style, not directly from the class label; it is predictive because environment and class are correlated. At the Task-1 boundary, the method caches only per-feature stability scores computed by residual environment correlation after conditioning on class. Task 2 then reverses the environment-label correlation and uses the cached scores, without old examples.

Five-seed result at score power 4 and shortcut penalty 50:

| Model | Old corr acc | Reversed acc | Neutral/random acc | Old shortcut sensitivity | Reversed shortcut sensitivity | Shortcut weight | Causal weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| task1_only | 0.9538 | 0.6913 | 0.8419 | 0.1097 | 0.1708 | 0.2755 | 0.2998 |
| finetune | 0.8637 | 0.8825 | 0.8731 | 0.0139 | 0.0122 | 0.0511 | 0.6319 |
| l2_all | 0.8700 | 0.9675 | 0.9175 | 0.0553 | 0.0415 | 0.1141 | 0.6617 |
| oracle_score_l2 | 0.8338 | 0.9638 | 0.9087 | 0.0838 | 0.0576 | 0.1570 | 0.6141 |
| streaming_score_l2 | 0.8400 | 0.9638 | 0.9094 | 0.0792 | 0.0552 | 0.1498 | 0.6175 |
| streaming_score_anti | 0.8638 | 0.9725 | 0.9231 | 0.0700 | 0.0511 | 0.1020 | 0.4813 |
| random_score_l2 | 0.8875 | 0.9563 | 0.9169 | 0.0393 | 0.0313 | 0.0859 | 0.6774 |

Cached score sanity check:

- Raw causal-dimension stability mean: 0.9713
- Raw shortcut-dimension stability mean: 0.6119
- Powered shortcut-dimension stability mean: 0.1492

Interpretation:

This is the first positive method signal after the toy CSR failure. The streaming score correctly separates stable causal signal from environment shortcut, and `streaming_score_anti` modestly improves neutral/random robustness over uniform L2 and random-score controls. However, shortcut sensitivity is still higher than fine-tuning and random-score L2, so the current method is not yet a decisive solution. The next iteration should combine streaming score with an explicit invariant-risk or environment-adversarial term during Task 2, because selective memory alone does not fully prevent learning the new shortcut.
