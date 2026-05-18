# Continual WSI

Research code for continual whole-slide image learning experiments.

## Paths

- Code: /home/zjj/code/continual_wsi
- Data: /data_1_16T/data_tcga
- Outputs / intermediate files: /data_2_4T/data_zjj/continual_wsi

## Current Direction

Primary idea: spurious consolidation in continual learning.

Standard continual-learning regularization may preserve shortcut correlations as well as causal mechanisms. WSI pathology provides shortcut sources such as stain, scanner/site artifacts, and tissue-preparation style.

## First Validation

1. Audit available TCGA features and labels.
2. Build a minimal feature-level MIL/classifier benchmark from pre-extracted CONCH features.
3. Test controlled shortcut reversal before implementing full causal-selective regularization.
## Environment

See docs/environment.md for server paths, conda environments, storage policy, and package mirror notes.

