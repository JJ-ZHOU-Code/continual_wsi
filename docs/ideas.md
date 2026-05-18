# Research Ideas

## 1. Spurious Consolidation in Continual Learning

Standard continual-learning regularization may preserve old shortcuts as strongly as causal mechanisms. In WSI, shortcuts include stain intensity, scanner/site artifacts, tissue folds, and hospital-specific sampling. The first go/no-go experiment is a controlled shortcut reversal benchmark using pre-extracted WSI features.

## 2. Taxonomy-Evolving Continual WSI Learning

Pathology label ontologies evolve: coarse labels split into fine subtypes, while historical slides may remain coarse-labeled. A continual WSI learner should preserve backward-compatible coarse predictions while learning new fine-grained subtype labels.

## 3. Supervision-Incremental WSI Learning

Clinical supervision evolves from slide labels to sparse ROI labels, masks, molecular labels, and reports. The model should absorb richer supervision without forgetting weak-supervision performance.

## 4. Concept-Rationale Drift in Continual WSI MIL

Existing WSI CL methods evaluate prediction and localization forgetting, but may miss drift in the pathology concepts used to justify old predictions.

## 5. Privacy-Preserving Concept Footprint Replay

Replace patch-feature replay with compact concept/prototype footprints to reduce memory and privacy risk in continual WSI learning.

## Current Recommendation

Start with Idea 1 if a controlled shortcut-carryover signal can be shown quickly. Use Idea 2 as backup because RCC subtype labels and multi-cancer CONCH features are already present.
