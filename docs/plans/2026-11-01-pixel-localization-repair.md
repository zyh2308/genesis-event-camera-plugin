# Pixel-localization repair implementation plan

**Goal:** Accept changes by lower target-centre pixel localization error on fixed event inputs, then confirm on new episodes; do not optimize event similarity as a substitute.

**Architecture:** Keep the scene, camera, inference permissions, five-bin representation, model architecture, controller, and sensor response fixed. Extend the existing evaluator with calibrated image-coordinate errors and explicit missing-output failures. Reuse completed training as a development candidate, distinguish dense observed-HDR input from the unrepaired sparse reconstruction path, and audit every comparison against the archived world-coordinate scores.

**Stack:** Python, NumPy, existing CPU PyTorch environment on g1, immutable trajectory/event banks, existing plugin candidate branch. No shared environment or main-branch edits.

## Tasks

1. Add a pure-NumPy world-to-pixel scorer and test known camera projections, invalid depth, perfect predictions, and missing-state denominators.
2. Add pixel coordinates, Euclidean pixel error, and diagnostic failure thresholds (2/5/10/20 pixels, all polls denominator) to the existing common-input evaluator. Keep its historical world-coordinate fields unchanged.
3. Evaluate old Ours, coverage-only Ours, the completed dense-observed candidate, and fixed V2E on the same nine development episodes / 581 polls. Verify input/checkpoint hashes and exact reproduction of historical world-coordinate errors. Export every query and episode; do not select examples to score.
4. Inspect model peaks versus projected GT after inference. If the completed candidate improves pixel localization, treat it as a functional dense-input candidate, not proof that sparse reconstruction is fixed or that a generator ranking has been established.
5. Confirm the selected candidate on fresh trajectories and/or an untouched existing episode bank, without using those labels to train or select a checkpoint. Rendering requires a genuinely available and registered GPU; use g3 if g1 is occupied. If unavailable, finish CPU audits and publish a precise pending validation item without fabricating new data.
6. Publish code, configuration, per-query metrics, input manifests, checkpoint lineage, and a short Chinese report. Preserve old core/tag and old results.

## Acceptance

- Pixel error uses projected physical target centre at the causal cutoff, in the original 320x240 image coordinates; never compare against another generator's predicted location.
- Every method receives identical test events and query times. GT is used only for training labels and post-inference scoring/overlays.
- Report mean, median, P95, missing outputs, and >10px errors, including per-episode breakdown. Thresholds are diagnostics, not retroactively chosen grasp success criteria.
- Lower image/event residuals alone cannot satisfy acceptance. New-episode confirmation is separate from already inspected development results.
- Do not change calibrated sensor thresholds/noise, target markers, evaluation labels, or controller settings to reduce the metric.
