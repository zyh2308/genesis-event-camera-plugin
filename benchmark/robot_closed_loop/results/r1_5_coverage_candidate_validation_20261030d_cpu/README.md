# R1.5 coverage-splat candidate validation

## Result

The diagnostic front-to-back coverage rule passes the current CPU NumPy/Torch parity check and the broader deterministic controls. This supports the proposed explanation for the tiny-positive-weight z-buffer failure; it is still a diagnostic candidate, not a production core change.

On identical projected coordinates, the candidate NumPy and Torch implementations differ by at most `6.94e-5` luminance, with no visibility-mask mismatches. When each backend uses its own projected coordinates, the maximum candidate difference is `7.45e-5` in episode 004 and `6.93e-4` in episode 005; no pixel-time differs by more than `1e-3`, `0.2`, or in visibility.

The native interpolator outputs are much less stable. For the same saved states, the active NumPy and Torch paths differ by as much as `0.949` / `1.064` luminance, with P99 absolute differences `0.0407` / `0.0830` and 61 / 81 visibility-mask disagreements across the ten samples. The projection maps themselves differ by at most `1.07e-4 px` and `3.58e-7 m`, with no source-validity disagreements. A `7.63e-6` or smaller bilinear contribution can therefore change which surface wins the old z-buffer.

One `+55 mm` endpoint example at `(266,10)`, `t=0.5905 s`, makes the rule visible: NumPy selects a contribution of weight `9.54e-7` at `z=0.586602 m`, producing `1.083` luminance; Torch selects a different near contribution of weight `7.63e-6` at `z=0.585714 m`, producing `0.01854`. Each tiny contribution displaces a much larger sample. The candidate gives the two backend-native reconstructions a maximum difference below `0.001` luminance at that scale.

## General controls

- **Moving tilted continuous surface:** a linearly shaded plane with smoothly varying depth was translated from `-0.5` to `+0.5 px`. Candidate output matches the analytic bilinear image within `6e-8` in the interior. A `+1e-6 px` shift changes the legacy center luminance by `0.003`; the NumPy candidate is unchanged at float precision and the Torch candidate changes by `3e-8`. NumPy/Torch candidate outputs differ by at most `3e-8`, with identical visibility.
- **Moving foreground over background:** a `0.8` foreground patch over `0.02` background was moved through `±0.5 px`. At `+0.25 px`, the two edge values are `0.605` and `0.215`, matching the expected `0.75/0.25` and `0.25/0.75` coverages. Foreground and background interiors remain `0.8` and `0.02`. NumPy/Torch candidate outputs match exactly in this control.
- **Real cached clips:** the candidate removes the previously reported 92.5% / 84.4% of pixel-times with absolute log-luminance error above `0.2`.

## Remaining cached residuals

Candidate residuals above `0.2` log error remain at 665 pixel-times in episode 004 and 1,377 in episode 005. None are unsupported/hold pixels; only 3 / 5 fall within the two-pixel image border. Of these residuals, 293 / 691 are within a one-pixel band around a saved segmentation boundary.

Another 274 / 370 residual pixel-times occur inside stable segmentation-ID-1 regions, away from all segmentation boundaries. In those regions, saved Direct-HDR luminance changes by more than `0.2` log at 89 / 125 pixels over this 10 ms interval; 274 / 368 interior residual pixel-times occur at pixels with that level of temporal change. This is consistent with a changing radiance component on otherwise stable image regions, such as a moving shadow, but the current comparison does not identify the physical cause. The other residuals include target and robot labels. Saved Direct-HDR images are exact renders at logged states; they are not physical ground truth between those states.

This residual profile explains why the coverage candidate is not yet a complete image reconstruction. A future causal lighting/shadow ablation or a true intermediate render would be needed to attribute the stable-region component.

## Provenance and reproduction

Run from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmark/robot_closed_loop/diagnostics/validate_r1_5_coverage_candidate.py \
  --cache-root benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059 \
  --output benchmark/robot_closed_loop/results/r1_5_coverage_candidate_validation_UNIQUE_ID
```

- Code commit: `724b894f47cac40ee37fee9648fb344ff4591295`.
- Validation script SHA256: `c88388da22f2f96495864d08043c4ac4084d258a4e7b4db8c87c04445c27932b`.
- Base pixel-attribution script SHA256: `a41661a7467d848095e644bde347bb0a2acaa392e063c0b6d01c5ce7048ea724`.
- Runtime: WSL CPU, Python `3.14.6`, NumPy `2.5.2`, Torch `2.13.0+cu130`; CUDA was not used.
- Imported `physics_interpolator.py` SHA256 `3870f984b603c69c7dab2bc1a26a8d35218135af2ad139e50b115257777f09ef`; `physics_warp.py` `a7efbbf1e339084ae9c92e5252257a059aaeac4ff1f8269c1c245330bd51b3b7`; `torch_physics.py` `af6e2e39dc4b124b32274dca028e6b2f02617ebdb1e1fb05e27f3af913567d1f`. Absolute imported paths and hashes are in `summary.json`.
- Episode 004 source/frame cache SHA256: `0525c18c8d89f5831ef0cb71ac9e3b6b130d60579daf374f06d5cfcee5b879d7` / `b3afe5b5badee3d2606d343bb750aa970887928c41b8001a5fc4e9ff32b9c321`.
- Episode 005 source/frame cache SHA256: `91c1397642e317d6bb51150b4107c518b52e4fa9fcf5f1caf6a2a91315c7d434` / `f161079a24c1bf9fee838a6fbb2a0fa5d785435b116f0ea5628e6efbba497236`.
- `summary.json` holds the per-frame backend/projection metrics, runtime module paths, residual categories, and top source-weight traces. `backend_projection_pixel_traces.csv` records the largest NumPy/Torch disagreements; `residual_top20_per_frame.csv` records representative remaining errors.
- g1 SSH succeeded for a read-only check. Two RTX 3090s were occupied; other cards had no explicit project assignment, so no GPU job was started. No remote code or files were changed.
- No renderer, physics, model inference, training, or frozen-core modification was used.

## Decision

The main cached seam is now tied to a general numerical rule: positive bilinear taps take part in the depth test before their area is considered. The coverage candidate passes the synthetic plane/occlusion controls and suppresses the NumPy/Torch seam on the two cached episodes. Keep it diagnostic until the residual radiance changes are causally separated, the actual CUDA backend is checked under an assigned GPU, and broader episode replay confirms the same behavior. Do not use event-count agreement alone as a fidelity claim.
