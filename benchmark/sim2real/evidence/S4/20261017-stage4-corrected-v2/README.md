# Stage 4 corrected v2 evidence

This evidence belongs to the active g1 campaign `20261017-stage4-flowdt1-seed0-corrected-v5`. Large input banks, raw event sets, checkpoints and full logs remain on the server; this directory contains small receipts, independent diagnosis and a path/hash index for research review.

## Run identity

- Branch: `exp/public-sim2real-v1`
- Training source commit: `567eec285551cc3054f54a7974a4112cfbb41d97`
- Frozen simulator core: `e79e656ac00f7847b98db9e12bf43b093f3afff1` (unchanged)
- g1 run root: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5`
- Shared bank: `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/training_bank_v3_geometry_corrected_20261017_32x2s`
- Real-data scoring: disabled

## Evidence status at 2026-10-17 18:21:59 UTC

- `predata_correctness.json`: six pre-data controls passed (segmentation mapping, depth/K/pixel center geometry, trajectory initialization, CUDA warp/model, EPE controls and CUDA correlation operator). SHA256 `471c8b3d67792acc9c59d9853a847719904bd87095b686794cbc27b48f178ff1`.
- `full_preflight_pass_summary.json`: all 13 full preflight checks passed at 17:56:44 UTC, exit 0. g1 receipt SHA256 `6179a2d0db502dfcd53c75d0544f4b819f24395d8d5e670a8add6a0651a732be`.
- `first_full_preflight_failed_summary.json`: preserves the original false rejection and its server report/stdout/stderr hashes. The checker bug was a wrong-axis reduction under NumPy mixed boolean indexing. The threshold and data were not changed.
- `ground_flow_indexing_diagnosis.json`: independent per-pixel recount of 24 fixed ground cases; correct max `4.76837158203125e-7 px/30ms`, legacy wrong-axis maximum `1.7672651665634476e-5`.
- `pilot_update20_verification.json`: both arms saved update-20 checkpoints, all first 20 losses and gradients finite, 70 finite gradient tensors per update, matching paired sample IDs and augmentation sequences. Receipt SHA256 `fb83abe65816e58312b722dff388c471c07e979a87200146fc689a0dd8ecedc4`.
- `paired_pilot_validation.json`: exact remote validator receipt copied into this repository; PASS for 198 optimizer updates per arm, complete eight-rollout synthetic validation, post-validation epoch-boundary checkpoints, fixed visuals and paired schedule. SHA256 `e8a7fe414b81f29e682707c1db95b9257e117ae356c0a192ef88af027a071004`.
- Synthetic validation EPE was 0.28709464 EventGenesis / 0.28606818 V2E px per 30ms; the zero-flow baseline was 0.28420805. Both models are slightly worse after the short epoch. This is a pipeline diagnostic, not a generator ranking.
- The fresh-process continuous-versus-resumed update-199 equivalence check is active on g1 GPU 2. Launcher negative controls, final full-phase gate and 45-epoch continuation remain pending. No real data is used.

The root `benchmark/sim2real/TRAINING_LIVE_STATUS.md` gives the current concise status. `artifact_index.json` records immutable inputs and large artifacts by g1 path, bytes and SHA256. The initial failed preflight is retained as historical evidence and is not overwritten by the corrected passing receipt.
