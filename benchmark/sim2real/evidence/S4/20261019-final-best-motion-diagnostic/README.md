# Corrected-v2 synthetic-best optical-flow diagnostic

Completed on g1 2026-10-18 17:17:15–17:18:46 UTC / 2026-10-19 02:17:15–02:18:46 JST. This is fixed synthetic-validation diagnosis only; no real dataset was opened, and it did not affect training or checkpoint selection.

## Run contract

- 64 fixed windows: 8 held-out physical rollouts × 8 registered adapter rows. The retained epoch-23 diagnostic and plan remain untouched.
- EventGenesis uses its independent synthetic-best epoch 33 / update 6,534 checkpoint; V2E uses epoch 20 / update 3,960. The checkpoint pair differs by epoch/update as expected under independent synthetic-validation selection; both share the same initial-state SHA-256 `acc6bbd11488a21868c5f147c7f397521480166c24914cc7988a6428d6ed71cf`.
- Four common strata and four regions: full physical validity, common EventGenesis∪V2E event-active pixels intersected with validity, and GT motion thresholds 0.5 / 1.0 px per 30 ms. One event-union mask is used for both predictors.
- Inference uses event-only inputs. Output stays `[u,v]` in native pixel displacement per 30 ms. The run checked paired sample IDs, GT, validity, gray endpoints, timestamps, K and label hashes; all matched, and every interval was 30 ms.
- g1 environment: Python 3.11.15, PyTorch 2.5.1+cu121, CUDA 12.1, Tesla V100S-PCIE-32GB, physical GPU 3. Elapsed time 90.901 seconds, exit code 0.

## Result

| Region pooled across four strata | EventGenesis EPE vs zero flow | V2E EPE vs zero flow | Coverage / reading |
|---|---:|---:|---|
| Full physical valid region | 0.276812 vs 0.277367 | 0.276169 vs 0.277367 | Improvements 0.20% / 0.43%; 98.6% / 98.5% of predictions are below 0.1 px. |
| Shared event-active union | 0.302791 vs 0.352044 | 0.315132 vs 0.352044 | Improvements 14.0% / 10.5% across 27,751 pixels, just 0.4866% of full valid pixels. |
| GT motion ≥0.5 px/30 ms | 0.816378 vs 0.821715 | 0.818644 vs 0.821715 | 17.43% of full-valid pixels; under 0.7% pooled improvement. |
| GT motion ≥1.0 px/30 ms | 1.245190 vs 1.255078 | 1.249889 vs 1.255078 | 4.27% of full-valid pixels; under 0.8% pooled improvement. |

On event-active support, three scene strata improve against zero flow. Independent rigid-object motion with a static camera regresses for both arms: EventGenesis 0.3451 vs 0.2805; V2E 0.3032 vs 0.2805. This stratum has only 0.15% event support. Both models remain near zero over most full-valid and GT-motion-threshold pixels. Evidence supports weak/partial event-supported motion learning, not robust dense flow or a generator ranking. No timing, labels, vector unit, direction, decoder, resize or crop error was found in these checks.

## Initial adapter failure and correction

The first execution exited 1 before inference because the fixed epoch-23 loader required checkpoints marked `resume_phase=train` and `resume_allowed=true`. The registered `synthetic_best.pt` checkpoints are intentionally `inference` / false. That failure and stdout are preserved as `attempt1_*`. The final adapter independently validates that inference-only lifecycle plus each checkpoint's own epoch, update, best score, scheduler, initial state and model-state hash, then restores the pinned STE model in ordinary eval mode. The fixed samples, base diagnostic script and plan hashes did not change. The corrected execution exited 0 and loaded both checkpoints.

## Reproduction

The fixed diagnostic self-test is:

```bash
python benchmark/sim2real/scripts/stage4_final_best_motion_diagnostic.py \
  --self-test --config benchmark/sim2real/configs/stage4_final_best_motion_diagnostic_20261019.json
```

The g1 inference command was:

```bash
CUDA_VISIBLE_DEVICES=3 python benchmark/sim2real/scripts/stage4_final_best_motion_diagnostic.py \
  --config benchmark/sim2real/configs/stage4_final_best_motion_diagnostic_20261019.json
```

The run log, timing, exit code and output JSON are in this folder. The output JSON records both g1 checkpoint paths, sizes and SHA-256 values. Script and config hashes are in `artifact_index.json`.

## Interpretation boundary

This is a preregistered, fixed-sample synthetic diagnosis. The event-active support is sparse, and the GT-threshold regions show that model magnitudes remain very small. It explains how full-image zeros can dominate a diagnostic, but does not justify replacing the primary selection metric or changing the fixed training protocol. Real MVSEC scores remain unauthorized.
