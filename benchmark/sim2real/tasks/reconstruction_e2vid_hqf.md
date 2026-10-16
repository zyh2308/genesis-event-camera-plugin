# Reconstruction route: E2VID+ → HQF

Status: Stage 2 synthetic training and validation closed on g1. HQF source/evaluator preparation is static; no HQF files were downloaded, parsed, or scored.

## Executed

- Six whole Genesis physical rollouts were shared between EventGenesis and V2E: four train and two independent validation. E2VID inputs use the pinned five-bin HDF5 interface, 31 frames per rollout, and forward synthetic flow labels from the shared bank.
- The pinned E2VID recurrent model and public synthetic recipe were run for two epochs per generator, sequence length 3, 112 sequences per epoch and 224 optimizer updates. Both sides used seed 71 and identical initial state SHA-256 `b07667dbbfe18c1432ee543501b2686fb026d0f02c3e69eb23ebd10e4a5cffb0`.
- The official LPIPS-VGG and temporal-consistency components were recorded separately. Temporal consistency ran 224 train calls and 56 validation calls per generator. All updates had finite nonzero gradients; six recurrent state tensors were present after a sequence. Both checkpoints restored and reproduced the saved model-state hash.
- R1 is repaired: three-frame sequences invoke temporal consistency during training. R2 is repaired: flow labels and the common geometry validity mask are loaded independently of event count and observed event span. Fixtures cover empty events, two sparse events, and three equal-time events; all three preserve the same flow-label hash and 64 valid pixels.
- The shared geometric fixture projects a +0.25 m x translation at 1 m depth and `fx=2` to +0.5 px. The E2VID warp fixture also records lower unmasked image-warp L1 for the forward label (0.00476) than for its reversal (0.05238). The upstream temporal loss uses a flow-based visibility mask, so its scalar alone is not used as the direction test; that caveat and source hashes are recorded in the Stage 2 report.

## Synthetic-only diagnostics

| Generator | Train loss components (perceptual / temporal) | Synthetic validation loss components | Train wall | Peak allocated GPU |
|---|---:|---:|---:|---:|
| EventGenesis | 0.3432 / 0.0370 | 0.2069 / 0.0153 | 14.08 s | 0.69 GB |
| V2E | 0.4429 / 0.0384 | 0.2401 / 0.0232 | 15.12 s | 1.08 GB |

These short-run values verify finite loss/gradient, sequence state, and checkpoint flow. They are not generator performance estimates or real-data evidence.

## HQF readiness and protocol candidate

- The pinned EVREAL source exposes `dataset.MemMapDataset`, an HQF config and `eval.py`; the manifest lists the 14 canonical sequences and expected event/frame timestamp files. This is source inspection only.
- The current candidate uses all 14 sequences for final test, continuous recurrent inference reset at sequence start, no RGB seed frame, LPIPS-Alex primary and SSIM/PSNR/MSE supporting metrics, aggregated frame → sequence → dataset with equal sequence weight. The candidate adapter and actual files still need verification and research freeze.
- HQF is the current candidate because it is a reconstruction-specific high-quality-frame benchmark and EVREAL has a native HQF route. HS-ERGB remains an extension: it originates in the TimeLens video-interpolation benchmark, so the reconstruction target timing/alignment needs a separate definition. This is a protocol rationale, not a claim that HQF is universally better.
- The public source estimate is 19.4 GB compressed / 94 GB uncompressed. These are estimates, not measured downloads. License and all 14 sequence manifests must be checked before testing.

## Reproduction and evidence

- Candidate config: [`../configs/stage2_pilot.json`](../configs/stage2_pilot.json).
- Run command and logs: [`evidence/S2/20261016-stage2-pilot/runs/20261016-s2-e2vid-pilot-v1/`](../evidence/S2/20261016-stage2-pilot/runs/20261016-s2-e2vid-pilot-v1/).
- Converted data hashes: [`evidence/S2/20261016-stage2-pilot/data_manifests/e2vid_inputs_v2/conversion_manifest.json`](../evidence/S2/20261016-stage2-pilot/data_manifests/e2vid_inputs_v2/conversion_manifest.json).
- External checkpoint paths, sizes, and hashes: [`evidence/S2/20261016-stage2-pilot/external_artifacts.json`](../evidence/S2/20261016-stage2-pilot/external_artifacts.json).

Real HQF scoring, checkpoint selection, and generator adjustment were not performed.
