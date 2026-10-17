# Stage 4 corrected v2: live training status

Updated: `2026-10-17T18:08:25Z` (last remote metric records). The paired synthetic pilot is running on g1. It is a pipeline check, not a generator-performance result.

## Run

- Campaign: `20261017-stage4-flowdt1-seed0-corrected-v5`
- Training source commit: `567eec285551cc3054f54a7974a4112cfbb41d97`, branch `exp/public-sim2real-v1`
- Frozen simulator core: `e79e656ac00f7847b98db9e12bf43b093f3afff1` (unchanged)
- Method: STE-FlowNet, dt1, seed 0, EventGenesis/V2E paired; 32 shared physical rollouts, 1,584 synthetic train and 528 synthetic validation windows per arm; registered budget 45 epochs × 198 updates/epoch.
- Supervisor: tmux `stage4-flow-corrected-20261017-v6`, continuation PID `3538277`; pilot PIDs `3540102` and `3540103`.
- g1 GPUs: EventGenesis on GPU 2, V2E on GPU 4. Both were verified idle before launch. Peak allocated memory observed was 24,580,958,208 bytes per arm.
- Real-data scoring is disabled. Flow labels are not consumed by the optimizer; inference input is five-bin events only.

## Correctness gate

The corrected full preflight ran `2026-10-17T17:54:14Z`–`17:56:44Z`, exit 0, 13/13 PASS. Fingerprint SHA256: `e147c8d8d3ca00f4b227c18228ceeb141dd522510df86e8bb31775d31c539930`. Machine-readable summary: [`evidence/S4/20261017-stage4-corrected-v2/full_preflight_pass_summary.json`](evidence/S4/20261017-stage4-corrected-v2/full_preflight_pass_summary.json).

The earlier preflight failure remains preserved. It came from a NumPy mixed-indexing shape change: the checker reduced the `(N,2)` selection across pixels instead of computing one 2D norm per pixel. The 24-case independent recount gives max static-ground flow `4.76837158203125e-7 px/30ms` under the unchanged `1e-5` threshold. A hand fixture reproduces the legacy wrong-axis maximum `19.621416` versus the correct per-pixel maximum `17`. No bank or labels were changed to clear the gate. See `first_full_preflight_failed_summary.json` and `ground_flow_indexing_diagnosis.json` in the evidence folder.

## Training observed

The independently verified update-20 receipt confirms both arms performed 20 optimizer updates, saved resumable checkpoints, had finite losses and gradients, and used identical seed, initialization, config, paired sample IDs and augmentation seeds. At the latest remote read (`18:08:25Z`), EventGenesis had logged update 91 and V2E update 93; both training processes were alive. Pilot logs and checkpoints remain on g1.

| Arm | Update-20 loss | Update-20 checkpoint SHA256 |
|---|---:|---|
| EventGenesis | 778.8129 | `9461d34a175afaae248b7bdd445f979b72d514fba76316842649b53996a3ef49` |
| V2E | 823.3140 | `f36e7ce1b4551409071afa744c3e17db0d91dcf7eb1cc27a09c14b5d846e9411` |

The two short-pilot losses are telemetry only and do not support a generator ranking. The saved checkpoints are 580,038,560 bytes each. Their paths are `.../pilot/eventgenesis/update_000020.pt` and `.../pilot/v2e/update_000020.pt` under the run root below. Verification receipt SHA256: `fb83abe65816e58312b722dff388c471c07e979a87200146fc689a0dd8ecedc4`.

## Still running / next gates

The supervised continuation is capped at 198 pilot updates per arm. It will then run synthetic validation, process-exit/resume equivalence across the epoch boundary, production-launcher negative controls, and a final full gate check. Only if all pass will it resume from the pilot checkpoint into the remaining registered 45-epoch budget. No real dataset is used for checkpoint selection or ranking.

## Paths and reproduction

- g1 run root: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5`
- Live logs: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/pilot/{eventgenesis,v2e}/training_metrics.jsonl`
- Corrected preflight: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/preflight_result_corrected_v2.json`
- Resume command used: `bash benchmark/sim2real/scripts/continue_stage4_corrected_v2_after_data_server.sh --continue-after-data`
- Immutable input manifests, command records, full logs and large checkpoints are indexed in [`evidence/S4/20261017-stage4-corrected-v2/artifact_index.json`](evidence/S4/20261017-stage4-corrected-v2/artifact_index.json).

The previously completed v4 status is preserved byte-for-byte in `evidence/S4/20261016-stage4-flowdt1-seed0-v4-update20/TRAINING_LIVE_STATUS_final_v4.md` (SHA256 `3c393bc3adcc5a2ee4de6208f1790a7d9f9a103daba03b1a0bb2a494cfb7b13e`).
