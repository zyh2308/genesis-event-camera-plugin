# Stage 4 corrected v2 live status

Last remote check: `2026-10-17T18:08:25Z`. The continuation process remains attached to tmux on g1; both paired pilot processes were alive at the last process check.

## Run identity and guardrails

- Campaign ID: `20261017-stage4-flowdt1-seed0-corrected-v5`
- Training source commit: `567eec285551cc3054f54a7974a4112cfbb41d97`
- Branch: `exp/public-sim2real-v1`
- Frozen core: `e79e656ac00f7847b98db9e12bf43b093f3afff1` (unchanged)
- Tmux: `stage4-flow-corrected-20261017-v6`; continuation PID `3538277`
- Pilot PIDs: EventGenesis `3540102` on GPU 2; V2E `3540103` on GPU 4
- Synthetic-only training; no real scoring; no flow labels in the optimizer; event-only model input.

## Completed

1. The shared v3 corrected bank contains 32/32 physical rollouts (HDF5 1,767,229,910 bytes; SHA256 `cca882ba912fda3b3243d49bc7e29e28186046e3b6591c424f385b0506047884`). V2E and both paired adapters are complete. The adapter manifest is SHA256 `1f9aff5164c71301a7406c9981c31ef0d5df381df81641cf2a1d397e8df246b1`.
2. The initial full preflight failed closed on one checker-only reduction bug. The legacy failure report is preserved; independent analysis isolated NumPy mixed-indexing from the data itself. Correcting the per-pixel norm implementation leaves the threshold and data unchanged.
3. The corrected full preflight passed 13/13 at `17:56:44Z` (exit 0), fingerprint `e147c8d8d3ca00f4b227c18228ceeb141dd522510df86e8bb31775d31c539930`. It checked renderer/link mapping, depth/K/pixel centers, trajectory initialization, CUDA model/loss warp and correlation operator, EPE controls, geometry, 4,224 independently recounted event windows, paired labels/adapters, edge cases, input fingerprints, frozen core and synthetic-only guard.
4. The update-20 verifier loaded each arm's checkpoint and verified optimizer state/position, finite losses and gradients, identical initialization/config/data hashes, identical update-20 sample IDs and augmentation sequences, and no real data. At `18:08:25Z`, logs had advanced to EventGenesis 91 and V2E 93 updates.

## Original checker failure and independent repair evidence

`flow[index, :, mask]` returns shape `(N,2)` under NumPy mixed advanced indexing. The old `norm(axis=0)` therefore accumulated over pixels instead of taking one vector norm per pixel. The corrected helper asserts the `(2,H,W)` / `(H,W)` contracts and computes each selected vector's two-dimensional norm. A hand control demonstrates old max `19.621416` versus correct max `17`; all 24 fixed static-ground bank cases have correct max `4.76837158203125e-7 px/30ms`, below the unchanged `1e-5` threshold. This was a preflight implementation error; the labels were not altered.

## Active training and remaining steps

The paired short pilot is registered for 198 updates/arm. Update-20 checkpoints are stable at 580,038,560 bytes each:

- EventGenesis SHA256 `9461d34a175afaae248b7bdd445f979b72d514fba76316842649b53996a3ef49`
- V2E SHA256 `f36e7ce1b4551409071afa744c3e17db0d91dcf7eb1cc27a09c14b5d846e9411`

At update 20, losses were 778.8129 / 823.3140; both are short-run optimization telemetry only. The next stages are paired 198-step synthetic validation, process exit/resume equivalence at updates 198→199, negative controls through the production launcher, final full-phase gate verification, then autonomous resume into the registered 45-epoch budget. No long-run continuation starts unless every required gate passes.

## Machine evidence and locations

Small receipts and index: `benchmark/sim2real/evidence/S4/20261017-stage4-corrected-v2/`.

Full preflight receipt on g1: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/preflight_result_corrected_v2.json` (1,422,918 bytes, SHA256 `6179a2d0db502dfcd53c75d0544f4b819f24395d8d5e670a8add6a0651a732be`).

Pilot metrics: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/pilot/{eventgenesis,v2e}/training_metrics.jsonl`.

Exact command and environment records, checkpoints, original failure receipt, logs and manifests are indexed by absolute path/size/hash where available in `artifact_index.json`. The old v4 root training status is preserved byte-for-byte under `evidence/S4/20261016-stage4-flowdt1-seed0-v4-update20/`.
