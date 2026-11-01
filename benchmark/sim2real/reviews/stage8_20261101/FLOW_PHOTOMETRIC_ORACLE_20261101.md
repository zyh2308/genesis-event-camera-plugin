# Optical flow improvement diagnostic: photometric oracle

**Status time:** 2026-11-01 11:14 JST (2026-11-01 02:14 UTC)
**Repository:** `exp/public-sim2real-v1`; diagnostic registered from source commit `db74ee225dc494fe06e53bb603027cd6aad1d847`, with final launcher/config registration pushed in `31d96cd76fe0993f90b33fca3eddcfcc15feffdb`.
**Frozen simulator core:** `e79e656ac00f7847b98db9e12bf43b093f3afff1` (unchanged).
**Primary run:** g1 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-stage8-flow-photo-oracle-run06-v1`, exit code `0`; GPU reservation was released at `2026-11-01T02:11:44Z`.

## Question and scope

This run asks whether the exact STE-FlowNet multi-scale photometric objective prefers the shared synthetic GT flow over zero or reversed flow on fixed textured-motion validation windows. It uses the pinned `objective_dt1`, the recorded align-corners compatibility patch, both normal event tensors, and the existing held-out ReplicaCAD layout family. It performs **no optimizer updates**, creates no model checkpoint, and does not use real data or scores.

The model is forwarded once from a fresh random initialization only to read its four scale shapes. The objective comparison then supplies the same physical candidate flow and radiance endpoints to both event arms. This is an input/label/warp oracle; the pinned photometric operator does not itself use event values, so this does not show that either generator's events contain enough information for the model to recover motion.

## Result

The fixed region was registered in the config before execution: shared physical flow validity, GT magnitude at least `0.5 px / 30 ms`, luma gradient above `1/255` at both radiance endpoints, and in-bounds warps for all three candidates. The official scalar loss remains unmasked, as in the existing training objective; regional MAE is reported separately.

| Held-out rollout | Shared valid pixels | Event-active valid pixels (EG ∪ V2E) | Valid GT-motion + texture pixels | GT photo loss | Zero-flow photo loss | Reverse-flow photo loss | GT vs zero reduction | Regional MAE: GT / zero / reverse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v3_sc2_x_060mm` | 74,402 | 7,661 (10.30%) | 26,246 (35.28%) | 522.892 | 1,034.850 | 1,409.030 | 49.5% | 0.00539 / 0.03778 / 0.05685 |
| `v3_sc2_y_040mm` | 75,191 | 6,024 (8.01%) | 25,729 (34.22%) | 476.588 | 801.214 | 1,076.447 | 40.5% | 0.00604 / 0.02948 / 0.04543 |

All candidates remained in bounds on both rows. The correct GT warp also had the lowest regional absolute luma error in both rows. At the coarsest 44×44 scale on the Y rollout, GT was slightly worse than zero (`35.790` vs `34.721`); GT was lower at the other three scales, so the four-scale mean still favored GT by 40.5%. The second-order smoothness penalty did not reverse the comparison: `photo + 10 × smoothness` was `524.339` vs `1,034.850` on X and `477.595` vs `801.214` on Y.

## What this confirms

- On these two fixed windows, the shared 30 ms flow label has the right direction and approximate pixel displacement for the saved scene-linear radiance endpoints under the actual STE warp. A gross endpoint, sign, or scale mismatch is unlikely to explain the earlier near-zero/over-large network predictions on these rows.
- A deliberately wrong direction is substantially worse than the shared GT, and the correct flow improves the official four-scale photometric loss over zero. The photometric target is therefore capable of rewarding motion on these examples.
- The normal paired event inputs are sparse over the valid image area: their union is active on 10.30% and 8.01% of valid pixels on X and Y. The GT-motion/texture region itself covers about one third of valid pixels. These are separate areas; this run did not calculate their exact intersection. Sparse event support remains a plausible limitation, but these counts alone do not diagnose either generator as defective.
- The source bundle verified all 14 STE-FlowNet files against pinned commit `eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6`. The registered memory-bounded PyTorch correlation fallback passed the target shape `[1,64,128,128]` comparison (forward max absolute error `5.72e-6`, small-shape backward max gradient error `1.12e-8`). The model produced the expected scale shapes `[352,352]`, `[176,176]`, `[88,88]`, `[44,44]`.

## Existing paired supervised-fit evidence

To avoid repeating an already completed diagnostic, this handoff also uses the Stage 7 paired GT-supervised run on the same bank manifest (`995aac5c…`), with the same four training rollouts, two independent `sc2` validation rollouts, STE-FlowNet, paired seed-0 initialization, and 100 updates per arm. The independent post-run audit confirmed both initialized-state hashes matched, both saved state hashes matched the loaded state hashes, both optimizer counters restored at 100, and logged validation predictions and EPE recomputed exactly.

| Arm | Held-out path | Full-valid EPE / zero-flow EPE | Common event-active EPE / zero-flow EPE |
|---|---|---:|---:|
| EventGenesis | sc2-X | 2.297 / 3.500 | 1.733 / 2.674 |
| EventGenesis | sc2-Y | 1.769 / 2.361 | 1.775 / 1.666 |
| V2E | sc2-X | 2.371 / 3.500 | 1.814 / 2.674 |
| V2E | sc2-Y | 1.949 / 2.361 | 1.901 / 1.666 |

This is limited positive evidence that STE-FlowNet can learn useful motion from the synthetic event representation on some independent regions. It is not uniformly better than zero flow on the event-active Y subset and does not establish generator ranking or sim-to-real performance. The training driver exited `1` only while serializing a CUDA UUID into its final JSON; the saved checkpoints and metrics were independently verified by the post-run audit. Original logs and weights remain preserved at the g1 paths recorded in that audit.

## Interpretation and next action

The current evidence points away from a basic GT direction/warp mismatch on the fixed windows. It does **not** establish that the self-supervised training signal is adequate, or that the issue is uniquely the model, renderer, or generator. Sparse activity, limited layout/motion coverage, and the self-supervised optimization remain live explanations. The GT-supervised pilot separates those questions only partially: the network improves on held-out full-valid regions, while one held-out active subset still favors zero flow.

Recommended next action: expand the synthetic training/validation bank using pre-registered richer layouts and camera paths, then rerun the same bounded paired self-supervised diagnostic. Freeze geometry and labels before event generation, keep both event arms paired, preserve the current synthetic-only rule, and judge motion response on the same shared masks. Do not start another long self-supervised run or use MVSEC scores to guide the scene or checkpoint choice yet.

## Reproduction and evidence

- Registered configuration: [`stage7_gt_zero_reverse_photometric_oracle_v1.json`](../../configs/stage7_gt_zero_reverse_photometric_oracle_v1.json), SHA256 `3ec4af7b30ffc82561ae1a64c413add2e9c772e176d89664f80bdeecbc0b09f5`.
- Diagnostic code: [`audit_stage7_gt_zero_reverse_photometric.py`](../../scripts/audit_stage7_gt_zero_reverse_photometric.py), SHA256 `56cf91f3c812a7102dbc025155934b5c81f2a3035e28b3bbebf68b49bee5bba5`.
- g1 run result: [`result.json`](../../evidence/20261101-flow-photo-oracle/remote_runs/20261101-stage8-flow-photo-oracle-run06-v1/output/result.json), SHA256 `f349ad5a83ebfd6f586f07e4d175d15e350c82003266ad134d5c070f01db8eae`.
- All run logs, GPU reservation checks, environment snapshot, and five failed harness attempts are preserved under [`flow oracle evidence`](../../evidence/20261101-flow-photo-oracle/).
- Hash and size index: [`EVIDENCE_INDEX.json`](../../evidence/20261101-flow-photo-oracle/EVIDENCE_INDEX.json).
- Exact source bank: g1 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1/flow_bank_retry01`, manifest SHA256 `995aac5c0163ef8efacfb1eb2599b02d05538ee8f05bc98104e4e25690522717`.
- Runtime: g1 conda env `Eventbased_WAM_torch28`, PyTorch `2.8.0+cu126`, NumPy `2.4.6`, OpenCV `5.0.0`; Tesla V100S-PCIE-32GB, GPU index 3 / UUID `GPU-253ab9b8-a0e5-4f12-0579-261818cf31dd`. Start `2026-11-01T02:11:38Z`; finish and release `2026-11-01T02:11:44Z`; exit code `0`.
- Execution snapshot used pinned helper checkout `d0e36796f04826a14a0931878405abed943a49a3`; no shared server worktree or frozen core was modified.
- Real data and real scoring: **not used**.
