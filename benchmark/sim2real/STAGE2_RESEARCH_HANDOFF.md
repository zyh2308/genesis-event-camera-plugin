# Stage 2 research handoff

Updated: 2026-10-16  
Branch: `exp/public-sim2real-v1`  
Evidence baseline commit: `d9a9a3f08b45ef1eb2faaabddc8513c361d25a4a`  
Frozen simulator core: `e79e656ac00f7847b98db9e12bf43b093f3afff1` (unchanged)

This file is the compact handoff for the research review. It summarizes what the Stage 2 evidence supports and links to the full reports, configs, run logs, manifests, and external artifact hashes. It is not a real-data result report: no HQF, MVSEC, or E-POSE candidate model was scored on real data.

## Decision summary

- **Reconstruction (E2VID+ → HQF):** the synthetic training path and the Stage 1 R1/R2 fixes passed targeted checks. HQF data and the candidate real evaluator have not been exercised. Keep HQF as a protocol candidate pending review of its sequence inventory, license, adapter, and metric aggregation.
- **Optical flow (STE-FlowNet → MVSEC):** paired synthetic pilots completed for dt=1 and dt=4, including checkpoint restoration and correlation-operator checks. The two generators' synthetic EPE values are nearly tied; this pilot gives no generator ranking. MVSEC data and the real support-mask protocol remain unchecked or unfrozen.
- **Object pose (SingleShotPose → E-POSE):** the event-only training/checkpoint interface ran, but validation keypoint errors remain unusably large and the synthetic cuboid has no matched E-POSE CAD/class. Pause expansion at the raw-sample/CAD contract check. Current E-POSE retrieval responses blocked access in this environment; they do not establish global dataset unavailability.
- **Scientific conclusion:** engineering readiness improved for reconstruction and flow. There is no evidence yet for or against EventGenesis sim-to-real superiority. Pose is not a viable result at this stage.

## What was actually run

All pilots ran on g1, Ubuntu 20.04.6, conda `Eventbased_WAM`, RTX 3090 GPU 1 (24 GB). Six complete 0.3 s Genesis rollouts at 100 Hz formed one shared bank: four whole rollouts for training and two for independent synthetic validation. EventGenesis and V2E used the same physical frames, times, labels, rollout split, seed-paired initialization, model, and update budget. Inference inputs were event-only.

The bank is at `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage2/20261016-s2-pilot-v1/shared/shared_trajectory.h5` (26,526,701 bytes; SHA-256 `68011b77d292ca5600fb10b3464af26c958c785d2258e8a8cfea0db35b15763d`). Its camera matrix at 160×128 is `[[137.24844291261175,0,80],[0,137.24844291261175,64],[0,0,1]]`. Time is seconds; world coordinates are Genesis z-up in meters; camera coordinates are OpenCV x-right/y-down/z-forward. Flow labels are forward source-to-target displacement. Common validity does not depend on event count/span, but this small pilot does not include a target-occlusion z-buffer.

### Reconstruction

- Pinned E2VID recurrent model, five bins, Adam `1e-4` with AMSGrad, LPIPS-VGG and temporal consistency; sequence length 3, two epochs, 224 updates and 56 validation sequences per generator.
- R1: temporal loss made 224 training and 56 validation calls per generator. Updates had finite nonzero gradients, recurrent state had six tensors, and restored checkpoints matched their saved state hashes.
- R2: labels and common geometry validity were independent of observed event span. Empty-event, sparse-event, and tied-timestamp fixtures retained the same flow-label hash and 64 valid pixels.
- Synthetic validation perceptual/temporal components: EventGenesis `0.2069 / 0.0153`; V2E `0.2401 / 0.0232`. These short-run loss values verify optimization behavior only; they do not measure reconstruction quality or real generalization.
- Direction check: projected geometry gives positive forward flow; an unmasked image warp gives L1 `0.004762` for forward flow and `0.052381` for reversed flow. The upstream temporal loss can mask incorrect correspondences, so its scalar alone is not a reliable direction check.

### Optical flow

| Interval | EventGenesis synthetic validation EPE | V2E synthetic validation EPE | Peak allocated GPU |
|---|---:|---:|---:|
| dt=1 | 0.8301 px | 0.8082 px | 3.63 GB |
| dt=4 | 2.8537 px | 2.8379 px | 12.56 GB |

Each generator/task ran two epochs and 16 updates. Losses and gradients were finite; all four checkpoints restored to matching hashes. A benchmark-local pure-PyTorch correlation implementation matched an explicit 81-shift reference (max forward error `7.63e-6`, max gradient error `1.49e-8`). The EPE differences are small for this pilot and do not support ranking. dt=4 memory was measured at 160×128; MVSEC is 346×260, so target-resolution memory still needs a pilot.

### Object pose

- Full-frame input used only three event-derived channels (ON, OFF, total log-count); no RGB, GT crop/mask/box, segmented events, depth, segmentation, or GT pose entered inference.
- The generic cuboid pilot used paired initialization and 16 updates. Its validation keypoint EPE remained about 158,581 px for EventGenesis and 59,141 px for V2E. These values are unusable and are not an E-POSE score or generator comparison.
- The E-POSE paper's annotation fields were source-checked, but no raw H5 or matching CAD was parsed. Timestamp unit/order/polarity, K convention, pose transform direction/axes/units, synchronization, CAD scale, and class mapping remain unverified. One sample plus its CAD is the next gate.

## Validity limits and protocol issues

1. The generator pipelines do not receive identically represented images: EventGenesis consumes linear radiance; V2E receives Rec.709 luminance followed by Reinhard and sRGB uint8 mapping. These settings are not calibrated to a real sensor. Any future difference is a pipeline comparison until this input-mapping issue is explicitly handled in the frozen protocol.
2. Six short rollouts at 160×128 are enough for interface checks, not broad scene or motion coverage. Synthetic validation uses held-out physical rollouts from this same small bank.
3. No HQF, MVSEC, or E-POSE files were used for training, validation, checkpoint selection, simulator tuning, or scoring. There are no real scores.
4. Real evaluator protocols remain candidates. HQF needs its complete inventory/license, candidate adapter, and aggregation frozen. MVSEC needs its primary support mask, zero-GT and empty-frame rules frozen. E-POSE needs a raw-sample/CAD audit and a class-matched event-only model/label contract.

## Recommended next work

1. Research review freezes the HQF and MVSEC evaluation manifests and the shared generator/input-mapping protocol.
2. Expand synthetic-only rollout coverage for E2VID and STE-FlowNet under an agreed training budget; first measure STE dt=4 at MVSEC target resolution.
3. Keep pose at the one-sample-plus-CAD contract gate. Diagnose the very large synthetic keypoint error before deciding whether to continue with SingleShotPose or request a model/task change.
4. Begin real scoring only after protocol freeze and synthetic-only checkpoint selection. Preserve the real sets as final test data.

## Evidence and reproduction

- Full execution report: [`reports/stage2_execution.md`](reports/stage2_execution.md)
- Current status and handoff state: [`sim2real_status.md`](sim2real_status.md)
- Route reports: [E2VID/HQF](tasks/reconstruction_e2vid_hqf.md), [STE-FlowNet/MVSEC](tasks/flow_steflownet_mvsec.md), [SingleShotPose/E-POSE](tasks/pose_yolo6d_epose.md)
- Evidence package, including successful and failed attempts: [`evidence/S2/20261016-stage2-pilot/README.md`](evidence/S2/20261016-stage2-pilot/README.md)
- External g1 dataset/checkpoint paths, byte sizes, and SHA-256 values: [`evidence/S2/20261016-stage2-pilot/external_artifacts.json`](evidence/S2/20261016-stage2-pilot/external_artifacts.json)
- Per-file evidence hashes: [`evidence/S2/20261016-stage2-pilot/artifact_index.json`](evidence/S2/20261016-stage2-pilot/artifact_index.json)

Reproduction wrappers are listed in the evidence README. Exact commands, package/source hashes, start/end times, exit codes, preflight records, stdout logs, configs, and manifests are archived under `evidence/S2/20261016-stage2-pilot/`. The scripts refuse to overwrite existing run/data paths.
