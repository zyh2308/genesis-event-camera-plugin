# Public-data Sim-to-Real benchmark

This workspace compares EventGenesis and V2E event streams generated from the same synthetic physical trajectories. Downstream models train and validate only on paired synthetic corpora. Public real-event datasets remain test-only. The Stage 7 handoff records completed fixed-protocol HQF testing, synthetic-only flow GT-fit diagnostics, and the E-POSE keypoint label audit; real scores were not used for checkpoint selection or tuning. See [current status](sim2real_status.md) and [Stage 7 handoff](STAGE7_EXECUTION_UPDATE_20261029.md).

The simulator core is frozen at `e79e656ac00f7847b98db9e12bf43b093f3afff1`. Benchmark adapters and compatibility fixes belong here and must not change that core. Historical quality-benchmark smoke results are not evidence for this branch.

## Current candidate routes

- Reconstruction: E2VID+ → HQF. The fixed all-14 HQF test has completed for the two seed-0 synthetic-best candidates; scores are report-only. HS-ERGB remains an extension.
- Optical flow: STE-FlowNet → MVSEC; the 45-epoch synthetic-only pair is retained, and a separate high-texture GT-fit pilot exposed weak independent-rollout generalization.
- Object pose: SingleShotPose → E-POSE. The raw sample and matching CAD are available; official 2D keypoint labels have a confirmed shared-depth geometry error and must be rebuilt from CAD + pose + K before training/evaluation.

Each route keeps its native event representation, but EventGenesis and V2E get identical physical trajectories, labels, rollout split, paired initialization, architecture, training schedule and synthetic validation. E-POSE GT masks, crops, boxes, segmented events, RGB, depth, segmentation and GT pose are never model inputs.

## Historical Stage 3 reference

The following is a historical snapshot. For current results see [Stage 7 handoff](STAGE7_EXECUTION_UPDATE_20261029.md).

E2VID+ reused the accepted synthetic checkpoints, parsed the preregistered HQF `bike_bay_hdr` sequence, and ran both full-sequence event-only candidate inferences. STE-FlowNet's Stage 2 EPE bug was corrected; all four existing checkpoints were re-scored on the complete synthetic validation set, and all four ran on the preregistered MVSEC `indoor_flying1` fixed fragment at dt1/dt4. Stage 2 pose windows were time-misaligned; the corrected paired pilot and confidence/BatchNorm diagnostics are recorded. E-POSE raw sample/CAD access remains blocked by current publisher WAF responses.

Read [`sim2real_status.md`](sim2real_status.md) and [`STAGE7_EXECUTION_UPDATE_20261029.md`](STAGE7_EXECUTION_UPDATE_20261029.md) for the current status. [`STAGE3_RESEARCH_HANDOFF.md`](STAGE3_RESEARCH_HANDOFF.md) and [`reports/stage3_execution.md`](reports/stage3_execution.md) are historical records. Stage 2 historical records remain under [`evidence/S2/20261016-stage2-pilot/`](evidence/S2/20261016-stage2-pilot/); its original EPE and pose figures are marked invalid in the report after the Stage 2 review. Stage 3 commands, environments, exit statuses, failed attempts, public file manifests, and checked reconstruction frames are under [`evidence/S3/20261016-stage3/`](evidence/S3/20261016-stage3/). Large public data, checkpoints and MVSEC per-window prediction bundles remain outside Git; see `external_artifacts_stage3.json` for their paths, sizes and hashes.

Stage 4 synthetic training configuration and paired seed-0 STE-FlowNet entrypoints are specified in [`STAGE4_TRAINING_START.md`](STAGE4_TRAINING_START.md). Current progress is in [`TRAINING_LIVE_STATUS.md`](TRAINING_LIVE_STATUS.md). The fixed-sample motion-learning diagnostic, four-stratum results, current interpretation, and reproduction details are in [`STAGE4_MOTION_DIAGNOSTIC_20261018.md`](STAGE4_MOTION_DIAGNOSTIC_20261018.md), with machine-readable evidence under [`evidence/S4/20261018-motion-diagnostic/`](evidence/S4/20261018-motion-diagnostic/). Full banks, raw events, growing logs, and checkpoints remain on g1; their paths and snapshot hashes are recorded in the evidence receipt/index.

The compact Stage 2 handoff is retained at [`STAGE2_RESEARCH_HANDOFF.md`](STAGE2_RESEARCH_HANDOFF.md) with corrections added. Research inputs and their original paths are versioned under `research/input_2026-10-16/`.

Research inputs and their original paths are versioned under `research/input_2026-10-16/`. Every report distinguishes source inspection, executed work and estimates. Short synthetic pilot values do not establish generator performance or real generalization.
