# Public-data Sim-to-Real benchmark

This workspace compares EventGenesis and V2E event streams generated from the same synthetic physical trajectories. Downstream models train and validate only on paired synthetic corpora. Public real-event datasets remain final-test data: Stage 3 runs used preregistered fixed samples only to verify file/time alignment, event-only candidate inference, output shapes, and per-window evaluation plumbing. No real aggregate ranking or checkpoint selection was performed.

The simulator core is frozen at `e79e656ac00f7847b98db9e12bf43b093f3afff1`. Benchmark adapters and compatibility fixes belong here and must not change that core. Historical quality-benchmark smoke results are not evidence for this branch.

## Candidate routes

- Reconstruction: E2VID recurrent architecture / E2VID+ public synthetic-training recipe → HQF. HS-ERGB is an optional extension.
- Optical flow: STE-FlowNet → MVSEC; primary route.
- Object pose: YOLO-6D / SingleShotPose → E-POSE; currently gated on a sample/CAD contract and an event-only model interface.

Each route keeps its native event representation, but EventGenesis and V2E get identical physical trajectories, labels, rollout split, paired initialization, architecture, training schedule and synthetic validation. E-POSE GT masks, crops, boxes, segmented events, RGB, depth, segmentation and GT pose are never model inputs.

## Stage 3 status

E2VID+ reused the accepted synthetic checkpoints, parsed the preregistered HQF `bike_bay_hdr` sequence, and ran both full-sequence event-only candidate inferences. STE-FlowNet's Stage 2 EPE bug was corrected; all four existing checkpoints were re-scored on the complete synthetic validation set, and all four ran on the preregistered MVSEC `indoor_flying1` fixed fragment at dt1/dt4. Stage 2 pose windows were time-misaligned; the corrected paired pilot and confidence/BatchNorm diagnostics are recorded. E-POSE raw sample/CAD access remains blocked by current publisher WAF responses.

Read [`sim2real_status.md`](sim2real_status.md) and [`STAGE3_RESEARCH_HANDOFF.md`](STAGE3_RESEARCH_HANDOFF.md) for the current handoff, and [`reports/stage3_execution.md`](reports/stage3_execution.md) for the detailed results and costs. Stage 2 historical records remain under [`evidence/S2/20261016-stage2-pilot/`](evidence/S2/20261016-stage2-pilot/); its original EPE and pose figures are marked invalid in the report after the Stage 2 review. Stage 3 commands, environments, exit statuses, failed attempts, public file manifests, and checked reconstruction frames are under [`evidence/S3/20261016-stage3/`](evidence/S3/20261016-stage3/). Large public data, checkpoints and MVSEC per-window prediction bundles remain outside Git; see `external_artifacts_stage3.json` for their paths, sizes and hashes.

The compact Stage 2 handoff is retained at [`STAGE2_RESEARCH_HANDOFF.md`](STAGE2_RESEARCH_HANDOFF.md) with corrections added. Research inputs and their original paths are versioned under `research/input_2026-10-16/`.

Research inputs and their original paths are versioned under `research/input_2026-10-16/`. Every report distinguishes source inspection, executed work and estimates. Short synthetic pilot values do not establish generator performance or real generalization.
