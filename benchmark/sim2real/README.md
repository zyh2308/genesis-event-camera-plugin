# Public-data Sim-to-Real benchmark

This workspace compares EventGenesis and V2E event streams generated from the same synthetic physical trajectories. Downstream models train and validate only on paired synthetic corpora. Public real-event datasets are held out for final testing; Stage 2 did not run candidate-model scoring on real test data.

The simulator core is frozen at `e79e656ac00f7847b98db9e12bf43b093f3afff1`. Benchmark adapters and compatibility fixes belong here and must not change that core. Historical quality-benchmark smoke results are not evidence for this branch.

## Candidate routes

- Reconstruction: E2VID recurrent architecture / E2VID+ public synthetic-training recipe → HQF. HS-ERGB is an optional extension.
- Optical flow: STE-FlowNet → MVSEC; primary route.
- Object pose: YOLO-6D / SingleShotPose → E-POSE; currently gated on a sample/CAD contract and an event-only model interface.

Each route keeps its native event representation, but EventGenesis and V2E get identical physical trajectories, labels, rollout split, paired initialization, architecture, training schedule and synthetic validation. E-POSE GT masks, crops, boxes, segmented events, RGB, depth, segmentation and GT pose are never model inputs.

## Stage 2 status

E2VID+ completed paired multi-rollout synthetic training and independent validation; the Stage 1 R1/R2 findings are fixed with recorded evidence. STE-FlowNet completed the same for dt1 and dt4, including checkpoint restore and correlation-operator checks. The generic full-frame event-only SingleShotPose pilot also executed, but its validation predictions are unusable and it does not match E-POSE classes/CAD. Public real-data manifests and evaluator routes are source checked only; no real dataset files were evaluated.

Read [`sim2real_status.md`](sim2real_status.md) for the handoff summary and [`reports/stage2_execution.md`](reports/stage2_execution.md) for results, resource costs, protocol candidates and open decisions. All successful and failed Stage 2 commands, logs, environments, exit codes and small manifests are indexed under [`evidence/S2/20261016-stage2-pilot/`](evidence/S2/20261016-stage2-pilot/). External g1 data and checkpoint paths/sizes/hashes are in its `external_artifacts.json`.

For a single-file research handoff with conclusions, key metrics, limits, next gates and evidence links, start at [`STAGE2_RESEARCH_HANDOFF.md`](STAGE2_RESEARCH_HANDOFF.md).

Research inputs and their original paths are versioned under `research/input_2026-10-16/`. Every report distinguishes source inspection, executed work and estimates. Short synthetic pilot values do not establish generator performance or real generalization.
