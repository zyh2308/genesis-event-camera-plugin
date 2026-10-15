# Public-data Sim-to-Real benchmark

This workspace compares EventGenesis and V2E event streams generated from the same synthetic physical trajectories. Downstream models train and validate only on paired synthetic corpora. Public real-event datasets are held out for final testing; this stage does not run candidate-model scoring on real test data.

The simulator core is frozen at `e79e656ac00f7847b98db9e12bf43b093f3afff1`. Benchmark adapters and compatibility fixes belong here and must not change that core. Historical quality-benchmark smoke results are not evidence for this branch.

## Candidate routes

- Reconstruction: E2VID recurrent architecture / E2VID+ public synthetic-training recipe → HQF. HS-ERGB is an optional extension.
- Optical flow: STE-FlowNet → MVSEC; primary route.
- Object pose: YOLO-6D / SingleShotPose multi-object → E-POSE; optional if data and adapter gates pass.

Each route keeps its native event representation, but EventGenesis and V2E get identical trajectory IDs, labels, split, model initialization, training schedule, and synthetic validation. E-POSE GT masks, crops, and segmented events are never model inputs.

## Evidence labels

Every report separates `source checked`, `actually run`, and `estimated`. Real dataset schema checks are not model scoring. A pilot does not count as a formal result.

Research inputs and their original paths are versioned under `research/input_2026-10-16/`. Current machine-readable state is in `sim2real_status.md`; evidence and reports are linked there.
