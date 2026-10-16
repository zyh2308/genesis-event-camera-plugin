# Protocol changelog

## 2026-10-16 — execution handoff

- Imported the two common research baselines and the 2026-10-16 research candidate bundle verbatim under `research/input_2026-10-16/`.
- The YAML and detailed numbers remain candidates pending engineering checks; they are not a formal freeze.
- Current execution instruction advances all three minimum pilots in parallel where feasible. This supersedes the older sequential execution order in the imported candidate notes, while preserving the research questions and fair-pairing constraints.
- Experiments must run on the research server at `/home/zhaoyuhan/Eventbased_WAM`; local WSL is used for repository preparation and review only.
- No simulator-core change is authorized or planned. Any necessary adapter/compatibility patch is confined to `benchmark/sim2real/`.
- Real data may be inspected for file format and evaluator compatibility only. No candidate-model real ranking, checkpoint selection, or simulator tuning from real-test results in this stage.

## 2026-10-16 — Stage 2 engineering closeout

- Built a six-rollout shared physical bank, paired EventGenesis/V2E task inputs, and short synthetic-only train/validation closures for E2VID+ and STE-FlowNet dt1/dt4. The Stage 1 E2VID R1/R2 review findings are repaired with boundary and label-hash evidence.
- Ran a full-frame event-only generic-cuboid SingleShotPose closure to expose the model/data contract. It does not establish an E-POSE class/CAD-compatible task and its predictions are unusable; a raw E-POSE sample/CAD remains a gate.
- HQF, MVSEC and E-POSE model evaluation was not performed. Dataset manifests remain source-level checks and evaluation rules remain candidate freeze items for the research session.
- Stage 2 report, command/environment logs, successful and failed runs, external artifact hashes and preserved Stage 1 review are indexed under `evidence/S2/20261016-stage2-pilot/`. Short pilot metrics are diagnostics only, not generator rankings.
