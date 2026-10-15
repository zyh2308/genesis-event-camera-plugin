# Protocol changelog

## 2026-10-16 — execution handoff

- Imported the two common research baselines and the 2026-10-16 research candidate bundle verbatim under `research/input_2026-10-16/`.
- The YAML and detailed numbers remain candidates pending engineering checks; they are not a formal freeze.
- Current execution instruction advances all three minimum pilots in parallel where feasible. This supersedes the older sequential execution order in the imported candidate notes, while preserving the research questions and fair-pairing constraints.
- Experiments must run on the research server at `/home/zhaoyuhan/Eventbased_WAM`; local WSL is used for repository preparation and review only.
- No simulator-core change is authorized or planned. Any necessary adapter/compatibility patch is confined to `benchmark/sim2real/`.
- Real data may be inspected for file format and evaluator compatibility only. No candidate-model real ranking, checkpoint selection, or simulator tuning from real-test results in this stage.
