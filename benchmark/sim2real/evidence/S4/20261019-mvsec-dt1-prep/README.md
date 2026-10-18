# MVSEC dt1 evaluation preparation (candidate)

Checked: 2026-10-19 (Asia/Tokyo). This folder contains a metric-contract proof and a pre-registration draft only. **No real MVSEC scores were computed.** The protocol and manifest remain candidates for research review.

## Current repository-side flow artifact audit

- `TRAINING_LIVE_STATUS.md` and `STAGE4_MOTION_DIAGNOSTIC_20261018.md` are an explicit 2026-10-18 02:01 UTC historical live snapshot. They record 4,900/8,910 EventGenesis and 4,950/8,910 V2E updates and must not be repeated as today's live state.
- The fixed motion diagnostic uses epoch 23 / update 4,554 checkpoints, not corrected-v2 synthetic-best epoch 33 / 20. The requested final-best common-event-active diagnostic is still outstanding.
- `STAGE4_SEED0_TRAINING_REPORT.md` describes a distinct earlier v4 run at commit `1fab7e17e201943c77703aa8a1c0c6eaefbb8f5a`; its approximately 0.94 px/30ms validation values and artifact hashes do not index the corrected-v2 run.
- No current corrected-v2 full 45-epoch curves, final/best checkpoint hashes, or terminal exit receipts are present in this local checkout. Those must be read from the current g1 run before the final archive is claimed. This audit worker did not connect to g1 and did not change global status pages.

## Pinned implementation findings

- The source is `ruizhao26/STE-FlowNet` commit `eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6`; the repository's existing `run_mvsec_candidate_inference.py` reproduces its timestamp propagation helper and its local test compares outputs against the pinned helper.
- `multiscaleloss.py::estimate_corresponding_gt_flow` treats every flow map as displacement to the next GT timestamp and propagates asynchronous maps to exact APS endpoints by nearest-neighbor flow sampling. Flow is endpoint displacement in pixels, not velocity.
- `main_steflow_dt1.py` uses `i > 20` and `i + 20 < N` as its ordinary sequence loader boundary. `outdoor_day1` validation uses the half-open APS index ranges `[9200,9600)` and `[10500,10900)`. `flow_error_dense(..., is_car=True)` keeps the top 190 rows for `outdoor_day1`; other listed sequences use all 260 rows.
- The pinned helper reports mean EPE and a standalone fraction with `EPE < 3 px`. The new candidate's Outlier metric is deliberately separate: `EPE > 3 px AND EPE > 0.05 * |GT|`. The old threshold fraction is retained only as a compatibility diagnostic.
- The candidate support is one shared mask: finite, nonzero GT intersected with active pixels in identical real event representations from both candidate arms, after the registered sequence crop. An empty support has undefined metrics and is preserved in the common exclusion list.

The exact candidate decisions and required review points are in [`mvsec_dt1_protocol_candidate_20261019.json`](../../../protocol/mvsec_dt1_protocol_candidate_20261019.json) and [`mvsec_dt1_test_manifest_candidate_20261019.json`](../../../manifests/mvsec_dt1_test_manifest_candidate_20261019.json). Sequence choices match the checked-in research baseline: `indoor_flying1/2/3` and `outdoor_day1`. The Stage 3 `indoor_flying1` one-second fragment is an interface smoke, not this final test; its old candidate outputs are explicitly excluded from the manifest.

## Controlled validation

Run from the repository root:

```bash
python benchmark/sim2real/scripts/stage4_mvsec_dt1_metrics.py --self-test
```

The self-test constructs a 260x346 analytic flow field and checks EPE values `[0,2,4,4]`, AEE `2.5 px/interval`, candidate Outlier `25%`, and legacy standalone `EPE >= 3` fraction `50%`. It also verifies zero-flow GT exclusion, non-finite GT exclusion, common event support, outdoor top-190-row crop, arm-input mismatch rejection, component-broadcast rejection, and empty-support handling. The evaluator reports all-valid-GT AEE separately as an optional diagnostic. It opens no real dataset, model, or checkpoint.

Captured self-test output: [`controlled_selftest.json`](controlled_selftest.json), exit code 0. The attempted candidate CLI invocation with no approval record exited 1 at the protocol status check and created no score output; see [`negative_real_score_gate.log`](negative_real_score_gate.log) and [`negative_real_score_gate.exit_code.txt`](negative_real_score_gate.exit_code.txt).

File sizes and SHA-256 values for this preparation bundle are indexed in [`artifact_index.json`](artifact_index.json).

Any real scoring command requires a frozen protocol, frozen exact test manifest, and a research authorization record containing the exact SHA-256 digests of both. The current candidate protocol and manifest explicitly set `real_scores_authorized=false`; no authorization record is present. The gate therefore rejects real scoring before it opens any per-window prediction artifact.

## Not yet resolved

- G1 paths, sizes, and SHA-256 hashes of all four full MVSEC sequence/GT files and calibration archive must be refreshed from actual server files; this worker performed no SSH/data operations.
- Exact full-sequence eligible APS index lists and timestamp ranges must be resolved from those file hashes before checkpoint loading. The candidate manifest currently has empty registered-index arrays by design.
- Corrected-v2 synthetic-best checkpoint paths, sizes, and hashes must be filled from the live g1 artifact audit. Expected epochs (EventGenesis 33, V2E 20) are from the supplied task baseline; Stage 3 fragment outputs are not reusable for the final model.
- Research must freeze the outdoor range/crop, support mask, empty-window aggregation, and equal-sequence weighting before real scoring.

## Provenance

Protocol source files: pinned `multiscaleloss.py`, `main_steflow_dt1.py`, and the checked-in baseline candidate sequence list. Reproduction artifacts are generated by the script above; captured output and the negative authorization-gate result are stored next to this README.
