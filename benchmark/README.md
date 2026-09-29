# Formal quality benchmark (v3 pre-formal readiness protocol)

This directory is the benchmark harness for the frozen v3.1 plugin at
`e79e656`. It is intentionally separate from the simulator core. Phase 4.7
is an execution-readiness gate and has not run formal metrics:

1. preserve the immutable v1/v2 registration and create a pre-result v3 revision;
2. verify the benchmark HEAD may advance while the simulator core is unchanged;
3. replace local 5 ms primary units with full-direction LR/RL comparisons;
4. freeze sequence-local time origins without method-specific offset search;
5. require complete real HDF5 reads, native Genesis HDR, Ours, and official V2E
   full-pipeline smokes before Phase 5.

It does **not** run the full benchmark, tune Ours, or create a composite score.
Formal execution remains blocked by `protocol_v3.yaml` and requires separate
human approval for Phase 5.

## Files

- `../docs/quality_benchmark_asset_audit.md`: evidence boundary and asset table.
- `protocol_v1.yaml`: immutable pre-result registration record.
- `protocol_v2.yaml`: immutable historical pre-formal protocol.
- `protocol_v3.yaml`: Phase 4.7 execution-readiness protocol.
- `FORMAL_RUN_APPROVAL.yaml`: independent preparation approval record.
- `evaluation_pairs_v2.yaml`: immutable historical LR/RL pair manifest.
- `evaluation_pairs_v3.yaml`: full-direction LR/RL readiness pair manifest.
- `replays/checkerboard_translation_v2.yaml`: formal replay contract.
- `replays/checkerboard_translation_v3.yaml`: executable checkerboard replay definition.
- `protocol_v2.sha256`, `evaluation_pairs_v2.sha256`: integrity sidecars.
- `protocol_v3.sha256`, `evaluation_pairs_v3.sha256`: integrity sidecars.
- `baselines/BASELINE_VERSIONS.md`: source/version/configuration ledger.
- `baselines/ESIM_BUILD.md`: exact official build attempt and failure.
- `baselines/esim_wrapper.py`: fail-closed official-output adapter only.
- `configs/`: frozen per-method configuration files and SHA-256 ledger.
- `metrics/event_statistics.py`: fixed distributional metrics and HDF5 loaders.
- `scripts/metric_smoke_test.py`: schema and metric self-check.
- `scripts/baseline_smoke_test.py`: existing-artifact and baseline-readiness check.
- `scripts/esim_smoke_test.py`: ESIM provenance and adapter-contract smoke test.
- `scripts/run_benchmark.py`: protocol-only, fail-closed formal entry point.
- `scripts/v2e_cadence_smoke.py`: official V2E IIR cadence contract smoke.
- `scripts/window_smoke_test.py`: fixed equal-duration window smoke.
- `scripts/real_timestamp_origin_smoke.py`: complete real LR/RL loader and origin smoke.
- `scripts/genesis_replay_smoke.py`: native HDR/depth/segmentation/pose/Ours smoke.
- `scripts/v2e_official_pipeline_smoke.py`: official SuperSloMo-to-EventEmulator smoke.
- `results/raw/`: machine-readable smoke outputs; not formal benchmark results.
- `results/tables/`, `results/visualizations/`: reserved for protocol-approved runs.

## Run the Phase 4.7 readiness gate

From the repository root:

```bash
PYTHONPATH=. python benchmark/scripts/run_benchmark.py --preflight
```

The gate first checks v1/v2 immutability, v3/pair hashes, clean git, and frozen
core integrity. It then requires complete EVK4 LR/RL HDF5 decoding, a native
Genesis HDR replay with the frozen Ours core, and the official V2E image-folder
path with SuperSloMo enabled. A missing vendor ECF filter, Genesis runtime, or
SuperSloMo checkpoint is a STOP condition. No formal metric is computed.

## Evidence boundary

The current assets support descriptive event-rate, polarity, active-pixel, and
fixed-grid spatial comparisons under the frozen protocol. Per-bin temporal
error and strict real–sim CD/GD remain disabled because the EVK4 dynamic stream
and Genesis replay do not share a measured trajectory and time reference. See
`PROTOCOL_CHANGELOG.md` for the v1/v2 boundary and
`results/tables/PHASE4_5_STATUS.md` for the historical Phase 4.5 decision.
