# Formal quality benchmark (v2 pre-formal protocol)

This directory is the benchmark harness for the frozen v3.1 plugin at
`e79e656`. It is intentionally separate from the simulator core. Phase 4.6
has completed the protocol-completion checks but has not run formal metrics:

1. preserve the immutable v1 registration and freeze a v2 execution protocol;
2. verify the benchmark HEAD may advance while the simulator core is unchanged;
3. freeze real/simulation pair definitions and equal-duration windows;
4. fix the V2E numerical input cadence from the official IIR guard;
5. smoke-test schemas, cadence, windows, and baseline provenance.

It does **not** run the full benchmark, tune Ours, or create a composite score.
Formal execution remains blocked by the explicit confirmation gate in
`protocol_v2.yaml`; the approval file authorizes preparation only.

## Files

- `../docs/quality_benchmark_asset_audit.md`: evidence boundary and asset table.
- `protocol_v1.yaml`: immutable pre-result registration record.
- `protocol_v2.yaml`: execution-complete pre-formal protocol.
- `FORMAL_RUN_APPROVAL.yaml`: independent preparation approval record.
- `evaluation_pairs_v2.yaml`: frozen LR/RL real-to-replay pair manifest.
- `replays/checkerboard_translation_v2.yaml`: formal replay contract.
- `protocol_v2.sha256`, `evaluation_pairs_v2.sha256`: integrity sidecars.
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
- `results/raw/`: machine-readable smoke outputs; not formal benchmark results.
- `results/tables/`, `results/visualizations/`: reserved for protocol-approved runs.

## Run the Phase 1–4 smoke checks

From the repository root:

```bash
PYTHONPATH=. python benchmark/scripts/metric_smoke_test.py
PYTHONPATH=. python benchmark/scripts/baseline_smoke_test.py
```

The second command deliberately reports ESIM and ICNS/IEBCS as unavailable when
their official executables are not present. It validates the existing V2E
artifact without launching a new full replay. It also checks the retained
Metavision HDF5 metadata and records the current missing vendor ECF filter if
the compressed event payload cannot be opened.

## Evidence boundary

The current assets support descriptive event-rate, polarity, active-pixel, and
fixed-grid spatial comparisons under the frozen protocol. Per-bin temporal
error and strict real–sim CD/GD remain disabled because the EVK4 dynamic stream
and Genesis replay do not share a measured trajectory and time reference. See
`PROTOCOL_CHANGELOG.md` for the v1/v2 boundary and
`results/tables/PHASE4_5_STATUS.md` for the historical Phase 4.5 decision.
