# Formal quality benchmark (v1 draft)

This directory is the benchmark harness for the frozen v3.1 plugin at
`e79e656`. It is intentionally separate from the simulator core. Phase 4.5
has now frozen the protocol and completed the final pre-run checks:

1. audit existing real/simulation assets;
2. freeze the protocol and configuration hashes;
3. verify metric schemas and mathematics;
4. smoke-test baseline availability and provenance;
5. attempt the official ESIM build without substituting another algorithm.

It does **not** run the full benchmark, tune Ours, or create a composite score.
Formal execution remains blocked by the explicit confirmation gate in
`protocol_v1.yaml`.

## Files

- `../docs/quality_benchmark_asset_audit.md`: evidence boundary and asset table.
- `protocol_v1.yaml`: pre-registered protocol candidate.
- `baselines/BASELINE_VERSIONS.md`: source/version/configuration ledger.
- `baselines/ESIM_BUILD.md`: exact official build attempt and failure.
- `baselines/esim_wrapper.py`: fail-closed official-output adapter only.
- `configs/`: frozen per-method configuration files and SHA-256 ledger.
- `metrics/event_statistics.py`: fixed distributional metrics and HDF5 loaders.
- `scripts/metric_smoke_test.py`: schema and metric self-check.
- `scripts/baseline_smoke_test.py`: existing-artifact and baseline-readiness check.
- `scripts/esim_smoke_test.py`: ESIM provenance and adapter-contract smoke test.
- `scripts/run_benchmark.py`: protocol-only, fail-closed formal entry point.
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
`results/tables/PHASE4_5_STATUS.md` for the final pre-run decision.
