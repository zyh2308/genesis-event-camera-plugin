# Phase 4.7 — Formal Execution Readiness status

This is a runtime-readiness record, not a benchmark result. No formal metric,
ranking, aggregate, temporal offset search, or Phase 5 execution was run.

## Provenance

| Item | Value |
|---|---|
| Current readiness commit | `5bb3d079b64118de4d76570ef98ca37841c5acab` |
| Runtime-gate implementation commit | `b6e965b41c65c4b6c950ddc899e126403a1bad73` |
| Check-only follow-up | `5bb3d07` fixes a false positive for the official `disable_slomo: False` log line; no V2E numerical parameter changed |
| Frozen core | `e79e656ac00f7847b98db9e12bf43b093f3afff1` |
| Phase 4.6 historical commit | `b8d5d81cee74066dfa24ad682e2820d30b35264c` |
| `protocol_v3.yaml` SHA-256 | `c6d0926147e3fb08b15892c1f9ee368c5d7380ad3326cca0b8d962ae1f78d763` |
| `evaluation_pairs_v3.yaml` SHA-256 | `86c9596bb3c6e79788e1ae5aa627319ade886a64c88571f4ccc5e7ede6fbc175` |
| Formal benchmark executed | **NO** |
| Formal metrics executed | **NO** |

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| Real LR complete loader | PASS | Vendor HDF5 ECF filter `/usr/lib/x86_64-linux-gnu/hdf5/serial/plugins`; complete `CD/events`, 23,675,387 events; first/last `6,272/1,756,258 us`; leading/trailing silence `6.272/0 ms` |
| Real RL complete loader | PASS | Same ECF filter; complete `CD/events`, 23,766,581 events; first/last `5,824/1,775,800 us`; leading/trailing silence `5.824/0 ms` |
| Real timestamp origin | PASS | `t_rel = t - declared_sequence_start`; origin is declared segment origin 0; first-event alignment disabled; all timestamps are inside the declared intervals |
| Genesis/Ours active smoke | PASS | Remote Genesis environment, LR, exactly `0.30 s`, 31 frames at 100 Hz; 0–0.15 s stationary hold retained and no event-count-based duration adaptation |
| Trajectory parity | PASS | YAML/runtime max absolute error `0.0 m`; explicit keyframes used |
| Segmentation entity resolution | PASS | `70/70` expected checkerboard square entities resolved through `segmentation_idx_dict`; coverage `1.0`; unresolved set empty |
| Background exclusion | PASS | Background segmentation ID `[1]`; motion-map/background intersection `[]`; visible background is not used as a moving object |
| Native HDR and independent display RGB | PASS | Native radiance `float32`, max `7.66015625`; depth/segmentation shapes `[48,64]`; display RGB archived separately; no radiance clipping fallback |
| Ours smoke output | PASS | Frozen core `e79e656`; smoke-only event count `1,072` |
| Genesis frame motion diagnostic | PASS | Relative to `t=0.15`: `t=0.20` mean/max/fraction `0.06836/51/0.1628%`; `t=0.25` `0.36100/54/0.9766%`; `t=0.30` `0.65918/99/1.5625%` |
| Official V2E full pipeline | PASS | Official SuperSloMo path executed; 31 source frames, 100 Hz, timestamp resolution `0.0001 s`, slowdown factor `100`; no cutoff/undersampling warning |
| Official V2E event output | PASS | 23 events; processed source duration `0.300 s` exactly; timestamps `0.042414–0.299099 s` inside `[0,0.300]`; event span `0.256685 s` is diagnostic only; leading/trailing silence `42.414/0.901 ms` |
| V2E checkpoint | PASS | `/home/zhaoyuhan/Eventbased_WAM/code/v2e_source/input/SuperSloMo39.ckpt`; size `158,447,937` bytes; SHA-256 `d4d3070431eff774a1c27038930121ffeae59ede3a7d184118c539b443ea27cb` |
| Headless shim provenance | PASS | `v2ecore.desktop.__file__ = /tmp/genesis_event_benchmark_phase4_7_repair/v2e_active_030/_v2e_runtime_compat/v2ecore/desktop.py`; probe used the same interpreter, cwd, and `PYTHONPATH` as the official V2E subprocess |
| Frozen core integrity | PASS | After fetching the exact known frozen object on the remote clone, `git diff e79e656 -- genesis_event_plugin/` is empty |
| Protocol/history integrity | PASS | `protocol_v1`, `protocol_v2`, `protocol_v3`, pair manifest, and frozen config hashes unchanged |

## Frame-difference diagnostics

The active smoke's replay image itself changes after the declared stationary
hold. The diagnostic was generated from the archived display RGB frames at
`t=0.15, 0.20, 0.25, 0.30 s`; it is a runtime sanity check only and is not a
formal metric.

Evidence on the remote host:

```text
/tmp/genesis_event_benchmark_phase4_7_repair/genesis_active_030/replay_smoke.json
/tmp/genesis_event_benchmark_phase4_7_repair/genesis_active_030/frame_difference_diagnostics.json
/tmp/genesis_event_benchmark_phase4_7_repair/v2e_active_030/v2e_pipeline_smoke.json
/tmp/genesis_event_benchmark_phase4_7_repair/v2e_active_030/v2e_cli.json
```

## Interpretation

All required Phase 4.7 runtime gates passed on the current readiness commit.
The old blockers (missing ECF filter, missing SuperSloMo checkpoint, and the
0.02 s probe) are no longer the current state. The active readiness smoke is
now fixed at 0.30 s and records event silence instead of requiring the first
event-to-last-event span to equal the replay duration.

This does not establish trajectory-matched fidelity or any formal method
ranking. The repository remains stopped before Phase 5 pending human approval.
