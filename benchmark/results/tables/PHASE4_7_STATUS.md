# Phase 4.7 — Formal Execution Readiness status

This is a readiness record, not a benchmark result. No formal metric, ranking,
aggregate, or method-specific time alignment was executed.

## Provenance

| Item | Value |
|---|---|
| Phase 4.7 readiness commit | `9dc0231ef805250efc932eda1ec17d3c8fb25b18` |
| Frozen core | `e79e656ac00f7847b98db9e12bf43b093f3afff1` |
| Phase 4.6 historical commit | `b8d5d81cee74066dfa24ad682e2820d30b35264c` |
| `protocol_v3.yaml` SHA-256 | `c6d0926147e3fb08b15892c1f9ee368c5d7380ad3326cca0b8d962ae1f78d763` |
| `evaluation_pairs_v3.yaml` SHA-256 | `86c9596bb3c6e79788e1ae5aa627319ade886a64c88571f4ccc5e7ede6fbc175` |
| formal benchmark executed | **NO** |
| formal metrics executed | **NO** |

## Gate results

| Gate | Result | Evidence / blocker |
|---|---|---|
| `protocol_v1` unchanged | PASS | SHA-256 remains `b1b4519446bbc821220b3502a0eb3811bf8c0baf8bd457eca022a0d082ebdd97` |
| `protocol_v2` unchanged | PASS | `benchmark/protocol_v2.sha256` still matches its bytes |
| frozen core diff | PASS | `git diff e79e656 -- genesis_event_plugin/` is empty |
| full-direction pair definition | PASS | only LR/RL; declared durations `1.756258 s` and `1.775800 s` |
| primary metric contract | PASS | four full-direction metrics; CD/GD/5 ms aggregate disabled |
| time-origin contract | PASS | sequence-local origin declared as segment origin 0; first-event/offset search disabled |
| real LR/RL complete loader | **STOP** | exact HDF5 files exist locally but complete `CD/events` read fails without official ECF filter `/usr/local/lib/plugin`; the remote Genesis host does not contain these exact segments |
| Genesis native HDR smoke | PASS (runtime probe) | remote `genesis_renderer_hdr_v2` overlay, 64×48, 0.02 s, `radiance=float32`, max `7.6875`, depth/segmentation/pose/timestamps present |
| Ours v3.1 smoke | PASS (runtime probe) | same Genesis replay; 1,621 smoke events; core commit recorded as `e79e656` |
| official V2E full pipeline | **STOP / NOT RUN** | official `SuperSloMo39.ckpt` is absent; direct EventEmulator shortcut was not used |

## Interpretation

Phase 4.7 is not execution-ready for Phase 5. The Genesis HDR/Ours runtime
contract is now demonstrated in the actual remote Genesis environment through
the repository's documented overlay, but the real-data gate and official V2E
chain remain unresolved. The next allowed actions are to restore the vendor
ECF filter or predeclare and validate an exact decoded binary fallback, and to
install/provide the official SuperSloMo checkpoint. After those gates pass, the
full LR/RL durations remain the declared pair durations above; no trajectory
matching or formal metric may be inferred from this smoke.
