# Phase 1–4 benchmark readiness status

**Date:** 2026-09-29  
**Frozen commit checked:** `e79e656ac00f7847b98db9e12bf43b093f3afff1`  
**Formal benchmark:** not run  
**Parameter search:** not run  

This is the completed Phase 1–4 checkpoint. It is superseded for protocol
state by `PHASE4_5_STATUS.md`, which records the frozen v1 protocol.

## Decision summary

| Phase | Status | Evidence |
|---|---|---|
| 1. Asset audit | complete | `docs/quality_benchmark_asset_audit.md` |
| 2. Protocol design | draft, awaiting confirmation | `benchmark/protocol_v1.yaml` |
| 3. Metric verification | distributional metrics and CD/GD mathematical smoke pass; CD/GD remain scientifically gated | `benchmark/results/raw/metric_smoke_test.json` |
| 4. Baseline smoke test | complete for existing Ours/V2E artifacts; ESIM/IEBCS unavailable | `benchmark/results/raw/baseline_smoke_test.json` |

## What is usable now

The six static EVK4 recordings can remain the fixed `real_v2` calibration
reference. The full dynamic checkerboard stream, the retained LR/RL one-round
HDF5 files, the `real_v2` Genesis replay, and the existing V2E official-core
artifact are all readable and use the declared 1280×720 geometry. The benchmark
loader validates the event schema and converts all supported inputs to
`[t_s, x, y, p_01]`.

The current materials can support a descriptive, duration-normalized comparison
of event rate/count, ON fraction, active-pixel ratio, fixed 8×8 spatial
distribution, and fixed 5 ms temporal rate—after the protocol is confirmed and
with the unpaired-data limitation stated in every result.

## What is not a valid formal result yet

The real dynamic sequence and the simulator replay have no shared time origin,
measured rail trajectory, hardware trigger, or archived matched HDR/depth/seg/
pose sequence. The old simulator metadata also reports a non-zero principal
point residual and references a calibration JSON that is not present in the
current workspace. The local benchmark environment now has the project’s
declared SciPy dependency and the CD/GD implementation passes an identical-cloud
mathematical smoke test; therefore the remaining CD/GD block is scientific, not
software availability. CD/GD are still not enabled, and the current V2E run is
not a final fair baseline because its 100 Hz input / 300 Hz cutoff combination
triggered its own numerical warning.

The current `n=1` dynamic family cannot justify confidence intervals or
statistical-significance claims. No composite score or method ranking is
reported.

## Baseline availability

- **Ours-v3.1 / real_v2:** existing HDF5 is valid and its frozen provenance is
  recorded.
- **V2E:** existing official `EventEmulator` HDF5 is valid; source snapshot is
  v1.5.1, but a cadence-corrected formal rerun remains required.
- **ESIM:** source commit exists, but no official executable/build is present.
- **ICNS/IEBCS:** no local official checkout.
- **PECS:** reference-only in this phase.

## New-recording conclusion

No new recording is needed to finish Phase 1–4 or to build the harness. One
targeted matched dynamic recording is needed later for a strict real–sim CD/GD
claim: record a repeatable trajectory/time reference, reuse the current EVK4,
board, camera and lighting, archive the corresponding Genesis radiance/depth/
segmentation/pose inputs, and keep a separate repeated sequence as held-out.
