# Phase 4.5 status: frozen protocol and ESIM readiness

**Date:** 2026-09-29  
**Ours frozen commit:** `e79e656ac00f7847b98db9e12bf43b093f3afff1`  
**Formal benchmark:** **NOT executed**  
**Ours parameter search:** **NOT executed**

## 1. Final frozen protocol

The frozen protocol is
[`benchmark/protocol_v1.yaml`](../protocol_v1.yaml), marked
`protocol_version: v1-frozen` and `status: frozen_before_formal_execution`.
Its SHA-256 is:

```text
b1b4519446bbc821220b3502a0eb3811bf8c0baf8bd457eca022a0d082ebdd97
```

The full history is in
[`PROTOCOL_CHANGELOG.md`](../PROTOCOL_CHANGELOG.md). The protocol execution
gate remains false until human confirmation; this prevents an accidental
formal run from changing the evidence boundary.

## 2. Calibration/development and test split

```text
Development/calibration:
  evk4_static_noise_reference_20260912
  (3 dark + 3 white-wall recordings, already used to freeze real_v2)

Evaluation/descriptive set:
  evk4_checkerboard_translation_full_20260912
  evk4_checkerboard_translation_one_round_20260914
```

The dynamic set is **not an independent held-out split**: the recordings lack
an auditable shared real–sim time origin and exact motion trajectory. Therefore
the frozen interpretation is `fixed-configuration descriptive comparison`,
not train/test generalization. No evaluation sequence may be used to modify
Ours or baseline parameters.

## 3. Formal main metrics

1. Event-rate relative error:
   `abs(rate_sim - rate_real) / max(rate_real, eps)`
2. Polarity-ratio error:
   `abs(ON_ratio_sim - ON_ratio_real)`; raw ON/OFF counts are retained.
3. Active-pixel-ratio error:
   `abs(APR_sim - APR_real)`.
4. Fixed 8×8 spatial-distribution L1:
   `sum_i abs(p_sim_i - p_real_i)`, with normalized cell histograms.

Resolution and crop are fixed at 1280×720 and `[0, 0, 1280, 720]`. The
visualization windows are fixed at normalized positions 0.25, 0.50, and 0.75;
short sequences use one full-sequence fallback and record that window ID.
Aggregation is per-window → per-sequence → descriptive mean/std. No composite
score or ranking score is permitted.

## 4. Temporal metric

Per-5-ms corresponding-bin error is **disabled**. Real and simulated streams do
not share a verified time origin, hardware trigger, or objectively detected
common motion onset. The formal result may report mean event rate, event-rate
standard deviation, and an explicitly labeled event-rate distribution
descriptor, but not `real(t)` versus `sim(t)` bin-by-bin error.

No method-specific temporal offset search is allowed.

## 5. CD/GD

Both Chamfer Distance and Gaussian Distance are **disabled** in the frozen
protocol. The mathematical implementation passes an identical-cloud smoke
test, but current spatial-temporal correspondence is not auditable. Enabling
them would mix trajectory, crop, timestamp, and simulator errors.

## 6. Baseline status

| Method | Status | Formal inclusion now |
|---|---|---|
| Ours-v3.1 `real_v2` | frozen measured/fitted configuration; HDF5 artifact valid | yes, after execution gate |
| V2E official core 1.5.1 | existing artifact valid; official defaults recorded; cadence warning remains | yes, after cadence issue is protocol-resolved |
| ESIM | `supported_but_not_reproducibly_built`; official source present, `catkin_make` unavailable | no executable row yet |
| ICNS/IEBCS | official source not locally checked out | no |
| PECS | reference for protocol/metrics, not executed baseline | no |

The ESIM adapter smoke test passed only its schema conversion contract. It did
not claim to run ESIM. The exact build failure is recorded in
[`ESIM_BUILD.md`](../baselines/ESIM_BUILD.md), and the output is in
[`esim_smoke_test.json`](../raw/esim_smoke_test.json).

## 7. New data decision

No new recording is required for protocol freeze or Phase 4.5. A future strict
CD/GD benchmark still needs one matched dynamic recording with a measurable
motion/time reference and one independent repeat for held-out evaluation. No
new experimental hardware is required by this decision.

## 8. Core-modification audit

The v3.1 simulator core was not modified. Phase 4.5 changes are restricted to
benchmark protocol/configuration, metrics naming, provenance, an official-ESIM
I/O adapter, smoke tests, and result-directory scaffolding.

## 9. Final statement

**Formal benchmark has NOT been executed.**

