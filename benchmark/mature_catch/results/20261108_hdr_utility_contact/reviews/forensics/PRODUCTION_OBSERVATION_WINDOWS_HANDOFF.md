# Production benchmark extraction: 10 ms observation windows

2026-11-08. Parent-authorized edits limited to three benchmark files; no SSH, core changes, policy changes, physics changes, commit or deployment by this child.

## Final files / handoff lock

Under `gec-robot-catch-hdr-utility-20261108/benchmark/mature_catch`:

| File | SHA256 |
|---|---|
| `event_observation_windows.py` | `1aad32cd931d2ed01b0888211d330f6304ff1819fc713abea2ff2707002cfb02` |
| `event_sphere_tracker.py` | `585ee86f66c01988c1058e5366acb603146bbba901758fd7a5f5667b9dae1464` |
| `test_event_observation_windows.py` | `fbb3f3684289a494652b35aaef1ea70ecd84fa7cecb81a4618d74cb7f0975e13` |

**The child stopped modifying these files after sending final hashes to root.** Root owns review, commit, source sealing, manifest/SparseSensor audit delivery and server actual-cache parity, followed by real closed-loop runs. Any later concern should be reported before changing the sealed source.

## API

- `CATCHIT_EVENT_OBSERVATION_WINDOW_MS=0` or unset: original `push`, `_detect`, `observe` path is directly delegated. The optional feature module is not imported by tracker construction. Existing `_predict`, `_start_motion`, partial checks, fit and thresholds remain untouched.
- `=10`: only the preselected 10 ms candidate is accepted; no new window sweep.
- `tracker.observation_window_ms`: active configuration.
- `tracker.observation_window_audit`: JSON-ready list; empty when disabled. Audit holds counts, time bounds, parent/child IDs, two-integer raw-index ranges, camera transform/source, availability and detected sample metadata. It never embeds per-event arrays or per-event index lists.
- `push(events, available_s, packet_time_s, camera_to_world, parent_packet_id=None)`: enabled path pre-registers the exact original camera endpoint and arrival parent. `push` registration and later consumption are idempotent.
- Direct `_detect(parent)` also auto-registers camera endpoints, so the existing cached replay can exercise production code without production reading HDF.
- Enabled `observe` supplies its real `now_s` as processing context. Direct `_detect` has `processing_now_s=None`, explicitly unmeasured rather than a fabricated online timestamp.

## Behavior retained from validated v2

True carry-in historical child windows, all events in their original arrival parent, all original event timestamps, original parent availability, available camera-history SE(3) interpolation at child end, exact available endpoint poses where applicable, and parent-relative input ranges proving no duplicate/dropped event partition.

Each child recursively passes the full original top-level `_detect`, including partial-contour handling and Kalman camera-origin assignment. A child guard prevents recursive repartitioning. A valid child sample is appended once and the existing `_predict(childend)` processes it before the next child. The parent detector returns None to prevent its caller appending it again. Control queries remain at the original polls. No release GT, target pose/velocity, extra RGB-D or HDR enters the estimator.

Repeated child endpoint times across distinct parents are kept. Truly out-of-sequence raw events or measurements stop explicitly rather than resetting the filter time. The 50 ms two-observation rule, 0.30 m association gate, priors, drag, fit code and original motion initialization are unchanged. No bootstrap candidate is included.

## Verification completed locally

- Production tests **6/6 PASS**: disabled legacy method delegation; reset registration/consumption; true carry-in, unique ranges and repeated child endpoint; rejection of future camera support/unavailable parent; out-of-order stop without rewind; full actual native push/observe/detector integration and JSON audit.
- Synthetic v2-diagnostic versus production extraction: **fit0 and fit1 both PASS**. Each has 12 detected samples with exact matching original sample fields, identical onset, and zero difference in query position/velocity/latest timestamp/count. Results in `PRODUCTION_SUBWINDOW_SYNTHETIC_COMPARISON.json`.
- Three edited Python files compile. Local WSL `git diff --check` passes.
- Actual eight-cache production parity has **not** been claimed by this child; root must run it from the sealed production source with explicit flag0 and flag10. Do not double-wrap the production tracker in the diagnostic monkeypatch.

## Independent reading of actual v2 cache result

`SUBWINDOW_V2_ACTUAL_STATUS.json` reports actual g1 exit0, wall32.134 s, all eight archive-baseline parity checks passing, and no pre-release moving control query. Whole-packet fit1 versus subwindow fit1:

| Cached case | Ours position median mm | Ours velocity median m/s | V2E position median mm | V2E velocity median m/s |
|---|---:|---:|---:|---:|
| c01 | 29.2 → 15.1 | .123 → .068 | 31.9 → 64.0 | .138 → .110 |
| c02 | 33.8 → 12.3 | .127 → .078 | 30.3 → 34.3 | .132 → .113 |
| c03 | 25.8 → 19.3 | .114 → .072 | 42.5 → 70.3 | .106 → .168 |
| c04 | 37.5 → 9.9 | .141 → .080 | 39.7 → 75.2 | .102 → .153 |

Ours improves all four position and velocity medians; V2E position worsens in all four, velocity improves in two and worsens in two. Do not hide these regressions or call this a closed-loop success result. The data are old development caches with candidate estimator outputs; original physical actions are fixed.

Historical onset improves without always advancing actual information use: c02 both methods first produce a moving state at the original query .76 instead of .80 s; c03 V2E and c04 both advance by one 40 ms poll. c01 both and c03 Ours retain the same first moving query despite an earlier estimated past onset. This distinguishes event timestamps from useful online latency.

The agreed next test is one fixed common 10 ms + existing fit1 + fixed common contact physics on the existing six development seeds, retaining all 12 outcomes. It can test whether better cached state estimates improve the physical approach and retention. A favorable development result still needs the unused bank for confirmation; it does not justify claiming a general generator ranking.
