# Benchmark protocol changelog

## v1-draft

- **Date:** 2026-09-29
- **Ours commit:** `e79e656ac00f7847b98db9e12bf43b093f3afff1`
- **State:** Phase 1–4 candidate; not frozen
- **Notes:** Asset limitations and candidate metrics were documented. Temporal
  per-bin error and CD/GD were not yet fully gated in the first draft.

## v1-frozen

- **Date:** 2026-09-29
- **Ours commit:** `e79e656ac00f7847b98db9e12bf43b093f3afff1`
- **State:** frozen before formal benchmark execution
- **Protocol file:** `benchmark/protocol_v1.yaml`
- **Protocol SHA-256:** `b1b4519446bbc821220b3502a0eb3811bf8c0baf8bd457eca022a0d082ebdd97`
- **Formal benchmark executed:** no
- **Changes from v1-draft:**
  - changed formal rate metric from raw event-count comparison to duration-normalized event-rate relative error;
  - fixed main metrics to rate, polarity ratio, active-pixel ratio, and 8×8 spatial L1;
  - moved temporal comparison to descriptive rate statistics because no auditable common time origin exists;
  - explicitly disabled per-bin temporal error, CD, and GD;
  - recorded the dynamic data split as fixed-configuration descriptive comparison rather than a valid independent held-out split;
  - fixed deterministic visualization windows at normalized positions 0.25, 0.50, and 0.75;
  - added explicit missing-data handling, method configuration files, and config hashes;
  - recorded ESIM as supported but not reproducibly built after the official catkin build attempt failed because `catkin_make` is unavailable.

This file is the protocol history. The frozen protocol must not be edited during
formal execution; any actual protocol bug requires `v2` or later plus a new
hash and an explicit changelog entry.

## v2-frozen — Phase 4.6

- **Date:** 2026-09-29
- **State:** created before any formal benchmark result
- **Protocol file:** `benchmark/protocol_v2.yaml`
- **Protocol SHA-256:** `c6382e3cd94ce02c2f92d8d12a8d5604affb8a4fd9a17b0694c7609120ab28df`
- **Pair manifest:** `benchmark/evaluation_pairs_v2.yaml`
- **Pair manifest SHA-256:** `e62e7cc0fb18730c53213277ceea20f7598b7f22cca102b0695201d857b2c9b9`
- **Formal benchmark executed:** no
- **Approval record:** `benchmark/FORMAL_RUN_APPROVAL.yaml`
- **Frozen core:** `e79e656ac00f7847b98db9e12bf43b093f3afff1`
- **Phase 4.5 harness commit:** `46ce49b941d36d2c145bd9e369dc8ef53362b8cc`

### Why v2 was created

`protocol_v1.yaml` was not edited and its SHA-256 remains unchanged. The v2
file was created before any Ours or V2E formal metric existed. It resolves
execution ambiguities rather than changing the scientific question:

1. the runner now permits benchmark commits after the frozen core while checking
   that `genesis_event_plugin/` has an empty diff against `e79e656`;
2. human approval is recorded independently, without modifying v1;
3. formal real/simulation pairs are frozen as LR and RL one-round pairs;
4. all methods use the same fixed 5 ms windows at normalized fractions 0.25,
   0.50, and 0.75;
5. V2E uses one fixed 10 kHz effective cadence (`timestamp_resolution=0.0001 s`)
   with its unchanged 300 Hz cutoff. This follows the official source guard
   `maxeps=0.3`: `eps=0.1885` at 10 kHz, while the previous 100 Hz input gives
   `eps=18.8496` and triggers the documented warning;
6. the formal Ours input contract requires native scene-linear HDR/radiance,
   depth, segmentation, poses, and timestamps. Display-RGB substitution is
   explicitly prohibited.

The current replay definition remains `definition_only_pending_formal_scene_generation`.
The Phase 4.6 smoke gate may validate the HDR adapter when Genesis is available,
but it must stop if the HDR overlay cannot actually run. No full evaluation
metric, baseline parameter search, or Ours parameter change is permitted in
this phase.
## Phase 4.7 — Formal execution readiness

This revision was created before any formal benchmark result. It preserves
`protocol_v1.yaml` and `protocol_v2.yaml` byte-for-byte and addresses execution
risks found during readiness review:

1. Primary units are complete LR and RL directions with equal declared
   duration per direction. The old 5 ms windows remain deterministic
   qualitative visualization only.
2. All streams use `t_rel = t - declared_sequence_start`; first-event and
   method-specific offset alignment are forbidden.
3. The real EVK4 loader must read all of `CD/events` for both HDF5 segments.
   The official vendor ECF filter is required unless a decoded fallback is
   declared and checked before any formal result.
4. Genesis must produce native scene-linear HDR/radiance, depth,
   segmentation, camera/object poses, and timestamps for the same replay used
   by both Ours and V2E.
5. V2E must execute its official image-folder pipeline with SuperSloMo and
   the frozen 100 Hz → 0.1 ms cadence. A direct EventEmulator call is not a
   passing smoke.

The v3 runner is fail-closed and reports `formal_benchmark_executed: false`.
It stops at the first missing runtime dependency or failed smoke.
