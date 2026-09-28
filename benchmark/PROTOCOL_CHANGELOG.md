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

