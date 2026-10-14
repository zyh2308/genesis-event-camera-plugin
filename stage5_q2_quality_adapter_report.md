# Stage 5 Q2 — QualityAdapter and v4 interface audit

**Audit time:** 2026-10-14 11:04 (Asia/Shanghai)
**Worktree:** `/home/科研/Eventbased_WAM/repositories/gec-quality`
**Branch:** `exp/quality-cdgd-v4`
**Implementation base HEAD:** `c9d7f1492b78d72fe0515f0948cf18ac45616895`
**Frozen simulator core:** `e79e656ac00f7847b98db9e12bf43b093f3afff1`

## Scope and execution boundary

This phase establishes the engineering boundary for a future v4 event-fidelity
benchmark. It implements only the raw-artifact-to-canonical-interface layer,
its provenance contract, and non-formal validation. It does not create or
freeze `protocol_v4`, run alignment, fit normalization, execute CD/GD, rank
simulators, add a baseline, or modify `genesis_event_plugin/`.

The current formal benchmark status remains:

```text
formal benchmark: NOT EXECUTED
formal CD/GD: NOT EXECUTED
simulator modified by Q2: NO
```

## 1. QualityAdapter design

The common path is:

```text
raw event artifact + artifact sidecar
        |
        v
source-specific QualityAdapter
        |
        v
canonical events [x, y, p, t_s] + validated metadata
        |
        v
future metric implementation
```

Three source adapters now exist:

| Source | Adapter | Existing loader reused | Conversion performed |
|---|---|---|---|
| Real EVK4 | `RealEVK4Adapter` | `benchmark.io.real_stream.load_real_hdf5` | loader output `[t_rel_us,x,y,p_01]` → `[x,y,p,t_s]`; `0/1` → `-1/+1` |
| EventGenesis | `GenesisAdapter` | `genesis_event_plugin.evaluation.event_reader.read_genesis_hdf5` | recorder `[t_us,x,y,p_01]` → canonical `[x,y,p,t_s]` |
| official V2E | `V2EAdapter` | `benchmark.metrics.event_statistics.load_aer_events` | loader output `[t_s,x,y,p_01]` → canonical `[x,y,p,t_s]`; `0/1` → `-1/+1` |

The adapter API returns `AdaptedEventStream(events, metadata)`. The base
contract validates shape, finite values, integer pixel coordinates, polarity,
timestamp ordering, resolution bounds, declared time bounds, event count, and
sidecar observed timestamps. No adapter invokes a metric or alters event
timing to align methods. Real EVK4 loader selection remains explicit through
`sequence_start_us` and `duration_s_declared`; the first event is not used as
an implicit time origin.

## 2. Canonical event schema

Defined in `benchmark/quality_v4/event_schema.yaml`:

- schema id: `genesis_event_quality_v4_canonical`
- schema version: `1.0`
- array shape: `(N, 4)`; `float64`
- column order: `[x, y, p, t]`
- timestamp: sequence-local seconds, with origin equal to the declared
  sequence start
- polarity: exactly `-1` for OFF and `+1` for ON
- coordinates: zero-based integer pixel coordinates, origin at top-left,
  `x` rightward and `y` downward
- event timestamps: nondecreasing
- empty streams: representable and validated as `(0, 4)`

The schema is explicitly an interface design artifact, not a frozen protocol.
Raw input layouts and units may differ, but metric code must receive only this
canonical representation through an adapter.

## 3. Artifact sidecar contract

Defined in `benchmark/quality_v4/artifact_schema.yaml`. Each artifact entering
the v4 interface must be associated with:

`sequence_id`, `generator_name`, `resolution_xy`, `event_layout`,
`polarity_encoding`, `timestamp_unit`, `declared_start`, `declared_end`,
`observed_first_timestamp`, `observed_last_timestamp`, `event_count`,
`config_sha256`, and `protocol_sha256`.

The declared interval and observed timestamps use the raw artifact unit. The
interval is sequence-local; an absolute acquisition origin is separate
loader metadata and is not a method-specific offset. Non-empty sidecars must
contain finite observed timestamps inside the declared interval. Empty
artifacts use null observed timestamps and event count zero.

At adapter execution time, the loaded artifact is checked against the sidecar:
the row count and observed first/last timestamp must match. A mismatch is a
hard validation error rather than a silent metadata rewrite. The adapter adds
canonical schema, layout, unit, polarity, adapter name, and artifact path
fields to the returned metadata.

## 4. Metric self-test status

`benchmark/quality_v4/tests/pecs_metric_selftest.py` was executed. It imports
the existing PECS-style implementation without changing its formulas.

```text
status: PASS
formal_benchmark_executed: false
formal_metrics_executed: false
identity CD: 0.0
identity GD: 0.0
empty input: ValueError for empty/non-empty combinations
timestamp-unit consistency: canonical max abs diff 0.0
polarity conversion consistency: canonical max abs diff 0.0
deterministic result: repeat outputs identical
```

An additional synthetic adapter-contract check passed: valid sidecar/event
metadata was accepted and an intentionally mismatched event count was
rejected. YAML parsing and Python compilation also passed. These are
interface checks only; no project event artifact was loaded and no formal
metric was calculated.

## 5. Compatibility audit

| Existing component | Q2 disposition |
|---|---|
| `genesis_event_plugin/evaluation/event_schema.py` | Legacy canonicalization remains unchanged; it is reused only by legacy metric/self-test paths, not edited. |
| `benchmark/metrics/event_statistics.py` | Legacy statistical loader/canonicalization remains unchanged; V2E adapter wraps it with explicit metadata and canonical output. |
| `genesis_event_plugin/evaluation/pecs_metrics.py` | Legacy PECS-style CD/GD remains unchanged; self-test checks only identity, empty behavior, unit/polarity consistency, and determinism. |
| `benchmark/io/real_stream.py` | Existing official Real EVK4 HDF5 loader remains unchanged; Real adapter wraps its `[t_rel_us,x,y,p_01]` output. |
| `genesis_event_plugin/evaluation/event_reader.py` | Existing Genesis HDF5 reader remains unchanged; Genesis adapter wraps its recorder output. |
| `benchmark/scripts/run_benchmark.py` | Existing benchmark/preflight runner remains unchanged and was not used for formal execution. |
| `ExperimentRunner` / prior `QualityAdapter` | No pre-existing implementation was found; Q2 introduces the scoped adapter skeleton only. |

No legacy file was deleted or rewritten. No existing loader was modified.

## 6. Frozen-core and historical-protocol safety

The Q2 patch is confined to `benchmark/quality_v4/`, the requested report,
and the coordination status update. It does not touch
`genesis_event_plugin/`. Historical `protocol_v1`, `protocol_v2`, and
`protocol_v3` remain immutable records. No `protocol_v4` was created or
claimed as frozen. The old v3 prohibition on formal CD/GD is unchanged.

## 7. Remaining research decisions

Before any formal benchmark, Quality Chat must separately freeze a new v4
protocol covering at least:

1. exact artifact sidecars and source paths for the paired sequences;
2. trajectory/time synchronization and the allowed alignment model;
3. the final metric definition and whether legacy PECS CD/GD are retained,
   replaced, or reported only as supplementary diagnostics;
4. fairness conditions for the official V2E execution;
5. missing-data and empty-stream policy at protocol level;
6. independent review of all hashes and provenance.

Until those decisions are frozen, the adapter layer is ready for integration
testing but not for formal ranking or quality claims.
