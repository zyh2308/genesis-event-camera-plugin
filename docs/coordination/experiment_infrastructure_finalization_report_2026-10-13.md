# EventGenesis Experiment Infrastructure Finalization Report

**Date:** 2026-10-13
**Branch:** `work/core-maintenance`
**Formal simulator core:** `e79e656ac00f7847b98db9e12bf43b093f3afff1` (`v3.1-quality-benchmark`)

## 1. Current status

EventGenesis now has a thin infrastructure layer outside
`genesis_event_plugin/`.  Quality validation, public Sim-to-Real, and robot
closed-loop runs can bind different metadata/state adapters to the same
`GenesisPhysicsEventPlugin` capture path:

```text
Genesis scene/episode loop
        -> ExperimentRunner
        -> GenesisPhysicsEventPlugin (one event-generation core)
        -> events/aer_events.h5
        -> metadata/metadata.json + metadata/state.jsonl
        -> Quality | Sim2Real | Robot adapter outputs
```

This change does not modify the frozen simulator physics, HDR/radiance path,
interpolator, visibility rules, or DVS algorithm.  It is therefore an
infrastructure change, not a new simulator version.  The branch remains a
development branch and is not a replacement for the formal frozen tag.

The currently supported v3 execution path is the one already audited:
`GenesisPhysicsEventPlugin -> PhysicsInterpolator/physics_warp ->
DvsEmulator -> GenesisEventRecorder`.  The legacy `GenesisEventPlugin` path
is not selected by the new runner.

## 2. Required modifications

The minimal additions are:

| Area | Addition | Purpose |
| --- | --- | --- |
| Configuration | `configs/eventgenesis_default.yaml`, `eventgenesis_experiment/config.py` | One explicit, validated parameter source pinned to the frozen core |
| Adapter boundary | `eventgenesis_experiment/adapter.py` | Keep experiment-specific state/label/action logic outside the simulator |
| Runner | `eventgenesis_experiment/runner.py` | Bind one adapter to one v3 plugin and preserve timestamp/event-index alignment |
| Sidecar | `eventgenesis_experiment/sidecar.py` | Store synchronized physical state, labels, and actions as JSONL |
| Export | `eventgenesis_experiment/export.py` | Create `events/`, `labels/`, `metadata/` and a hash manifest |
| Verification | `tests/test_experiment_infrastructure.py` | Mock Genesis-compatible integration test without a Genesis installation |
| Packaging | `pyyaml>=6.0` | Make the YAML contract installable rather than an environment accident |

No file under `genesis_event_plugin/` is changed by this finalization.  The
runner translates the frozen YAML into the existing plugin constructor; it
does not create a second event generator.

## 3. Directory proposal

```text
gec-core/
├── configs/
│   └── eventgenesis_default.yaml       # frozen experiment contract
├── eventgenesis_experiment/
│   ├── config.py                       # strict YAML/dataclass loader
│   ├── adapter.py                      # Quality/Sim2Real/Robot callbacks
│   ├── runner.py                       # shared capture interface
│   ├── sidecar.py                      # metadata.json + state.jsonl
│   └── export.py                       # dataset layout + manifest hashes
├── genesis_event_plugin/               # frozen simulator implementation
└── tests/test_experiment_infrastructure.py

dataset/<run-id>/
├── events/aer_events.h5                # existing recorder, AER [t_us,x,y,p_01]
├── labels/                             # optional files/references (e.g. flow)
├── metadata/metadata.json              # run/config/provenance contract
├── metadata/state.jsonl                # one synchronized row per capture
└── manifest.json                       # SHA-256 of all package files
```

`ExperimentRunner` deliberately does not step Genesis.  The owning quality,
Sim-to-Real, or robot environment keeps control of reset/step/action timing;
it calls `capture()` at each endpoint.  This avoids accidentally changing a
task's physical trajectory while unifying event generation.  The runner also
rejects an endpoint interval that differs from the frozen
`reconstruction.source_dt_s` (within a numerical tolerance), so a variable-
step task must declare a separately reviewed configuration rather than
silently changing the temporal sampling contract.

## 4. Config design

`configs/eventgenesis_default.yaml` is the only default parameter source and
pins `core_commit` to the formal frozen commit.  Unknown keys are rejected and
the runner checks that the actual Genesis camera resolution agrees with the
declared scene/sensor resolution (including any declared over-render margin).

The frozen sections are:

- **Sensor:** resolution, DVS positive/negative threshold, threshold mismatch,
  cutoff, leakage, shot/background noise, refractory period, noise-map
  provenance, seed, and readout model.
- **Radiometry:** linear input domain, response model, scale/white level,
  numerical floor, calibration status/id, and optional transfer path.  The
  default status remains `uncalibrated`; it must not be reported as complete
  EVK4 unit calibration.
- **Reconstruction:** source endpoint interval, adaptive-sampling guards,
  visibility policy, warp composition/backend, and strict-fidelity policy.
- **Scene:** camera profile/id, sensor/render resolution, and declared object
  and articulated-link IDs.
- **Output:** timestamp units (`s` in memory/sidecar, `us` in AER HDF5), event
  layout, and optional RGB recording.

These are frozen for a formal run.  An experiment may select only an adapter,
episode/trajectory, scene identity, and output location/run id.  It may not
silently change thresholds, reconstruction settings, camera intrinsics,
resolution, seeds, or radiometric transfer.  A deliberate ablation must use a
separately named, reviewed configuration and must never overwrite the default
run.

The YAML contains nominal EVK4-compatible values inherited from the existing
`real_v2` project preset.  The `pixel_noise_map_path: null` value means “use
the preset's repository-resolved map”; the resolved path is captured in the
recorder/plugin metadata.  This preserves the established preset semantics
without embedding a machine-specific absolute path in the YAML.

## 5. Metadata format

`metadata/metadata.json` records the dataset schema, run id, experiment kind,
Genesis version, frozen core commit, complete parsed configuration, config
SHA-256 (when a config file is supplied), scene identity, and product paths.

`metadata/state.jsonl` has one row per `capture()` call:

```json
{
  "episode_index": 0,
  "frame_index": 17,
  "timestamp_s": 0.170,
  "event_index": {"start": 2310, "end": 2488},
  "event_timestamp_s": {"start": 0.161, "end": 0.170},
  "state": {"camera_pose": "...", "object_pose": "..."},
  "labels": {"flow_path": "labels/flow_000017.npz"},
  "action": [0.0, 0.1],
  "action_timestamp_s": 0.170
}
```

The values inside `state`, `labels`, and `action` are adapter-owned.  Quality
uses camera/object state; Sim-to-Real adds optical-flow/motion and pose labels;
Robot adds joint state, object pose, camera pose, and action.  Large arrays
should be saved under `labels/` and referenced by path rather than expanded in
JSON.  Event timestamps remain the authoritative AER times, while sidecar
rows provide an explicit event-index and endpoint-time join, so downstream
code never guesses synchronization from frame order.

`manifest.json` hashes every output file (excluding itself), and repeats the
core commit, run id, experiment kind, and config hash.  It is the final
integrity check before a dataset is published or consumed by another session.

## 6. Migration plan

1. **Quality line:** wrap the existing v3 frame/motion providers in
   `QualityAdapter`; call `start_episode()`, `capture()`, and `end_episode()`
   from the existing scene loop.  Compare the produced AER file against the
   current frozen quality harness before changing any benchmark code.
2. **Sim-to-Real line:** reuse the same runner and supply a label callback that
   writes flow/motion arrays into `labels/` and returns their relative paths;
   supply pose/state through `state_provider`.
3. **Robot line:** keep the robot task's stepping/controller untouched; pass
   qpos/qvel/object/camera snapshots and the action taken at the same scene
   timestamp through `RobotAdapter`.
4. Each line must use the same default YAML, seed, camera contract, and event
   schema.  Only adapter metadata and trajectory/scene identity differ.
5. Freeze a dataset only after `manifest.json`, config hash, core commit, and
   validation checklist are archived together.  Any output-affecting change
   follows the existing three matched-experiment approval rule and cannot
   silently replace `v3.1-quality-benchmark`.

## 7. Validation checklist

- [x] `load_config()` parses the default YAML and rejects unknown keys.
- [x] The config rejects any core commit other than the formal frozen commit.
- [x] The runner has one event-generation call path and selects v3 explicitly.
- [x] Quality, Sim2Real, and Robot adapters share the runner interface.
- [x] Camera/render and sensor resolutions are checked at attachment time.
- [x] Endpoint intervals are checked against frozen `source_dt_s`.
- [x] AER events retain the existing `uint64 [t_us,x,y,p_01]` recorder schema.
- [x] Sidecar rows include endpoint time, event index range, and event time range.
- [x] Robot action timestamps are stored with the corresponding physical time.
- [x] Dataset export creates stable directories and a SHA-256 manifest.
- [x] Mock integration test passes with no Genesis renderer or server job.
- [x] `git diff e79e656 -- genesis_event_plugin/` remains empty.
- [ ] Before a formal paper run: verify Genesis version/renderer overlay,
  camera intrinsics/distortion, calibration assets, trajectory seed, and
  adapter-provided state/flow labels in the actual experiment environment.

The remaining unchecked item is deliberately an experiment-owner validation,
not a simulator implementation gap.  It cannot be inferred from a generic
runner and must be recorded per published dataset.
