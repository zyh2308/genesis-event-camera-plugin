# Quality benchmark asset audit

**Audit date:** 2026-09-29  
**Frozen plugin commit:** `e79e656ac00f7847b98db9e12bf43b093f3afff1`  
**Scope:** Phase 1 asset audit only; no formal benchmark result and no model
parameter search was run.

This audit remains the evidence source for the frozen Phase 4.5 protocol; the
final protocol state and ESIM decision are summarized in
`benchmark/results/tables/PHASE4_5_STATUS.md`.

## Executive decision

The current project contains enough material to build and smoke-test a
reproducible distributional benchmark, but not enough to claim a strict
PECS-style real–simulation fidelity result. The decisive missing conditions
are a common time reference, a measured motion trajectory, and an archived
matched Genesis input sequence. The current dynamic real EVK4 stream and the
Genesis/V2E streams have compatible nominal resolution and event schemas, but
they are different recordings/replays rather than the same physical trajectory.

Accordingly:

1. **No new recording is required for Phase 1–4.** The existing assets support
   the audit, frozen protocol, metric implementation smoke test, and baseline
   availability test.
2. **A new recording is required before a formal strict CD/GD claim.** It need
   not be a full PECS six-scene replication; one repeatable matched dynamic
   sequence with an auditable motion/time reference is the minimum useful next
   acquisition.
3. **CD/GD are implemented in the repository, their input-order behavior is
   covered by existing test code, and the mathematical smoke test now passes
   with the project-declared `scipy` dependency. They are nevertheless
   scientifically gated off in `benchmark/protocol_v1.yaml`.** Enabling them now would measure
   camera/trajectory/time mismatch as much as simulator quality.

## Inventory

| sequence | real/sim | resolution | duration | timestamp | RGB | HDR | depth | seg | pose | events | alignment status | usable metrics |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|
| `evk4_static_dark_20260912_x3` | real EVK4 | 1280×720 | 20.108–20.384 s each | EVT3 relative `t_us`; reliable within run | no | no | no | no | no | yes; 3 runs, 72,621–78,564 events | no dynamic correspondence required | noise rate, hot-pixel, polarity, static temporal diagnostics |
| `evk4_static_white_wall_20260912_x3` | real EVK4 | 1280×720 | 20.400–21.552 s each | EVT3 relative `t_us`; reliable within run | no | no | no | no | no | yes; 3 runs, 16,083,409–17,220,579 events | no dynamic correspondence required | white-wall noise/polarity/Fano/ACF calibration reference |
| `evk4_checkerboard_translation_20260912_full` | real EVK4 | 1280×720 | 21.655999 s | EVT3 relative `t_us`, 15,253,408–36,909,407 raw us | no paired frame stream | no | no | no | no | yes; 244,714,861 events | no encoder, trigger, or common Genesis time origin | descriptive event rate, polarity, active pixels, coarse spatial/temporal statistics |
| `evk4_checkerboard_translation_20260914_one_round` | real EVK4 | 1280×720 | LR 1.756258 s; RL 1.775800 s | EVT3 HDF5 `CD/events.t` in us | no paired frame stream | no | no | no | no | yes; 23,675,387 + 23,766,581 | correlated subset of the same acquisition family; no paired sim timestamps | descriptive per-direction statistics only |
| `genesis_real_v2_replay_20260913` | sim | 1280×720 | 4.190 s nominal; event stream 4.1875 s | simulator-generated `timestamp_us`; starts at its own origin | yes; 420 frames at 10 ms | absent from archived run | absent from archived run | absent from archived run | absent from archived run | yes; 42,187,462 events | approximate triangle-wave replay, not paired to EVK4 timestamps | smoke/distributional comparison only |
| `v2e_official_core_replay_20260913` | sim baseline | 1280×720 | 4.189779 s | simulator-generated `timestamp_us`; starts at its own origin | yes; same RGB HDF5 as above | not used | not used | not used | not used | yes; 37,946,993 events | same unpaired RGB replay; V2E also warned about 100 Hz input vs 300 Hz cutoff | baseline artifact/schema smoke only until cadence is fixed |
| `genesis_legacy_real_replay_20260913` | sim, historical | 1280×720 | 4.1875 s | simulator-generated | yes | absent | absent | absent | absent | yes; separate historical run | older `real` configuration, not frozen `real_v2` | historical provenance only; excluded from formal set |

The two one-round Metavision HDF5 files are:

```text
/home/科研/Eventbased_WAM/data/real/evk4_checkerboard_translation_20260914/derived/one_round_h5/segment_01_LR.hdf5
/home/科研/Eventbased_WAM/data/real/evk4_checkerboard_translation_20260914/derived/one_round_h5/segment_02_RL.hdf5
```

The local audit can inspect their datasets, fields and camera metadata, but the
current Python/HDF5 environment cannot read the compressed `CD/events` payload:
the vendor ECF filter plugin is absent (`/usr/local/lib/plugin` is missing).
This is recorded as an access blocker in
`benchmark/results/raw/baseline_smoke_test.json`; it is not silently treated as
an empty or decoded stream. The complete dynamic source is still available in
decoded binary columns under `/root/eventcamera_lab_staging/decoded/`, so a
future protocol-approved runner can use a dedicated binary loader or restore
the vendor filter environment.

The full decoded dynamic source and the six static runs remain in the local
staging tree under `/root/eventcamera_lab_staging/decoded/`. They are referenced
by the protocol but intentionally not copied into this code repository.

## What is and is not aligned

### Confirmed compatible

- All primary streams declare the native EVK4 geometry 1280×720.
- The plugin/V2E HDF5 schema is `[timestamp_us, x, y, polarity]` with polarity
  encoded as `0=OFF, 1=ON`.
- The Metavision EVT3 structured schema is `{x, y, p, t}` and can be converted
  without ambiguity to the benchmark canonical schema `[t_s, x, y, p_01]` once
  the vendor filter exposes the payload.
- The benchmark loader performs this conversion explicitly and rejects invalid
  coordinates, timestamps, or polarity encodings; its smoke test currently
  stops at the missing-filter boundary rather than bypassing it.

### Not established

- **Common time origin:** real and simulated timestamps are each internally
  ordered but have no trigger/offset relation. Matching the same duration is
  not synchronization.
- **Same trajectory:** the real rail has no encoder trace. The Genesis replay
  uses an explicit triangle-wave assumption (`travel_m=0.165`, `period_s=4.2`)
  rather than an observed `x(t)`.
- **Same optical projection:** the stored run metadata reports a principal
  point residual of approximately `[+29.25, -11.77]` pixels between the target
  and Genesis camera matrices. The referenced intrinsics JSON path is not
  present in the current workspace, so this cannot be independently reloaded
  from the archived run.
- **Same radiometry:** the archived Genesis replay is marked `input_space=srgb`
  and has no HDR/radiance buffer. The v3 plugin explicitly refuses to call
  display RGB a calibrated radiance measurement.
- **Same scene state:** no matched depth, segmentation, or pose arrays are
  archived alongside the dynamic RGB/event outputs.
- **Same crop and field of view in practice:** the array shapes match, but the
  missing matched calibration/trajectory prevents claiming pixel-to-pixel
  correspondence.

The appropriate current label is therefore **coarse distributional
comparison**, not paired event-level ground truth.

## Existing calibration and noise assets

The project has a useful fixed `real_v2` noise reference:

- Three dark streams and three white-wall streams were used once to fit the
  noise map and shared AR(1) activity model.
- The fitted map is
  `/home/科研/Eventbased_WAM/validation/simulator-quality/real_v2/calibration/evk4_real_noise_map_v2.npz`.
- The fit record is
  `/home/科研/Eventbased_WAM/validation/simulator-quality/real_v2/calibration/evk4_real_noise_fit_v2.json`.
- The frozen main parameters are `pos=0.205`, `neg=0.195`, `sigma=0.04`,
  `cutoff=50 Hz`, `alpha=0.701838977169265`, `scale=0.016`, seed `42`; uniform
  leak and shot terms remain zero to avoid double counting the measured map.

The current preliminary geometry record is documented in the project reports
(28 valid views, RMS approximately 2.8168 px), but its JSON is not archived at
the path embedded in the old run metadata. It must not be silently treated as
formal calibration acceptance.

## Metric capability assessment

### Safe now, after protocol confirmation

For equal-duration or explicitly normalized windows, the following are
scientifically interpretable as distributional diagnostics:

- duration-normalized event count error;
- ON-fraction/polarity error;
- active-pixel ratio error;
- fixed 8×8 spatial distribution L1 error;
- fixed 5 ms temporal event-rate normalized L1 error.

They are not independent physical proofs: the current unpaired motion and
illumination can affect all of them. They should be reported as descriptive
coarse statistics, with the data limitation next to the table.

### Gated now

The repository already contains PECS-style CD/GD code and its input-order
correction is covered by the metric smoke test. However, CD/GD use nearest
neighbors in joint space/time. Without common time, same trajectory, same crop,
and a registration policy frozen in advance, a low/high value cannot be
attributed to the simulator. The protocol therefore records both metrics as
**disabled pending a matched rig**, not because a result was unfavorable.

### Not available from current archive

- exact per-pixel trajectory correspondence;
- event latency/readout/refractory calibration;
- absolute radiance/illumination calibration;
- independent held-out dynamic sequence;
- bootstrap uncertainty across independent motion repeats.

## Calibration/development and held-out recommendation

The existing assets do not provide enough independent dynamic sequences for a
valid calibration/test split. The only defensible current split is:

```text
S_cal  = 3 dark + 3 white-wall runs, used once to establish frozen real_v2
S_test = the dynamic checkerboard runs, descriptive only and not independent
```

The retained LR/RL one-round files must not be called a held-out test if the
full 20260912 sequence is used, because they come from the same acquisition
family and have no independently recorded motion/time reference. A future
formal split should contain at least one separate repeated dynamic sequence;
the parameter freeze must happen before it is inspected.

## Baseline readiness

- V2E official `EventEmulator` core has an existing artifact and its parameters
  are recorded in `benchmark/baselines/BASELINE_VERSIONS.md`.
- ESIM source and commit are present, but no official ROS/catkin build or
  executable is available; it is not a result row yet.
- ICNS/IEBCS is not checked out; no substitute will be labeled as it.
- PECS is a protocol/metric reference for this phase, not an executed baseline.

The existing V2E artifact is not a final fair comparison because its own warning
identifies a 100-Hz input / 300-Hz cutoff numerical mismatch. This is a concrete
rerun condition, not a reason to discard V2E or tune it selectively.

## New-recording decision

**For the requested Phase 1–4 preparation: no new recording.** The data audit,
protocol freeze candidate, metric smoke test, and baseline smoke test can be
completed with existing files.

**For a formal strict real–sim benchmark: yes, one targeted recording is
needed.** Use the current EVK4, current checkerboard/rail and current lighting;
no new high-end hardware is required. Record one repeatable motion with:

1. a visible motion marker or rail position trace sufficient to reconstruct
   `x(t)` at the event timeline;
2. one common timestamp/trigger reference, or a documented flash/edge marker;
3. the same board size, camera distance, crop, and resolution in Genesis;
4. archived Genesis HDR/radiance, depth, segmentation, pose and raw events;
5. one separate repeat for held-out evaluation.

That recording is needed to unlock CD/GD and any claim stronger than coarse
distributional agreement. It is not needed to start the benchmark framework.
