# Genesis Event Camera Core Contract

**Status:** coordination contract for the Core / Architecture workflow
**Effective date:** 2026-10-08
**Formal simulator frozen commit:** `e79e656ac00f7847b98db9e12bf43b093f3afff1`
**Formal tag:** `v3.1-quality-benchmark`

This document defines the reproducible boundary between the Genesis simulator,
the event-camera plugin, and downstream experiments. It is intentionally
stricter than a software package description: it prevents benchmark or robot
task changes from silently changing the simulator used by formal comparisons.

## 1. Scope and baseline

The Core is the physics/radiance v3 event-camera path, its calibration schema,
Genesis renderer integration, simulator-correctness tests, and architecture
documentation. The original/direct baseline remains preserved in its existing
workflow and is not replaced by this branch.

The formal quality-comparison simulator is exactly the commit and tag above.
`work/core-maintenance` is a development/coordination branch. A commit on this
branch is **development only; NOT adopted as formal frozen core** unless the
project control process explicitly approves it.

Genesis runtime assumptions for the HDR path are:

- Genesis `1.2.2` (or the exact version recorded by a future experiment);
- the repository's optional `integrations/genesis_renderer_hdr_v2_overlay/`
  applied to a matching Genesis source tree;
- an offscreen RGB/depth renderer with explicit `radiance=True` for the
  scene-linear HDR attachment;
- the default Genesis installation and the direct baseline remain unchanged.

The adapter is deliberately a **single-environment compatibility path**.
The Torch batch kernels do not constitute an end-to-end Genesis B>1 adapter;
the Genesis-facing wrapper currently selects environment 0.

## 2. Simulator input contract

Each capture interval is defined by two timestamped endpoint states. A valid
provider supplies, or explicitly marks unavailable:

1. **Scene-linear radiance/HDR**: finite, non-negative floating-point RGB
   buffer `[H, W, 3]`. Native HDR comes from the Genesis overlay or a measured
   provider. Display RGB converted to linear values is an explicitly labelled
   fallback and must not be reported as calibrated EVK4 radiance.
2. **Depth**: finite metric depth `[H, W]`, in the renderer camera coordinate
   convention used by the projection code.
3. **Segmentation/material support**: `[H, W]` IDs or a documented validity
   mask, used to distinguish visible support and disocclusion.
4. **Camera state**: intrinsics, distortion/crop metadata, and camera pose
   (translation plus orientation) at each endpoint.
5. **Object/link state**: world pose for every moving object or articulated
   link needed by the correspondence path; endpoint velocities alone are not
   accepted as a hidden continuous trajectory.
6. **Timestamps**: monotonically ordered endpoint and sub-interval times with
   declared units and an episode origin.

Missing or unsupported geometry must produce a `RenderRequest` in strict mode;
the plugin must not silently fabricate a Genesis sub-frame.

## 3. Intermediate-frame and sampling contract

For every requested interpolation alpha, the reference path:

```text
Genesis physical state
  -> object and camera SE(3) motion
  -> depth-aware 3-D reprojection
  -> bidirectional motion warp
  -> z-buffer visibility test
  -> real-render fallback for occluded/disoccluded regions
  -> DVS event generation
```

The current endpoint path uses a constant-twist SE(3) approximation between
known endpoints. It records unsupported/disoccluded pixels and can fail closed
with `RenderRequest`; `hold`/`nan` are compatibility policies, not physical
ground truth.

The adaptive sample count is the maximum of independent numerical and sensor
guards:

\[
U = \max(U_{motion}, U_{radiance}, U_{visibility}, U_{depth},
U_{camera}, U_{articulation}, U_{sensor}).
\]

The sampler must retain diagnostics for motion, correspondence depth,
`z_alpha`, radiance residual, disocclusion/unknown support, and IIR convergence.

## 4. Output contract

The plugin exposes a compatibility `capture()` result and recorder outputs:

- raw AER event stream: `aer_events.h5`, dataset `events` with rows
  `[t_us, x, y, p_01]` and `uint64` storage;
- compatibility in-memory event rows: `[t(s), x, y, p]` as documented by the
  public API;
- optional event frames: `event_frames.h5`, `(T, H, W, 2)` ON/OFF channels;
- optional RGB/reference frames and timestamped metadata;
- calibration status, sensor configuration, resolution, camera profile,
  random seed, and input-domain provenance.

Event polarity, timestamp units, coordinate origin, resolution, and episode
ranges are part of the schema and must not be inferred from downstream reward.

## 5. Frozen Core directories

The following are Core-owned and may be changed on this branch only under the
approval rule below:

- `genesis_event_plugin/`;
- simulator correctness / physics / renderer / API tests under `tests/`;
- `integrations/genesis_renderer_hdr_v2_overlay/`;
- `calibration/` schema and validators;
- Core architecture and reproducibility documentation under `docs/`.

The following are intentionally outside the Core migration boundary and must
not be tuned or copied as part of a Core change:

- formal benchmark protocols, results, and quality-evaluation assets under
  `benchmark/`;
- Dreamer/DreamerV3/WAM or other robot-task training code;
- generated HDF5/video/event datasets, checkpoints, and caches;
- server-specific logs, credentials, private keys, tokens, and paths.

The current maintenance branch inherits the repository's existing tracked
history, including benchmark files from `origin/main`; this migration commit
does not modify or import those assets into the Core workflow.

## 6. What is output-affecting

A change is output-affecting if it can change event values, event timestamps,
polarity, event ordering, visibility support, or the input radiometry. Examples
include renderer attachments or shaders, RGB/linear/HDR transfer, camera
intrinsics/distortion/crop, pose interpolation, depth/z-buffer/occlusion rules,
adaptive-sampling guards, DVS threshold/noise/leakage/IIR/refractory/readout
models, event quantization, coordinate conventions, or timestamp handling.

Pure documentation, test-only assertions, formatting, and non-executed
diagnostic labels are not output-affecting provided they do not alter runtime
defaults.

## 7. Approval and upgrade rule

No output-affecting development commit may replace the formal frozen commit
for quality comparisons without project-control approval. An approved upgrade
must use the same unified protocol and be evaluated in **three matched
experiments** (same scene/trajectory, camera profile, resolution, seed,
timestamps, metrics, and resource budget) before a new frozen tag is named.
The three results and the exact commit must be recorded together. Benchmark
parameter tuning cannot be used to justify a simulator change.
