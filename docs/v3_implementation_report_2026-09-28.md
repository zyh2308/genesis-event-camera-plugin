# Physics/radiance v3 implementation report

## Scope and evidence boundary

This branch turns the earlier physics/radiance prototype into a testable
mainline. It does **not** claim complete Prophesee EVK4 reproduction, absolute
scene radiometry, or superiority over PECS/V2E/EVIS. Those claims require a
matched renderer/camera protocol and unit calibration data that are not in
this repository.

The public API now distinguishes three response modes:

| mode | meaning | status |
| --- | --- | --- |
| `video_linlog` | historical V2E-compatible piecewise transfer with DN=20 knee | legacy baseline |
| `physical_log` | `log(max(gain * I + offset, floor))` on explicit linear input | uncalibrated/effective until measured |
| `calibrated_transfer` | monotone measured input-to-log lookup table | fail-closed outside table |

Every bundle/output records `radiometric_calibration_status`. A Genesis
display RGB buffer is still an explicitly marked sRGB fallback; it is not
called EVK4 radiance.

## Physics path

`PhysicsInterpolator` now interpolates endpoint camera poses with linear
translation and an SO(3) log/exp rotation path. For every adaptive alpha it:

1. reconstructs endpoint pixels as 3-D camera points;
2. applies an object endpoint transform (absolute `object_world0/1`, or the
   legacy delta map);
3. transforms to the interpolated camera pose;
4. projects to image coordinates and uses the projected `z_alpha` for the
   forward-splat z-buffer;
5. fuses the two endpoint projections without fabricating a Genesis render.

This is a constant-twist endpoint approximation, not the hidden continuous
physics trajectory. If endpoint geometry leaves unknown/disoccluded pixels,
the pipeline records a `RenderRequest(t, alpha, reason, mask, diagnostics)`.
`hold` and `nan` remain backward-compatible policies; `raise`,
`request_render`, or `strict_fidelity=True` fail closed and require a real
Genesis subframe from the caller.

## Adaptive sampling

The independent bounds are now combined as

\[
U=\max(U_{motion},U_{radiance},U_{visibility},U_{depth},U_{camera},
U_{articulation},U_{sensor}).
\]

The diagnostics include boundary motion from both segmentation endpoints,
correspondence depth and `z_alpha` depth change, endpoint radiance residual
max/P99, disocclusion fraction, and an IIR convergence guard. The sensor guard
is a numerical target (`sensor_eps_target`), not a hard-coded EVK4 constant.

## GPU/batch path

`depth_aware_bidirectional_warp_projected_batch_torch` accepts endpoint frames
`[B,H,W]` or `[B,H,W,C]` and projections `[B,T,2,H,W]`/`[B,T,H,W]`. It flattens
`B*T` into one device-side z-buffer and uses scatter reductions for the four
bilinear candidates; it does not loop over environments or pixels and does not
copy intermediate frames to CPU.

`DvsBatchEmulator` is intentionally experimental and limited to the noiseless,
no-IIR, no-refractory, ideal-readout subset. It returns packed device events
`[batch,t,x,y,p]` and device offsets. Full single-environment V2E-style noise,
calibrated maps, photoreceptor noise, leakage, and readout behavior still use
`DvsEmulator`; no unsupported feature silently falls back.

The Torch backend of `PhysicsInterpolator` now creates the projected alpha
stack on-device as well. Its legacy `return_torch=False` API necessarily
converts each final frame back to NumPy for the existing `DvsEmulator`; callers
that use `return_torch=True` retain the `[B,T,H,W]` tensor. The Genesis-facing
single-environment wrapper is not yet wired to the experimental batch DVS, so
the final plugin capture path still has this explicit compatibility boundary.

## Genesis renderer overlay

The optional 1.2.2 overlay can expose an offscreen `RGBA16F` scene-linear
buffer through `camera.render(radiance=True)`, alongside depth and
segmentation. BatchRenderer and ray-tracer calls reject this request clearly.
`camera.render(motion_vectors=True)` also rejects explicitly: the overlay does
not retain current/previous clip-space positions for every visible primitive,
so returning a camera-only or endpoint-reconstructed vector would be wrong for
articulated links. The projected physics reference remains the documented
fallback until a native attachment is implemented.

## Calibration schema

`calibration/<camera_id>/v3/` is validated by `CalibrationBundle`. It separates
intrinsics, radiometry, ON/OFF threshold distributions, background noise maps,
temporal response, and readout evidence, with source labels such as measured,
nominal, fitted, or assumed. The repository contains no fabricated unit
calibration values.

## Verification performed

- Python bytecode compilation for package and tests: passed.
- Existing dependency-light physics tests (`tests/test_physics_core.py`):
  passed when invoked directly in the current environment.
- New v3 checks (`tests/test_v3_mainline.py`): passed directly, including
  physical-log distinction, axial `z_alpha`, B=2/T=3 projected Torch warp,
  packed batch DVS, combined sampler guards, strict render request, and
  calibration schema validation.
- Full `tests/test_core_correctness.py` remains dependency-gated here because
  the environment lacks SciPy; no dependency was installed as part of this
  branch. Run it after installing the declared requirements.

## What remains before a scientific claim

1. Add a Torch-native SE(3) projection kernel and benchmark against the NumPy
   reference on the target GPU.
2. Implement measured readout/dead-time only after a timestamped camera
   calibration is available.
3. Connect real Genesis HDR/depth/segmentation/link-ID/timestamp providers and
   validate the overlay on the exact Genesis commit/backend used in experiments.
4. Run matched PECS/V2E/EVIS and real-EVK4 protocols; report event rate,
   polarity, timing, spatial CD/GD, and uncertainty rather than relying on
   downstream robot reward alone.
