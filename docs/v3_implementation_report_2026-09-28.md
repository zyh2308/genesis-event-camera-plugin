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

## Correctness patch / v3.1

### Target-space visibility and disocclusion

The previous implementation used `valid01 & valid10`. Although the arrays had
the same shape, `valid01` was indexed on the t0 source grid and `valid10` on
the t1 source grid; equal indices therefore did not identify the same surface
point. That expression was not a geometrically meaningful disocclusion mask.

The projected splat now exposes separate diagnostics:

- `source_valid`: validity on the source image grid;
- `occupancy`: any source sample that lands in a target pixel;
- `visibility`: samples that survive the target-domain `z_alpha` z-buffer;
- `intermediate_visibility`: the merged endpoint support at an interpolated
  target image;
- `unknown_region`: target pixels not covered by reliable projected support;
- `disocclusion`: target pixels that are valid in the real endpoint but are not
  covered by the opposite endpoint projection.

For t0→t1, the canonical mask is computed in t1 coordinates as
`target_valid1 & ~visibility_0_to_1`; the reverse direction is computed in t0
coordinates independently. The sampler's `disocclusion_fraction` uses the
target-domain unsupported mask (`~visibility`, including invalid target depth),
and strict `RenderRequest` decisions consume that result. The code and
diagnostics explicitly state that disocclusion is defined in the projected
target image domain.

### Radiance residual support

`endpoint_radiance_residual()` warps `log(I0)` into the t1 target domain and
computes residual only where the projected splat has target visibility and the
target frame is finite/non-negative. It returns NaN elsewhere (or an explicit
support mask with `return_support=True`). It no longer intersects a target
mask with `valid0` from the t0 source grid. A perfect geometrically warped
target therefore produces approximately zero residual only on the projected
correspondence support.

### Depth correspondence

The old pair `endpoint0[1]`/`endpoint1[1]` was not a correspondence: the two
arrays were generated from opposite source grids and different target poses.
The v3.1 sampler instead accepts a per-source-material-point quantity
`|z_target - z_source| / z_source`. For the t0→t1 direction, each t0 source
point's original depth is compared with its own projected `z_alpha` at t1.
`used_correspondence_depth=True` now reports method
`source_material_point`, and axial-motion tests check the analytic value.

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

The Torch projected batch API can also return occupancy, visibility, z-buffer,
and intermediate-unknown diagnostics. These were compared against the NumPy
reference with explicit array/mask tolerances in the v3.1 tests.

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
  physical-log distinction, pure rotation, axial `z_alpha`, articulated-link
  separation, two-object z-buffer occlusion, target-space disocclusion,
  target-domain residual support, source-material-point depth correspondence,
  B=2/T=3 projected Torch warp, packed batch DVS, combined sampler guards,
  strict render request, and calibration schema validation.
- CUDA is available in the current environment. CUDA projection, warp
  diagnostics, and end-to-end `PhysicsInterpolator` parity against NumPy were
  run successfully (projection/mask tolerance at `3e-5`; mask equality exact).
- Full `tests/test_core_correctness.py` remains dependency-gated here because
  the environment lacks SciPy; no dependency was installed as part of this
  branch. Run it after installing the declared requirements.

## What remains before a scientific claim

1. Implement measured readout/dead-time only after a timestamped camera
   calibration is available.
2. Connect real Genesis HDR/depth/segmentation/link-ID/timestamp providers and
   validate the overlay on the exact Genesis commit/backend used in experiments.
3. Run matched PECS/V2E/EVIS and real-EVK4 protocols; report event rate,
   polarity, timing, spatial CD/GD, and uncertainty rather than relying on
   downstream robot reward alone.

The remaining Genesis adapter is intentionally single-environment: its
batched pose compatibility code selects environment 0. The B>1 Torch kernels
are therefore not an end-to-end Genesis batch claim. The endpoint interpolation
is also still an endpoint constant-twist approximation, not a recovered hidden
physics trajectory; unsupported content still requires a real intermediate
Genesis render.
