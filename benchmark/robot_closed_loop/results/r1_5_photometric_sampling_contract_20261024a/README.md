# R1.5 photometric timing and sampling contract

**Date:** 2026-10-24 (Asia/Tokyo)  
**Status:** structural diagnosis and interface candidate only; no renderer was started in this run.  
**Execution branch:** `exp/robot-closed-loop-v1`, local code commit `a7b276487e82aaae80f430d08594bf496f174fda`.  
**Frozen core:** unchanged.  
**Evidence source:** copied research audit bundle in [`evidence/`](evidence/), with its original manifest preserved. The outer `SHA256SUMS` covers this report and every archived file.

## What the new evidence establishes

The continuous-surface image test and the static-floor event test ask different questions. The former checks geometry and pixel radiance at a chosen frame. The latter bypasses spatial warping and checks whether endpoint brightness can represent the light history between two frames. The results show that good geometric support does not establish a correct temporal light signal.

At 0.5855 s, the two-endpoint continuous-surface candidate has valid same-label support for 83.53% / 83.93% of the image in episodes 004 / 005. Within the common valid region, 63 / 92 pixels differ from exact cached Direct-HDR by more than 0.2 log units; 56 / 62 are ground pixels, and 36 / 62 lie outside the segmentation edge band. The legacy point/`next_primary` candidate has 530 / 484 such pixels on that same common mask. The image comparison uses only real recorded times and does not fill unsupported pixels into the score.

The event test uses 61,322 / 61,039 stationary floor pixels, unchanged camera pose, identical endpoint depth, and ten actual cached radiance observations from 0.5815 through 0.5905 s. It initializes the same ideal DVS model from the real 0.5805 s endpoint and changes only the brightness sequence. Endpoint log-linear interpolation is an interval model; the direct rows are actual saved renderer observations.

## Image, event, timing and cost measurements

| Quantity | Episode 004, center | Episode 005, +55 mm | Scope |
|---|---:|---:|---|
| Continuous-surface support at 0.5855 s | 83.53% | 83.93% | Two endpoint source views only |
| Common valid same-label pixels | 64,152 | 64,459 | Exact reference and candidate both valid |
| Common valid pixels with `abs(log error) > 0.2` | 63 | 92 | Among those, ground pixels 56 / 62; outside 2 px segmentation band 36 / 62 |
| Static floor pixels in temporal test | 61,322 | 61,039 | Fixed mask; always visible and at least 2 px from geometry-label boundaries |
| Events from direct cached radiance | 308 | 451 | Ten actual intermediate observations |
| Events from endpoint log-linear radiance | 189 | 216 | Same times and same DVS response |
| Event count difference | -119 (-38.6%) | -235 (-52.1%) | Endpoint estimate minus direct cache |
| Direct events on pixels whose endpoint log delta is below threshold | 71 | 169 | These middle changes are not inferable from the two endpoints alone |
| Same-pixel/polarity paired timing error, median / P95 | 2.583 / 6.537 ms | 2.593 / 6.353 ms | Temporal-order pairing; unmatched events remain a count difference |
| Pairs within 0.1 ms / within 1 ms | 5 / 41 of 189 | 4 / 50 of 216 | Tolerance counts are not global optimal matching |
| New renderer / candidate end-to-end cost | Not measured | Not measured | g1 connection failed before resource inspection; no GPU work was launched |

The cached-region image diagnosis itself took 4.113 s on the research WSL CPU. This is array analysis time, not a renderer or production cost. Earlier, separately configured surface-raster profiles measured 4.293 s candidate vs 3.151 s Direct-HDR (1.36x) for CPU profile AA, and 13.540 s vs 2.619 s (5.17x) for CUDA profile AE. Those old profiles are context only and cannot be transferred to the proposed shadow/MSAA ablation or this OpenGL candidate.

The source and direct cache HDF5 hashes used by the image partition diagnostic are:

| Episode | Source cache SHA256 | Direct-frame evidence SHA256 |
|---|---|---|
| 004 | `0525c18c8d89f5831ef0cb71ac9e3b6b130d60579daf374f06d5cfcee5b879d7` | `b3afe5b5badee3d2606d343bb750aa970887928c41b8001a5fc4e9ff32b9c321` |
| 005 | `91c1397642e317d6bb51150b4107c518b52e4fa9fcf5f1caf6a2a91315c7d434` | `f161079a24c1bf9fee838a6fbb2a0fa5d785435b116f0ea5628e6efbba497236` |

## What is a bug, and what is a model limit

**Proven candidate implementation defect, already addressed in the diagnostic surface path:** point splats could let tiny foreground contributions win visibility and produce whole-pixel changes on continuous surfaces. The triangle surface candidate passes its CPU general controls; the real two-endpoint example still leaves roughly 16% unsupported. This is a geometry/coverage problem, independent of the temporal result below.

**Proven endpoint-model limit:** on fixed ground with constant geometry, the exact cached radiance produces 308 / 451 events while endpoint log-linear radiance produces 189 / 216. Events also occur at pixels whose endpoints are below the DVS threshold. An endpoint-only function cannot recover an unobserved intermediate brighten-then-darken history from identical/subthreshold endpoints. The current endpoint residual/sampling bound therefore cannot certify the absence of such a history. This does not prove that dynamic shadows are the only source: the shadow-off intervention has not run, and reflection or another illumination/render-pass effect remains possible.

**Sampling-contract omission:** the local HDR renderer source requests `min(GL_MAX_SAMPLES, 4)` for its multisample attachments but does not log the actual attachment sample count, queried sample locations, or resolve mode with the output. The radiance path resolves the multisample color buffer into a single-sample `RGBA16F` buffer; segmentation is a separate pass with `GL_MULTISAMPLE` disabled, and depth uses a separate resolve. In this overlay, public `Camera.render(antialiasing=...)` is forwarded to BatchRenderer only; the headless rasterizer path does not receive it and uses its own fixed multisample setup. The current OpenGL surface candidate uses a single-sample private FBO. It changes the rasterization backend but does not retain the renderer's per-sample color/coverage data or add intermediate light observations.

The current local overlay source hashes are `renderer.py` `0c7fe474b49bbcc88e747090629174fb97f7f0ceb0d2693f74c721be40dccd9d`, `camera.py` `78c18ffada07a98657ddf11326e833fc928431ecead674434aadb25dc023b714`, `rasterizer.py` `16ad98c2392cc86d07b230c3fa2a20c720679223bfd4213110efe570d71026e3`, and `mesh.frag` `1dc5d8fe568e0569927b6605eb26041128b0ff964d834a5709a1d394664ec32b`. These are hashes of the checked-out overlay source, **not proof that the same files were active on g1 for the earlier cache run**.

The Khronos OpenGL 4.6 specification says that resolving multisample floating-point color to a single-sample buffer is implementation-dependent; the `filter` parameter is ignored for that resolve. It also distinguishes queryable shading sample positions from rasterization coverage positions, which need not coincide; coverage positions cannot be queried by `GetMultisamplefv`. Therefore a run should log the queried shading positions and use a diagnostic sample-mask pass if it needs empirical per-pixel coverage, without describing the former as the latter. See [Khronos OpenGL 4.6 specification, framebuffer blits](https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf#page=561) and [multisample positions/shading](https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf#page=487).

## Versioned FramePacket candidate

Keep this as a benchmark/adapter-level candidate first. A packet represents one actual renderer observation, not a fabricated intermediate state.

```text
FramePacket v1
  identity: episode_id, camera_id, frame_id
  clocks: scene_state_time_s, capture_started_monotonic_ns,
          capture_completed_monotonic_ns, downstream_available_monotonic_ns
  radiance: float data, HxW or HxWxC, channel order, scene-linear domain,
            gain/floor/white-level metadata, validity mask
  camera: resolution, intrinsics, pixel-center convention, ray convention
  geometry (internal generation only): depth + explicit camera-Z/ray-range
            semantics, surface ID/segmentation, validity
  sampling: renderer-requested and attachment-actual sample count, GL vendor/
            renderer/version, queried shading sample positions, per-sample
            radiance/depth/surface ID when retained
  resolve: color, depth and label resolve rules recorded separately
  provenance: renderer/source hash, render flags (shadow/reflection), pass IDs,
              exposure and any post-resolve filtering
  mode: strict_sampled | legacy_resolved
```

An interval estimate carries two independent masks: `geometry_support` and `radiance_temporal_support`, plus reason codes, source state times, and event availability time. `radiance_temporal_support` is false by default between two legacy endpoints. It becomes supported only by an actual observation at that query time or by an explicit, validated temporal-radiance model; a small endpoint residual alone cannot turn it true. Geometry and radiance request masks are unioned for repair, while the event estimator still receives only the resulting event stream, never the internal depth or segmentation.

The event timestamp remains the state time represented by the radiance; availability is the time the render/DVS result could first be consumed. Both must be logged. Offline replay may use an exact cached state at its actual time. Online code may not retrieve a past state after it has passed and label a later approximation as that observation; it must mark the missing interval, delay delivery, or make a new observation at a state that still exists.

## Minimal compatibility-preserving change proposal

1. Add a `FramePacket` sidecar/API in the adapter layer and validate it before array conversion. Keep the old four-array API through an adapter that labels packets `legacy_resolved`; do not silently assign sample count, sample positions, or a resolve rule.
2. Add an `IntervalEvidence`/typed `RenderRequest` with separate geometry and photometry masks/reasons, time and availability fields. Preserve the old `mask` field as a compatibility projection of their union.
3. In permissive legacy mode, preserve current results byte-for-byte but log `photometry=unknown_between_frames`. In strict mode, request an actual render/subsample sample when temporal photometry is unsupported; fail closed if the requested historical state is unavailable. Never use future GT or scene labels as the downstream estimator input.
4. Version manifests and output directories by packet and resolve protocol. Existing caches, scores and checkpoints remain reproducible as legacy artifacts; new packet/resolve outputs form a separate distribution and must not be pooled with them without an explicit paired retraining/evaluation plan.

This proposal does not yet select the final temporal-light model or promise a production sample buffer. Retaining S per-sample RGB/depth/identity values has memory and readback costs not measured here.

## Bounded next experiment, when g1 can be inspected safely

No cadence sweep or general six-control matrix is planned. Reuse episodes 004/005 and their saved states. First compare only source / midpoint / endpoint (`0.5805`, `0.5855`, `0.5905` s), then extend to the ten already saved intermediate rows only if this locates a contribution. Add one simple high-contrast contour control.

| Factor | Levels | Must stay matched within each cell |
|---|---|---|
| Shadows | on / off | same state, light, camera, radiance/depth/seg passes; reflection setting separately recorded |
| MSAA | off / actual renderer sample count | direct reference and candidate use the same mode and resolve; query and record actual attachment samples |
| Frames | source / midpoint / endpoint | real cached or exact logged states only; no synthetic physics rows |
| Regions | ground interior, moving-surface interior, 2 px geometry contour, newly exposed/unsupported | analysis masks only; they never enter the perception model |

Each run records active renderer commit and source hashes, GL vendor/renderer/version, requested and actual samples, queried shading sample positions, whether a sample-mask coverage pass ran, FBO formats and readback pass, independent color/depth/label resolves, render flags including reflection, and per-pass plus end-to-end cost. At the event level, report counts, unmatched events, and same-pixel/polarity timing-order matches at predeclared tolerances 0.1 / 1 / 5 ms. Candidate and direct reference must use identical settings inside each cell; the formal Direct-HDR reference protocol remains preserved and unchanged.

If shadow-off removes the floor variation, shadowing is implicated for these clips, not proven the only cause elsewhere. If it persists, next isolate reflective-floor pass and other light/material terms. Only after this diagnostic should a small photometric-unknown-region sampling candidate be measured against Direct-HDR cost.

## Execution and resource boundary

On 2026-10-24, this execution session made one serialized read-only SSH preflight to `zhaoyuhan@10.129.163.68:22`; it timed out before the remote command ran. I stopped remote operations. The research-side earlier successful snapshot reported utilization 85/83/94/94/95/94% on GPUs 0–5, but that is not a current resource check. No GPU availability is claimed and no renderer was started. The local array analysis was CPU-only. Before any future render, repeat a single read-only GPU/process ownership preflight; if no GPU is idle, continue only CPU/interface work.

## Reproduction and archived evidence

The exact audit scripts and outputs are preserved under [`evidence/`](evidence/). They depend on the two archived HDF5 source/direct caches named and hashed above, which are not duplicated in this small report bundle. The original research workspace remains `C:\Users\zhaoyuhan\Documents\ChatGPT\New project\docs\robot_structural_diagnosis_2026-10-24\`.

| Item | SHA256 |
|---|---|
| `evidence/FINDINGS.md` | `4e6c074d34843e61583a9bb3df8c43768145c84df48996b68763f6e9e38c8681` |
| `evidence/cached_regions.json` | `84d98c5310859dff9c625cfb9c8008084157ffeb68394aafa44145a3382414c5` |
| `evidence/static_surface_time.json` | `68141686afd62093bf0719b3a4bf6d9669158223a21204fb0a9f2255733ca8e2` |
| Episode 004 event NPZ | `211ebf6b409c9d1b3e0a2c09f90733dc5c39171d655bf2280fcdaadc7e4aa30d` |
| Episode 005 event NPZ | `9d31d2066bf9f8f3e50920f0eaddf6a1b4e9f29cd0961fc87a07aba54b05c63d` |

The source archive’s `SHA256SUMS` is retained unchanged; the parent `SHA256SUMS` validates the complete copy. No supplied script was rerun in this execution turn, and the old six-control JSON was archived only as part of provenance.
