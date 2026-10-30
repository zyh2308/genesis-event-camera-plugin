# R1.5 CUDA support parity and strict-sampling run F

**Result:** CPU/CUDA parity passes for the two archived clips and the public `PhysicsInterpolator` path. This is not a full renderer-fidelity or core-freeze pass.

## What caused the 106-point disagreement

The 106 points were accumulated at a strict `0.995` projected-footprint coverage boundary. Feeding **identical projected coordinates** into the CPU and CUDA compositors gave identical support, surface IDs, depth decisions, and surface-consistency decisions. Coverage differed by at most `1.20e-7`; supported log-luminance differed by at most `4.77e-7`.

The mismatch was introduced earlier, in projection. The NumPy path retained double-precision geometry, while the public Torch caller had already converted interpolation time, camera intrinsics/poses, and object poses to float32 before calling the projector. This produced up to `6.11e-5 px` UV and `2.39e-7 m` depth differences in the earlier trace. A tiny change was enough to move a pixel across the coverage cutoff or a depth/surface comparison. The issue was precision consistency at the caller/projector boundary; lowering the support threshold would have hidden it.

The candidate now preserves float64 geometry through the public caller and projector, with float32 image/compositor outputs. In run F, projected UV differed by at most `2.85e-14 px` (numerical round-off), projected depth had zero measured difference, and the same-projection compositor checks had zero component mismatches. All 106 historical XOR locations now receive the same CPU/CUDA decision at the unchanged `0.995` cutoff. This does **not** mean all 106 become supported: 69 are supported by both paths and 37 are consistently rejected/unknown by both (35 coverage rejections, one surface-consistency rejection, and one cross-endpoint surface/depth rejection). The per-location projection, both coverages, depth layers, surface IDs, and rejection reason are in `point_components.csv` and `layer_contributors.jsonl`.

## Before and after

| Check | Before candidate correction | Run F |
|---|---:|---:|
| Historical support-mask XOR, episode 004 | 79 px | 0 px |
| Historical support-mask XOR, episode 005 | 27 px | 0 px |
| Maximum same-projection CPU/CUDA coverage delta | up to `1.20e-7` | `1.20e-7` (within exact decision margin) |
| Maximum supported log-luminance delta | `2.21e-4` / `2.17e-4` in run 30c | `4.77e-7` |
| CPU/CUDA DVS threshold probes | 12/12 matched in run 30c | 12/12 exact raw-event sequence |
| Public interpolator support-mask XOR across all 7 samples | not previously tested end-to-end | 0 px for both episodes |

Run F used the existing acceptance criteria unchanged: exact support masks, supported log-luminance error `<=0.02`, exact CPU/CUDA event sequence on threshold probes, and cached-clip event-count delta `<=2%`.

## Complete event-chain result

For each clip, all four combinations (NumPy reconstruction + CPU DVS, NumPy + CUDA DVS, CUDA reconstruction + CPU DVS, CUDA + CUDA) agree in event count, polarity, and per-pixel/polarity histogram. The two reconstruction outputs also agree through the public `PhysicsInterpolator` at every adaptive output time.

| Clip | Candidate events per reconstruction | ON / OFF | Cross-backend pixel-polarity L1 | Max cross-path event-time error |
|---|---:|---:|---:|---:|
| 004 | 397 | 243 / 154 | 0 | `9.95 ns` |
| 005 | 582 | 361 / 221 | 0 | `11.8 ns` |

The 7 adaptive timestamps are exactly shared between backends. Unknown fractions also match frame-by-frame: 004 ranges from 4.88% to 0%; 005 from 4.97% to 0%. The CPU/CUDA interpolator diagnostic took 1.118/0.226 s for 004 and 0.844/0.158 s for 005; these are replay-path timings, not a live Genesis renderer throughput claim.

The local controls also pass: coverage at and one float32 ULP around `0.995`, a tiny positive footprint that remains visible but unsupported, front/back occlusion, and positive/negative DVS threshold-adjacent inputs.

## Intermediate radiance, causality, and cost

The exact archived Direct-HDR sequence contains 10 actual scene-linear frames from `0.5815` through `0.5905 s` at 1 ms spacing. Feeding these frames to the strict path reproduces the archived images and raw events exactly: 625 events for 004 and 939 for 005; pixel/polarity histograms and event timestamps match exactly. Outputs are marked available at the right endpoint (`0.5905 s`), and the strict path does not use future state.

A single 5 ms midpoint is not enough: it yields 413/625 events in 004 and 615/939 in 005, with pixel-polarity L1 differences of 212/324 and event-time P95 errors of 4.013/4.275 ms. The motion-derived schedule asks for six additional real radiance samples in this 10 ms interval (seven output samples total). When all 1 ms archived frames are available, they already satisfy the spacing requirement. Their availability time is the corresponding observation time, while the event sequence for the full bracket is only available at the right endpoint. Any live-render wall delay must be added separately.

The strict API replay using cached frames took 1.132 s / 0.871 s on CPU for 004/005 and required zero **new** renderer calls because the frames already existed. This is not the cost of acquiring the frames. A prior actual Genesis benchmark measured full-frame fallback for 42/42 states over a 40 ms window: 0.677/0.950 s fallback rendering and 7.102/6.107 s candidate total pipeline versus 1.220/1.713 s direct-render-plus-DVS. That is useful evidence that the prior fallback path is costly, but it is **not** the exact six-request, 10 ms strict-sampling schedule. The run F config records this cost reference and its scope.

**Freeze decision: do not switch the formal default core yet.** Backend consistency is now accepted for these clips. The remaining freeze gate is a fresh, same-state Genesis rerender of the strict intermediate samples with render, transfer, DVS, and total wall time measured separately, plus confirmation of when those observations can be delivered in an online causal path. Cached replay proves that the representation and timing contract work when actual samples are available; it does not prove their production cost or real-time availability. Renderer fidelity and downstream perception are separate gates too.

## Reproduction and artifacts

- Candidate source branch: `exp/r1_5-core-candidate-20261030`.
- Numerical run source commit: `9e6b5d7e2689d541087a56ff1fbf0fc904571b69`.
- Result archive commit: `52c4b2280914710c00fc36899f745dca724dfdc3` (subsequent documentation commit updates this README and `SHA256SUMS`).
- Device/environment: g1, RTX 3090 UUID `GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460`, PyTorch `2.5.1+cu121`, CUDA runtime `12.1`.
- Config SHA256: `649219ea880b73e5c9dcf73c1cbd39165d5c6a309e386a97cc3fe94ef65dccf8`.
- Episode 004 state/HDR cache SHA256: `0525c18c8d89f5831ef0cb71ac9e3b6b130d60579daf374f06d5cfcee5b879d7` / `b3afe5b5badee3d2606d343bb750aa970887928c41b8001a5fc4e9ff32b9c321`.
- Episode 005 state/HDR cache SHA256: `91c1397642e317d6bb51150b4107c518b52e4fa9fcf5f1caf6a2a91315c7d434` / `f161079a24c1bf9fee838a6fbb2a0fa5d785435b116f0ea5628e6efbba497236`.
- The output `SHA256SUMS` covers the summary, configuration, raw forensic arrays, per-point diagnostics, and event metrics. The runtime-log bundle has its own `SHA256SUMS`.

From the repository root on g1, using the recorded candidate source and the cache path above:

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460 \
  /home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python \
  benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_cuda_support_forensics.py \
  --cache-root /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-diag-20261019/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059 \
  --config benchmark/robot_closed_loop/results/r1_5_cuda_support_forensics_20261031f/config.json \
  --output benchmark/robot_closed_loop/results/r1_5_cuda_support_forensics_20261031f_rerun
```

The original run's detached launch, preflight, allocation, exit status, and checksums are under `benchmark/robot_closed_loop/results/runtime_logs/r1_5_cuda_support_forensics_20261031f/`. No formal training or generator ranking was run.
