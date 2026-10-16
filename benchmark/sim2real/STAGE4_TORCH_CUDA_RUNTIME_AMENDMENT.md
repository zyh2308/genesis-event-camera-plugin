# Stage 4 EventGenesis CUDA runtime amendment

Date: 2026-10-16 UTC  
Purpose: preserve the registered Stage 4 physical/sensor/training protocol while using the existing batched CUDA implementation after the registered CPU NumPy implementation proved too slow to reach the first interval.

## Evidence and decision

The first Stage 4 campaign (`20261016-stage4-flowdt1-seed0-v3`) passed environment and source pin checks and rendered endpoint 0 of rollout `s0_00`. It had no next-endpoint progress during the recorded ~8 minute run. Its generator used the Stage 4 JSON setting `warp_backend=numpy`, `device=cpu`. The partial bank and all logs are preserved under:

`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v3/`

The sampler arithmetic is fixed by the existing config: for a 10 ms endpoint interval, 300 Hz cutoff gives `tau=1/(2*pi*300)=0.0005305164769729845 s`; with `sensor_eps_target=0.05`, the bound is 377 samples, capped by the configured maximum at 300. The NumPy reference then performs per-alpha projection and bidirectional visibility splats.

A bounded remote diagnostic used one 346×260 synthetic interval, the same 10 ms duration, 300-sample sampler, locked EventGenesis sensor settings, and the existing Torch CUDA projected-stack/warp backend. It ran on physical GPU 1 (RTX 3090), with no other GPU job on that device. Result JSON and logs are at:

`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v3/diagnostics/backend_torch_cuda_v2.json`

Measured results:

| Check | Result |
|---|---:|
| Projection + batched warp for 300 intermediate samples | 2.4154 s |
| DVS integration of those samples | 0.5668 s |
| Combined interval | 2.9822 s |
| Peak CUDA allocated / reserved | 6.60 / 7.47 GiB |
| NumPy-reference projection max error at alpha indices 0, 150, 299 | 3.052e-5 px |
| Projected depth max error | 2.384e-7 m |
| Projection/warp validity mask disagreements | 0 |
| Warped radiance max absolute difference at those samples | 4.247e-4 |

This is a bounded backend diagnostic, not a complete Genesis rollout or a generator-quality comparison. The 3-sample comparison establishes close numerical agreement and exact checked validity masks; it does not establish bitwise event equality. At the measured interval time, 6,400 intervals imply about 5.3 hours for projection/warp/DVS alone, before Genesis rendering, I/O, V2E export, and training. This is a planning estimate, not measured full-bank runtime.

The next campaign uses the existing batched Torch CUDA implementation. The explicit Stage 4 runtime config records `warp_backend=torch`, `device=cuda:0` (logical index 0 with `CUDA_VISIBLE_DEVICES=1`, physical GPU 1). It retains the 300-sample rule, thresholds, noise/readout settings, physical rollouts and labels, event windows, both generators, downstream model, optimizer, initialization, data split and training budget. It changes the numerical execution backend and float precision path; the measured differences above are retained rather than described as exact equality.

## Registered config

- Config: [`configs/stage4_training_launch_v1_torch_cuda.json`](configs/stage4_training_launch_v1_torch_cuda.json)
- SHA-256: `c0ab0d17f024b00106957c3e44d7a013e920f6e760f70266c43a9cf904634685`
- Checksum file: [`protocol/training_launch_v1_torch_cuda.sha256`](protocol/training_launch_v1_torch_cuda.sha256)
- Bank output directory: `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/training_bank_v1_torch_cuda_32x2s/`
- Frozen simulator core unchanged: `e79e656ac00f7847b98db9e12bf43b093f3afff1`

Training parameters and all synthetic-only / real-test isolation rules are byte-for-byte inherited from the registered v1 config except the explicit bank version label, EventGenesis warp backend/device, and this amendment metadata. The original `stage4_training_launch_v1.json` remains unchanged as the failed CPU NumPy attempt's config record.

## Scope limits

- The diagnostic fixture is synthetic and tests only backend execution/selected numerical outputs at the registered raster and sampler size.
- The full Stage 4 bank, V2E export, STE adapter, and optimizer updates have not yet completed at the time of this amendment.
- No real data was read, scored, or used to choose this runtime path.
- The result is an engineering feasibility decision. It makes no claim about EventGenesis superiority or real-data generalization.
