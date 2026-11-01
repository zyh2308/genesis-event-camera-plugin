# R1.5 localization recovery: source readiness

## What is ready

The nine saved robot state traces have been replayed at 0.5 ms without rendering. They match the prior physical traces and the saved 100 Hz HDR cache after launch. The training-source episodes and the old 581-query Direct-HDR development reference are **different physical episodes**. For example, training-source episodes 000/004/005 use seeds 2026104130/4134/4135, while matching episode numbers in the old reference use seeds 2026103420/3424/3425. The old 581-query set therefore remains evaluation-only; it must not be used to build training windows.

The training cache contains the 100 Hz scene-linear HDR/depth/segmentation frames. The paired event archives for the old common reference are from other seeds and have a different NPZ key (`events_1000hz`) from the training loader's expected `events_2000hz`. Neither is a valid substitute for rendering the training episodes from their exact 0.5 ms state traces. The strict-vs-Direct runner will make those matched streams once the renderer can use an assigned GPU.

## Render readiness finding

At 2026-11-01 04:02 UTC, all six g1 GPUs had active processes owned by another user's jobs, each using about 2.45–2.61 GiB and reporting 64–93% utilization. The project reservation directory was empty. No renderer or model training was started.

A bounded software-renderer probe was attempted because g1 had ample host RAM and low CPU load. Despite `LIBGL_ALWAYS_SOFTWARE=1`, `GALLIUM_DRIVER=llvmpipe`, and `EGL_PLATFORM=surfaceless`, EGL reported `NVIDIA GeForce RTX 3090/PCIe/SSE2`. The probe exited immediately before any scene render; a fresh process check showed only the six pre-existing user-owned GPU jobs. `ldconfig` did not advertise `libOSMesa`, so no CPU-only graphics path is currently verified. Do not repeat this EGL probe while GPUs are occupied. The probe also emitted a warning that the shared Quadrants compilation cache lock was held; it was left untouched.

## Runner correction before first render

The strict/direct runner initially placed the frozen tag before the selected benchmark-support snapshot on `sys.path`. The frozen tag also contains an older `benchmark/robot_closed_loop`, so that order would have imported the wrong `robot_env` and failed its provenance guard. The runner now imports and pins the frozen `genesis_event_plugin` package first, then prioritizes the isolated benchmark-support snapshot. This is a benchmark-layer fix; no frozen-core file changed. The corrected runner has passed `py_compile` and `git diff --check`; it has not yet rendered.

## Next exact step

After a coordinator assigns a GPU UUID and time slot, rerun the live ownership check, set `EGL_DEVICE_ID` to that assigned device, and run the strict/direct source pair for episodes 004 and 005 from the same verified 0.5 ms state traces. Save both raw streams, query-time five-bin hashes, strict availability times, image-to-cache checks, and per-stage costs. Only after this first divergence check should the already fixed episode split and 40-epoch budget be used for the single justified training run.

The full remote audit values, source hashes, and renderer observations are in [`source_readiness.json`](source_readiness.json). No new checkpoint, generator ranking, or localization-recovery result is claimed here.
