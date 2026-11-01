# R1.5 localization recovery: source readiness

## What is ready

The nine saved robot state traces have been replayed at 0.5 ms without rendering. They match the prior physical traces and the saved 100 Hz HDR cache after launch. The training-source episodes and the old 581-query Direct-HDR development reference are **different physical episodes**. For example, training-source episodes 000/004/005 use seeds 2026104130/4134/4135, while matching episode numbers in the old reference use seeds 2026103420/3424/3425. The old 581-query set therefore remains evaluation-only; it must not be used to build training windows.

The training cache contains the 100 Hz scene-linear HDR/depth/segmentation frames. The paired event archives for the old common reference are from other seeds and have a different NPZ key (`events_1000hz`) from the training loader's expected `events_2000hz`. Neither is a valid substitute for rendering the training episodes from their exact 0.5 ms state traces. The strict-vs-Direct runner will make those matched streams once the renderer can use an assigned GPU.

## Render readiness finding

The latest read-only g1 snapshot was 2026-11-01 07:32:56 UTC. GPU 0/1 (RTX 3090) and GPU 2 (V100S) reported 0% utilization, 1–2 MiB use, and no compute process; GPU 3–5 still had active compute processes at 91% utilization (PIDs 2829582/2830439/2829588). The project `.gpu_reservations` directory does not exist. No renderer or model training was started: an idle snapshot is not a resource assignment, so the task still needs an explicit UUID and time slot from the coordinator.

A bounded software-renderer probe was attempted because g1 had ample host RAM and low CPU load. Despite `LIBGL_ALWAYS_SOFTWARE=1`, `GALLIUM_DRIVER=llvmpipe`, and `EGL_PLATFORM=surfaceless`, EGL reported `NVIDIA GeForce RTX 3090/PCIe/SSE2`. The probe exited immediately before any scene render; a fresh process check showed only the six pre-existing user-owned GPU jobs. `ldconfig` did not advertise `libOSMesa`, so no CPU-only graphics path is currently verified. Do not repeat this EGL probe while GPUs are occupied. The probe also emitted a warning that the shared Quadrants compilation cache lock was held; it was left untouched.

## Runner correction before first render

The strict/direct runner initially placed the frozen tag before the selected benchmark-support snapshot on `sys.path`. The frozen tag also contains an older `benchmark/robot_closed_loop`, so that order would have imported the wrong `robot_env` and failed its provenance guard. The runner now imports and pins the frozen `genesis_event_plugin` package first, then prioritizes the isolated benchmark-support snapshot. A subsequent import-only check exposed another deployment mismatch: that support snapshot omits `diagnostics/run_paired_generator_smoke.py`, which supplied only pose and segmentation-ID helpers. Those helpers are now local to the diagnostic runner. The corrected snapshot also includes the previously omitted `event_state_model.py` and its small Python dependencies. On g1, the `Eventbased_WAM` Python environment imported Genesis 1.2.2, all frozen plugin modules, all selected support modules, and the runner; provenance assertions passed. This import-only run did not initialize a Genesis scene, render, or train. Genesis emitted a warning that the installed PyTorch version is below its supported range; actual rendering remains unverified. No frozen-core file changed. Runner SHA256: `82d11c694f35ec7e42d31a56a535f514950c8b139163fb5ca8b1c8c4ab2af62e`.

## Next exact step

After a coordinator assigns a GPU UUID and time slot, rerun the live ownership check, set `EGL_DEVICE_ID` to that assigned device, and run the strict/direct source pair for episodes 004 and 005 from the same verified 0.5 ms state traces. Save both raw streams, query-time five-bin hashes, strict availability times, image-to-cache checks, and per-stage costs. Only after this first divergence check should the already fixed episode split and 40-epoch budget be used for the single justified training run.

The full remote audit values, source hashes, and renderer observations are in [`source_readiness.json`](source_readiness.json). No new checkpoint, generator ranking, or localization-recovery result is claimed here.
