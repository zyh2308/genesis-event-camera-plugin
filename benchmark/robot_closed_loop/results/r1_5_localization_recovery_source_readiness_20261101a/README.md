# R1.5 localization recovery: source readiness

## What is ready

The nine saved robot state traces have been replayed at 0.5 ms without rendering. They match the prior physical traces and the saved 100 Hz HDR cache after launch. The training-source episodes and the old 581-query Direct-HDR development reference are **different physical episodes**. For example, training-source episodes 000/004/005 use seeds 2026104130/4134/4135, while matching episode numbers in the old reference use seeds 2026103420/3424/3425. The old 581-query set therefore remains evaluation-only; it must not be used to build training windows.

The training cache contains the 100 Hz scene-linear HDR/depth/segmentation frames. The paired event archives for the old common reference are from other seeds and have a different NPZ key (`events_1000hz`) from the training loader's expected `events_2000hz`. Neither is a valid substitute for rendering the training episodes from their exact 0.5 ms state traces. The strict-vs-Direct runner will make those matched streams once the renderer can use an assigned GPU.

## Render readiness finding

The user assigned g1 GPU 0 (RTX 3090) for a 30-minute episode 004/005 diagnostic slot. Immediately before launch at 07:38:05 UTC, UUID `GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23` reported 0% use and 2 MiB, with no compute process. A shared reservation record was created at `/home/zhaoyuhan/Eventbased_WAM/.gpu_reservations/r1_5_localization_recovery_20261101b_gpu0_20261101T073805Z.json`. GPU 3–5 still had active processes (PIDs 2829582/2830439/2829588). The first invocation exited at argument parsing; follow-up at 07:39:41 UTC confirmed no renderer process on GPU 0. The corrected runner must be redeployed within the reserved slot before rendering begins.

A bounded software-renderer probe was attempted because g1 had ample host RAM and low CPU load. Despite `LIBGL_ALWAYS_SOFTWARE=1`, `GALLIUM_DRIVER=llvmpipe`, and `EGL_PLATFORM=surfaceless`, EGL reported `NVIDIA GeForce RTX 3090/PCIe/SSE2`. The probe exited immediately before any scene render; a fresh process check showed only the six pre-existing user-owned GPU jobs. `ldconfig` did not advertise `libOSMesa`, so no CPU-only graphics path is currently verified. Do not repeat this EGL probe while GPUs are occupied. The probe also emitted a warning that the shared Quadrants compilation cache lock was held; it was left untouched.

## Runner correction before first render

The strict/direct runner initially placed the frozen tag before the selected benchmark-support snapshot on `sys.path`. The frozen tag also contains an older `benchmark/robot_closed_loop`, so that order would have imported the wrong `robot_env` and failed its provenance guard. The runner now imports and pins the frozen `genesis_event_plugin` package first, then prioritizes the isolated benchmark-support snapshot. A subsequent import-only check exposed another deployment mismatch: that support snapshot omits `diagnostics/run_paired_generator_smoke.py`, which supplied only pose and segmentation-ID helpers. Those helpers are now local to the diagnostic runner. The corrected snapshot also includes the previously omitted `event_state_model.py` and its small Python dependencies. On g1, the `Eventbased_WAM` Python environment imported Genesis 1.2.2, all frozen plugin modules, all selected support modules, and the runner; provenance assertions passed. This import-only run did not initialize a Genesis scene, render, or train. Genesis emitted a warning that the installed PyTorch version is below its supported range; actual rendering remains unverified. The first GPU-slot invocation exposed an argument parser bug: the bootstrap parser consumed `--frozen-core-root` before a second parser required it again, so episode 004 exited before scene initialization or rendering. The second parser now defaults to the already resolved core path. Local syntax and parser smoke checks pass; the new version is not yet redeployed. Latest runner SHA256: `aeeb6ed8e0f4f16e22484377712072c96511e3718cf258225bd1f006f01b88ee`.

## Next exact step

After a coordinator assigns a GPU UUID and time slot, rerun the live ownership check. Run this pair from a uniquely named tmux session, replacing the GPU placeholders and output run ID; the session should be stopped after episodes 004/005 for an initial evidence review:

```bash
export CUDA_VISIBLE_DEVICES=GPU-<assigned-uuid>
export EGL_DEVICE_ID=<assigned-host-index>
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
SRC=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/robot-r1-localization-20261101a-b7ba943/support
CORE=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/robot-r1-localization-20261101a-914ffff/frozen_core
TRACE=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/r1_5_localization_recovery_20261101a/state_trace_full9_20261101a
CACHE=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-ba137bd/benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018a/trajectory_cache
OUT=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/r1_5_localization_recovery_20261101a/strict_direct_pair_<unique-run-id>
mkdir -p "$OUT"
for EP in 4 5; do
  "$PY" -u "$SRC/benchmark/robot_closed_loop/diagnostics/render_r1_5_strict_direct_episode.py" \
    --frozen-core-root "$CORE" --source-root "$SRC" \
    --support-source-commit b7ba943ef3d72c75315c1065141044bf5df47c72 \
    --trace-root "$TRACE" --legacy-cache-root "$CACHE" \
    --scene-config "$SRC/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json" \
    --control-config "$SRC/benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json" \
    --experiment-config "$SRC/benchmark/robot_closed_loop/configs/r1_5_localization_recovery_20261101a.json" \
    --episode-id "$EP" --output "$OUT" --genesis-backend gpu \
    > "$OUT/episode_${EP}.log" 2>&1
done
```

The runner saves both raw streams, query-time five-bin hashes, strict availability times, image-to-cache checks, and per-stage costs. Only after this first divergence check should the already fixed episode split and 40-epoch budget be used for the single justified training run.

The full remote audit values, source hashes, and renderer observations are in [`source_readiness.json`](source_readiness.json). No new checkpoint, generator ranking, or localization-recovery result is claimed here.
