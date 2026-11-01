#!/usr/bin/env bash
set -Eeuo pipefail
RUN_ROOT=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/r1_5_localization_recovery_20261101a/strict_direct_pair_r1_5_localization_recovery_20261101b_gpu0_20261101T073805Z
SRC=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/robot-r1-localization-20261101a-b7ba943/support
CORE=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/robot-r1-localization-20261101a-914ffff/frozen_core
TRACE=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/r1_5_localization_recovery_20261101a/state_trace_full9_20261101a
CACHE=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-ba137bd/benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018a/trajectory_cache
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
GPU_UUID=GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export EGL_DEVICE_ID=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export ROBOT_EXECUTION_CODE_COMMIT=b7ba943ef3d72c75315c1065141044bf5df47c72
cd /home/zhaoyuhan/Eventbased_WAM
exec > "$RUN_ROOT/launcher.log" 2>&1
printf 'start_utc=%s\nlauncher_pid=%s\nhost=%s\ngpu_uuid=%s\ngpu_index=0\nrun_id=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$(hostname)" "$GPU_UUID" 'r1_5_localization_recovery_20261101b_gpu0_20261101T073805Z'
current=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits | awk -F, -v u="$GPU_UUID" 'BEGIN{n=0} {gsub(/^[ \t]+|[ \t]+$/, "", $1); if($1==u){print; n++}} END{if(n) exit 1}') || { echo 'assigned GPU acquired another compute process; refusing'; exit 90; }
for EP in 4 5; do
  echo "episode_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) episode=$EP"
  "$PY" -u "$SRC/benchmark/robot_closed_loop/diagnostics/render_r1_5_strict_direct_episode.py" \
    --frozen-core-root "$CORE" --source-root "$SRC" \
    --support-source-commit b7ba943ef3d72c75315c1065141044bf5df47c72 \
    --trace-root "$TRACE" --legacy-cache-root "$CACHE" \
    --scene-config "$SRC/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json" \
    --control-config "$SRC/benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json" \
    --experiment-config "$SRC/benchmark/robot_closed_loop/configs/r1_5_localization_recovery_20261101a.json" \
    --episode-id "$EP" --output "$RUN_ROOT" --genesis-backend gpu \
    > "$RUN_ROOT/episode_${EP}.log" 2>&1 &
  pid=$!
  echo "$pid" > "$RUN_ROOT/episode_${EP}.pid"
  echo "episode_pid=$pid"
  if wait "$pid"; then echo "episode_exit=0 episode=$EP"; else rc=$?; echo "episode_exit=$rc episode=$EP"; exit "$rc"; fi
  echo "episode_end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) episode=$EP"
done
echo "pair_status=complete end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
