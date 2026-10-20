#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1_5_surface_20261021a
PY=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
CACHE=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-diag-20261019/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059
OUTPUT="$ROOT/benchmark/robot_closed_loop/results/r1_5_surface_raster_cached_replay_20261021b"
RUNTIME="$ROOT/benchmark/robot_closed_loop/results/runtime_logs"
SESSION=r1_5_surface_raster_cached_replay_20261021b
LOG="$RUNTIME/$SESSION.log"
PID_FILE="$RUNTIME/$SESSION.pid"
EXIT_FILE="$RUNTIME/$SESSION.exit"
SCRIPT="$ROOT/benchmark/robot_closed_loop/diagnostics/launch_r1_5_surface_raster_cached_replay_20261021b.sh"

if [[ "${1:-}" == "--worker" ]]; then
  env PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TI_NUM_THREADS=4 \
    "$PY" "$ROOT/benchmark/robot_closed_loop/diagnostics/run_r1_5_surface_raster_cached_replay.py" \
      --cache-root "$CACHE" \
      --scene-config "$ROOT/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json" \
      --control-config "$ROOT/benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json" \
      --ours-checkpoint "$ROOT/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/ours_fit/spatial_event_state.pt" \
      --v2e-checkpoint "$ROOT/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/v2e_fit/spatial_event_state.pt" \
      --output "$OUTPUT" >"$LOG" 2>&1 &
  child=$!
  printf '%s\n' "$child" >"$PID_FILE"
  set +e
  wait "$child"
  status=$?
  set -e
  printf '%s\n' "$status" >"$EXIT_FILE"
  exit "$status"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 2
fi
if [[ -e "$OUTPUT" || -e "$PID_FILE" || -e "$EXIT_FILE" || -e "$LOG" ]]; then
  echo "run output already exists; refusing to overwrite: $OUTPUT" >&2
  exit 2
fi
for episode in 004 005; do
  for input in trace_window.jsonl same_state_source_cache.h5 rendered_frame_evidence.h5; do
    [[ -f "$CACHE/episode_$episode/$input" ]] || { echo "missing input $CACHE/episode_$episode/$input" >&2; exit 2; }
  done
done
[[ -f "$ROOT/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/ours_fit/spatial_event_state.pt" ]]
[[ -f "$ROOT/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/v2e_fit/spatial_event_state.pt" ]]
mkdir -p "$RUNTIME"
tmux new-session -d -s "$SESSION" "$SCRIPT --worker"
sleep 2
tmux has-session -t "$SESSION"
echo "detached=1 session=$SESSION pid=$(cat "$PID_FILE")"
ps -p "$(cat "$PID_FILE")" -o pid=,ppid=,stat=,etime=,args=
