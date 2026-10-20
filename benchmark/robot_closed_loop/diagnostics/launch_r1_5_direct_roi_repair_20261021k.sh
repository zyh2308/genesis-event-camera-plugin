#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1_5_direct_roi_20261021k
PY=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
CACHE=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-diag-20261019/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059
CHECKPOINT_ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1_5_multiref_20261021i
RUN_ID=r1_5_direct_roi_repair_20261021k
OUTPUT="$ROOT/benchmark/robot_closed_loop/results/$RUN_ID"
RUNTIME="$ROOT/benchmark/robot_closed_loop/results/runtime_logs/$RUN_ID"
SESSION=$RUN_ID
SCRIPT="$ROOT/benchmark/robot_closed_loop/diagnostics/launch_r1_5_direct_roi_repair_20261021k.sh"

if [[ "${1:-}" == "--worker" ]]; then
  env PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TI_NUM_THREADS=4 \
    "$PY" "$ROOT/benchmark/robot_closed_loop/diagnostics/run_r1_5_multiref_surface_cached_replay.py" \
      --cache-root "$CACHE" \
      --scene-config "$ROOT/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json" \
      --control-config "$ROOT/benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json" \
      --candidate-config "$ROOT/benchmark/robot_closed_loop/configs/r1_5_multiref_surface_candidate_20261021k.json" \
      --roi-repair-config "$ROOT/benchmark/robot_closed_loop/configs/r1_5_direct_roi_repair_20261021k.json" \
      --ours-checkpoint "$CHECKPOINT_ROOT/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/ours_fit/spatial_event_state.pt" \
      --v2e-checkpoint "$CHECKPOINT_ROOT/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/v2e_fit/spatial_event_state.pt" \
      --source-interval 0.005 \
      --max-source-views 9 \
      --moving-source-views 5 \
      --output "$OUTPUT" \
      >"$RUNTIME/run.log" 2>&1 &
  child=$!
  printf '%s\n' "$child" >"$RUNTIME/run.pid"
  set +e
  wait "$child"
  status=$?
  set -e
  printf '%s\n' "$status" >"$RUNTIME/run.exit"
  exit "$status"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 2
fi
for path in "$OUTPUT" "$RUNTIME"; do
  if [[ -e "$path" ]]; then
    echo "run path already exists; refusing to overwrite: $path" >&2
    exit 2
  fi
done
for episode in 004 005; do
  for input in trace_window.jsonl same_state_source_cache.h5 rendered_frame_evidence.h5; do
    [[ -f "$CACHE/episode_$episode/$input" ]] || { echo "missing $CACHE/episode_$episode/$input" >&2; exit 2; }
  done
done
[[ -f "$CHECKPOINT_ROOT/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/ours_fit/spatial_event_state.pt" ]]
[[ -f "$CHECKPOINT_ROOT/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/v2e_fit/spatial_event_state.pt" ]]
mkdir -p "$RUNTIME"
tmux new-session -d -s "$SESSION" "bash '$SCRIPT' --worker"
sleep 2
tmux has-session -t "$SESSION"
echo "detached=1 session=$SESSION pid=$(cat "$RUNTIME/run.pid")"
ps -p "$(cat "$RUNTIME/run.pid")" -o pid=,ppid=,stat=,etime=,args=
