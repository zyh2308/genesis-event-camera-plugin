#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1_5_scissor_roi_20261021n
PY=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
CACHE=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-diag-20261019/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059
PRIOR=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1_5_direct_roi_20261021k/benchmark/robot_closed_loop/results/r1_5_direct_roi_repair_20261021k
RUN_ID=r1_5_scissor_roi_probe_20261021n
OUTPUT="$ROOT/benchmark/robot_closed_loop/results/$RUN_ID"
RUNTIME="$ROOT/benchmark/robot_closed_loop/results/runtime_logs/$RUN_ID"
SESSION=$RUN_ID
SCRIPT="$ROOT/benchmark/robot_closed_loop/diagnostics/launch_r1_5_scissor_roi_probe_20261021n.sh"

if [[ "${1:-}" == "--worker" ]]; then
  env PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TI_NUM_THREADS=1 \
    nice -n 10 "$PY" "$ROOT/benchmark/robot_closed_loop/diagnostics/run_r1_5_scissor_roi_probe.py" \
      --cache-root "$CACHE" \
      --scene-config "$ROOT/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json" \
      --control-config "$ROOT/benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json" \
      --probe-config "$ROOT/benchmark/robot_closed_loop/configs/r1_5_scissor_roi_probe_20261021n.json" \
      --prior-run "$PRIOR" \
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
[[ -d "$PRIOR" && -f "$PRIOR/summary.json" ]]
for episode in 004 005; do
  [[ -f "$CACHE/episode_$episode/trace_window.jsonl" ]]
  [[ -f "$PRIOR/episode_$episode/candidate_render_request_masks.npz" ]]
  [[ -f "$PRIOR/episode_$episode/candidate_roi_replay_arrays.npz" ]]
done
mkdir -p "$RUNTIME"
tmux new-session -d -s "$SESSION" "bash '$SCRIPT' --worker"
sleep 2
tmux has-session -t "$SESSION"
echo "detached=1 session=$SESSION pid=$(cat "$RUNTIME/run.pid")"
ps -p "$(cat "$RUNTIME/run.pid")" -o pid=,ppid=,stat=,etime=,args=
