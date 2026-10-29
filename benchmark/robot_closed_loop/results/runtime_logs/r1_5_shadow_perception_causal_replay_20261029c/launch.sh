#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r1_5-20261029-f6a2b1e
RUN_ID=r1_5_shadow_perception_causal_replay_20261029c
OUT="$ROOT/benchmark/robot_closed_loop/results/$RUN_ID"
LOGDIR="$ROOT/benchmark/robot_closed_loop/results/runtime_logs/$RUN_ID"
LOCK=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/.gpu_reservations/GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23
ALLOC=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/$RUN_ID/resource_allocation.json
SESSION=r15shadcause20261029c
SCRIPT="$ROOT/benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_shadow_perception_causal_replay.py"
SCENE="$ROOT/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"
EXPERIMENT="$ROOT/benchmark/robot_closed_loop/configs/r1_5_shadow_perception_causal_replay_20261029c.json"
CACHE="$ROOT/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059"
CHECKPOINT="$ROOT/benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018b/paired_r1/ours_fit/spatial_event_state.pt"
LOGDIR_WORKER="$LOGDIR/worker.sh"

if [ -e "$OUT" ]; then echo "refusing existing output: $OUT" >&2; exit 40; fi
if [ ! -f "$LOCK/reservation.json" ] || ! grep -Fq "$RUN_ID" "$LOCK/reservation.json"; then
  echo "GPU lease missing or owned by another run" >&2; exit 41
fi
GPU_LINE=$(nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader -i 0)
GPU_UTIL=$(printf '%s' "$GPU_LINE" | cut -d, -f2 | tr -d ' %')
GPU_MEM=$(printf '%s' "$GPU_LINE" | cut -d, -f3 | tr -dc '0-9')
if [ "$(printf '%s' "$GPU_LINE" | cut -d, -f1 | tr -d ' ')" != GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23 ] || [ "$GPU_UTIL" != 0 ] || [ "${GPU_MEM:-999999}" -gt 128 ]; then
  echo "selected GPU no longer idle: $GPU_LINE" >&2; exit 42
fi
GPU_APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader)
if printf '%s\n' "$GPU_APPS" | grep -Fq GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23; then
  echo "selected GPU has a compute process: $GPU_APPS" >&2; exit 43
fi
for required in "$SCRIPT" "$SCENE" "$EXPERIMENT" "$CACHE/episode_004/trace_window.jsonl" "$CACHE/episode_005/trace_window.jsonl" "$CHECKPOINT"; do
  if [ ! -f "$required" ]; then echo "missing required input: $required" >&2; exit 44; fi
done
command -v tmux >/dev/null
printf '%s\n' "$(date -Is)" > "$LOGDIR/launch_started_at.txt"
printf 'GPU_PRELAUNCH=%s\nGPU_APPS=%s\n' "$GPU_LINE" "$GPU_APPS"
tmux new-session -d -s "$SESSION" "bash '$LOGDIR_WORKER'"
sleep 1
tmux has-session -t "$SESSION"
PANE_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
printf '%s\n' "$PANE_PID" > "$LOGDIR/tmux_pane_pid.txt"
python3 - "$ALLOC" "$SESSION" "$PANE_PID" "$GPU_LINE" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
data.update({"status": "running", "tmux_session": sys.argv[2], "tmux_pane_pid": int(sys.argv[3]),
             "gpu_prelaunch_snapshot": sys.argv[4], "launcher_started_at_epoch": time.time()})
p.write_text(json.dumps(data, indent=2) + "\n")
PY
sleep 1
if ! tmux has-session -t "$SESSION"; then echo "tmux exited immediately" >&2; exit 45; fi
printf 'TMUX_SESSION=%s\nPANE_PID=%s\nOUT=%s\n' "$SESSION" "$PANE_PID" "$OUT"
