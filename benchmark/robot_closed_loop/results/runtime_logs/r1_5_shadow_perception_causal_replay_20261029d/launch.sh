#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r1_5-20261029-f6a2b1e
LOGDIR="$ROOT/benchmark/robot_closed_loop/results/runtime_logs/r1_5_shadow_perception_causal_replay_20261029d"
OUT="$ROOT/benchmark/robot_closed_loop/results/r1_5_shadow_perception_causal_replay_20261029d"
LOCK=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/.gpu_reservations/GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23
SESSION=r15shadowd20261029
EXPECTED_SCRIPT=b4f366ab798abc12926e863bedbe666f5c0de401ce4bff6adcf720e319ca58a1
EXPECTED_CONFIG=ca83a66dec75a63b8cbc11c1bf02fee29b367bc3825a5872b98c7f9de26d923d
cd "$ROOT"
test ! -e "$OUT"
test "$(git rev-parse HEAD)" = 855b4b0bc7de96170b4bfee8d7002b370782f346
test "$(sha256sum benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_shadow_perception_causal_replay.py | awk '{print $1}')" = "$EXPECTED_SCRIPT"
test "$(sha256sum benchmark/robot_closed_loop/configs/r1_5_shadow_perception_causal_replay_20261029d.json | awk '{print $1}')" = "$EXPECTED_CONFIG"
grep -q 'r1_5_shadow_perception_causal_replay_20261029d' "$LOCK/allocation.json"
ROBOT_PYTHON=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
TRACE=benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059/episode_005/trace_window.jsonl
"$ROBOT_PYTHON" -c 'import json,sys; t=[json.loads(x)["time_s"] for x in open(sys.argv[1])]; q=[x for x in t if .5855-1e-9<=x<=.5905+1e-9]; assert len(q)==7,(q,len(q)); assert any(abs(x-.5605)<1e-9 for x in t); print("exact_query_rows",q,"first_causal_row",.5605)' "$TRACE"
printf '%s\n' '--- final GPU0 gate ---'
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader -i 0
PMON=$(nvidia-smi pmon -c 1 -i 0 -s um)
printf '%s\n' "$PMON"
if printf '%s\n' "$PMON" | awk '$1==0 && $2!="-" && $2!="pid" {found=1} END {exit !found}'; then
  echo 'GPU0 has a compute PID; aborting launch.' >&2
  exit 73
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo 'Run tmux session already exists; refusing to overwrite.' >&2
  exit 74
fi
mkdir -p "$LOGDIR"
tmux new-session -d -s "$SESSION" "bash '$LOGDIR/worker.sh'"
WORKER_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
printf '%s\n' "$WORKER_PID" > "$LOGDIR/tmux_pane_pid.txt"
python3 - "$LOCK/allocation.json" "$WORKER_PID" <<'PY'
import json,sys,datetime
p=sys.argv[1]
d=json.load(open(p)); d.update(status='running', worker_pid=int(sys.argv[2]), tmux_session='r15shadowd20261029', launched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
json.dump(d,open(p,'w'),indent=2)
PY
sleep 2
ps -o user,pid,ppid,etime,cmd -p "$WORKER_PID"
tmux has-session -t "$SESSION"
printf 'session=%s worker_pid=%s output=%s\n' "$SESSION" "$WORKER_PID" "$OUT"
