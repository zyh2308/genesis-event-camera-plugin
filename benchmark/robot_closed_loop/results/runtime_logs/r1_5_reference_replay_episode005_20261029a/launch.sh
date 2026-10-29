#!/usr/bin/env bash
set -euo pipefail

RUN_ID=r1_5_reference_replay_episode005_20261029a
HOST=g1.pkuml.org
OLD_ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a
ROBOT_PYTHON=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
RUN_PY=benchmark/robot_closed_loop/run.py
SCENE_CONFIG=benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json
OUT="$OLD_ROOT/benchmark/robot_closed_loop/results/$RUN_ID"
LOGDIR="$OLD_ROOT/benchmark/robot_closed_loop/results/runtime_logs/$RUN_ID"
GPU_UUID=GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23
LOCK_PARENT=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/.gpu_reservations
LOCK="$LOCK_PARENT/$GPU_UUID"
SESSION=r15ref00520261029a
EXPECTED_COMMIT=2100645b0473c294760789bc4475b644f261d9cf
EXPECTED_RUN_SHA=0b5b9fe1d00c746895750b3acace7466d079e17e35279c8999fcb759d1f358eb
EXPECTED_EVENT_STATE_SHA=f87599c46755a06b3fe7afb09c45d1d2ecae9527937e1ef39ff463576871d322
EXPECTED_SCENE_SHA=dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd

cd "$OLD_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --short --untracked-files=no)"
test -x "$ROBOT_PYTHON"
test "$(sha256sum "$RUN_PY" | awk '{print $1}')" = "$EXPECTED_RUN_SHA"
test "$(sha256sum benchmark/robot_closed_loop/event_state.py | awk '{print $1}')" = "$EXPECTED_EVENT_STATE_SHA"
test "$(sha256sum "$SCENE_CONFIG" | awk '{print $1}')" = "$EXPECTED_SCENE_SHA"
test -f benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/episodes/episode_005/direct_hdr_events_1khz_2khz.npz
test "$(sha256sum benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/episodes/episode_005/direct_hdr_events_1khz_2khz.npz | awk '{print $1}')" = b698817a8347ca48ad75015f0cb81c3bfc711958a60d3387d468c70456a7b2b6
test "$(sha256sum benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/episodes/episode_005/state_control_trace.jsonl | awk '{print $1}')" = 310837ea07ab2f819f590ccce91b3952dd9e85ea5005537bc1f4d1a339bff28c
test ! -e "$OUT"
test ! -e "$LOGDIR"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "refusing to reuse tmux session $SESSION" >&2
  exit 2
fi

mkdir -p "$LOGDIR"
if [[ ! -d "$LOCK_PARENT" ]]; then
  mkdir -m 700 -p "$LOCK_PARENT"
fi
if ! mkdir -m 700 "$LOCK"; then
  echo "GPU UUID already has a reservation: $GPU_UUID" >&2
  exit 73
fi
KEEP_LOCK=0
cleanup_unlaunched_reservation() {
  if [[ "$KEEP_LOCK" == 0 && -f "$LOCK/allocation.json" ]]; then
    cp "$LOCK/allocation.json" "$LOGDIR/resource_allocation_aborted.json"
    rm -f "$LOCK/allocation.json"
    rmdir "$LOCK" 2>/dev/null || true
  fi
}
trap cleanup_unlaunched_reservation EXIT

cat > "$LOCK/allocation.json" <<EOF
{
  "run_id": "$RUN_ID",
  "agent": "robot R1.5 diagnosis",
  "host": "$HOST",
  "gpu_index": 0,
  "gpu_uuid": "$GPU_UUID",
  "status": "reserved_before_final_gate",
  "reserved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "worktree": "$OLD_ROOT",
  "output": "$OUT",
  "source_commit": "$EXPECTED_COMMIT",
  "scene_config_sha256": "$EXPECTED_SCENE_SHA"
}
EOF
chmod 600 "$LOCK/allocation.json"

{
  hostname
  date -u +%Y-%m-%dT%H:%M:%SZ
  uptime
  free -h
  df -h /home/zhaoyuhan/Eventbased_WAM
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  nvidia-smi pmon -c 1 -i 0 -s um
  printf '\n--- live renderer overlay hashes ---\n'
  PKG=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/lib/python3.11/site-packages/genesis
  sha256sum "$PKG/vis/camera.py" "$PKG/vis/rasterizer.py" "$PKG/ext/pyrender/constants.py" "$PKG/ext/pyrender/renderer.py" "$PKG/ext/pyrender/offscreen.py"
} > "$LOGDIR/preflight.txt" 2>&1

PMON=$(nvidia-smi pmon -c 1 -i 0 -s um)
if printf '%s\n' "$PMON" | awk '$1==0 && $2 ~ /^[0-9]+$/ {found=1} END {exit !found}'; then
  python3 -c 'import json,sys,datetime; p=sys.argv[1]; d=json.load(open(p)); d.update(status="aborted_gpu_busy",finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); json.dump(d,open(p,"w"),indent=2)' "$LOCK/allocation.json"
  cat "$LOCK/allocation.json" > "$LOGDIR/resource_allocation_aborted.json"
  exit 73
fi

cat > "$LOGDIR/worker.sh" <<'WORKER'
#!/usr/bin/env bash
set -u
RUN_ID=r1_5_reference_replay_episode005_20261029a
OLD_ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a
ROBOT_PYTHON=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
OUT="$OLD_ROOT/benchmark/robot_closed_loop/results/$RUN_ID"
LOGDIR="$OLD_ROOT/benchmark/robot_closed_loop/results/runtime_logs/$RUN_ID"
LOCK=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/.gpu_reservations/GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23
cd "$OLD_ROOT"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOGDIR/worker_started_at.txt"
printf '%s\n' "$$" > "$LOGDIR/worker_pid.txt"
env CUDA_VISIBLE_DEVICES=0 EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  "$ROBOT_PYTHON" benchmark/robot_closed_loop/run.py \
  --mode reference \
  --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --speeds 0.4 --reference-episodes 1 --reference-y-values 0.055 \
  --event-controller-hz 2000 --reference-controller-source gt_current \
  --visibility-hz 50 --seed 2026103425 --out "$OUT" \
  > "$LOGDIR/run.log" 2>&1 &
PYTHON_PID=$!
printf '%s\n' "$PYTHON_PID" > "$LOGDIR/python_pid.txt"
python3 -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); assert d.get("run_id")==sys.argv[2]; d.update(status="running",worker_pid=int(sys.argv[3]),python_pid=int(sys.argv[4]),tmux_session=sys.argv[5],launched_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()); json.dump(d,open(p,"w"),indent=2)' "$LOCK/allocation.json" "$RUN_ID" "$$" "$PYTHON_PID" r15ref00520261029a
wait "$PYTHON_PID"
RC=$?
printf '%s\n' "$RC" > "$LOGDIR/exit_code.txt"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOGDIR/worker_finished_at.txt"
python3 -c 'import json,sys,datetime; p=sys.argv[1]; d=json.load(open(p)); assert d.get("run_id")==sys.argv[2]; d.update(status=("completed" if int(sys.argv[3])==0 else "failed"),exit_code=int(sys.argv[3]),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); json.dump(d,open(p,"w"),indent=2)' "$LOCK/allocation.json" "$RUN_ID" "$RC"
cp "$LOCK/allocation.json" "$LOGDIR/resource_allocation.json"
if python3 -c 'import json,sys; assert json.load(open(sys.argv[1])).get("run_id")==sys.argv[2]' "$LOCK/allocation.json" "$RUN_ID"; then
  rm -f "$LOCK/allocation.json"
  rmdir "$LOCK"
fi
exit "$RC"
WORKER
chmod 700 "$LOGDIR/worker.sh"

tmux new-session -d -s "$SESSION" "bash '$LOGDIR/worker.sh'"
WORKER_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
printf '%s\n' "$WORKER_PID" > "$LOGDIR/tmux_pane_pid.txt"
for _ in $(seq 1 20); do
  [[ -s "$LOGDIR/python_pid.txt" ]] && break
  sleep 0.25
done
test -s "$LOGDIR/python_pid.txt"
PYTHON_PID=$(cat "$LOGDIR/python_pid.txt")
ps -o user,pid,ppid,etime,cmd -p "$WORKER_PID,$PYTHON_PID" > "$LOGDIR/detach_verification.txt"
tmux has-session -t "$SESSION"
printf '%s\n' "session=$SESSION worker_pid=$WORKER_PID python_pid=$PYTHON_PID output=$OUT"
KEEP_LOCK=1
