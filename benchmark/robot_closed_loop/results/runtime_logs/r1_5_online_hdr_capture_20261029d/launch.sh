#!/usr/bin/env bash
set -euo pipefail

RUN_ID=r1_5_online_hdr_capture_20261029d
HOST=g1.pkuml.org
R0_ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a
E_ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r1_5-20261029-f6a2b1e
ROBOT_PYTHON=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
OUT="$R0_ROOT/benchmark/robot_closed_loop/results/$RUN_ID"
LOGDIR="$R0_ROOT/benchmark/robot_closed_loop/results/runtime_logs/$RUN_ID"
STAGING="$LOGDIR/staging"
GPU_UUID=GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23
LOCK_PARENT=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/.gpu_reservations
LOCK="$LOCK_PARENT/$GPU_UUID"
SESSION=r15onlinehdr20261029d
SOURCE_COMMIT=bac8463fea68dd0ae78b193190f2407e9f3ed85f
EXPECTED_RUN_SHA=0b5b9fe1d00c746895750b3acace7466d079e17e35279c8999fcb759d1f358eb
EXPECTED_EVENT_STATE_SHA=f87599c46755a06b3fe7afb09c45d1d2ecae9527937e1ef39ff463576871d322
EXPECTED_SCENE_SHA=aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a
EXPECTED_TRACE_SHA=310837ea07ab2f819f590ccce91b3952dd9e85ea5005537bc1f4d1a339bff28c
EXPECTED_EVENTS_SHA=b698817a8347ca48ad75015f0cb81c3bfc711958a60d3387d468c70456a7b2b6

test "$(hostname)" = g1
test "$(git -C "$R0_ROOT" rev-parse HEAD)" = 2100645b0473c294760789bc4475b644f261d9cf
test -z "$(git -C "$R0_ROOT" status --short --untracked-files=no)"
test "$(git -C "$E_ROOT" rev-parse HEAD)" = 855b4b0bc7de96170b4bfee8d7002b370782f346
test -z "$(git -C "$E_ROOT" status --short --untracked-files=no)"
test -x "$ROBOT_PYTHON"
test "$(sha256sum "$R0_ROOT/benchmark/robot_closed_loop/run.py" | awk '{print $1}')" = "$EXPECTED_RUN_SHA"
test "$(sha256sum "$R0_ROOT/benchmark/robot_closed_loop/event_state.py" | awk '{print $1}')" = "$EXPECTED_EVENT_STATE_SHA"
test "$(sha256sum "$STAGING/scene_pilot_r0_5_candidate.json" | awk '{print $1}')" = "$EXPECTED_SCENE_SHA"
ARCHIVE_ROOT="$R0_ROOT/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/episodes/episode_005"
test "$(sha256sum "$ARCHIVE_ROOT/state_control_trace.jsonl" | awk '{print $1}')" = "$EXPECTED_TRACE_SHA"
test "$(sha256sum "$ARCHIVE_ROOT/direct_hdr_events_1khz_2khz.npz" | awk '{print $1}')" = "$EXPECTED_EVENTS_SHA"
test ! -e "$OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "refusing to reuse tmux session $SESSION" >&2
  exit 2
fi

mkdir -p "$LOGDIR"
if [[ ! -d "$LOCK_PARENT" ]]; then mkdir -m 700 -p "$LOCK_PARENT"; fi
if ! mkdir -m 700 "$LOCK"; then
  echo "GPU UUID already reserved: $GPU_UUID" >&2
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

python3 - "$LOCK/allocation.json" <<PY
import datetime,json,sys
json.dump({'run_id':'$RUN_ID','agent':'robot R1.5 online HDR diagnosis','host':'$HOST','gpu_index':0,'gpu_uuid':'$GPU_UUID','status':'reserved_preflight','reserved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':'$SOURCE_COMMIT','original_runner_commit':'2100645b0473c294760789bc4475b644f261d9cf','renderer_checkout':'$E_ROOT','renderer_commit':'855b4b0bc7de96170b4bfee8d7002b370782f346','output':'$OUT'},open(sys.argv[1],'w'),indent=2)
PY
chmod 600 "$LOCK/allocation.json"

{
  hostname
  date -u +%Y-%m-%dT%H:%M:%SZ
  uptime
  free -h
  df -h /home/zhaoyuhan/Eventbased_WAM
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  nvidia-smi pmon -c 1 -i 0 -s um
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader -i 0 || true
  "$ROBOT_PYTHON" -V
  "$ROBOT_PYTHON" -c 'import genesis,torch,numpy; print("genesis_module",genesis.__file__); print("torch",torch.__version__,"cuda",torch.cuda.is_available()); print("numpy",numpy.__version__)'
} > "$LOGDIR/preflight.txt" 2>&1

PMON=$(nvidia-smi pmon -c 1 -i 0 -s um)
printf '%s\n' "$PMON" > "$LOGDIR/final_gpu_pmon.txt"
COMPUTE_APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader -i 0 2>&1 || true)
printf '%s\n' "$COMPUTE_APPS" > "$LOGDIR/final_gpu_compute_apps.txt"
if printf '%s\n' "$PMON" | awk '$1==0 && $2 ~ /^[0-9]+$/ {found=1} END {exit !found}' || [[ -n "$(printf '%s' "$COMPUTE_APPS" | tr -d '[:space:]')" ]]; then
  python3 -c 'import json,sys,datetime; p=sys.argv[1]; d=json.load(open(p)); d.update(status="aborted_gpu_busy",final_pmon=open(sys.argv[2]).read(),final_compute_apps=open(sys.argv[3]).read(),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); json.dump(d,open(p,"w"),indent=2)' "$LOCK/allocation.json" "$LOGDIR/final_gpu_pmon.txt" "$LOGDIR/final_gpu_compute_apps.txt"
  cp "$LOCK/allocation.json" "$LOGDIR/resource_allocation_aborted.json"
  rm -f "$LOCK/allocation.json"
  rmdir "$LOCK"
  KEEP_LOCK=1
  exit 73
fi

mkdir -p "$OUT/scripts" "$OUT/online_hdr_capture/episode_000"
cp "$STAGING/capture_r1_5_online_hdr_window.py" "$OUT/scripts/"
cp "$STAGING/experiment_config.json" "$OUT/experiment_config.json"
cp "$STAGING/scene_pilot_r0_5_candidate.json" "$OUT/scene_pilot_r0_5_candidate.json"
cat > "$LOGDIR/worker.sh" <<'WORKER'
#!/usr/bin/env bash
set -u
RUN_ID=r1_5_online_hdr_capture_20261029d
R0_ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a
ROBOT_PYTHON=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
OUT="$R0_ROOT/benchmark/robot_closed_loop/results/$RUN_ID"
LOGDIR="$R0_ROOT/benchmark/robot_closed_loop/results/runtime_logs/$RUN_ID"
LOCK=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/.gpu_reservations/GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23
cd "$R0_ROOT" || exit 91
printf 'started_utc=%s\n' "$(date -u +%FT%TZ)" > "$LOGDIR/worker_started_at.txt"
printf '%s\n' "$$" > "$LOGDIR/worker_pid.txt"
set +e
env CUDA_VISIBLE_DEVICES= EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$R0_ROOT" \
  R15_ONLINE_HDR_CAPTURE_WINDOW_S=0.5505,0.5905 \
  R15_ONLINE_HDR_CAPTURE_PATH="$OUT/online_hdr_capture/episode_000/online_hdr_frames_2khz.npz" \
  "$ROBOT_PYTHON" -u "$OUT/scripts/capture_r1_5_online_hdr_window.py" \
  --mode reference --config "$OUT/scene_pilot_r0_5_candidate.json" \
  --speeds 0.4 --reference-episodes 1 --reference-y-values 0.055 \
  --event-controller-hz 2000 --reference-controller-source gt_current \
  --visibility-hz 50 --seed 2026103425 --out "$OUT/source_replay" \
  2>&1 | tee "$LOGDIR/run.log"
RC=${PIPESTATUS[0]}
if [[ "$RC" == 0 ]]; then
  TRACE="$OUT/source_replay/episodes/episode_000/state_control_trace.jsonl"
  EVENTS="$OUT/source_replay/episodes/episode_000/direct_hdr_events_1khz_2khz.npz"
  if [[ "$(sha256sum "$TRACE" | awk '{print $1}')" != 310837ea07ab2f819f590ccce91b3952dd9e85ea5005537bc1f4d1a339bff28c || "$(sha256sum "$EVENTS" | awk '{print $1}')" != b698817a8347ca48ad75015f0cb81c3bfc711958a60d3387d468c70456a7b2b6 ]]; then
    echo "exact source reproduction hash mismatch" | tee -a "$LOGDIR/run.log"
    RC=92
  fi
fi
printf '%s\n' "$RC" > "$LOGDIR/exit_code.txt"
printf 'finished_utc=%s\nexit_code=%s\n' "$(date -u +%FT%TZ)" "$RC" > "$LOGDIR/worker_finished_at.txt"
(cd "$OUT" && find . -type f ! -path './SHA256SUMS' -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
python3 - "$LOCK/allocation.json" "$RUN_ID" "$RC" <<'PY'
import datetime,json,sys
p,run_id,rc=sys.argv[1],sys.argv[2],int(sys.argv[3])
d=json.load(open(p)); assert d.get('run_id')==run_id
d.update(status=('completed' if rc==0 else 'failed'),exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
json.dump(d,open(p,'w'),indent=2)
PY
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
for _ in $(seq 1 20); do [[ -s "$LOGDIR/worker_pid.txt" ]] && break; sleep 0.25; done
test -s "$LOGDIR/worker_pid.txt"
ps -o user,pid,ppid,etime,cmd -p "$WORKER_PID" > "$LOGDIR/detach_verification.txt"
tmux has-session -t "$SESSION"
python3 - "$LOCK/allocation.json" "$RUN_ID" "$WORKER_PID" "$SESSION" <<'PY'
import datetime,json,sys
p,run_id,pid,session=sys.argv[1:]
d=json.load(open(p)); assert d.get('run_id')==run_id
d.update(status='running',worker_pid=int(pid),tmux_session=session,launched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
json.dump(d,open(p,'w'),indent=2)
PY
printf 'session=%s worker_pid=%s output=%s\n' "$SESSION" "$WORKER_PID" "$OUT"
KEEP_LOCK=1
