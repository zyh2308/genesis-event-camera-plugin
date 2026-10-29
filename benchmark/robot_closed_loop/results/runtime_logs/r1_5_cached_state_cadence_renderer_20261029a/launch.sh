#!/usr/bin/env bash
set -euo pipefail

RUN_ID=r1_5_cached_state_cadence_renderer_20261029a
HOST=g1.pkuml.org
E_ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r1_5-20261029-f6a2b1e
R05_ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a
ROBOT_PYTHON=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
OUT="$R05_ROOT/benchmark/robot_closed_loop/results/$RUN_ID"
LOGDIR="$R05_ROOT/benchmark/robot_closed_loop/results/runtime_logs/$RUN_ID"
GPU_UUID=GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23
LOCK_PARENT=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/.gpu_reservations
LOCK="$LOCK_PARENT/$GPU_UUID"
SESSION=r15cadencemsaa20261029a
SOURCE_COMMIT=${R15_SOURCE_COMMIT:?set R15_SOURCE_COMMIT to the local committed source version}
EXPECTED_CONFIG_SHA256=893779fe80b29e1bcfdde0a729194d2d4050f3991fbb3b3aa3a1237a71b093e6
EXPECTED_DIAGNOSTIC_SHA256=6c571aad7489992103c263dd6999bedb1de6b925f02815ce8b490b417d5a3fe4
ARCHIVE_TRACE="$R05_ROOT/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/episodes/episode_005/state_control_trace.jsonl"
ARCHIVE_EVENTS="$R05_ROOT/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/episodes/episode_005/direct_hdr_events_1khz_2khz.npz"
E_CACHE="$E_ROOT/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059/episode_005"
E_EVENTS="$E_ROOT/benchmark/robot_closed_loop/results/r1_5_shadow_perception_causal_replay_20261029e/episode_005_shadow_on_raw_events.npz"
SCENE_CONFIG="$E_ROOT/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"

test "$(hostname)" = g1
test "$(git -C "$E_ROOT" rev-parse HEAD)" = 855b4b0bc7de96170b4bfee8d7002b370782f346
test -z "$(git -C "$E_ROOT" status --short --untracked-files=no)"
test -x "$ROBOT_PYTHON"
test "$(sha256sum "$LOGDIR/experiment_config.json" | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA256"
test "$(sha256sum "$LOGDIR/diagnose_r1_5_cached_state_cadence_renderer.py" | awk '{print $1}')" = "$EXPECTED_DIAGNOSTIC_SHA256"
test "$(sha256sum "$ARCHIVE_TRACE" | awk '{print $1}')" = 310837ea07ab2f819f590ccce91b3952dd9e85ea5005537bc1f4d1a339bff28c
test "$(sha256sum "$ARCHIVE_EVENTS" | awk '{print $1}')" = b698817a8347ca48ad75015f0cb81c3bfc711958a60d3387d468c70456a7b2b6
test "$(sha256sum "$E_CACHE/trace_window.jsonl" | awk '{print $1}')" = f0f556842029cd54b3fc082ee0e2848dd194f3322a8a9c0769c2c889a6f6ae5d
test "$(sha256sum "$E_EVENTS" | awk '{print $1}')" = c46f7bc28504ed9f43e76c334533534291d0de64b0c0f9bd429fa1a49eab58d8
test "$(sha256sum "$SCENE_CONFIG" | awk '{print $1}')" = aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a
test ! -e "$OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "refusing to reuse tmux session $SESSION" >&2
  exit 2
fi

mkdir -p "$LOGDIR" "$OUT"
if [[ ! -d "$LOCK_PARENT" ]]; then mkdir -m 700 -p "$LOCK_PARENT"; fi
if ! mkdir -m 700 "$LOCK"; then
  echo "GPU UUID already reserved: $GPU_UUID" >&2
  rmdir "$OUT" 2>/dev/null || true
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
json.dump({'run_id':'$RUN_ID','agent':'robot R1.5 diagnosis','host':'$HOST','gpu_index':0,'gpu_uuid':'$GPU_UUID','status':'reserved_before_final_gate','reserved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':'$SOURCE_COMMIT','renderer_checkout':'$E_ROOT','renderer_commit':'855b4b0bc7de96170b4bfee8d7002b370782f346','output':'$OUT'},open(sys.argv[1],'w'),indent=2)
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
  printf '\n--- runtime ---\n'
  "$ROBOT_PYTHON" -V
  "$ROBOT_PYTHON" -c 'import genesis,torch,numpy; print("genesis_module",genesis.__file__); print("torch",torch.__version__,"cuda",torch.cuda.is_available()); print("numpy",numpy.__version__)'
  printf '\n--- renderer helper hashes ---\n'
  sha256sum "$E_ROOT/benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_shadow_multisample_cached_states.py" \
    "$E_ROOT/benchmark/robot_closed_loop/diagnostics/diagnose_r1_ours_localization_cadence.py"
} > "$LOGDIR/preflight.txt" 2>&1

PMON=$(nvidia-smi pmon -c 1 -i 0 -s um)
if printf '%s\n' "$PMON" | awk '$1==0 && $2 ~ /^[0-9]+$/ {found=1} END {exit !found}'; then
  python3 -c 'import json,sys,datetime; p=sys.argv[1]; d=json.load(open(p)); d.update(status="aborted_gpu_busy",finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); json.dump(d,open(p,"w"),indent=2)' "$LOCK/allocation.json"
  cat "$LOCK/allocation.json" > "$LOGDIR/resource_allocation_aborted.json"
  rmdir "$OUT" 2>/dev/null || true
  exit 73
fi

cp "$LOGDIR/experiment_config.json" "$OUT/experiment_config.json"
cp "$LOGDIR/diagnose_r1_5_cached_state_cadence_renderer.py" "$OUT/diagnose_r1_5_cached_state_cadence_renderer.py"
cat > "$LOGDIR/worker.sh" <<'WORKER'
#!/usr/bin/env bash
set -u
RUN_ID=r1_5_cached_state_cadence_renderer_20261029a
E_ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r1_5-20261029-f6a2b1e
R05_ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a
ROBOT_PYTHON=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
OUT="$R05_ROOT/benchmark/robot_closed_loop/results/$RUN_ID"
LOGDIR="$R05_ROOT/benchmark/robot_closed_loop/results/runtime_logs/$RUN_ID"
LOCK=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/.gpu_reservations/GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23
ARCHIVE_TRACE="$R05_ROOT/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/episodes/episode_005/state_control_trace.jsonl"
ARCHIVE_EVENTS="$R05_ROOT/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/episodes/episode_005/direct_hdr_events_1khz_2khz.npz"
E_CACHE="$E_ROOT/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059/episode_005"
E_EVENTS="$E_ROOT/benchmark/robot_closed_loop/results/r1_5_shadow_perception_causal_replay_20261029e/episode_005_shadow_on_raw_events.npz"
SCENE_CONFIG="$E_ROOT/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"
cd "$E_ROOT" || exit 91
printf 'started_utc=%s\nsource_commit=%s\nrenderer_commit=855b4b0bc7de96170b4bfee8d7002b370782f346\n' \
  "$(date -u +%FT%TZ)" "$(cat "$LOGDIR/source_commit.txt")" > "$LOGDIR/worker_started_at.txt"
printf '%s\n' "$$" > "$LOGDIR/worker_pid.txt"
set +e
env CUDA_VISIBLE_DEVICES= EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$E_ROOT" \
  "$ROBOT_PYTHON" -u "$OUT/diagnose_r1_5_cached_state_cadence_renderer.py" \
  --archive-trace "$ARCHIVE_TRACE" --e-trace-window "$E_CACHE/trace_window.jsonl" \
  --archive-events "$ARCHIVE_EVENTS" --e-shadow-on-events "$E_EVENTS" \
  --scene-config "$SCENE_CONFIG" --output "$OUT/artifacts" \
  2>&1 | tee "$LOGDIR/run.log"
RC=${PIPESTATUS[0]}
printf '%s\n' "$RC" > "$LOGDIR/exit_code.txt"
printf 'finished_utc=%s\nexit_code=%s\n' "$(date -u +%FT%TZ)" "$RC" > "$LOGDIR/worker_finished_at.txt"
if [[ -d "$OUT/artifacts" ]]; then
  (cd "$OUT/artifacts" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
fi
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
printf '%s\n' "$SOURCE_COMMIT" > "$LOGDIR/source_commit.txt"
chmod 600 "$LOGDIR/source_commit.txt"
tmux new-session -d -s "$SESSION" "bash '$LOGDIR/worker.sh'"
WORKER_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
printf '%s\n' "$WORKER_PID" > "$LOGDIR/tmux_pane_pid.txt"
for _ in $(seq 1 20); do [[ -s "$LOGDIR/worker_pid.txt" ]] && break; sleep 0.25; done
test -s "$LOGDIR/worker_pid.txt"
python3 - "$LOCK/allocation.json" "$RUN_ID" "$WORKER_PID" "$SESSION" <<'PY'
import datetime,json,sys
p,run_id,pid,session=sys.argv[1:]
d=json.load(open(p)); assert d.get('run_id')==run_id
d.update(status='running',worker_pid=int(pid),tmux_session=session,launched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
json.dump(d,open(p,'w'),indent=2)
PY
ps -o user,pid,ppid,etime,cmd -p "$WORKER_PID" > "$LOGDIR/detach_verification.txt"
tmux has-session -t "$SESSION"
printf 'session=%s worker_pid=%s output=%s\n' "$SESSION" "$WORKER_PID" "$OUT"
KEEP_LOCK=1
