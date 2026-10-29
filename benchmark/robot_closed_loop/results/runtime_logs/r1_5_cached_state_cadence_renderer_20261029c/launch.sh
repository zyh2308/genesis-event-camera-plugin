#!/usr/bin/env bash
set -euo pipefail
RUN_ID=r1_5_cached_state_cadence_renderer_20261029c
R0=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a
E=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r1_5-20261029-f6a2b1e
PY=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
OUT="$R0/benchmark/robot_closed_loop/results/$RUN_ID"
LOG="$R0/benchmark/robot_closed_loop/results/runtime_logs/$RUN_ID"
SRC="$R0/benchmark/robot_closed_loop/results/r1_5_cadence_factorial_20261029b/source_replay/episodes/episode_000"
ARCH="$R0/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/episodes/episode_005"
ECACHE="$E/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059/episode_005"
EEVENTS="$E/benchmark/robot_closed_loop/results/r1_5_shadow_perception_causal_replay_20261029e/episode_005_shadow_on_raw_events.npz"
SCENE="$E/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"
GPU=GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23
LOCKP=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/.gpu_reservations
LOCK="$LOCKP/$GPU"
SESSION=r15renderer20261029c
SOURCE_COMMIT=${R15_SOURCE_COMMIT:?set R15_SOURCE_COMMIT}
CFG_SHA=767f7396486ee98589b751cde53cb58c5d14a7170be022aedb0ad0785958ca94
DIAG_SHA=a4226abf6ef5032890fd5565c22e46f12a887749013faabfff9e5998ad687efa
TRACE_SHA=310837ea07ab2f819f590ccce91b3952dd9e85ea5005537bc1f4d1a339bff28c
EVENT_SHA=b698817a8347ca48ad75015f0cb81c3bfc711958a60d3387d468c70456a7b2b6
CACHE_SHA=b87efd4d7b5bfda94b0987f0d682671a4230246c5f9886eb086600542bc075bb

test "$(hostname)" = g1
test "$(git -C "$R0" rev-parse HEAD)" = 2100645b0473c294760789bc4475b644f261d9cf
test -z "$(git -C "$R0" status --short --untracked-files=no)"
test "$(git -C "$E" rev-parse HEAD)" = 855b4b0bc7de96170b4bfee8d7002b370782f346
test -z "$(git -C "$E" status --short --untracked-files=no)"
test "$(sha256sum "$LOG/experiment_config.json" | awk '{print $1}')" = "$CFG_SHA"
test "$(sha256sum "$LOG/diagnose_r1_5_cached_state_cadence_renderer.py" | awk '{print $1}')" = "$DIAG_SHA"
test "$(sha256sum "$SRC/state_control_trace.jsonl" | awk '{print $1}')" = "$TRACE_SHA"
test "$(sha256sum "$SRC/direct_hdr_events_1khz_2khz.npz" | awk '{print $1}')" = "$EVENT_SHA"
test "$(sha256sum "$SRC/cached_physical_states_window.npz" | awk '{print $1}')" = "$CACHE_SHA"
test "$(sha256sum "$ARCH/state_control_trace.jsonl" | awk '{print $1}')" = "$TRACE_SHA"
test "$(sha256sum "$ARCH/direct_hdr_events_1khz_2khz.npz" | awk '{print $1}')" = "$EVENT_SHA"
test "$(sha256sum "$ECACHE/trace_window.jsonl" | awk '{print $1}')" = f0f556842029cd54b3fc082ee0e2848dd194f3322a8a9c0769c2c889a6f6ae5d
test "$(sha256sum "$EEVENTS" | awk '{print $1}')" = c46f7bc28504ed9f43e76c334533534291d0de64b0c0f9bd429fa1a49eab58d8
test "$(sha256sum "$SCENE" | awk '{print $1}')" = aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a
test ! -e "$OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "session exists" >&2; exit 2; fi
mkdir -p "$LOG" "$LOCKP"
if ! mkdir -m 700 "$LOCK"; then echo "GPU reserved: $GPU" >&2; exit 73; fi
KEEP=0
trap 'if [[ "$KEEP" == 0 && -f "$LOCK/allocation.json" ]]; then cp "$LOCK/allocation.json" "$LOG/resource_allocation_aborted.json"; rm -f "$LOCK/allocation.json"; rmdir "$LOCK" 2>/dev/null || true; fi' EXIT
python3 - "$LOCK/allocation.json" <<PY
import datetime,json,sys
json.dump({'run_id':'$RUN_ID','agent':'robot R1.5 cached renderer diagnostic','host':'g1.pkuml.org','gpu_uuid':'$GPU','gpu_index':0,'source_commit':'$SOURCE_COMMIT','renderer_commit':'855b4b0bc7de96170b4bfee8d7002b370782f346','source_capture_run':'r1_5_cadence_factorial_20261029b','status':'reserved_preflight','reserved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'output':'$OUT'},open(sys.argv[1],'w'),indent=2)
PY
chmod 600 "$LOCK/allocation.json"
{ hostname; date -u +%FT%TZ; uptime; free -h; df -h /home/zhaoyuhan/Eventbased_WAM; nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader; nvidia-smi pmon -c 1 -i 0 -s um; "$PY" -V; "$PY" -c 'import genesis,torch,numpy; print(genesis.__file__,torch.__version__,torch.cuda.is_available(),numpy.__version__)'; } > "$LOG/preflight.txt" 2>&1
PMON=$(nvidia-smi pmon -c 1 -i 0 -s um); printf '%s\n' "$PMON" > "$LOG/final_gpu_pmon.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader -i 0 > "$LOG/final_gpu_compute_apps.txt" 2>&1 || true
if printf '%s\n' "$PMON" | awk '$1==0 && $2~/^[0-9]+$/{f=1} END{exit !f}'; then python3 -c 'import json,sys,datetime; p=sys.argv[1]; x=json.load(open(p)); x.update(status="aborted_gpu_busy",pmon=open(sys.argv[2]).read(),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); json.dump(x,open(p,"w"),indent=2)' "$LOCK/allocation.json" "$LOG/final_gpu_pmon.txt"; cp "$LOCK/allocation.json" "$LOG/resource_allocation_aborted.json"; rm -f "$LOCK/allocation.json"; rmdir "$LOCK"; KEEP=1; exit 73; fi
mkdir -p "$OUT/scripts"
cp "$LOG/experiment_config.json" "$OUT/experiment_config.json"
cp "$LOG/diagnose_r1_5_cached_state_cadence_renderer.py" "$OUT/scripts/diagnose_r1_5_cached_state_cadence_renderer.py"
cat > "$LOG/worker.sh" <<'WORKER'
#!/usr/bin/env bash
set -u
RUN_ID=r1_5_cached_state_cadence_renderer_20261029c
R0=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a
E=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r1_5-20261029-f6a2b1e
PY=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
OUT="$R0/benchmark/robot_closed_loop/results/$RUN_ID"
LOG="$R0/benchmark/robot_closed_loop/results/runtime_logs/$RUN_ID"
SRC="$R0/benchmark/robot_closed_loop/results/r1_5_cadence_factorial_20261029b/source_replay/episodes/episode_000"
ARCH="$R0/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/episodes/episode_005"
ECACHE="$E/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059/episode_005"
EEVENTS="$E/benchmark/robot_closed_loop/results/r1_5_shadow_perception_causal_replay_20261029e/episode_005_shadow_on_raw_events.npz"
SCENE="$E/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"
LOCK=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/.gpu_reservations/GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23
SOURCE_COMMIT=$(cat "$LOG/source_commit.txt")
cd "$E" || exit 91
printf 'started_utc=%s\nsource_commit=%s\nrenderer_commit=855b4b0bc7de96170b4bfee8d7002b370782f346\n' "$(date -u +%FT%TZ)" "$SOURCE_COMMIT" > "$LOG/worker_started_at.txt"
printf '%s\n' "$$" > "$LOG/worker_pid.txt"
set +e
env CUDA_VISIBLE_DEVICES= EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$E" "$PY" -u "$OUT/scripts/diagnose_r1_5_cached_state_cadence_renderer.py" \
 --archive-trace "$ARCH/state_control_trace.jsonl" --e-trace-window "$ECACHE/trace_window.jsonl" \
 --dense-state-cache "$SRC/cached_physical_states_window.npz" --capture-rerun-trace "$SRC/state_control_trace.jsonl" \
 --capture-rerun-events "$SRC/direct_hdr_events_1khz_2khz.npz" --archive-events "$ARCH/direct_hdr_events_1khz_2khz.npz" \
 --e-shadow-on-events "$EEVENTS" --scene-config "$SCENE" --output "$OUT/artifacts" 2>&1 | tee "$LOG/run.log"
RC=${PIPESTATUS[0]}
printf '%s\n' "$RC" > "$LOG/exit_code.txt"
printf 'finished_utc=%s\nexit_code=%s\n' "$(date -u +%FT%TZ)" "$RC" > "$LOG/worker_finished_at.txt"
(cd "$OUT" && find . -type f ! -path './SHA256SUMS' -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
python3 - "$LOCK/allocation.json" "$RUN_ID" "$RC" <<'PY'
import datetime,json,sys
p,r,c=sys.argv[1],sys.argv[2],int(sys.argv[3]); x=json.load(open(p)); assert x.get('run_id')==r
x.update(status='completed' if c==0 else 'failed',exit_code=c,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); json.dump(x,open(p,'w'),indent=2)
PY
cp "$LOCK/allocation.json" "$LOG/resource_allocation.json"
if python3 -c 'import json,sys; assert json.load(open(sys.argv[1])).get("run_id")==sys.argv[2]' "$LOCK/allocation.json" "$RUN_ID"; then rm -f "$LOCK/allocation.json"; rmdir "$LOCK"; fi
exit "$RC"
WORKER
chmod 700 "$LOG/worker.sh"
printf '%s\n' "$SOURCE_COMMIT" > "$LOG/source_commit.txt"; chmod 600 "$LOG/source_commit.txt"
tmux new-session -d -s "$SESSION" "bash '$LOG/worker.sh'"
PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}'); printf '%s\n' "$PID" > "$LOG/tmux_pane_pid.txt"
for _ in $(seq 1 20); do [[ -s "$LOG/worker_pid.txt" ]] && break; sleep 0.25; done
test -s "$LOG/worker_pid.txt"
python3 - "$LOCK/allocation.json" "$RUN_ID" "$PID" "$SESSION" <<'PY'
import datetime,json,sys
p,r,pid,s=sys.argv[1:]; x=json.load(open(p)); assert x.get('run_id')==r
x.update(status='running',worker_pid=int(pid),tmux_session=s,launched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); json.dump(x,open(p,'w'),indent=2)
PY
ps -o user,pid,ppid,etime,cmd -p "$PID" > "$LOG/detach_verification.txt"
tmux has-session -t "$SESSION"
printf 'session=%s worker_pid=%s output=%s\n' "$SESSION" "$PID" "$OUT"
KEEP=1
