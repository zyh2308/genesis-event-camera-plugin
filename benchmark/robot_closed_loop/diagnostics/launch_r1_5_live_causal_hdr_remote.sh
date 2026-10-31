#!/usr/bin/env bash
set -euo pipefail
HOST=${1:?host label}
RUN_ID=${2:?run id}
SOURCE_COMMIT=${3:?source commit}
GPU_INDEX=${4:?GPU index}
GPU_UUID=${5:?GPU UUID}
HOME_ROOT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT="$HOME_ROOT/runs/robot_scenes/genesis_event_camera_plugin/$RUN_ID"
ROOT="$RUN_ROOT/source"
PYTHON="$HOME_ROOT/repositories/gec-robot/.venv-robot/bin/python"
LOCK_PARENT="$HOME_ROOT/runs/robot_scenes/.gpu_reservations"
GPU_LOCK="$LOCK_PARENT/$GPU_UUID"
SLOT="$LOCK_PARENT/.r1_5_core_candidate_cuda_slot"
OUT="$RUN_ROOT/output"
LOGDIR="$RUN_ROOT/logs"
CONFIG="benchmark/robot_closed_loop/configs/${RUN_ID}.json"
SCENE_CONFIG=benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json
SCRIPT=benchmark/robot_closed_loop/diagnostics/run_r1_5_live_causal_hdr_plugin.py
WORKER="$ROOT/benchmark/robot_closed_loop/diagnostics/run_r1_5_live_causal_hdr_worker.sh"
SESSION="r15causal${RUN_ID##*_}"
KEEP=0
preflight_fail() { echo "PREFLIGHT_FAIL: $*" >&2; exit 70; }
cleanup_unlaunched() {
  if [[ "$KEEP" == 0 ]]; then
    if [[ -f "$GPU_LOCK/allocation.json" ]] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("run_id")==sys.argv[2] and d.get("gpu_uuid")==sys.argv[3] else 1)' "$GPU_LOCK/allocation.json" "$RUN_ID" "$GPU_UUID"; then
      rm -f "$GPU_LOCK/allocation.json" "$GPU_LOCK/created_utc.txt"
      rmdir "$GPU_LOCK" 2>/dev/null || true
    fi
    if [[ "$(cat "$SLOT/owner.txt" 2>/dev/null || true)" == "$RUN_ID $HOST $GPU_UUID" ]]; then
      rm -f "$SLOT/owner.txt"; rmdir "$SLOT" 2>/dev/null || true
    fi
  fi
}
trap cleanup_unlaunched EXIT

[[ "$(hostname -s)" == "$HOST" ]] || preflight_fail "host identity expected=$HOST actual=$(hostname -s)"
[[ "$(findmnt -T "$HOME_ROOT" -no FSTYPE)" == nfs ]] || preflight_fail "project home is not the verified NFS mount"
[[ -x "$PYTHON" && -x "$WORKER" && -r "$ROOT/$SCRIPT" ]] || preflight_fail "HDR Python/worker/runner unavailable"
[[ -f "$ROOT/source_provenance.json" && -f "$ROOT/$CONFIG" && -f "$ROOT/$SCENE_CONFIG" ]] || preflight_fail "staged source/config missing"
grep -Fq "$SOURCE_COMMIT" "$ROOT/source_provenance.json" || preflight_fail "source commit mismatch"
[[ ! -e "$OUT" && ! -e "$LOGDIR" ]] || preflight_fail "refusing to overwrite run output/log directory"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("run_id")==sys.argv[2]' "$ROOT/$CONFIG" "$RUN_ID"
python3 -c 'import hashlib,json,sys; d=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert h==d["expected_scene_config_sha256"],(h,d["expected_scene_config_sha256"])' "$ROOT/$CONFIG" "$ROOT/$SCENE_CONFIG"
"$PYTHON" -c 'import sys,inspect,genesis as gs; from genesis.vis.camera import Camera; s=inspect.signature(Camera.render); assert "radiance" in s.parameters,(gs.__file__,s); print("HDR_API_OK",sys.executable,gs.__file__,s)'
command -v tmux >/dev/null || preflight_fail "tmux unavailable"
command -v nvidia-smi >/dev/null || preflight_fail "nvidia-smi unavailable"

host_resources_free() {
  local cores load avail disk_kb jobs
  cores=$(nproc); load=$(awk '{print $1}' /proc/loadavg)
  avail=$(awk '/MemAvailable:/ {print $2*1024}' /proc/meminfo)
  disk_kb=$(df -Pk "$HOME_ROOT" | awk 'NR==2 {print $4}')
  awk -v l="$load" -v c="$cores" 'BEGIN {exit !(l <= c*1.25)}' || return 1
  (( avail >= 8*1024*1024*1024 && disk_kb >= 2*1024*1024 )) || return 1
  if command -v squeue >/dev/null 2>&1; then
    jobs=$(squeue -h -u zhaoyuhan -o '%T' 2>/dev/null | awk '$1=="PENDING" || $1=="RUNNING" {n++} END {print n+0}')
    (( jobs == 0 )) || return 1
  fi
}
gpu_free() {
  local row idx uuid util mem apps pmon
  row=$(nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits -i "$GPU_UUID" | head -1)
  [[ -n "$row" ]] || return 1
  IFS=, read -r idx uuid util mem <<< "$row"
  idx=$(printf '%s' "$idx" | xargs); uuid=$(printf '%s' "$uuid" | xargs)
  util=$(printf '%s' "$util" | xargs); mem=$(printf '%s' "$mem" | xargs)
  [[ "$idx" == "$GPU_INDEX" && "$uuid" == "$GPU_UUID" && "$util" =~ ^[0-9]+$ && "$mem" =~ ^[0-9]+$ ]] || return 1
  (( util <= 5 && mem <= 512 )) || return 1
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null || true)
  printf '%s\n' "$apps" | grep -Fq "$GPU_UUID" && return 1
  pmon=$(nvidia-smi pmon -c 1 -i "$GPU_INDEX" -s um 2>/dev/null || true)
  printf '%s\n' "$pmon" | awk -v i="$GPU_INDEX" '$1==i && $2 ~ /^[0-9]+$/ {found=1} END {exit !found}' && return 1
  return 0
}
if ! host_resources_free || ! gpu_free; then
  echo "ABORT final resource gate failed host=$HOST gpu=$GPU_UUID"; exit 72
fi
mkdir -m 700 "$SLOT" || { echo "ABORT shared candidate slot already reserved"; exit 73; }
printf '%s %s %s\n' "$RUN_ID" "$HOST" "$GPU_UUID" > "$SLOT/owner.txt"
mkdir -m 700 "$GPU_LOCK" || { echo "ABORT GPU UUID already reserved"; exit 74; }
date -u +%FT%TZ > "$GPU_LOCK/created_utc.txt"
python3 -c 'import datetime,json,sys; p,run,host,idx,uuid,commit,root,out=sys.argv[1:]; json.dump({"run_id":run,"agent":"R1.5 live causal HDR plugin smoke","host":host,"gpu_index":int(idx),"gpu_uuid":uuid,"pid":None,"status":"reserved_before_final_gate","reserved_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"source_commit":commit,"source_root":root,"output":out,"mode":"live_causal"},open(p,"w"),indent=2)' "$GPU_LOCK/allocation.json" "$RUN_ID" "$HOST" "$GPU_INDEX" "$GPU_UUID" "$SOURCE_COMMIT" "$ROOT" "$OUT"
chmod 600 "$GPU_LOCK/allocation.json"
if ! host_resources_free || ! gpu_free; then echo "ABORT post-reservation resource gate failed"; exit 75; fi
mkdir -m 700 "$LOGDIR"
{
  echo "host=$HOST"; echo "hostname=$(hostname -f)"; echo "started_utc=$(date -u +%FT%TZ)"
  echo "run_id=$RUN_ID"; echo "source_commit=$SOURCE_COMMIT"; echo "python=$PYTHON"
  echo "gpu_index=$GPU_INDEX"; echo "gpu_uuid=$GPU_UUID"
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -i "$GPU_UUID"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader
  echo "load=$(cat /proc/loadavg)"; free -h; df -h "$HOME_ROOT"
  "$PYTHON" -c 'import torch,genesis,inspect; from genesis.vis.camera import Camera; print("torch",torch.__version__); print("genesis",genesis.__file__); print(inspect.signature(Camera.render))'
  sha256sum "$ROOT/$SCRIPT" "$ROOT/$CONFIG" "$ROOT/$SCENE_CONFIG"
  cat "$ROOT/source_provenance.json"
} > "$LOGDIR/preflight.txt" 2>&1
if tmux has-session -t "$SESSION" 2>/dev/null; then preflight_fail "tmux session already exists"; fi
tmux new-session -d -s "$SESSION" "bash '$WORKER' '$RUN_ID' '$HOST' '$GPU_INDEX' '$GPU_UUID' '$SOURCE_COMMIT'"
sleep 2
if tmux has-session -t "$SESSION" 2>/dev/null; then
  PANE_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
  python3 -c 'import datetime,json,sys; p,run,pid,sess=sys.argv[1:]; d=json.load(open(p)); assert d.get("run_id")==run; d.update(status="running",tmux_pane_pid=int(pid),tmux_session=sess,launched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); json.dump(d,open(p,"w"),indent=2)' "$GPU_LOCK/allocation.json" "$RUN_ID" "$PANE_PID" "$SESSION"
  ps -o user,pid,ppid,etime,cmd -p "$PANE_PID" > "$LOGDIR/detach_verification.txt"
  KEEP=1
  echo "LAUNCHED run_id=$RUN_ID gpu=$GPU_UUID tmux=$SESSION pane_pid=$PANE_PID output=$OUT logs=$LOGDIR"
elif [[ -f "$LOGDIR/status.json" ]]; then
  KEEP=1; echo "FINISHED_QUICKLY run_id=$RUN_ID output=$OUT"
else
  preflight_fail "tmux disappeared before worker status"
fi
