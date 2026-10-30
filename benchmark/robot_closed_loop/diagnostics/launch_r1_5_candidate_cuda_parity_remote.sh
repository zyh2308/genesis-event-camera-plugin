#!/usr/bin/env bash
set -euo pipefail
HOST=${1:?host label}
RUN_ID=${2:?run id}
SOURCE_COMMIT=${3:?source commit}
GPU_INDEX=${4:?GPU index}
GPU_UUID=${5:?GPU UUID}
HOME_ROOT=/home/zhaoyuhan/Eventbased_WAM
BASE="$HOME_ROOT/repositories/genesis-event-camera-plugin-r1-diag-20261019"
CACHE="$BASE/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059"
RUN_ROOT="$HOME_ROOT/runs/robot_scenes/genesis_event_camera_plugin/$RUN_ID"
ROOT="$RUN_ROOT/source"
LOCK_PARENT="$HOME_ROOT/runs/robot_scenes/.gpu_reservations"
GPU_LOCK="$LOCK_PARENT/$GPU_UUID"
SLOT="$LOCK_PARENT/.r1_5_core_candidate_cuda_slot"
OUT="$RUN_ROOT/output"
LOGDIR="$RUN_ROOT/logs"
PYTHON=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
WORKER="$ROOT/benchmark/robot_closed_loop/diagnostics/launch_r1_5_candidate_cuda_parity_worker.sh"
SESSION=r15cuda20261030c
KEEP_LOCK=0
preflight_fail() { echo "PREFLIGHT_FAIL: $*" >&2; exit 70; }

cleanup_unlaunched() {
  if [[ "$KEEP_LOCK" == 0 ]]; then
    if [[ -f "$GPU_LOCK/allocation.json" ]] && grep -Fq "$RUN_ID" "$GPU_LOCK/allocation.json"; then
      rm -f "$GPU_LOCK/allocation.json" "$GPU_LOCK/created_utc.txt"
      rmdir "$GPU_LOCK" 2>/dev/null || true
    fi
    if [[ -f "$SLOT/owner.txt" ]] && [[ "$(cat "$SLOT/owner.txt")" == "$RUN_ID $HOST $GPU_UUID" ]]; then
      rm -f "$SLOT/owner.txt"
      rmdir "$SLOT" 2>/dev/null || true
    fi
  fi
}
trap cleanup_unlaunched EXIT

[[ "$(hostname -s)" == "$HOST" ]] || preflight_fail "host identity expected=$HOST actual=$(hostname -s)"
[[ "$(findmnt -T "$HOME_ROOT" -no FSTYPE)" == nfs ]] || preflight_fail "project home is not on verified NFS"
[[ -d "$ROOT/genesis_event_plugin" && -f "$ROOT/source_provenance.json" ]] || preflight_fail "staged source/provenance missing"
[[ ! -e "$OUT" && ! -e "$LOGDIR" ]] || preflight_fail "run output/log path already exists"
grep -Fq "$SOURCE_COMMIT" "$ROOT/source_provenance.json" || preflight_fail "staged source commit mismatch"
[[ -x "$PYTHON" && -x "$WORKER" ]] || preflight_fail "python or worker is not executable"
[[ -r "$CACHE/episode_004/same_state_source_cache.h5" && -r "$CACHE/episode_005/same_state_source_cache.h5" ]] || preflight_fail "archived cache missing at $CACHE"
printf '%s  %s\n' \
  0525c18c8d89f5831ef0cb71ac9e3b6b130d60579daf374f06d5cfcee5b879d7 "$CACHE/episode_004/same_state_source_cache.h5" \
  91c1397642e317d6bb51150b4107c518b52e4fa9fcf5f1caf6a2a91315c7d434 "$CACHE/episode_005/same_state_source_cache.h5" | sha256sum -c - || preflight_fail "archived cache hash mismatch"
"$PYTHON" -c 'import torch,numpy,h5py; assert torch.cuda.is_available(); print("torch",torch.__version__,"cuda",torch.version.cuda,"numpy",numpy.__version__,"h5py",h5py.__version__)' || preflight_fail "CUDA runtime/import check failed"
command -v tmux >/dev/null || preflight_fail "tmux unavailable"
command -v nvidia-smi >/dev/null || preflight_fail "nvidia-smi unavailable"

gpu_free() {
  local row idx uuid util mem apps pmon
  row=$(nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits -i "$GPU_UUID" | head -1)
  [[ -n "$row" ]] || return 1
  IFS=, read -r idx uuid util mem <<< "$row"
  idx=$(printf '%s' "$idx" | xargs); uuid=$(printf '%s' "$uuid" | xargs)
  util=$(printf '%s' "$util" | xargs); mem=$(printf '%s' "$mem" | xargs)
  [[ "$idx" =~ ^[0-9]+$ && "$idx" == "$GPU_INDEX" && "$uuid" == "$GPU_UUID" ]]
  [[ "$util" =~ ^[0-9]+$ && "$mem" =~ ^[0-9]+$ ]]
  (( util <= 5 && mem <= 512 )) || return 1
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null || true)
  printf '%s\n' "$apps" | grep -Fq "$GPU_UUID" && return 1
  pmon=$(nvidia-smi pmon -c 1 -i "$GPU_INDEX" -s um 2>/dev/null || true)
  printf '%s\n' "$pmon" | awk -v i="$GPU_INDEX" '$1==i && $2 ~ /^[0-9]+$/ {found=1} END {exit !found}' && return 1
  return 0
}
host_resources_free() {
  local cores load avail disk_kb jobs
  cores=$(nproc)
  load=$(awk '{print $1}' /proc/loadavg)
  avail=$(awk '/MemAvailable:/ {print $2*1024}' /proc/meminfo)
  disk_kb=$(df -Pk "$HOME_ROOT" | awk 'NR==2 {print $4}')
  awk -v l="$load" -v c="$cores" 'BEGIN {exit !(l <= c*1.25)}' || return 1
  (( avail >= 8*1024*1024*1024 && disk_kb >= 2*1024*1024 )) || return 1
  if command -v squeue >/dev/null 2>&1; then
    jobs=$(squeue -h -u zhaoyuhan -o '%T' 2>/dev/null | awk '$1=="PENDING" || $1=="RUNNING" {n++} END {print n+0}')
    (( jobs == 0 )) || return 1
  fi
}

if ! host_resources_free || ! gpu_free; then
  echo "ABORT final resource gate failed host=$HOST gpu=$GPU_UUID"
  exit 72
fi
if ! mkdir -m 700 "$SLOT" 2>/dev/null; then
  echo "ABORT shared diagnostic slot already reserved"
  exit 73
fi
printf '%s %s %s\n' "$RUN_ID" "$HOST" "$GPU_UUID" > "$SLOT/owner.txt"
if ! mkdir -m 700 "$GPU_LOCK" 2>/dev/null; then
  echo "ABORT GPU UUID already reserved"
  exit 74
fi
date -u +%FT%TZ > "$GPU_LOCK/created_utc.txt"
python3 - "$GPU_LOCK/allocation.json" <<PY
import json,datetime
json.dump({'run_id':'$RUN_ID','agent':'robot R1.5 candidate CUDA parity diagnostic','host':'$HOST','gpu_index':int('$GPU_INDEX'),'gpu_uuid':'$GPU_UUID','status':'reserved_before_final_gate','reserved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':'$SOURCE_COMMIT','candidate_checkout':'$ROOT','output':'$OUT'},open('$GPU_LOCK/allocation.json','w'),indent=2)
PY
chmod 600 "$GPU_LOCK/allocation.json"
if ! host_resources_free || ! gpu_free; then
  echo "ABORT post-reservation gate failed; releasing own reservation"
  exit 75
fi

mkdir -p "$LOGDIR"
{
  echo "host=$HOST"; echo "hostname=$(hostname -f)"; echo "started_utc=$(date -u +%FT%TZ)"
  echo "source_commit=$SOURCE_COMMIT"; echo "candidate_checkout=$ROOT"; echo "gpu_index=$GPU_INDEX"; echo "gpu_uuid=$GPU_UUID"
  echo "gpu_snapshot=$(nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -i "$GPU_UUID")"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader 2>&1
  echo "load=$(cat /proc/loadavg)"; free -h; df -h "$HOME_ROOT"
  "$PYTHON" -c 'import torch,numpy,h5py; print("torch",torch.__version__,"cuda",torch.version.cuda,"available",torch.cuda.is_available()); print("numpy",numpy.__version__,"h5py",h5py.__version__)'
  cat "$ROOT/source_provenance.json"
  sha256sum "$ROOT/benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_core_candidate_cuda_parity.py" "$ROOT/benchmark/robot_closed_loop/configs/r1_5_core_cuda_parity_20261030c.json"
} > "$LOGDIR/preflight.txt" 2>&1

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ABORT tmux session exists"
  exit 76
fi
tmux new-session -d -s "$SESSION" "bash '$WORKER' '$RUN_ID' '$HOST' '$GPU_INDEX' '$GPU_UUID' '$SOURCE_COMMIT'"
sleep 1
tmux has-session -t "$SESSION"
PANE_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
python3 - "$GPU_LOCK/allocation.json" "$RUN_ID" "$PANE_PID" "$SESSION" <<'PY'
import json,sys,datetime
p,run_id,pid,session=sys.argv[1:]
d=json.load(open(p)); assert d.get('run_id')==run_id
d.update(status='running',tmux_pane_pid=int(pid),tmux_session=session,launched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
json.dump(d,open(p,'w'),indent=2)
PY
sleep 1
tmux has-session -t "$SESSION"
ps -o user,pid,ppid,etime,cmd -p "$PANE_PID" > "$LOGDIR/detach_verification.txt"
echo "LAUNCHED run_id=$RUN_ID host=$HOST gpu_index=$GPU_INDEX gpu_uuid=$GPU_UUID tmux=$SESSION pane_pid=$PANE_PID output=$OUT log=$LOGDIR"
KEEP_LOCK=1
