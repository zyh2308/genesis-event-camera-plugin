#!/usr/bin/env bash
set -euo pipefail
RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-replicad-v3-sc2-v2e-gpu-run21
PROJECT=/home/zhaoyuhan/Eventbased_WAM
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
SESSION=sim2real_replicad_sc2_v2e_cuda_20261030_21
GPU_INDEX=0
GPU_UUID=GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23
LOCK=$PROJECT/runs/sim2real/resource_reservations/$GPU_UUID
INPUT_RAD=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-replicad-v3_sc2_staging_00-fast-run16/render_output/camera_motion_30ms/radiance_frames.npy
CONVERSION_SOURCE=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030/benchmark/sim2real/scripts/generate_stage5_flow_coverage_bank.py
V2E_SOURCE=$PROJECT/external/sources/v2e-cac6384445/v2e-pinned
V2E_MODEL=$PROJECT/external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt
cd "$PROJECT"
test "$(whoami)" = zhaoyuhan
test -x "$PY" && test -s "$INPUT_RAD" && test -s "$CONVERSION_SOURCE" && test -f "$V2E_SOURCE/v2e.py" && test -s "$V2E_MODEL"
test ! -e "$RUN/started_utc.txt"
LOAD1=$(awk '{print $1}' /proc/loadavg); NCPU=$(getconf _NPROCESSORS_ONLN); MEM=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if ! awk -v l="$LOAD1" -v c="$NCPU" 'BEGIN {exit !(l < c * .8)}'; then echo "CPU gate refused: load=$LOAD1 cpus=$NCPU" >&2; exit 5; fi
if [[ "$MEM" -lt 16777216 ]]; then echo "memory gate refused: available_kb=$MEM" >&2; exit 5; fi
ACTUAL_UUID=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
if [[ "$ACTUAL_UUID" != "$GPU_UUID" ]]; then echo "GPU UUID mismatch: $ACTUAL_UUID != $GPU_UUID" >&2; exit 5; fi
GPU_APPS=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [[ -n "$GPU_APPS" ]]; then echo "selected GPU has compute-app ownership; refusing allocation" >&2; echo "$GPU_APPS" >&2; exit 5; fi
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "refusing duplicate tmux session $SESSION" >&2; exit 6; fi
mkdir -p "$RUN" "$PROJECT/runs/sim2real/resource_reservations"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/preflight_utc.txt"
{ printf 'host=%s\nproject=%s\npython=%s\nscene_config=v3_sc2_staging_00.scene_instance.json\n' "$(hostname)" "$PROJECT" "$PY"
 printf 'cuda_visible_devices=0\ngpu_index=%s\ngpu_uuid=%s\n' "$GPU_INDEX" "$ACTUAL_UUID"
 printf 'base_env=Eventbased_WAM; V2E inherited CUDA device 0\nload1=%s\ncpu_count=%s\nmem_available_kb=%s\n' "$LOAD1" "$NCPU" "$MEM"
 printf 'source_radiance_sha256='; sha256sum "$INPUT_RAD"
 printf 'conversion_source_sha256='; sha256sum "$CONVERSION_SOURCE"
 printf 'v2e_source_commit=%s\n' cac638444557a224ba6ee5c29260916f152c00bd
 sha256sum "$V2E_SOURCE/v2e.py" "$V2E_MODEL"
 nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
 df -h "$RUN" /tmp
} > "$RUN/environment_and_inputs.txt" 2>&1
if ! mkdir "$LOCK" 2>/dev/null; then echo "GPU reservation already exists: $LOCK" >&2; exit 6; fi
printf '%s\n' "$RUN" > "$LOCK/owner.txt"
"$PY" - "$RUN/resource_reservation.json" "$LOCK/resource_reservation.json" "$ACTUAL_UUID" "$RUN" <<'PY'
import json, sys
from datetime import datetime, timezone
d={"schema":"sim2real_gpu_reservation_v1","status":"reserved","host":"g1.pkuml.org","gpu_uuid":sys.argv[3],"run_id":"20261030-stage7-replicad-v3-sc2-v2e-gpu-run21","run_path":sys.argv[4],"created_utc":datetime.now(timezone.utc).isoformat(),"owner":"zhaoyuhan / EventGenesis Sim-to-Real engineering","purpose":"paired V2E conversion for preregistered sc2 validation candidate","cuda_visible_devices":"0"}
for p in sys.argv[1:3]: open(p,'w').write(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
tmux new-session -d -s "$SESSION" "bash '$RUN/driver.sh'"
if ! tmux has-session -t "$SESSION"; then
  if [[ -f "$LOCK/owner.txt" ]] && [[ "$(cat "$LOCK/owner.txt")" == "$RUN" ]]; then
    rm -f -- "$LOCK/owner.txt" "$LOCK/resource_reservation.json"
    rmdir -- "$LOCK" 2>/dev/null || true
  fi
  echo "detached tmux session did not persist" >&2
  exit 7
fi
PANE_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
printf '%s\n' "$SESSION" > "$RUN/tmux_session.txt"
printf '%s\n' "$PANE_PID" > "$RUN/tmux_pane_pid.txt"
ps -o pid,ppid,etime,args -p "$PANE_PID" > "$RUN/detach_verification.txt"
cat "$RUN/environment_and_inputs.txt"
printf 'tmux_session=%s pane_pid=%s reservation=%s\n' "$SESSION" "$PANE_PID" "$LOCK"
