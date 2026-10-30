#!/usr/bin/env bash
set -euo pipefail
RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-replicad-v3-sc0-v2e-cpu-run17
PROJECT=/home/zhaoyuhan/Eventbased_WAM
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
SESSION=sim2real_replicad_sc0_v2e_20261030_17
INPUT_RAD=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-replicad-v3_sc0_staging_00-fast-run14/render_output/camera_motion_30ms/radiance_frames.npy
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
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "refusing duplicate tmux session $SESSION" >&2; exit 6; fi
mkdir -p "$RUN"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/preflight_utc.txt"
{ printf 'host=%s\nproject=%s\npython=%s\nscene_config=v3_sc0_staging_00.scene_instance.json\n' "$(hostname)" "$PROJECT" "$PY"
 printf 'base_env=Eventbased_WAM; CUDA_VISIBLE_DEVICES empty for child V2E\nload1=%s\ncpu_count=%s\nmem_available_kb=%s\n' "$LOAD1" "$NCPU" "$MEM"
 printf 'source_radiance_sha256='; sha256sum "$INPUT_RAD"
 printf 'conversion_source_sha256='; sha256sum "$CONVERSION_SOURCE"
 printf 'v2e_source_commit=%s\n' cac638444557a224ba6ee5c29260916f152c00bd
 sha256sum "$V2E_SOURCE/v2e.py" "$V2E_MODEL"
 nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
 df -h "$RUN" /tmp
} > "$RUN/environment_and_inputs.txt" 2>&1
tmux new-session -d -s "$SESSION" "bash '$RUN/run_v2e_same_radiance_camera_motion_driver_sc0_run17.sh'"
tmux has-session -t "$SESSION"
PANE_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
printf '%s\n' "$SESSION" > "$RUN/tmux_session.txt"
printf '%s\n' "$PANE_PID" > "$RUN/tmux_pane_pid.txt"
ps -o pid,ppid,etime,args -p "$PANE_PID" > "$RUN/detach_verification.txt"
cat "$RUN/environment_and_inputs.txt"
printf 'tmux_session=%s pane_pid=%s\n' "$SESSION" "$PANE_PID"
