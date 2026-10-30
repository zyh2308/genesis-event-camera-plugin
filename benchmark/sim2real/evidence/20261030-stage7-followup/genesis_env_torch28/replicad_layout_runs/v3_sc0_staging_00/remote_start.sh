#!/usr/bin/env bash
set -euo pipefail
RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-replicad-v3_sc0_staging_00-fast-run14
TMP_ROOT=/tmp/sim2real-stage7-replicad-v3_sc0_staging_00-fast-run14
ASSET_ROOT=/home/zhaoyuhan/Eventbased_WAM/external/sim2real_assets/replicacad_interactive_v1_5/extracted_v1_5
SCENE_CONFIG=v3_sc0_staging_00.scene_instance.json
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python
SESSION=sim2real_replicad_v3_sc0_staging_00_fast_20261030_14
CACHE=$TMP_ROOT/cache
GENESIS_ROOT=/home/zhaoyuhan/Eventbased_WAM/runtime/sim2real/genesis_hdr_v2
PROJECT=/home/zhaoyuhan/Eventbased_WAM
cd "$PROJECT"
test "$(pwd)" = "$PROJECT"
test -x "$PY" && test -d "$ASSET_ROOT" && test -f "$GENESIS_ROOT/genesis/vis/camera.py"
test -d "$RUN" && test -f "$RUN/driver.sh" && test -f "$RUN/replicad_layout_camera_motion_pilot.py" && test ! -e "$TMP_ROOT"
LOAD1=$(awk '{print $1}' /proc/loadavg); NCPU=$(getconf _NPROCESSORS_ONLN); MEM=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if ! awk -v l="$LOAD1" -v c="$NCPU" 'BEGIN {exit !(l < c * .8)}'; then echo "CPU gate refused: load=$LOAD1 cpus=$NCPU" >&2; exit 5; fi
if [[ "$MEM" -lt 16777216 ]]; then echo "memory gate refused: available_kb=$MEM" >&2; exit 5; fi
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "refusing duplicate session $SESSION" >&2; exit 6; fi
mkdir -p "$RUN" "$CACHE/quadrants" "$CACHE/genesis"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/preflight_utc.txt"
{
 printf 'host=%s\nproject=%s\n' "$(hostname)" "$PROJECT"
 printf 'execution_checkout_commit='; git -C "$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030" rev-parse HEAD
 printf 'local_source_repo_commit='; cat "$RUN/local_repo_commit.txt"
 printf 'scene_config=%s\nscene_role=%s\npython=%s\nasset_root=%s\nload1=%s\ncpu_count=%s\nmem_available_kb=%s\n' "$SCENE_CONFIG" "train_candidate" "$PY" "$ASSET_ROOT" "$LOAD1" "$NCPU" "$MEM"
 printf 'device_mode=CPU; CUDA_VISIBLE_DEVICES empty; Mesa software rasterizer requested\n'
 printf 'tmp_root=%s\ncache=%s\n' "$TMP_ROOT" "$CACHE"
 sha256sum "$ASSET_ROOT/../ReplicaCAD_dataset_v1.5.zip" "$ASSET_ROOT/configs/scenes/$SCENE_CONFIG" "$GENESIS_ROOT/genesis/vis/camera.py" "$RUN/replicad_layout_camera_motion_pilot.py" "$RUN/replicad_layout_camera_motion_pilot_registration_v1.json" "$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030/genesis_event_plugin/physics_plugin.py" "$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030/genesis_event_plugin/physics_warp.py" "$PY"
 nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
 df -h "$RUN" /tmp
} > "$RUN/environment_and_inputs.txt" 2>&1
tmux new-session -d -s "$SESSION" "bash '$RUN/driver.sh'"
tmux has-session -t "$SESSION"
PANE_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
printf '%s\n' "$SESSION" > "$RUN/tmux_session.txt"
printf '%s\n' "$PANE_PID" > "$RUN/tmux_pane_pid.txt"
ps -o pid,ppid,etime,args -p "$PANE_PID" > "$RUN/detach_verification.txt"
cat "$RUN/environment_and_inputs.txt"
printf 'tmux_session=%s pane_pid=%s\n' "$SESSION" "$PANE_PID"
