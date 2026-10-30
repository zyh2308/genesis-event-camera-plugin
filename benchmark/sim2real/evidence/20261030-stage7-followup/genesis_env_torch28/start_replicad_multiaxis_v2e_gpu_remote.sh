#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT=$PROJECT/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1
SESSION=sim2real_stage7_multiaxis_v2e_20261031_v1
cd "$PROJECT"
test "$(pwd)" = "$PROJECT"
test -f "$RUN_ROOT/run_replicad_multiaxis_v2e_gpu_pilot.sh"
test -f "$RUN_ROOT/run_v2e_same_radiance_camera_motion.py"
for scene in v3_sc0_staging_00 v3_sc1_staging_00 v3_sc2_staging_00; do
  test -f "$RUN_ROOT/$scene/camera_motion_30ms/radiance_frames.npy"
done
test ! -e "$RUN_ROOT/v2e_started_utc.txt"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "refusing duplicate tmux session $SESSION" >&2; exit 6
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/v2e_launcher_gpu_check_utc.txt"
{
  nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
  uptime; free -h; df -h "$RUN_ROOT" /tmp
} > "$RUN_ROOT/v2e_launcher_gpu_check.txt" 2>&1
tmux new-session -d -s "$SESSION" "bash '$RUN_ROOT/run_replicad_multiaxis_v2e_gpu_pilot.sh'"
tmux has-session -t "$SESSION"
PANE_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
printf '%s\n' "$SESSION" > "$RUN_ROOT/v2e_tmux_session.txt"
printf '%s\n' "$PANE_PID" > "$RUN_ROOT/v2e_tmux_pane_pid.txt"
ps -o pid,ppid,etime,args -p "$PANE_PID" > "$RUN_ROOT/v2e_detach_verification.txt"
cat "$RUN_ROOT/v2e_detach_verification.txt"
printf 'tmux_session=%s pane_pid=%s\n' "$SESSION" "$PANE_PID"
