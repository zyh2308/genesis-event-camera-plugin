#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT=$PROJECT/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1
SESSION=sim2real_stage7_replicad_multiaxis_20261031_v1
cd "$PROJECT"
test "$(pwd)" = "$PROJECT"
test -x "$PROJECT/../miniconda3/envs/Eventbased_WAM_torch28/bin/python"
test -f "$RUN_ROOT/run_replicad_multiaxis_cpu_pilot.sh"
test -f "$RUN_ROOT/replicad_layout_camera_motion_pilot.py"
test -f "$RUN_ROOT/multiaxis_paths_registration_v1.json"
test ! -e "$RUN_ROOT/started_utc.txt"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "refusing duplicate tmux session $SESSION" >&2; exit 6
fi
tmux new-session -d -s "$SESSION" "bash '$RUN_ROOT/run_replicad_multiaxis_cpu_pilot.sh'"
tmux has-session -t "$SESSION"
PANE_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
printf '%s\n' "$SESSION" > "$RUN_ROOT/tmux_session.txt"
printf '%s\n' "$PANE_PID" > "$RUN_ROOT/tmux_pane_pid.txt"
ps -o pid,ppid,etime,args -p "$PANE_PID" > "$RUN_ROOT/detach_verification.txt"
cat "$RUN_ROOT/detach_verification.txt"
printf 'tmux_session=%s pane_pid=%s\n' "$SESSION" "$PANE_PID"
