#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT=$PROJECT/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1
RETRY_ROOT=$RUN_ROOT/v2e_retry01
SESSION=sim2real_stage7_gtfit_retry02_checkpoint_audit_20261031
test "$(cat "$RETRY_ROOT/gtfit_retry02_exit_code.txt")" = 1
test -f "$RETRY_ROOT/run_stage7_gtfit_retry02_checkpoint_audit_gpu.sh"
test -f "$RETRY_ROOT/finalize_stage7_gtfit_retry02.py"
test ! -e "$RETRY_ROOT/checkpoint_audit_started_utc.txt"
test ! -e "$RUN_ROOT/flow_gtfit_retry02_final_audit"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "refusing duplicate tmux session $SESSION" >&2
  exit 6
fi
tmux new-session -d -s "$SESSION" "bash '$RETRY_ROOT/run_stage7_gtfit_retry02_checkpoint_audit_gpu.sh' > '$RETRY_ROOT/checkpoint_audit_driver.log' 2>&1"
tmux has-session -t "$SESSION"
PANE_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
printf '%s\n' "$SESSION" > "$RETRY_ROOT/checkpoint_audit_tmux_session.txt"
printf '%s\n' "$PANE_PID" > "$RETRY_ROOT/checkpoint_audit_tmux_pane_pid.txt"
ps -o pid,ppid,etime,args -p "$PANE_PID" > "$RETRY_ROOT/checkpoint_audit_detach_verification.txt"
cat "$RETRY_ROOT/checkpoint_audit_detach_verification.txt"
