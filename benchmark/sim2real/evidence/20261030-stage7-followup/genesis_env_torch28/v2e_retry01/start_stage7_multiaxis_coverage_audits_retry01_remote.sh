#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT=$PROJECT/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1
RETRY_ROOT=$RUN_ROOT/v2e_retry01
SESSION=sim2real_stage7_coverage_audit_retry01_20261031
test "$(cat "$RETRY_ROOT/exit_code.txt")" = 0
for scene in v3_sc0_staging_00 v3_sc1_staging_00 v3_sc2_staging_00; do
  test "$(cat "$RUN_ROOT/$scene/exit_code.txt")" = 0
  test "$(cat "$RETRY_ROOT/$scene/v2e_gpu/exit_code.txt")" = 0
done
test -f "$RETRY_ROOT/run_stage7_multiaxis_coverage_audits_retry01.sh"
test ! -e "$RUN_ROOT/coverage_audits_retry01"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "refusing duplicate tmux session $SESSION" >&2
  exit 6
fi
tmux new-session -d -s "$SESSION" "bash '$RETRY_ROOT/run_stage7_multiaxis_coverage_audits_retry01.sh' > '$RETRY_ROOT/coverage_audit_driver.log' 2>&1"
tmux has-session -t "$SESSION"
PANE_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
printf '%s\n' "$SESSION" > "$RETRY_ROOT/coverage_audit_tmux_session.txt"
printf '%s\n' "$PANE_PID" > "$RETRY_ROOT/coverage_audit_tmux_pane_pid.txt"
ps -o pid,ppid,etime,args -p "$PANE_PID" > "$RETRY_ROOT/coverage_audit_detach_verification.txt"
cat "$RETRY_ROOT/coverage_audit_detach_verification.txt"
