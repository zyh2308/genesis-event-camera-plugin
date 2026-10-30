#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT=$PROJECT/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1
RETRY_ROOT=$RUN_ROOT/v2e_retry01
SESSION=sim2real_stage7_replicad_gtfit_retry01_20261031
test "$(cat "$RETRY_ROOT/bank_builder_exit_code.txt")" = 0
test -f "$RUN_ROOT/flow_bank_retry01/training_bank_manifest.json"
test -f "$RUN_ROOT/run_stage7_replicad_gtfit.py"
test -f "$RUN_ROOT/stage7_replicad_multiaxis_gtfit_v1.json"
test -f "$RETRY_ROOT/run_stage7_replicad_gtfit_gpu_retry01.sh"
test ! -e "$RETRY_ROOT/gtfit_started_utc.txt"
test ! -e "$RUN_ROOT/flow_gtfit_retry01"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "refusing duplicate tmux session $SESSION" >&2; exit 6
fi
tmux new-session -d -s "$SESSION" "bash '$RETRY_ROOT/run_stage7_replicad_gtfit_gpu_retry01.sh' > '$RETRY_ROOT/gtfit_driver.log' 2>&1"
tmux has-session -t "$SESSION"
PANE_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
printf '%s\n' "$SESSION" > "$RETRY_ROOT/gtfit_tmux_session.txt"
printf '%s\n' "$PANE_PID" > "$RETRY_ROOT/gtfit_tmux_pane_pid.txt"
ps -o pid,ppid,etime,args -p "$PANE_PID" > "$RETRY_ROOT/gtfit_detach_verification.txt"
cat "$RETRY_ROOT/gtfit_detach_verification.txt"
