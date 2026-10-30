#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN=$PROJECT/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1/flow_selfsup_diagnostic_v1
SESSION=sim2real_stage7_replicad_selfsup_20261031_v1
test ! -e "$RUN/session_name.txt" && test ! -e "$RUN/driver_started_utc.txt"
printf '%s\n' "$SESSION" > "$RUN/session_name.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/launcher_started_utc.txt"
tmux new-session -d -s "$SESSION" \
  "SIM2REAL_RUN_DIR='$RUN' SIM2REAL_RUN_ID=20261031-stage7-replicad-selfsup-diagnostic-v1 bash '$RUN/run_stage7_replicad_selfsup_diagnostic_g1.sh' >> '$RUN/driver.log' 2>&1"
sleep 1
tmux has-session -t "$SESSION"
tmux display-message -p -t "$SESSION" '#{pane_pid}' > "$RUN/tmux_pane_pid.txt"
kill -0 "$(cat "$RUN/tmux_pane_pid.txt")"
printf 'session=%s\npane_pid=%s\n' "$SESSION" "$(cat "$RUN/tmux_pane_pid.txt")"
cat "$RUN/launcher_started_utc.txt"
