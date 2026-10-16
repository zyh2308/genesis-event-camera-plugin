#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/zhaoyuhan/Eventbased_WAM
REPO="$PROJECT/repositories/gec-sim2real"
SIM="$REPO/benchmark/sim2real"
SESSION=stage4_flowdt1_seed0
OUTER_LOG="$PROJECT/logs/sim2real_stage4_flowdt1_seed0_tmux.log"
mkdir -p "$(dirname "$OUTER_LOG")"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 2
fi
tmux new-session -d -s "$SESSION" -c "$REPO" "bash '$SIM/scripts/run_stage4_campaign_server.sh'"
tmux pipe-pane -o -t "$SESSION" "cat >> '$OUTER_LOG'"
sleep 1
tmux has-session -t "$SESSION"
printf 'session=%s\n' "$SESSION"
printf 'pane_pid=%s\n' "$(tmux display-message -p -t "$SESSION" '#{pane_pid}')"
printf 'outer_log=%s\n' "$OUTER_LOG"
