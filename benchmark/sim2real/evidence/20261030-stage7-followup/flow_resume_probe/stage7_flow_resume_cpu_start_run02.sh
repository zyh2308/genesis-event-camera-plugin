#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/home/zhaoyuhan/Eventbased_WAM
REPO=$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
RUN_ROOT=$PROJECT/runs/sim2real/20261030-stage7-flow-resume-cpu-200to201-g1-run02
RUN_ID=20261030-stage7-flow-resume-cpu-200to201-g1-run02

if [ -e "$RUN_ROOT" ]; then
  echo "refusing to overwrite existing CPU probe run directory: $RUN_ROOT"
  exit 4
fi
if tmux has-session -t "$RUN_ID" 2>/dev/null; then
  echo "tmux session already exists: $RUN_ID"
  exit 5
fi
mkdir -p "$RUN_ROOT"
printf 'start_request_utc=%s\nrun_id=%s\nhost=%s\nresource=CPU-only, two threads\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ID" "$(hostname -f)" > "$RUN_ROOT/start_request.txt"
tmux new-session -d -s "$RUN_ID" \
  "bash $REPO/benchmark/sim2real/scripts/stage7_flow_resume_cpu_driver_run02.sh"
tmux has-session -t "$RUN_ID"
sleep 2
echo "tmux_session=$RUN_ID"
echo "driver_pid=$(cat "$RUN_ROOT/driver.pid" 2>/dev/null || echo starting)"
echo "run_state=$(cat "$RUN_ROOT/run_state.txt" 2>/dev/null || echo starting)"
if tmux has-session -t "$RUN_ID"; then tmux capture-pane -t "$RUN_ID" -p | tail -20; else echo "tmux_session=completed_or_exited"; fi
