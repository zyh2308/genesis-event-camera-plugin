#!/usr/bin/env bash
set -euo pipefail
CHECKOUT=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-recon-long-state-cpu-run03
DRIVER=$CHECKOUT/benchmark/sim2real/evidence/20261030-stage7-followup/reconstruction_long_state/driver.sh
SESSION=sim2real_recon_longstate_20261030_03
if [[ -e "$RUN" ]]; then
  echo "refusing to overwrite existing run: $RUN" >&2
  exit 3
fi
if [[ ! -f "$DRIVER" ]]; then
  echo "driver missing: $DRIVER" >&2
  exit 4
fi
if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "$SESSION" "bash '$DRIVER'"
  tmux list-sessions | grep -F "$SESSION"
  printf 'tmux_session=%s\n' "$SESSION"
else
  nohup bash "$DRIVER" > "$RUN/launcher.log" 2>&1 < /dev/null &
  PID=$!
  printf '%s\n' "$PID" > "$RUN/launcher_pid.txt"
  sleep 1
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "detached process exited immediately; inspect $RUN/launcher.log" >&2
    exit 5
  fi
  ps -p "$PID" -o pid=,user=,lstart=,args=
fi
