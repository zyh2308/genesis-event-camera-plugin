#!/usr/bin/env bash
set -euo pipefail
CHECKOUT=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-genesis-torch28-setup-run01
DRIVER=$CHECKOUT/benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28/driver.sh
SESSION=sim2real_genesis_torch28_setup_20261030_01
if [[ -e "$RUN" ]]; then
  echo "refusing to overwrite existing setup run: $RUN" >&2
  exit 3
fi
if [[ ! -f "$DRIVER" ]]; then
  echo "environment setup driver is missing: $DRIVER" >&2
  exit 4
fi
tmux new-session -d -s "$SESSION" "bash '$DRIVER'"
tmux list-sessions | grep -F "$SESSION"
printf 'tmux_session=%s\n' "$SESSION"
