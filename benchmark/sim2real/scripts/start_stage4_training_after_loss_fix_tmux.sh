#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/zhaoyuhan/Eventbased_WAM
REPO="$PROJECT/repositories/gec-sim2real"
SIM="$REPO/benchmark/sim2real"
SESSION=stage4_flowdt1_seed0_lossfix
RUN_ID=${STAGE4_RUN_ID:-20261016-stage4-flowdt1-seed0-v4}
CONFIG_PATH=${STAGE4_CONFIG_PATH:-$SIM/configs/stage4_training_launch_v1_torch_cuda.json}
BANK_ROOT=${STAGE4_BANK_ROOT:-$PROJECT/data/sim2real/stage4/training_bank_v1_torch_cuda_32x2s}
CONFIG_CHECKSUM=${STAGE4_CONFIG_CHECKSUM:-training_launch_v1_torch_cuda.sha256}
OUTER_LOG="$PROJECT/logs/sim2real_stage4_flowdt1_seed0_lossfix_tmux.log"
mkdir -p "$(dirname "$OUTER_LOG")"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 2
fi
tmux new-session -d -s "$SESSION" -c "$REPO" \
  "env STAGE4_RUN_ID='$RUN_ID' STAGE4_CONFIG_PATH='$CONFIG_PATH' STAGE4_BANK_ROOT='$BANK_ROOT' STAGE4_CONFIG_CHECKSUM='$CONFIG_CHECKSUM' bash '$SIM/scripts/resume_stage4_after_loss_fix_server.sh'"
tmux pipe-pane -o -t "$SESSION" "cat >> '$OUTER_LOG'"
sleep 1
tmux has-session -t "$SESSION"
printf 'session=%s\n' "$SESSION"
printf 'pane_pid=%s\n' "$(tmux display-message -p -t "$SESSION" '#{pane_pid}')"
printf 'outer_log=%s\n' "$OUTER_LOG"
