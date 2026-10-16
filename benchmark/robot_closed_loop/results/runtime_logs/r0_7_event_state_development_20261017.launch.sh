#!/usr/bin/env bash
set -euo pipefail

REPO=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_7-learned
PYTHON=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
SOURCE=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016
OUT=$REPO/benchmark/robot_closed_loop/results/r0_7_event_state_development_20261017
NAME=r0_7_event_state_development_20261017
LOG_DIR=$REPO/benchmark/robot_closed_loop/results/runtime_logs
LOG=$LOG_DIR/$NAME.log
PID_FILE=$LOG_DIR/$NAME.pid
EXIT_FILE=$LOG_DIR/$NAME.exit
RUNTIME_FILE=$LOG_DIR/$NAME.runtime

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" > "$PID_FILE"
cd "$REPO"
set +e
/usr/bin/time -v -o "$RUNTIME_FILE" env CUDA_VISIBLE_DEVICES= PYTHONPATH=. \
  "$PYTHON" -m benchmark.robot_closed_loop.diagnostics.train_event_state_development \
  --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --source-results "$SOURCE" \
  --out "$OUT" \
  --train-episodes 0,2,3,4,7,8 \
  --validation-episodes 1,5,6 \
  --epochs 240 --patience 35 --batch-size 64 --seed 606 --threads 2 \
  > "$LOG" 2>&1
EXIT_CODE=$?
set -e
printf '%s\n' "$EXIT_CODE" > "$EXIT_FILE"
exit "$EXIT_CODE"
