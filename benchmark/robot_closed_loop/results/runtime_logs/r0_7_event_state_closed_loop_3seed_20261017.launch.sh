#!/usr/bin/env bash
set -euo pipefail

REPO=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_7-learned
PYTHON=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
CHECKPOINT=$REPO/benchmark/robot_closed_loop/results/r0_7_event_state_development_20261017/spatial_event_state.pt
OUT=$REPO/benchmark/robot_closed_loop/results/r0_7_event_state_closed_loop_3seed_20261017
NAME=r0_7_event_state_closed_loop_3seed_20261017
LOG_DIR=$REPO/benchmark/robot_closed_loop/results/runtime_logs
LOG=$LOG_DIR/$NAME.log
PID_FILE=$LOG_DIR/$NAME.pid
EXIT_FILE=$LOG_DIR/$NAME.exit
RUNTIME_FILE=$LOG_DIR/$NAME.runtime

mkdir -p "$LOG_DIR"
rm -f "$EXIT_FILE" "$RUNTIME_FILE"
printf '%s\n' "$$" > "$PID_FILE"
cd "$REPO"
set +e
/usr/bin/time -v -o "$RUNTIME_FILE" env EGL_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3 \
  PYOPENGL_PLATFORM=egl PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$PYTHON" -m benchmark.robot_closed_loop.run \
  --mode reference \
  --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --speeds 0.3,0.4,0.5 --reference-episodes 1 \
  --reference-y-values=0,0.055,-0.055 \
  --event-controller-hz 2000 --reference-controller-source events \
  --event-model-checkpoint "$CHECKPOINT" --visibility-hz 50 \
  --seed 2026103800 --out "$OUT" \
  > "$LOG" 2>&1
EXIT_CODE=$?
set -e
printf '%s\n' "$EXIT_CODE" > "$EXIT_FILE"
exit "$EXIT_CODE"
