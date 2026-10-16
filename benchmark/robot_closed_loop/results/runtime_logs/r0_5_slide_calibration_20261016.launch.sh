#!/usr/bin/env bash
set -u

WORKTREE="$HOME/Eventbased_WAM/repositories/gec-robot-r0_5-b757a83"
PYTHON="$HOME/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python"
TAG="$WORKTREE/benchmark/robot_closed_loop/results/runtime_logs/r0_5_slide_calibration_20261016"
OUT="$WORKTREE/benchmark/robot_closed_loop/results/r0_5_slide_calibration_20261016"

echo "$$" > "${TAG}.pid"
cd "$WORKTREE"
export CUDA_VISIBLE_DEVICES=2
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=.

"$PYTHON" -m benchmark.robot_closed_loop.run \
  --mode slide \
  --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --split calibration \
  --episodes 20 \
  --speeds 0.3,0.4,0.5 \
  --seed 2026103100 \
  --out "$OUT" > "${TAG}.log" 2>&1
rc=$?
echo "$rc" > "${TAG}.exit"
exit "$rc"
