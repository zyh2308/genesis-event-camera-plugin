#!/usr/bin/env bash
set -u

WORKTREE="$HOME/Eventbased_WAM/repositories/gec-robot-r0_5-6cd7c70"
PYTHON="$HOME/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python"
TAG="$WORKTREE/benchmark/robot_closed_loop/results/runtime_logs/r0_5_oracle_pilot3x3_20261016"
OUT="$WORKTREE/benchmark/robot_closed_loop/results/r0_5_oracle_pilot3x3_20261016"

echo "$$" > "${TAG}.pid"
cd "$WORKTREE"
export EGL_DEVICE_ID=2
export CUDA_VISIBLE_DEVICES=2
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=.

"$PYTHON" -m benchmark.robot_closed_loop.run \
  --mode oracle \
  --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --episodes 3 \
  --speeds 0.3,0.4,0.5 \
  --a-eff 0.09803606718731761 \
  --seed 2026103200 \
  --out "$OUT" > "${TAG}.log" 2>&1
rc=$?
echo "$rc" > "${TAG}.exit"
exit "$rc"
