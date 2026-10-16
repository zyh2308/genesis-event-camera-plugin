#!/usr/bin/env bash
set -u

WORKTREE="$HOME/Eventbased_WAM/repositories/gec-robot-r0_5-3c60932"
PYTHON="$HOME/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python"
TAG="$WORKTREE/benchmark/robot_closed_loop/results/runtime_logs/r0_5_oracle_strict100x3_20261016"
OUT="$WORKTREE/benchmark/robot_closed_loop/results/r0_5_oracle_strict100x3_20261016"

echo "$$" > "${TAG}.pid"
cd "$WORKTREE"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=.

/usr/bin/time -o "${TAG}.runtime" -f 'python_process_wall_s=%e\npython_process_user_cpu_s=%U\npython_process_system_cpu_s=%S\nmax_rss_kb=%M' \
  "$PYTHON" -m benchmark.robot_closed_loop.run \
  --mode oracle \
  --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --episodes 100 \
  --speeds 0.3,0.4,0.5 \
  --a-eff 0.09803606718731761 \
  --seed 2026103300 \
  --out "$OUT" > "${TAG}.log" 2>&1
rc=$?
echo "$rc" > "${TAG}.exit"
exit "$rc"
