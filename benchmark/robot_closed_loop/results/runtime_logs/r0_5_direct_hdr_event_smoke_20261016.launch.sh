#!/usr/bin/env bash
set -u

WORKTREE="$HOME/Eventbased_WAM/repositories/gec-robot-r0_5-02b1afc"
PYTHON="$HOME/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python"
TAG="$WORKTREE/benchmark/robot_closed_loop/results/runtime_logs/r0_5_direct_hdr_event_smoke_20261016"
OUT="$WORKTREE/benchmark/robot_closed_loop/results/r0_5_direct_hdr_event_smoke_20261016"

echo "$$" > "${TAG}.pid"
cd "$WORKTREE"
export EGL_DEVICE_ID=2
export CUDA_VISIBLE_DEVICES=2
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=.

/usr/bin/time -o "${TAG}.runtime" -f 'python_process_wall_s=%e\npython_process_user_cpu_s=%U\npython_process_system_cpu_s=%S\nmax_rss_kb=%M' \
  "$PYTHON" -m benchmark.robot_closed_loop.run \
  --mode reference \
  --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --speeds 0.3,0.4,0.5 \
  --reference-episodes 3 \
  --reference-y-values=-0.055,0,0.055,-0.055,0,0.055,-0.055,0,0.055 \
  --event-controller-hz 2000 \
  --visibility-hz 50 \
  --seed 2026103400 \
  --out "$OUT" > "${TAG}.log" 2>&1
rc=$?
echo "$rc" > "${TAG}.exit"
exit "$rc"
