#!/usr/bin/env bash
set -u

WORKTREE="/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_6-overfit-3eada66"
PYTHON="/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python"
SOURCE="/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016"
TAG="$WORKTREE/benchmark/robot_closed_loop/results/runtime_logs/r0_6_event_state_overfit_20261017"
OUT="$WORKTREE/benchmark/robot_closed_loop/results/r0_6_event_state_overfit_20261017"

echo "$$" > "${TAG}.pid"
cd "$WORKTREE"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=4
export PYTHONPATH=.

/usr/bin/time -o "${TAG}.runtime" -f 'python_process_wall_s=%e\npython_process_user_cpu_s=%U\npython_process_system_cpu_s=%S\nmax_rss_kb=%M' \
  "$PYTHON" benchmark/robot_closed_loop/diagnostics/train_event_state_overfit.py \
  --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --source-results "$SOURCE" \
  --out "$OUT" \
  --episodes 0,4,8 \
  --epochs 120 \
  --batch-size 64 \
  --seed 606 \
  --device cpu > "${TAG}.log" 2>&1
rc=$?
echo "$rc" > "${TAG}.exit"
exit "$rc"
