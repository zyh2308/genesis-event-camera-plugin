#!/usr/bin/env bash
set -u
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$script_dir/../../../.." && pwd)"
cd "$repo"
logs=$repo/benchmark/robot_closed_loop/results/runtime_logs
name=r0_8_proprio_monitor60_negative_regression_20261017a
echo "launcher_pid=$$" > "$logs/$name.pid"
/usr/bin/time -v -o "$logs/$name.runtime" env EGL_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3 PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python -m benchmark.robot_closed_loop.diagnostics.run_proprioceptive_grip_gate_regression --scene-config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json --control-config benchmark/robot_closed_loop/configs/shared_control_r0_8_monitor60_diagnostic.json --output benchmark/robot_closed_loop/results/r0_8_proprio_monitor60_negative_regression_20261017a --negative-duration 0.35 --seed 2026104350 > "$logs/$name.log" 2>&1
status=$?
echo "$status" > "$logs/$name.exit"
exit "$status"
