#!/usr/bin/env bash
set -u
repo=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_8-lift-probe-20261017
cd "$repo"
logs=$repo/benchmark/robot_closed_loop/results/runtime_logs
name=r0_8_shared_gate_oracle_3speed3lateral_20261017a
echo "launcher_pid=$$" > "$logs/$name.pid"
/usr/bin/time -v -o "$logs/$name.runtime" env EGL_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3 PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode oracle --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json --control-config benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json --episodes 3 --speeds 0.3,0.4,0.5 --initial-y-values=-0.055,0.0,0.055,-0.055,0.0,0.055,-0.055,0.0,0.055 --a-eff 0.09803606718731761 --seed 2026104300 --out benchmark/robot_closed_loop/results/r0_8_shared_gate_oracle_3speed3lateral_20261017a > "$logs/$name.log" 2>&1
status=$?
echo "$status" > "$logs/$name.exit"
exit "$status"
