#!/usr/bin/env bash
set -u
repo=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_8-lift-probe-20261017
cd "$repo"
logs=$repo/benchmark/robot_closed_loop/results/runtime_logs
name=r0_8_paired_pipeline_smoke_20261017c
echo "launcher_pid=$$" > "$logs/$name.pid"
/usr/bin/time -v -o "$logs/$name.runtime" env EGL_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3 PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python -m benchmark.robot_closed_loop.diagnostics.run_paired_generator_smoke --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json --control-config benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json --output benchmark/robot_closed_loop/results/r0_8_paired_pipeline_smoke_20261017c --speed 0.4 --y-values -0.055 0.055 --seed 2026104130 --v2e-source /home/zhaoyuhan/Eventbased_WAM/code/v2e_source --slomo-model /home/zhaoyuhan/Eventbased_WAM/code/v2e_source/input/SuperSloMo39.ckpt --python /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python --train-epochs 3 > "$logs/$name.log" 2>&1
status=$?
echo "$status" > "$logs/$name.exit"
exit "$status"
