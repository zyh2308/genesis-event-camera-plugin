#!/usr/bin/env bash
set -u
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$script_dir/../../../.." && pwd)"
cd "$repo"
logs=$repo/benchmark/robot_closed_loop/results/runtime_logs
name=r0_8_paired_r2_ours_20261017a
checkpoint=$repo/benchmark/robot_closed_loop/results/r0_8_paired_pipeline_smoke_20261017c/ours_fit/spatial_event_state.pt
echo "launcher_pid=$$" > "$logs/$name.pid"
/usr/bin/time -v -o "$logs/$name.runtime" env EGL_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3 PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode reference --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json --control-config benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json --speeds 0.4 --reference-episodes 1 --reference-y-values=0.0 --event-controller-hz 2000 --reference-controller-source events --event-model-checkpoint "$checkpoint" --visibility-hz 50 --seed 2026104400 --out benchmark/robot_closed_loop/results/r0_8_paired_r2_ours_20261017a > "$logs/$name.log" 2>&1
status=$?
echo "$status" > "$logs/$name.exit"
exit "$status"
