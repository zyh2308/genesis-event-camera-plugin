#!/usr/bin/env bash
set -u
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$script_dir/../../../.." && pwd)"
cd "$repo"
logs=$repo/benchmark/robot_closed_loop/results/runtime_logs
name=r0_8_five_bin_sampling_20261017b
echo "launcher_pid=$$" > "$logs/$name.pid"
/usr/bin/time -v -o "$logs/$name.runtime" env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python -m benchmark.robot_closed_loop.diagnostics.compare_five_bin_sampling_decisions --rollout-root benchmark/robot_closed_loop/results/r0_7_heatmap_event_state_closed_loop_3seed_20261017 --checkpoint benchmark/robot_closed_loop/results/r0_7_heatmap_event_state_development_20261017/spatial_event_state.pt --scene-config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json --control-config benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json --output benchmark/robot_closed_loop/results/r0_8_five_bin_sampling_20261017b > "$logs/$name.log" 2>&1
status=$?
echo "$status" > "$logs/$name.exit"
exit "$status"
