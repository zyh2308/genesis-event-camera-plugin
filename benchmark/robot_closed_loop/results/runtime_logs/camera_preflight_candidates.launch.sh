#!/usr/bin/env bash
set -u
root=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-preflight
python=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
variant=${1:?pass one camera candidate name}
run_id=camera_preflight_${variant}_20261016
out=benchmark/robot_closed_loop/results/${run_id}
log=benchmark/robot_closed_loop/results/runtime_logs/${run_id}.log
pid_file=benchmark/robot_closed_loop/results/runtime_logs/${run_id}.pid
exit_file=benchmark/robot_closed_loop/results/runtime_logs/${run_id}.exit
cd "$root" || exit 2
echo "$$" > "$pid_file"
set +e
CUDA_VISIBLE_DEVICES=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. "$python" -m benchmark.robot_closed_loop.run \
  --mode preflight --episodes 1 --config benchmark/robot_closed_loop/configs/${variant}.json \
  --out "$out" > "$log" 2>&1
rc=$?
echo "$rc" > "$exit_file"
exit "$rc"
