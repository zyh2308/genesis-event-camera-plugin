#!/usr/bin/env bash
set -u
root=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot
run_id=r0_oracle_holdout100x3_20261016
out=benchmark/robot_closed_loop/results/${run_id}
log=benchmark/robot_closed_loop/results/runtime_logs/${run_id}.log
pid_file=benchmark/robot_closed_loop/results/runtime_logs/${run_id}.pid
exit_file=benchmark/robot_closed_loop/results/runtime_logs/${run_id}.exit
cd "$root" || exit 2
echo "$$" > "$pid_file"
set +e
CUDA_VISIBLE_DEVICES=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run \
  --mode oracle --episodes 100 --speeds 0.3,0.4,0.5 \
  --a-eff 0.09804843035177806 --seed 2026102800 \
  --out "$out" > "$log" 2>&1
rc=$?
echo "$rc" > "$exit_file"
exit "$rc"
