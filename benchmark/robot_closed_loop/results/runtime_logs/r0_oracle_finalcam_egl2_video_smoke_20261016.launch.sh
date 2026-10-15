#!/usr/bin/env bash
set +e
run_id=r0_oracle_finalcam_egl2_video_smoke_20261016
log="benchmark/robot_closed_loop/results/runtime_logs/${run_id}.log"
pid_file="benchmark/robot_closed_loop/results/runtime_logs/${run_id}.pid"
exit_file="benchmark/robot_closed_loop/results/runtime_logs/${run_id}.exit"
echo $$ > "$pid_file"
env CUDA_VISIBLE_DEVICES=2 EGL_DEVICE_ID=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. \
  /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python \
  -m benchmark.robot_closed_loop.run --mode oracle --episodes 1 --speeds 0.4 \
  --a-eff 0.09804843035177806 --seed 2026102920 \
  --out benchmark/robot_closed_loop/results/r0_oracle_finalcam_egl2_video_smoke_20261016 \
  > "$log" 2>&1
rc=$?
echo "$rc" > "$exit_file"
exit "$rc"
