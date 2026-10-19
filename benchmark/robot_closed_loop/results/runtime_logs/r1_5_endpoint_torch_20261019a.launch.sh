#!/usr/bin/env bash
set -u
cd /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-cadence-sweep-20261019d
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH=. nice -n 10 /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python benchmark/robot_closed_loop/diagnostics/verify_r1_5_endpoint_torch_candidate.py --scene-config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json --evidence-root /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-diag-20261019/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059 --output benchmark/robot_closed_loop/results/r1_5_endpoint_torch_candidate_20261019a_g1 > /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-cadence-sweep-20261019d/benchmark/robot_closed_loop/results/runtime_logs/r1_5_endpoint_torch_20261019a.log 2>&1 &
child=$!
echo $child > /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-cadence-sweep-20261019d/benchmark/robot_closed_loop/results/runtime_logs/r1_5_endpoint_torch_20261019a.pid
wait $child
rc=$?
echo $rc > /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-cadence-sweep-20261019d/benchmark/robot_closed_loop/results/runtime_logs/r1_5_endpoint_torch_20261019a.exit
exit $rc
