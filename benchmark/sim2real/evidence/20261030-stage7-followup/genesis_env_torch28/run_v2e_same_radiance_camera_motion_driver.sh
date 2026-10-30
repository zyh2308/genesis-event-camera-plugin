#!/usr/bin/env bash
set -euo pipefail

RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-apt1-v2e-camera-motion-cpu-run08
PROJECT=/home/zhaoyuhan/Eventbased_WAM
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
SESSION=sim2real_apt1_v2e_cpu_20261030_08
INPUT_RAD=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-replicad-apt1-camera-motion-run07/render_output/camera_motion_30ms/radiance_frames.npy
CONVERSION_SOURCE=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030/benchmark/sim2real/scripts/generate_stage5_flow_coverage_bank.py
V2E_SOURCE=$PROJECT/external/sources/v2e-cac6384445/v2e-pinned
V2E_MODEL=$PROJECT/external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt
exec > >(tee -a "$RUN/driver.log") 2>&1
set +e
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_utc.txt"
timeout 900s env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  "$PY" -u "$RUN/run_v2e_same_radiance_camera_motion.py" \
  --radiance-frames "$INPUT_RAD" --conversion-source "$CONVERSION_SOURCE" \
  --project-root "$PROJECT" --output-root "$RUN/artifacts"
RC=$?
printf '%s\n' "$RC" > "$RUN/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
echo v2e_driver_exit_code=$RC
exit "$RC"
