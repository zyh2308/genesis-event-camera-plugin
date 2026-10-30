#!/usr/bin/env bash
set -euo pipefail

RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-replicad-apt1-camera-motion-run06
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python
ASSET_ROOT=/home/zhaoyuhan/Eventbased_WAM/external/sim2real_assets/replicacad_interactive_v1_5/extracted_v1_5
CACHE=/tmp/sim2real-stage7-replicad-apt1-camera-motion-run06
GENESIS_ROOT=/home/zhaoyuhan/Eventbased_WAM/runtime/sim2real/genesis_hdr_v2

exec > >(tee -a "$RUN/driver.log") 2>&1
set +e
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_utc.txt"
timeout 900s env \
  CUDA_VISIBLE_DEVICES= \
  LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
  PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  QD_OFFLINE_CACHE=1 QD_OFFLINE_CACHE_FILE_PATH="$CACHE/quadrants" \
  QD_OFFLINE_CACHE_CLEANING_POLICY=never QD_NUM_THREADS=2 \
  GS_CACHE_FILE_PATH="$CACHE/genesis" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$GENESIS_ROOT:/home/zhaoyuhan/Eventbased_WAM" \
  "$PY" -u "$RUN/replicad_apt1_camera_motion_pilot.py" \
    --asset-root "$ASSET_ROOT" --output-dir "$RUN/render_output" \
  > "$RUN/stdout.txt" 2> "$RUN/stderr.txt"
RC=$?
printf '%s\n' "$RC" > "$RUN/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
printf 'python_timeout_exit_code=%s\n' "$RC"
exit "$RC"
