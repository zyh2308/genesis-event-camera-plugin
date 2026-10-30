#!/usr/bin/env bash
set -euo pipefail
RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-replicad-v3_sc0_staging_00-fast-run14
TMP_ROOT=/tmp/sim2real-stage7-replicad-v3_sc0_staging_00-fast-run14
OUT_TMP=$TMP_ROOT/render_output
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python
ASSET_ROOT=/home/zhaoyuhan/Eventbased_WAM/external/sim2real_assets/replicacad_interactive_v1_5/extracted_v1_5
SCENE_CONFIG=v3_sc0_staging_00.scene_instance.json
CACHE=$TMP_ROOT/cache
GENESIS_ROOT=/home/zhaoyuhan/Eventbased_WAM/runtime/sim2real/genesis_hdr_v2
exec > >(tee -a "$RUN/driver.log") 2>&1
set +e
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_utc.txt"
timeout 900s env CUDA_VISIBLE_DEVICES= LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless QD_OFFLINE_CACHE=1 QD_OFFLINE_CACHE_FILE_PATH="$CACHE/quadrants" QD_OFFLINE_CACHE_CLEANING_POLICY=never QD_NUM_THREADS=2 GS_CACHE_FILE_PATH="$CACHE/genesis" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$GENESIS_ROOT:/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030:/home/zhaoyuhan/Eventbased_WAM" "$PY" -u "$RUN/replicad_layout_camera_motion_pilot.py" --asset-root "$ASSET_ROOT" --scene-config "$SCENE_CONFIG" --output-dir "$OUT_TMP" > "$RUN/stdout.txt" 2> "$RUN/stderr.txt"
RC=$?
printf '%s\n' "$RC" > "$RUN/exit_code.txt"
if [[ -d "$OUT_TMP/camera_motion_30ms" ]]; then
  mkdir -p "$RUN/render_output"
  cp -a "$OUT_TMP/camera_motion_30ms" "$RUN/render_output/"
  for f in result.json layout_rgb.png; do [[ ! -f "$OUT_TMP/$f" ]] || cp "$OUT_TMP/$f" "$RUN/render_output/$f"; done
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
printf 'python_timeout_exit_code=%s\n' "$RC"
exit "$RC"
