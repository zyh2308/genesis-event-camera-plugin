#!/usr/bin/env bash
set -euo pipefail

RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-genesis-radiance-cpu-probe-run02
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python
GENESIS_ROOT=/home/zhaoyuhan/Eventbased_WAM/runtime/sim2real/genesis_hdr_v2
CACHE=/tmp/sim2real-stage7-genesis-radiance-cpu-probe-run02

cd /home/zhaoyuhan/Eventbased_WAM
test -x "$PY"
test -f "$GENESIS_ROOT/genesis/vis/camera.py"
test ! -e "$RUN/started_utc.txt"
LOAD1=$(awk '{print $1}' /proc/loadavg)
NCPU=$(getconf _NPROCESSORS_ONLN)
MEM=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if ! awk -v l="$LOAD1" -v c="$NCPU" 'BEGIN {exit !(l < c * .8)}'; then
  echo "CPU gate refused: load=$LOAD1 cpus=$NCPU" >&2; exit 5
fi
if [[ "$MEM" -lt 16777216 ]]; then
  echo "memory gate refused: available_kb=$MEM" >&2; exit 5
fi
mkdir -p "$RUN" "$CACHE/quadrants" "$CACHE/genesis"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_utc.txt"
{
  printf 'host=%s\nproject=/home/zhaoyuhan/Eventbased_WAM\n' "$(hostname)"
  printf 'python=%s\nload1=%s\ncpu_count=%s\nmem_available_kb=%s\n' "$PY" "$LOAD1" "$NCPU" "$MEM"
  printf 'device_mode=CPU; CUDA_VISIBLE_DEVICES empty; Mesa software rasterizer requested\n'
  printf 'genesis_overlay=%s\n' "$GENESIS_ROOT"
  sha256sum "$GENESIS_ROOT/genesis/vis/camera.py"
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  sha256sum "$RUN/probe_genesis_radiance_cpu.py" "$PY"
} > "$RUN/environment.txt" 2>&1
set +e
timeout 180s env \
  CUDA_VISIBLE_DEVICES= \
  LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
  PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  QD_OFFLINE_CACHE=1 QD_OFFLINE_CACHE_FILE_PATH="$CACHE/quadrants" \
  QD_OFFLINE_CACHE_CLEANING_POLICY=never QD_NUM_THREADS=2 \
  GS_CACHE_FILE_PATH="$CACHE/genesis" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$GENESIS_ROOT:/home/zhaoyuhan/Eventbased_WAM" \
  "$PY" -u "$RUN/probe_genesis_radiance_cpu.py" --output-dir "$RUN/render_output" \
  > "$RUN/stdout.txt" 2> "$RUN/stderr.txt"
RC=$?
set -e
printf '%s\n' "$RC" > "$RUN/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
printf 'python_exit_code=%s\n' "$RC"
cat "$RUN/stdout.txt"
exit "$RC"
