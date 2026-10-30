#!/usr/bin/env bash
set -euo pipefail

REPO=/home/科研/Eventbased_WAM/repositories/gec-sim2real
SSH_EXE=/mnt/c/Windows/System32/OpenSSH/ssh.exe
HOST=zhaoyuhan@g1.pkuml.org
PORT=22
REMOTE_RUN=${REMOTE_RUN:-/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-genesis-cpu-render-probe-run02}

cd "$REPO"
[[ -x "$SSH_EXE" ]]

PAYLOAD_DIR=$(mktemp -d)
trap 'rm -rf "$PAYLOAD_DIR"' EXIT
mkdir -p "$PAYLOAD_DIR"
cp benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28/probe_genesis_cpu_render.py "$PAYLOAD_DIR/"
tar -czf - -C "$PAYLOAD_DIR" . | \
  "$SSH_EXE" -o ConnectTimeout=10 -p "$PORT" "$HOST" \
  "set -euo pipefail
   RUN='$REMOTE_RUN'
   PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python
   test ! -e \"\$RUN\"
   mkdir -p \"\$RUN\"
   tar -xzf - -C \"\$RUN\"
   LOAD1=\$(awk '{print \$1}' /proc/loadavg)
   NCPU=\$(getconf _NPROCESSORS_ONLN)
   MEM=\$(awk '/MemAvailable:/ {print \$2}' /proc/meminfo)
   if ! awk -v l=\"\$LOAD1\" -v c=\"\$NCPU\" 'BEGIN {exit !(l < c * .8)}'; then echo \"CPU gate refused: load=\$LOAD1 cpus=\$NCPU\" >&2; exit 5; fi
   if [[ \"\$MEM\" -lt 8388608 ]]; then echo \"memory gate refused: available_kb=\$MEM\" >&2; exit 5; fi
   date -u +%Y-%m-%dT%H:%M:%SZ > \"\$RUN/started_utc.txt\"
   { printf 'host=%s\\nproject=/home/zhaoyuhan/Eventbased_WAM\\npython=%s\\ncuda_visible_devices=empty\\nlibgl_always_software=1\\nmesa_driver=llvmpipe\\nthreads=1\\nload1=%s\\ncpu_count=%s\\nmem_available_kb=%s\\n' \"\$(hostname)\" \"\$PY\" \"\$LOAD1\" \"\$NCPU\" \"\$MEM\"; nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } > \"\$RUN/environment.txt\" 2>&1
   mkdir -p /tmp/sim2real-stage7-cpu-render-probe-run02/quadrants /tmp/sim2real-stage7-cpu-render-probe-run02/genesis
   sha256sum \"\$RUN/probe_genesis_cpu_render.py\" \"\$PY\" > \"\$RUN/input_hashes.txt\"
   set +e
   timeout 180s env CUDA_VISIBLE_DEVICES= LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless QD_OFFLINE_CACHE=1 QD_OFFLINE_CACHE_FILE_PATH=/tmp/sim2real-stage7-cpu-render-probe-run02/quadrants QD_OFFLINE_CACHE_CLEANING_POLICY=never GS_CACHE_FILE_PATH=/tmp/sim2real-stage7-cpu-render-probe-run02/genesis OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \"\$PY\" -u \"\$RUN/probe_genesis_cpu_render.py\" --output-dir \"\$RUN/render_output\" > \"\$RUN/stdout.txt\" 2> \"\$RUN/stderr.txt\"
   RC=\$?
   printf '%s\\n' \"\$RC\" > \"\$RUN/exit_code.txt\"
   date -u +%Y-%m-%dT%H:%M:%SZ > \"\$RUN/finished_utc.txt\"
   nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader > \"\$RUN/gpu_after.csv\" 2>&1
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > \"\$RUN/processes_after.csv\" 2>&1
   cat \"\$RUN/stdout.txt\"
   if [[ -s \"\$RUN/stderr.txt\" ]]; then cat \"\$RUN/stderr.txt\" >&2; fi
   printf 'exit_code=%s\\n' \"\$RC\"
   exit \"\$RC\""
