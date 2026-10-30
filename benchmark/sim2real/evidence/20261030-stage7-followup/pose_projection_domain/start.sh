#!/usr/bin/env bash
set -u

CHECKOUT=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-epose-projection-domain-run02
PYTHON=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
SCRIPT=$CHECKOUT/benchmark/sim2real/scripts/audit_epose_projection_domains.py
MAT=/home/zhaoyuhan/Eventbased_WAM/data/real/epose/figshare_28106717_v1/extracted/block/block/r_01_gl/0001_data.mat
SOURCE=/home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/E_Pose
PREVIOUS_AUDIT=$CHECKOUT/benchmark/sim2real/evidence/20261024-execution-refresh/epose_block_r01gl_semantic_audit_v1.json

if [[ -e "$RUN" ]]; then
  echo "refusing to overwrite existing run: $RUN" >&2
  exit 3
fi
if [[ ! -x "$PYTHON" || ! -f "$SCRIPT" || ! -f "$MAT" || ! -f "$SOURCE/CreateSaveDataset.m" || ! -f "$PREVIOUS_AUDIT" ]]; then
  echo "one or more registered sample/source paths are missing" >&2
  exit 4
fi
LOAD1=$(awk '{print $1}' /proc/loadavg)
NCPU=$(getconf _NPROCESSORS_ONLN)
MEM_AVAILABLE_KB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if ! awk -v l1="$LOAD1" -v cpus="$NCPU" 'BEGIN { exit !(l1 < cpus * 0.8) }'; then
  echo "CPU load gate refused bounded MAT projection audit: load=$LOAD1 cpus=$NCPU" >&2
  exit 5
fi
if [[ "$MEM_AVAILABLE_KB" -lt 8388608 ]]; then
  echo "memory gate refused audit: available_kb=$MEM_AVAILABLE_KB" >&2
  exit 5
fi
mkdir -p "$RUN"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_utc.txt"
git -C "$CHECKOUT" rev-parse HEAD > "$RUN/checkout_commit.txt"
sha256sum "$SCRIPT" > "$RUN/audit_script.sha256"
{
  "$PYTHON" --version
  printf 'device=cpu\nthreads=1\ncpu_load1=%s\ncpu_count=%s\nmem_available_kb=%s\n' "$LOAD1" "$NCPU" "$MEM_AVAILABLE_KB"
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  sha256sum "$MAT" "$SOURCE/CreateSaveDataset.m" "$PREVIOUS_AUDIT"
} > "$RUN/environment_and_inputs.txt" 2>&1
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
set +e
"$PYTHON" -u "$SCRIPT" --mat "$MAT" --epose-source-root "$SOURCE" \
  --previous-audit "$PREVIOUS_AUDIT" --output "$RUN/corrected_projection_sidecar.json" \
  > "$RUN/driver.log" 2>&1
RC=$?
printf '%s\n' "$RC" > "$RUN/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
cat "$RUN/driver.log"
exit "$RC"
