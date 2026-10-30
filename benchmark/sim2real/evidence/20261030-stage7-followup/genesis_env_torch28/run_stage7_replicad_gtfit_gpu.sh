#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT=$PROJECT/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1
EXEC=$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
MINI_REPO=$RUN_ROOT/source_v2/mini_repo
PY=$PROJECT/../miniconda3/envs/Eventbased_WAM_torch28/bin/python
MODEL_ROOT=$PROJECT/external/sources/downstream/STE-FlowNet
SCRIPT=$RUN_ROOT/run_stage7_replicad_gtfit.py
CONFIG=$RUN_ROOT/stage7_replicad_multiaxis_gtfit_v1.json
DATA=$RUN_ROOT/flow_bank_v1
OUT=$RUN_ROOT/flow_gtfit_v1
RESERVATIONS=$PROJECT/runs/sim2real/resource_reservations
SESSION_ID=20261031-stage7-replicad-multiaxis-gtfit-v1

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/gtfit_preflight_utc.txt"
test "$(realpath "$PROJECT")" = /home/zhaoyuhan/Eventbased_WAM
test -x "$PY" && test -d "$MODEL_ROOT" && test -f "$SCRIPT" && test -f "$CONFIG"
test -f "$DATA/training_bank_manifest.json"
test ! -e "$OUT"
LOAD1=$(awk '{print $1}' /proc/loadavg)
NCPU=$(getconf _NPROCESSORS_ONLN)
MEM=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if ! awk -v l="$LOAD1" -v c="$NCPU" 'BEGIN {exit !(l < c * .8)}'; then
  echo "CPU gate refused: load=$LOAD1 cpus=$NCPU" >&2
  printf '5\n' > "$RUN_ROOT/gtfit_preflight_exit_code.txt"; exit 5
fi
if [[ "$MEM" -lt 16777216 ]]; then
  echo "memory gate refused: available_kb=$MEM" >&2
  printf '5\n' > "$RUN_ROOT/gtfit_preflight_exit_code.txt"; exit 5
fi
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>&1 || true)
mkdir -p "$RESERVATIONS"
GPU_INDEX=""; GPU_UUID=""; LOCK=""
while IFS=',' read -r idx _name uuid util used _total; do
  idx=${idx//[[:space:]]/}; uuid=${uuid//[[:space:]]/}
  util=${util//[[:space:]]/}; util=${util%%%}
  used=${used//[[:space:]]/}
  if [[ "$util" != 0 || ! "$used" =~ ^[0-9]+$ || "$used" -gt 64 ]] || grep -Fq "$uuid" <<< "$APPS"; then
    continue
  fi
  candidate="$RESERVATIONS/$uuid"
  if mkdir "$candidate" 2>/dev/null; then
    GPU_INDEX=$idx; GPU_UUID=$uuid; LOCK=$candidate
    break
  fi
done < <(nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
if [[ -z "$GPU_INDEX" ]]; then
  echo "GPU gate refused: no unowned idle GPU available at this live check" >&2
  printf '5\n' > "$RUN_ROOT/gtfit_preflight_exit_code.txt"; exit 5
fi
RUN_ID="$SESSION_ID"
printf '%s\n' "$RUN_ID" > "$LOCK/run_id.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOCK/acquired_utc.txt"
cp "$LOCK/run_id.txt" "$RUN_ROOT/gtfit_reservation_run_id.txt"
cp "$LOCK/acquired_utc.txt" "$RUN_ROOT/gtfit_reservation_acquired_utc.txt"
release_lock() {
  if [[ -f "$LOCK/run_id.txt" ]] && [[ "$(cat "$LOCK/run_id.txt")" == "$RUN_ID" ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/gtfit_reservation_released_utc.txt"
    rm -f "$LOCK/run_id.txt" "$LOCK/acquired_utc.txt"
    rmdir "$LOCK"
  fi
}
trap release_lock EXIT
UTIL2=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' %')
USED2=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' MiB')
APPS2=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>&1 || true)
{
  printf 'host=%s\nproject=%s\nexecution_checkout=' "$(hostname)" "$PROJECT"
  git -C "$EXEC" rev-parse HEAD
  printf 'gpu_index=%s\ngpu_uuid=%s\nutilization_after_reservation=%s\nmemory_after_reservation_mib=%s\n' "$GPU_INDEX" "$GPU_UUID" "$UTIL2" "$USED2"
  printf 'compute_app_rows_after_reservation:\n%s\n' "$APPS2"
  printf 'model_root=%s\nscript_sha256=' "$MODEL_ROOT"; sha256sum "$SCRIPT"
  printf 'config_sha256='; sha256sum "$CONFIG"
  printf 'bank_manifest_sha256='; sha256sum "$DATA/training_bank_manifest.json"
  printf 'registered_training_updates_per_arm='; "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["diagnostic_budget"]["updates_per_arm"])' "$CONFIG"
  nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader
  free -h; df -h "$RUN_ROOT" /tmp
} > "$RUN_ROOT/gtfit_preflight_and_fingerprints.txt" 2>&1
if [[ "$UTIL2" != 0 || "$USED2" -gt 64 ]] || grep -Fq "$GPU_UUID" <<< "$APPS2"; then
  echo "GPU changed after reservation; refuse training" >&2
  printf '5\n' > "$RUN_ROOT/gtfit_preflight_exit_code.txt"; exit 5
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/gtfit_started_utc.txt"
set +e
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$MINI_REPO:$EXEC:$PROJECT" "$PY" -u "$SCRIPT" \
    --repo-root "$MINI_REPO" --model-root "$MODEL_ROOT" --data-root "$DATA" \
    --config "$CONFIG" --output-root "$OUT" --device cuda:0 \
    > "$RUN_ROOT/gtfit.stdout.log" 2> "$RUN_ROOT/gtfit.stderr.log"
RC=$?
set -e
printf '%s\n' "$RC" > "$RUN_ROOT/gtfit_exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/gtfit_finished_utc.txt"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gtfit_gpu_after.txt"
if [[ "$RC" -ne 0 ]]; then
  exit "$RC"
fi
sha256sum "$OUT/summary.json" > "$RUN_ROOT/gtfit_summary.sha256"
echo "GT-fit pilot completed on physical GPU $GPU_INDEX ($GPU_UUID)"
