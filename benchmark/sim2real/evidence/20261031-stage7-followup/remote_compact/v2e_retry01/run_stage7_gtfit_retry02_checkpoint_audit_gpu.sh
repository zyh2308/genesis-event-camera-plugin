#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT=$PROJECT/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1
RETRY_ROOT=$RUN_ROOT/v2e_retry01
PY=$PROJECT/../miniconda3/envs/Eventbased_WAM_torch28/bin/python
SCRIPT=$RETRY_ROOT/finalize_stage7_gtfit_retry02.py
EXEC=$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
MODEL_ROOT=$PROJECT/external/sources/downstream/STE-FlowNet
DATA=$RUN_ROOT/flow_bank_retry01
CONFIG=$RUN_ROOT/stage7_replicad_multiaxis_gtfit_v1.json
OUT=$RUN_ROOT/flow_gtfit_retry02_final_audit
RESERVATIONS=$PROJECT/runs/sim2real/resource_reservations
RUN_ID=20261031-stage7-gtfit-retry02-checkpoint-audit-v1

test "$(cat "$RETRY_ROOT/gtfit_retry02_exit_code.txt")" = 1
test -f "$RUN_ROOT/flow_gtfit_retry02/eventgenesis/checkpoint_final.pt"
test -f "$RUN_ROOT/flow_gtfit_retry02/v2e/checkpoint_final.pt"
test ! -e "$OUT"
test -x "$PY" && test -f "$SCRIPT" && test -f "$DATA/training_bank_manifest.json"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RETRY_ROOT/checkpoint_audit_preflight_utc.txt"

APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>&1 || true)
mkdir -p "$RESERVATIONS"
GPU_INDEX=""; GPU_UUID=""; LOCK=""; UTIL=""; USED=""
while IFS=',' read -r idx _name uuid util used _total; do
  idx=${idx//[[:space:]]/}; uuid=${uuid//[[:space:]]/}
  util=${util//[[:space:]]/}; util=${util%%%}; used=${used//[[:space:]]/}
  if [[ "$util" != 0 || ! "$used" =~ ^[0-9]+$ || "$used" -gt 64 ]] || grep -Fq "$uuid" <<< "$APPS"; then
    continue
  fi
  candidate="$RESERVATIONS/$uuid"
  if mkdir "$candidate" 2>/dev/null; then
    GPU_INDEX=$idx; GPU_UUID=$uuid; LOCK=$candidate; UTIL=$util; USED=$used
    break
  fi
done < <(nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
if [[ -z "$GPU_INDEX" ]]; then
  echo "GPU gate refused: no currently unowned idle GPU" >&2
  printf '5\n' > "$RETRY_ROOT/checkpoint_audit_exit_code.txt"
  exit 5
fi
printf '%s\n' "$RUN_ID" > "$LOCK/run_id.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOCK/acquired_utc.txt"
cp "$LOCK/run_id.txt" "$RETRY_ROOT/checkpoint_audit_reservation_run_id.txt"
cp "$LOCK/acquired_utc.txt" "$RETRY_ROOT/checkpoint_audit_reservation_acquired_utc.txt"
release_lock() {
  if [[ -f "$LOCK/run_id.txt" && "$(cat "$LOCK/run_id.txt")" == "$RUN_ID" ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$RETRY_ROOT/checkpoint_audit_reservation_released_utc.txt"
    rm -f "$LOCK/run_id.txt" "$LOCK/acquired_utc.txt"
    rmdir "$LOCK"
  fi
}
trap release_lock EXIT
UTIL2=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' %')
USED2=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' MiB')
APPS2=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>&1 || true)
{
  printf 'host=%s\nproject=%s\ngpu_index=%s\ngpu_uuid=%s\nutil_before=%s\nmem_before_mib=%s\n' "$(hostname)" "$PROJECT" "$GPU_INDEX" "$GPU_UUID" "$UTIL" "$USED"
  printf 'util_after_reservation=%s\nmem_after_reservation_mib=%s\ncompute_app_rows=%s\n' "$UTIL2" "$USED2" "$APPS2"
  printf 'audit_script_sha256='; sha256sum "$SCRIPT"
  printf 'executed_training_script_sha256='; sha256sum "$RETRY_ROOT/run_stage7_replicad_gtfit_retry02.py"
  printf 'data_manifest_sha256='; sha256sum "$DATA/training_bank_manifest.json"
  printf 'config_sha256='; sha256sum "$CONFIG"
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  free -h; df -h "$RUN_ROOT" /tmp
} > "$RETRY_ROOT/checkpoint_audit_preflight.txt" 2>&1
if [[ "$UTIL2" != 0 || "$USED2" -gt 64 ]] || grep -Fq "$GPU_UUID" <<< "$APPS2"; then
  echo "GPU changed after reservation; refusing checkpoint audit" >&2
  printf '5\n' > "$RETRY_ROOT/checkpoint_audit_exit_code.txt"
  exit 5
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$RETRY_ROOT/checkpoint_audit_started_utc.txt"
set +e
CUDA_VISIBLE_DEVICES="$GPU_INDEX" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$RUN_ROOT/source_v2/mini_repo:$EXEC:$PROJECT:$RETRY_ROOT" \
  "$PY" -u "$SCRIPT" \
    --repo-root "$RUN_ROOT/source_v2/mini_repo" --run-root "$RUN_ROOT" \
    --model-root "$MODEL_ROOT" --data-root "$DATA" --config "$CONFIG" \
    --executed-script "$RETRY_ROOT/run_stage7_replicad_gtfit_retry02.py" \
    --output-root "$OUT" > "$RETRY_ROOT/checkpoint_audit.stdout.log" 2> "$RETRY_ROOT/checkpoint_audit.stderr.log"
RC=$?
set -e
printf '%s\n' "$RC" > "$RETRY_ROOT/checkpoint_audit_exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RETRY_ROOT/checkpoint_audit_finished_utc.txt"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader > "$RETRY_ROOT/checkpoint_audit_gpu_after.txt"
if [[ "$RC" -ne 0 ]]; then exit "$RC"; fi
sha256sum "$OUT/restore_and_validation_audit.json" > "$RETRY_ROOT/checkpoint_audit_result.sha256"
