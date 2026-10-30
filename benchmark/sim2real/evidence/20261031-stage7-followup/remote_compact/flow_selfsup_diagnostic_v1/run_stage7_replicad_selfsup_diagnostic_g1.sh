#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT=$PROJECT/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1
RUN=${SIM2REAL_RUN_DIR:-$RUN_ROOT/flow_selfsup_diagnostic_v1}
EXEC=$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
PY=$PROJECT/../miniconda3/envs/Eventbased_WAM_torch28/bin/python
MODEL_ROOT=$PROJECT/external/sources/downstream/STE-FlowNet
DATA=$RUN_ROOT/flow_bank_retry01
CONFIG=$RUN/config.json
SCRIPT=$RUN/train_stage7_replicad_selfsup_diagnostic.py
OUT=$RUN/training_output
RESERVATIONS=$PROJECT/runs/sim2real/resource_reservations
RUN_ID=${SIM2REAL_RUN_ID:-20261031-stage7-replicad-selfsup-diagnostic-v1}

test "$(hostname -s)" = g1
test -x "$PY" && test -d "$EXEC" && test -d "$MODEL_ROOT"
test -f "$DATA/training_bank_manifest.json" && test -f "$CONFIG" && test -f "$SCRIPT"
test ! -e "$OUT" && test ! -e "$RUN/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/driver_started_utc.txt"
{
  printf 'host=%s\nproject=%s\nrun=%s\nrun_id=%s\n' "$(hostname -f)" "$PROJECT" "$RUN" "$RUN_ID"
  printf 'execution_checkout=%s\nexecution_commit=' "$EXEC"; git -C "$EXEC" rev-parse HEAD
  printf 'execution_branch='; git -C "$EXEC" branch --show-current
  printf 'python=%s\n' "$PY"; "$PY" --version
  printf 'training_script_sha256='; sha256sum "$SCRIPT"
  printf 'config_sha256='; sha256sum "$CONFIG"
  printf 'bank_manifest_sha256='; sha256sum "$DATA/training_bank_manifest.json"
  printf 'objective_helper_sha256='; sha256sum "$EXEC/benchmark/sim2real/scripts/run_steflow_stage2_pilot.py"
  printf 'warp_patch_sha256='; sha256sum "$EXEC/benchmark/sim2real/compat/steflow_align_corners.py"
  printf 'source_verifier_sha256='; sha256sum "$EXEC/benchmark/sim2real/scripts/verify_pinned_source.py"
  printf '\n-- resources before GPU reservation --\n'
  uptime; free -h; df -h "$RUN_ROOT" /tmp
} > "$RUN/preflight.txt" 2>&1
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits > "$RUN/gpu_before_reservation.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader > "$RUN/compute_apps_before_reservation.csv" 2>&1 || true
mkdir -p "$RESERVATIONS"

GPU_INDEX=""; GPU_UUID=""; GPU_NAME=""; LOCK=""
while IFS=',' read -r idx name uuid util used total; do
  idx=${idx//[[:space:]]/}; name=${name# }; name=${name% }; uuid=${uuid//[[:space:]]/}
  util=${util//[[:space:]]/}; used=${used//[[:space:]]/}
  if [[ "$util" != 0 || ! "$used" =~ ^[0-9]+$ || "$used" -gt 64 ]]; then continue; fi
  if grep -Fq "$uuid" "$RUN/compute_apps_before_reservation.csv"; then continue; fi
  candidate=$RESERVATIONS/$uuid
  if mkdir "$candidate" 2>/dev/null; then GPU_INDEX=$idx; GPU_UUID=$uuid; GPU_NAME=$name; LOCK=$candidate; break; fi
done < "$RUN/gpu_before_reservation.csv"
if [[ -z "$GPU_INDEX" ]]; then
  echo "No idle unowned GPU was available; refusing to train." > "$RUN/gpu_gate.txt"
  printf '5\n' > "$RUN/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
  exit 5
fi

printf '%s\n' "$RUN_ID" > "$LOCK/run_id.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOCK/acquired_utc.txt"
printf '{\n  "schema": "sim2real_gpu_reservation_v1",\n  "status": "reserved",\n  "host": "%s",\n  "gpu_uuid": "%s",\n  "run_id": "%s",\n  "run_path": "%s",\n  "created_utc": "%s",\n  "owner": "zhaoyuhan / EventGenesis Sim-to-Real engineering",\n  "purpose": "paired 100-update synthetic-only STE-FlowNet self-supervised diagnostic",\n  "cuda_visible_devices": "%s"\n}\n' \
  "$(hostname -f)" "$GPU_UUID" "$RUN_ID" "$RUN" "$(cat "$LOCK/acquired_utc.txt")" "$GPU_INDEX" > "$LOCK/resource_reservation.json"
cp "$LOCK/resource_reservation.json" "$RUN/resource_reservation.json"
printf 'gpu_index=%s\ngpu_name=%s\ngpu_uuid=%s\nlock=%s\n' "$GPU_INDEX" "$GPU_NAME" "$GPU_UUID" "$LOCK" > "$RUN/gpu_reservation.txt"

release_own_reservation() {
  local rc=$?
  if [[ ! -f "$RUN/exit_code.txt" ]]; then printf '%s\n' "$rc" > "$RUN/exit_code.txt"; fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader > "$RUN/gpu_after.csv" 2>&1 || true
  if [[ -n "$LOCK" && -f "$LOCK/run_id.txt" && "$(cat "$LOCK/run_id.txt")" = "$RUN_ID" ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/reservation_released_utc.txt"
    rm -f -- "$LOCK/run_id.txt" "$LOCK/acquired_utc.txt" "$LOCK/resource_reservation.json"
    rmdir -- "$LOCK"
  fi
  return 0
}
trap release_own_reservation EXIT

UTIL2=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' %')
USED2=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' MiB')
APPS2=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>&1 || true)
printf 'post_reservation_utilization=%s\npost_reservation_memory_mib=%s\npost_reservation_compute_apps:\n%s\n' "$UTIL2" "$USED2" "$APPS2" > "$RUN/post_reservation_check.txt"
if [[ "$UTIL2" != 0 || ! "$USED2" =~ ^[0-9]+$ || "$USED2" -gt 64 ]] || grep -Fq "$GPU_UUID" <<< "$APPS2"; then
  echo "GPU changed after reservation; refusing training" >> "$RUN/post_reservation_check.txt"
  printf '5\n' > "$RUN/exit_code.txt"
  exit 5
fi
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits >> "$RUN/post_reservation_check.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/training_started_utc.txt"
set +e
CUDA_VISIBLE_DEVICES="$GPU_INDEX" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$EXEC" "$PY" -u "$SCRIPT" \
    --repo-root "$EXEC" --model-root "$MODEL_ROOT" --run-root "$RUN_ROOT" \
    --config "$CONFIG" --output-root "$OUT" --device cuda:0 \
    > "$RUN/trainer.stdout.log" 2> "$RUN/trainer.stderr.log"
RC=$?
set -e
printf '%s\n' "$RC" > "$RUN/exit_code.txt"
if [[ "$RC" -eq 0 ]]; then
  sha256sum "$OUT/summary.json" > "$RUN/summary.sha256"
  find "$OUT" -maxdepth 2 -type f -name 'checkpoint_final.pt' -exec sha256sum {} \; > "$RUN/checkpoints.sha256"
fi
exit "$RC"
