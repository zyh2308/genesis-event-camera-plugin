#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ID=20261101-stage8-flow-photo-oracle-run02-v1
RUN=$PROJECT/runs/sim2real/$RUN_ID
REPO=$PROJECT/execution_checkouts/sim2real-flow-photo-oracle-20261101
MODEL=$PROJECT/external/sources/downstream/STE-FlowNet
BANK=$PROJECT/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python
SCRIPT=$RUN/audit_stage7_gt_zero_reverse_photometric.py
CONFIG=$RUN/configs/stage7_gt_zero_reverse_photometric_oracle_v1.json
OUT=$RUN/output
RESERVATIONS=$PROJECT/runs/sim2real/resource_reservations

test "$(hostname -s)" = g1
test -d "$RUN" && test -d "$REPO" && test -d "$MODEL" && test -x "$PY"
test -f "$SCRIPT" && test -f "$CONFIG"
test -f "$BANK/flow_bank_retry01/training_bank_manifest.json"
test ! -e "$OUT" && test ! -e "$RUN/exit_code.txt"
EXPECTED_LAUNCHER=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["code"]["launcher_sha256"])' "$CONFIG")
EXPECTED_SCRIPT=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["code"]["script_sha256"])' "$CONFIG")
ACTUAL_LAUNCHER=$(sha256sum "$0" | awk '{print $1}')
ACTUAL_SCRIPT=$(sha256sum "$SCRIPT" | awk '{print $1}')
test "$EXPECTED_LAUNCHER" = "$ACTUAL_LAUNCHER"
test "$EXPECTED_SCRIPT" = "$ACTUAL_SCRIPT"

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/driver_started_utc.txt"
{
  printf 'host=%s\nrun_id=%s\nrun=%s\n' "$(hostname -f)" "$RUN_ID" "$RUN"
  printf 'execution_snapshot=%s\nexecution_snapshot_source_commit=' "$REPO"
  sed -n 's/^source_commit=//p' "$RUN/execution_snapshot_manifest.txt"
  printf 'diagnostic_script_sha256='; sha256sum "$SCRIPT"
  printf 'launcher_script_sha256='; sha256sum "$0"
  printf 'config_sha256='; sha256sum "$CONFIG"
  printf 'manifest_sha256='; sha256sum "$BANK/flow_bank_retry01/training_bank_manifest.json"
  if [[ -e "$MODEL/.git" ]]; then
    printf 'model_source_type=git_checkout\nmodel_commit='
    git -C "$MODEL" rev-parse HEAD
  else
    printf 'model_source_type=verified_file_bundle (checked by the pinned-source verifier before the objective runs)\n'
  fi
  printf 'python='; "$PY" -c 'import torch, numpy, cv2; print(torch.__version__, torch.cuda.is_available(), numpy.__version__, cv2.__version__)'
  printf '\n-- resources before reservation --\n'
  uptime; free -h; df -h "$RUN" /tmp
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
  echo "No idle unowned GPU was available; refusing the diagnostic." > "$RUN/gpu_gate.txt"
  printf '5\n' > "$RUN/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
  exit 5
fi

printf '%s\n' "$RUN_ID" > "$LOCK/run_id.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOCK/acquired_utc.txt"
printf '{\n  "schema": "sim2real_gpu_reservation_v1",\n  "status": "reserved",\n  "host": "%s",\n  "gpu_uuid": "%s",\n  "run_id": "%s",\n  "run_path": "%s",\n  "created_utc": "%s",\n  "owner": "zhaoyuhan / EventGenesis Sim-to-Real engineering",\n  "purpose": "fixed synthetic validation actual photometric-loss GT/zero/reverse oracle",\n  "cuda_visible_devices": "%s"\n}\n' \
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
  echo "GPU changed after atomic reservation; refusing the diagnostic." >> "$RUN/post_reservation_check.txt"
  printf '5\n' > "$RUN/exit_code.txt"
  exit 5
fi
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits >> "$RUN/post_reservation_check.txt"

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_utc.txt"
set +e
CUDA_VISIBLE_DEVICES="$GPU_INDEX" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$REPO" "$PY" -u "$SCRIPT" \
    --repo-root "$REPO" --model-root "$MODEL" --run-root "$BANK" \
    --config "$CONFIG" --output-root "$OUT" --device cuda:0 \
    > "$RUN/stdout.log" 2> "$RUN/stderr.log" &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$RUN/trainer_pid.txt"
wait "$TRAIN_PID"
RC=$?
set -e
printf '%s\n' "$RC" > "$RUN/exit_code.txt"
exit "$RC"
