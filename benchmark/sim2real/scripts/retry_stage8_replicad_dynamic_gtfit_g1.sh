#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/zhaoyuhan/Eventbased_WAM
EXEC=${SIM2REAL_EXEC_CHECKOUT:?set SIM2REAL_EXEC_CHECKOUT to the immutable isolated checkout}
PY=$PROJECT/../miniconda3/envs/Eventbased_WAM_torch28/bin/python
MODEL_ROOT=$PROJECT/external/sources/downstream/STE-FlowNet
CONFIG=$EXEC/benchmark/sim2real/configs/stage8_replicad_dynamic_gtfit_v2.json
TRAINER=$EXEC/benchmark/sim2real/scripts/run_stage8_replicad_dynamic_gtfit.py
COMPAT=$EXEC/benchmark/sim2real/compat/steflow_device_placement.py
DATA_ROOT=$PROJECT/runs/sim2real/20261101-stage8-replicad-dynamic-v2/flow_bank
RUN_ROOT=$PROJECT/runs/sim2real/20261101-stage8-replicad-dynamic-v3-cpucompat
RUN_ID=20261101-stage8-replicad-dynamic-v3-cpucompat
RESERVATIONS=$PROJECT/runs/sim2real/resource_reservations

case "$(hostname -s)" in g1|g3) ;; *) echo "refusing unapproved compute host: $(hostname -s)" >&2; exit 2 ;; esac
test -x "$PY" && test -d "$MODEL_ROOT" && test -f "$CONFIG" && test -f "$TRAINER" && test -f "$COMPAT"
test -f "$DATA_ROOT/training_bank_manifest.json"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

"$PY" - "$CONFIG" "$DATA_ROOT/training_bank_manifest.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
manifest = json.load(open(sys.argv[2]))
assert config["diagnostic_budget"]["updates_per_arm"] == 100
assert config["real_data_used"] is False and config["real_scoring_enabled"] is False
assert manifest["status"] == "complete" and manifest["real_data_used"] is False
assert set(manifest["train_rollouts"]).isdisjoint(manifest["validation_rollouts"])
assert len(manifest["entries"]) == 4
print("stage8 retry preflight: paired synthetic bank and 100-update budget verified")
PY

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  printf 'host=%s\nproject=%s\nexec=%s\n' "$(hostname -f)" "$PROJECT" "$EXEC"
  printf 'commit='; git -C "$EXEC" rev-parse HEAD
  printf 'python=%s\n' "$PY"; "$PY" --version
  printf 'retry_of=%s\n' "$PROJECT/runs/sim2real/20261101-stage8-replicad-dynamic-v2"
  printf '\n-- source/config/data hashes --\n'
  sha256sum "$CONFIG" "$TRAINER" "$COMPAT" "$DATA_ROOT/training_bank_manifest.json"
  printf '\n-- environment and disk --\n'; uptime; free -h; df -h "$RUN_ROOT"
} > "$RUN_ROOT/preflight.txt" 2>&1

GPU_INDEX= GPU_UUID= GPU_NAME= LOCK=
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>&1 || true)
while IFS=',' read -r idx name uuid util used total; do
  idx=${idx//[[:space:]]/}; name=${name# }; name=${name% }; uuid=${uuid//[[:space:]]/}
  util=${util//[[:space:]]/}; used=${used//[[:space:]]/}
  if [[ "$util" != 0 || ! "$used" =~ ^[0-9]+$ || "$used" -gt 64 ]] || grep -Fq "$uuid" <<< "$APPS"; then continue; fi
  candidate=$RESERVATIONS/$uuid
  mkdir -p "$RESERVATIONS"
  if mkdir "$candidate" 2>/dev/null; then GPU_INDEX=$idx; GPU_UUID=$uuid; GPU_NAME=$name; LOCK=$candidate; break; fi
done < <(nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)

DEVICE=cpu CUDA_DEVICE= CPU_FALLBACK=true
if [[ -n "$GPU_INDEX" ]]; then
  DEVICE=cuda:0 CUDA_DEVICE=$GPU_INDEX CPU_FALLBACK=false
  printf '%s\n' "$RUN_ID" > "$LOCK/run_id.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$LOCK/acquired_utc.txt"
  printf '{"schema":"sim2real_gpu_reservation_v1","status":"reserved","host":"%s","gpu_uuid":"%s","run_id":"%s","run_path":"%s","purpose":"paired 100-update synthetic-only dynamic ReplicaCAD GT-flow diagnostic","cuda_visible_devices":"%s"}\n' "$(hostname -f)" "$GPU_UUID" "$RUN_ID" "$RUN_ROOT" "$GPU_INDEX" > "$LOCK/resource_reservation.json"
  cp "$LOCK/resource_reservation.json" "$RUN_ROOT/resource_reservation.json"
  UTIL2=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' %')
  USED2=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' MiB')
  APPS2=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>&1 || true)
  printf 'index=%s\nuuid=%s\nutil=%s\nmemory_mib=%s\ncompute_apps=%s\n' "$GPU_INDEX" "$GPU_UUID" "$UTIL2" "$USED2" "$APPS2" > "$RUN_ROOT/post_reservation_check.txt"
  if [[ "$UTIL2" != 0 || ! "$USED2" =~ ^[0-9]+$ || "$USED2" -gt 64 ]] || grep -Fq "$GPU_UUID" <<< "$APPS2"; then
    echo 'GPU became busy after reservation; refusing training' >> "$RUN_ROOT/post_reservation_check.txt"
    printf '5\n' > "$RUN_ROOT/exit_code.txt"; exit 5
  fi
else
  CPU_COUNT=$(getconf _NPROCESSORS_ONLN)
  LOAD1=$(awk '{print $1}' /proc/loadavg)
  MEM_AVAILABLE_KB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if ! python3 -c 'import sys; n=int(sys.argv[1]); load=float(sys.argv[2]); mem=int(sys.argv[3]); sys.exit(0 if n >= 16 and load <= n * 0.7 and mem >= 40000000 else 1)' "$CPU_COUNT" "$LOAD1" "$MEM_AVAILABLE_KB"; then
    printf '{"status":"blocked_cpu_gate","cpu_count":%s,"load1":%s,"mem_available_kb":%s}\n' "$CPU_COUNT" "$LOAD1" "$MEM_AVAILABLE_KB" > "$RUN_ROOT/device_fallback.json"
    printf '5\n' > "$RUN_ROOT/exit_code.txt"; exit 5
  fi
  printf '{"status":"cpu_fallback","cpu_count":%s,"load1":%s,"mem_available_kb":%s,"threads":2,"reason":"no idle unowned GPU at retry start; previous bounded five-check GPU wait already completed"}\n' "$CPU_COUNT" "$LOAD1" "$MEM_AVAILABLE_KB" > "$RUN_ROOT/device_fallback.json"
fi

release_lock() {
  local rc=$?
  if [[ -n "${LOCK:-}" && -f "$LOCK/run_id.txt" && "$(cat "$LOCK/run_id.txt")" = "$RUN_ID" ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/reservation_released_utc.txt"
    rm -f "$LOCK/run_id.txt" "$LOCK/acquired_utc.txt" "$LOCK/resource_reservation.json"
    rmdir "$LOCK"
  fi
  [[ -e "$RUN_ROOT/exit_code.txt" ]] || printf '%s\n' "$rc" > "$RUN_ROOT/exit_code.txt"
  return 0
}
trap release_lock EXIT
if [[ "$CPU_FALLBACK" != true ]]; then
  printf 'device=%s\ncuda_visible_devices=%s\n' "$DEVICE" "$CUDA_DEVICE" > "$RUN_ROOT/execution_device.txt"
  nvidia-smi -i "$GPU_INDEX" --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader >> "$RUN_ROOT/post_reservation_check.txt"
else
  printf 'device=cpu\ncuda_visible_devices=\n' > "$RUN_ROOT/execution_device.txt"
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/training_started_utc.txt"
set +e
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$EXEC:$PROJECT" \
  "$PY" -u "$TRAINER" --repo-root "$EXEC" --model-root "$MODEL_ROOT" --data-root "$DATA_ROOT" \
    --config "$CONFIG" --output-root "$RUN_ROOT/training_output" --device "$DEVICE" \
    > "$RUN_ROOT/trainer.stdout.log" 2> "$RUN_ROOT/trainer.stderr.log"
RC=$?
set -e
printf '%s\n' "$RC" > "$RUN_ROOT/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/training_finished_utc.txt"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu_after.csv" 2>&1 || true
if [[ "$RC" -eq 0 ]]; then
  sha256sum "$RUN_ROOT/training_output/summary.json" > "$RUN_ROOT/summary.sha256"
  find "$RUN_ROOT/training_output" -maxdepth 2 -type f -name 'checkpoint_final.pt' -exec sha256sum {} \; > "$RUN_ROOT/checkpoints.sha256"
fi
exit "$RC"
