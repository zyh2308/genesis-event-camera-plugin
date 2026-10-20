#!/usr/bin/env bash
# Run the registered 100-update synthetic-only STE objective diagnostic on a paired GPU set.
set -Eeuo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 PROJECT_ROOT CLEAN_REPO_ROOT MODEL_ROOT DATA_ROOT CONFIG RUN_ROOT GPU_EG GPU_V2E" >&2
  exit 2
fi

PROJECT_ROOT=$1
REPO_ROOT=$2
MODEL_ROOT=$3
DATA_ROOT=$4
CONFIG=$5
RUN_ROOT=$6
GPU_EG=$7
GPU_V2E=$8

PYTHON=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
SCRIPT=$REPO_ROOT/benchmark/sim2real/scripts/train_stage5_flow_selfsup_diagnostic.py

[[ "$(whoami)" == zhaoyuhan ]] || { echo "unexpected server user" >&2; exit 3; }
[[ "$PROJECT_ROOT" == /home/zhaoyuhan/Eventbased_WAM ]] || { echo "unexpected project root" >&2; exit 3; }
[[ "$REPO_ROOT" == "$PROJECT_ROOT"/repositories/gec-sim2real-exec-* ]] || {
  echo "diagnostic must run from a separate versioned execution checkout" >&2; exit 3;
}
[[ -x "$PYTHON" && -f "$SCRIPT" && -d "$MODEL_ROOT" && -d "$DATA_ROOT" && -f "$CONFIG" ]] || {
  echo "registered code/model/data/config missing" >&2; exit 3;
}
[[ "$GPU_EG" =~ ^[0-9]+$ && "$GPU_V2E" =~ ^[0-9]+$ && "$GPU_EG" != "$GPU_V2E" ]] || {
  echo "two distinct physical GPU IDs are required" >&2; exit 3;
}
[[ ! -e "$RUN_ROOT" ]] || { echo "refusing to overwrite diagnostic run: $RUN_ROOT" >&2; exit 3; }

ensure_idle_gpu() {
  local gpu="$1" stats used util uuid pid gpu_uuid owner
  uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
  stats=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r used util <<< "$(awk -F, '{gsub(/ /,"",$1);gsub(/ /,"",$2);print $1,$2}' <<< "$stats")"
  [[ "$used" -le 1024 && "$util" -le 5 ]] || { echo "GPU $gpu is occupied: $stats" >&2; return 1; }
  while IFS=, read -r pid gpu_uuid; do
    pid=${pid// /}; gpu_uuid=${gpu_uuid// /}
    [[ "$gpu_uuid" == "$uuid" ]] || continue
    if [[ -n "$pid" ]]; then
      owner=$(ps -o user= -p "$pid" | tr -d ' ' || true)
      echo "GPU $gpu has compute PID $pid owner=${owner:-unknown}" >&2
      return 1
    fi
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader)
}

ensure_idle_gpu "$GPU_EG" || exit 4
ensure_idle_gpu "$GPU_V2E" || exit 4
[[ -f "$DATA_ROOT/training_bank_manifest.json" ]] || { echo "paired adapter manifest missing" >&2; exit 5; }

mkdir -p "$RUN_ROOT/launcher"
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$STARTED" > "$RUN_ROOT/started_utc.txt"
printf '%q ' "$PYTHON" -u "$SCRIPT" --model-root "$MODEL_ROOT" --data-root "$DATA_ROOT" \
  --config "$CONFIG" --run-root "$RUN_ROOT" --generator eventgenesis --seed 0 --max-updates 100 \
  > "$RUN_ROOT/launcher/eventgenesis.command.txt"
printf '\n' >> "$RUN_ROOT/launcher/eventgenesis.command.txt"
printf '%q ' "$PYTHON" -u "$SCRIPT" --model-root "$MODEL_ROOT" --data-root "$DATA_ROOT" \
  --config "$CONFIG" --run-root "$RUN_ROOT" --generator v2e --seed 0 --max-updates 100 \
  > "$RUN_ROOT/launcher/v2e.command.txt"
printf '\n' >> "$RUN_ROOT/launcher/v2e.command.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN_ROOT/exit_code.txt"; date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/finished_utc.txt"; exit "$rc"' EXIT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  printf 'project=%s\nrepo=%s\ncommit=%s\nbranch=%s\nmodel_root=%s\n' "$PROJECT_ROOT" "$REPO_ROOT" \
    "$(git -C "$REPO_ROOT" rev-parse HEAD)" "$(git -C "$REPO_ROOT" branch --show-current)" "$MODEL_ROOT"
  git -C "$REPO_ROOT" status --short --branch
  printf 'config_sha256='; sha256sum "$CONFIG"
  printf 'adapter_manifest_sha256='; sha256sum "$DATA_ROOT/training_bank_manifest.json"
  printf 'model_commit=%s\n' "$(git -C "$MODEL_ROOT" rev-parse HEAD)"
  git -C "$MODEL_ROOT" status --short --branch
  nvidia-smi
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv
  "$PYTHON" -c 'import sys, torch, numpy, cv2; print("python",sys.version); print("torch",torch.__version__); print("numpy",numpy.__version__); print("opencv",cv2.__version__)'
} > "$RUN_ROOT/launcher/environment_preflight.txt" 2>&1

CUDA_VISIBLE_DEVICES=$GPU_EG STAGE5_PHYSICAL_GPU=$GPU_EG "$PYTHON" -u "$SCRIPT" \
  --model-root "$MODEL_ROOT" --data-root "$DATA_ROOT" --config "$CONFIG" --run-root "$RUN_ROOT" \
  --generator eventgenesis --seed 0 --max-updates 100 \
  > "$RUN_ROOT/launcher/eventgenesis.stdout.log" 2> "$RUN_ROOT/launcher/eventgenesis.stderr.log" &
PID_EG=$!
printf '%s\n' "$PID_EG" > "$RUN_ROOT/launcher/eventgenesis.pid"
CUDA_VISIBLE_DEVICES=$GPU_V2E STAGE5_PHYSICAL_GPU=$GPU_V2E "$PYTHON" -u "$SCRIPT" \
  --model-root "$MODEL_ROOT" --data-root "$DATA_ROOT" --config "$CONFIG" --run-root "$RUN_ROOT" \
  --generator v2e --seed 0 --max-updates 100 \
  > "$RUN_ROOT/launcher/v2e.stdout.log" 2> "$RUN_ROOT/launcher/v2e.stderr.log" &
PID_V2E=$!
printf '%s\n' "$PID_V2E" > "$RUN_ROOT/launcher/v2e.pid"

RC_EG=0; wait "$PID_EG" || RC_EG=$?
RC_V2E=0; wait "$PID_V2E" || RC_V2E=$?
printf '%s\n' "$RC_EG" > "$RUN_ROOT/launcher/eventgenesis.exit_code.txt"
printf '%s\n' "$RC_V2E" > "$RUN_ROOT/launcher/v2e.exit_code.txt"
[[ "$RC_EG" -eq 0 && "$RC_V2E" -eq 0 ]] || {
  echo "paired diagnostic failure: eventgenesis=$RC_EG v2e=$RC_V2E" >&2; exit 6;
}
