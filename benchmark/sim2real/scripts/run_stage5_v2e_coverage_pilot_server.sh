#!/usr/bin/env bash
# Paired, balanced V2E shards for the registered Stage-5 eight-rollout bank.
set -Eeuo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 PROJECT_ROOT CLEAN_REPO_ROOT BANK_ROOT V2E_ROOT GPU_A GPU_B" >&2
  exit 2
fi

PROJECT_ROOT=$1
REPO_ROOT=$2
BANK_ROOT=$3
V2E_ROOT=$4
GPU_A=$5
GPU_B=$6
ENV_BIN=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin
PYTHON=$ENV_BIN/python
V2E_SOURCE=$PROJECT_ROOT/external/sources/v2e-cac6384445/v2e-pinned
V2E_MODEL=$PROJECT_ROOT/external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt
SIM=$REPO_ROOT/benchmark/sim2real

[[ "$(whoami)" == zhaoyuhan ]] || { echo "unexpected server user" >&2; exit 3; }
[[ "$PROJECT_ROOT" == /home/zhaoyuhan/Eventbased_WAM ]] || { echo "unexpected project root" >&2; exit 3; }
[[ "$REPO_ROOT" == "$PROJECT_ROOT"/repositories/gec-sim2real-exec-* ]] || {
  echo "V2E must run from a separate versioned execution checkout" >&2; exit 3;
}
[[ -x "$PYTHON" && -f "$V2E_SOURCE/v2e.py" && -f "$V2E_MODEL" ]] || {
  echo "pinned Python/V2E source/model missing" >&2; exit 3;
}
[[ "$GPU_A" =~ ^[0-9]+$ && "$GPU_B" =~ ^[0-9]+$ && "$GPU_A" != "$GPU_B" ]] || {
  echo "two distinct physical GPU IDs are required" >&2; exit 3;
}
[[ ! -e "$V2E_ROOT" ]] || { echo "refusing to overwrite V2E output root: $V2E_ROOT" >&2; exit 3; }

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

ensure_idle_gpu "$GPU_A" || exit 4
ensure_idle_gpu "$GPU_B" || exit 4

BANK_MANIFEST=$BANK_ROOT/bank_manifest.json
BANK_H5=$BANK_ROOT/shared_trajectory.h5
SPLIT=$BANK_ROOT/registered_split.json
CONFIG=$BANK_ROOT/stage5_coverage_pilot_config.json
[[ -s "$BANK_MANIFEST" && -s "$BANK_H5" && -s "$SPLIT" && -s "$CONFIG" ]] || {
  echo "complete physical bank and registered config/split are required" >&2; exit 5;
}
"$PYTHON" - "$BANK_MANIFEST" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
if (record.get("schema") != "stage5_flow_scene_coverage_shared_bank_v3"
        or record.get("scope") != "complete_balanced_8_rollout_pilot"
        or record.get("complete_balanced_pilot") is not True
        or len(record.get("episodes", [])) != 8):
    raise SystemExit("the registered eight-rollout v3 physical bank is incomplete")
PY

mkdir -p "$V2E_ROOT"
if [[ "$GPU_A" == 0 ]]; then OUT_A=$V2E_ROOT; else OUT_A=$V2E_ROOT/gpu$GPU_A; fi
if [[ "$GPU_B" == 0 ]]; then OUT_B=$V2E_ROOT; else OUT_B=$V2E_ROOT/gpu$GPU_B; fi
mkdir -p "$OUT_A" "$OUT_B"
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$STARTED" > "$V2E_ROOT/started_utc.txt"
printf '%q ' "$PYTHON" "$SIM/scripts/run_stage4_v2e_shard.py" --project-root "$PROJECT_ROOT" \
  --bank-root "$BANK_ROOT" --output-root "$OUT_A" --gpu "$GPU_A" \
  --episodes s5q_00_train s5q_00_val s5q_02_train s5q_02_val \
  > "$V2E_ROOT/gpu${GPU_A}.command.txt"
printf '\n' >> "$V2E_ROOT/gpu${GPU_A}.command.txt"
printf '%q ' "$PYTHON" "$SIM/scripts/run_stage4_v2e_shard.py" --project-root "$PROJECT_ROOT" \
  --bank-root "$BANK_ROOT" --output-root "$OUT_B" --gpu "$GPU_B" \
  --episodes s5q_01_train s5q_01_val s5q_03_train s5q_03_val \
  > "$V2E_ROOT/gpu${GPU_B}.command.txt"
printf '\n' >> "$V2E_ROOT/gpu${GPU_B}.command.txt"

trap 'rc=$?; printf "%s\n" "$rc" > "$V2E_ROOT/exit_code.txt"; date -u +%Y-%m-%dT%H:%M:%SZ > "$V2E_ROOT/finished_utc.txt"; exit "$rc"' EXIT
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  printf 'project=%s\nrepo=%s\ncommit=%s\nbranch=%s\n' "$PROJECT_ROOT" "$REPO_ROOT" \
    "$(git -C "$REPO_ROOT" rev-parse HEAD)" "$(git -C "$REPO_ROOT" branch --show-current)"
  git -C "$REPO_ROOT" status --short --branch
  nvidia-smi
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv
  printf 'bank_sha256='; sha256sum "$BANK_H5"
  printf 'bank_manifest_sha256='; sha256sum "$BANK_MANIFEST"
  printf 'split_sha256='; sha256sum "$SPLIT"
  printf 'config_sha256='; sha256sum "$CONFIG"
  printf 'v2e_entry_sha256='; sha256sum "$V2E_SOURCE/v2e.py"
  printf 'slomo_sha256='; sha256sum "$V2E_MODEL"
  "$PYTHON" -c 'import sys, torch, h5py, cv2, numpy; print("python",sys.version); print("torch",torch.__version__); print("h5py",h5py.__version__); print("opencv",cv2.__version__); print("numpy",numpy.__version__)'
} > "$V2E_ROOT/environment_preflight.txt" 2>&1

"$PYTHON" -u "$SIM/scripts/run_stage4_v2e_shard.py" --project-root "$PROJECT_ROOT" \
  --bank-root "$BANK_ROOT" --output-root "$OUT_A" --gpu "$GPU_A" \
  --episodes s5q_00_train s5q_00_val s5q_02_train s5q_02_val \
  > "$V2E_ROOT/gpu${GPU_A}.worker.stdout.log" 2> "$V2E_ROOT/gpu${GPU_A}.worker.stderr.log" &
PID_A=$!
printf '%s\n' "$PID_A" > "$V2E_ROOT/gpu${GPU_A}.worker.pid"
"$PYTHON" -u "$SIM/scripts/run_stage4_v2e_shard.py" --project-root "$PROJECT_ROOT" \
  --bank-root "$BANK_ROOT" --output-root "$OUT_B" --gpu "$GPU_B" \
  --episodes s5q_01_train s5q_01_val s5q_03_train s5q_03_val \
  > "$V2E_ROOT/gpu${GPU_B}.worker.stdout.log" 2> "$V2E_ROOT/gpu${GPU_B}.worker.stderr.log" &
PID_B=$!
printf '%s\n' "$PID_B" > "$V2E_ROOT/gpu${GPU_B}.worker.pid"

RC_A=0; wait "$PID_A" || RC_A=$?
RC_B=0; wait "$PID_B" || RC_B=$?
printf '%s\n' "$RC_A" > "$V2E_ROOT/gpu${GPU_A}.worker.exit_code.txt"
printf '%s\n' "$RC_B" > "$V2E_ROOT/gpu${GPU_B}.worker.exit_code.txt"
[[ "$RC_A" -eq 0 && "$RC_B" -eq 0 ]] || {
  echo "V2E shard failure: gpu${GPU_A}=$RC_A gpu${GPU_B}=$RC_B" >&2; exit 6;
}

"$PYTHON" -u "$SIM/scripts/finalize_stage5_v2e_coverage_pilot.py" \
  --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
  > "$V2E_ROOT/finalize.stdout.log" 2> "$V2E_ROOT/finalize.stderr.log"
