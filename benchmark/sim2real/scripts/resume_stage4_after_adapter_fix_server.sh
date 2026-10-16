#!/usr/bin/env bash
# Rebuild the Stage-4 adapter from a verified V2E manifest, then launch seed 0.
set -u

PROJECT=/home/zhaoyuhan/Eventbased_WAM
REPO="$PROJECT/repositories/gec-sim2real"
SIM="$REPO/benchmark/sim2real"
CONFIG=${STAGE4_CONFIG_PATH:-$SIM/configs/stage4_training_launch_v1_torch_cuda.json}
CONFIG_CHECKSUM=${STAGE4_CONFIG_CHECKSUM:-training_launch_v1_torch_cuda.sha256}
RUN_ID=${STAGE4_RUN_ID:-20261016-stage4-flowdt1-seed0-v4}
RUN_ROOT="$PROJECT/runs/sim2real/$RUN_ID"
BANK_ROOT=${STAGE4_BANK_ROOT:-$PROJECT/data/sim2real/stage4/training_bank_v1_torch_cuda_32x2s}
V2E_ROOT="$RUN_ROOT/v2e_events"
ADAPTER_ROOT="$PROJECT/data/sim2real/stage4/$RUN_ID/steflow_30ms"
ADAPTER_PARTIAL="$RUN_ROOT/steflow_30ms_partial_adapter_failure1"
MODEL_ROOT="$PROJECT/external/sources/downstream/STE-FlowNet"
CONDA_ROOT=/home/zhaoyuhan/miniconda3
ENV_BIN="$CONDA_ROOT/envs/Eventbased_WAM/bin"
TRAIN_ROOT="$RUN_ROOT/flow_dt1_seed0"

if [[ ! -f "$V2E_ROOT/v2e_complete_manifest.json" || ! -f "$BANK_ROOT/bank_manifest.json" || ! -f "$BANK_ROOT/shared_trajectory.h5" ]]; then
  echo "complete V2E manifest or shared bank is missing" >&2
  exit 2
fi
if [[ ! -f "$RUN_ROOT/postv2e.exit_code.txt" || "$(cat "$RUN_ROOT/postv2e.exit_code.txt")" != "1" \
   || ! -f "$RUN_ROOT/steflow_adapter.exit_code.txt" || "$(cat "$RUN_ROOT/steflow_adapter.exit_code.txt")" != "1" ]]; then
  echo "expected a recorded failed post-V2E attempt and adapter attempt before retry" >&2
  exit 2
fi
if [[ -f "$ADAPTER_ROOT/training_bank_manifest.json" || -e "$ADAPTER_PARTIAL" ]]; then
  echo "a complete adapter or partial-output archive already exists; refusing overwrite" >&2
  exit 2
fi
if [[ -e "$RUN_ROOT/adapter_retry1.started_utc.txt" ]]; then
  echo "adapter retry already attempted; inspect its logs before another run" >&2
  exit 2
fi
if [[ -e "$ADAPTER_ROOT" ]]; then
  mv "$ADAPTER_ROOT" "$ADAPTER_PARTIAL"
  printf '%s\n' "$ADAPTER_PARTIAL" > "$RUN_ROOT/adapter_partial_archive_path.txt"
fi

cd "$REPO" || exit 2
COMMIT=$(git rev-parse HEAD)
BRANCH=$(git branch --show-current)
CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
printf '%s\n' "$$" > "$RUN_ROOT/adapter_retry1.pid"
date -u +%FT%TZ > "$RUN_ROOT/adapter_retry1.started_utc.txt"
trap 'rc=$?; if [[ -f "$RUN_ROOT/adapter_retry1_watcher.pid" ]]; then watcher=$(cat "$RUN_ROOT/adapter_retry1_watcher.pid"); if kill -0 "$watcher" 2>/dev/null; then kill "$watcher"; fi; fi; echo "$rc" > "$RUN_ROOT/adapter_retry1.exit_code.txt"; date -u +%FT%TZ > "$RUN_ROOT/adapter_retry1.finished_utc.txt"' EXIT

{
  date -u +%FT%TZ
  hostname
  printf 'repo=%s\nbranch=%s\ncommit=%s\nconfig=%s\nconfig_sha256=%s\n' "$REPO" "$BRANCH" "$COMMIT" "$CONFIG" "$CONFIG_SHA"
  git status --short --branch
  nvidia-smi
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv
  free -h
  df -h /home
} > "$RUN_ROOT/adapter_retry1_preflight.txt" 2>&1

if ! (cd "$SIM" && sha256sum -c "protocol/$CONFIG_CHECKSUM") \
  > "$RUN_ROOT/adapter_retry1_config_verification.txt" 2>&1; then
  exit 4
fi
if ! "$ENV_BIN/python" - "$V2E_ROOT" <<'PY' > "$RUN_ROOT/adapter_retry1_shard_verification.json" 2>&1
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
manifest=json.loads((root/"v2e_complete_manifest.json").read_text())
if manifest.get("schema") != "stage4_v2e_raw_events_v1" or len(manifest.get("episodes", [])) != 32:
    raise SystemExit("V2E manifest does not contain the registered 32 rollout rows")
seen={}
for row in manifest["episodes"]:
    rid=row["rollout_id"]
    if rid in seen or row.get("exit_code") != 0:
        raise SystemExit(f"duplicate or failed V2E manifest row: {rid}")
    seen[rid]=row
if len(seen) != 32:
    raise SystemExit("V2E manifest rollout IDs are incomplete")
for gpu in (1,2,4,5):
    folder=root/f"gpu{gpu}"
    progress=json.loads((folder/"shard_progress.json").read_text())
    exit_code=int((folder/"worker.exit_code.txt").read_text())
    if progress.get("status") != "complete" or progress.get("completed_rollouts") != 8 or exit_code != 0:
        raise SystemExit(f"GPU {gpu} V2E shard incomplete")
    if any(item.get("exit_code") != 0 for item in progress.get("episodes", [])):
        raise SystemExit(f"GPU {gpu} contains a failed episode row")
print(json.dumps({"status":"verified","rollouts":len(seen),"workers":4},indent=2))
PY
then
  exit 4
fi

source "$CONDA_ROOT/bin/activate" Eventbased_WAM || exit 90
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg" GS_CACHE_FILE_PATH="$RUN_ROOT/cache/genesis"
export TI_OFFLINE_CACHE=1 TI_OFFLINE_CACHE_FILE_PATH="$RUN_ROOT/cache/taichi"
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PROJECT/runtime/sim2real/genesis_hdr_v2:$REPO"

python -u "$SIM/scripts/watch_stage4_training_status.py" \
  --repo-root "$REPO" --bank-root "$BANK_ROOT" --adapter-root "$ADAPTER_ROOT" \
  --run-root "$RUN_ROOT" --config "$CONFIG" --config-sha "$CONFIG_SHA" \
  --branch "$BRANCH" --commit "$COMMIT" --poll-seconds 15 \
  > "$RUN_ROOT/adapter_retry1_status_watcher.log" 2>&1 &
echo $! > "$RUN_ROOT/adapter_retry1_watcher.pid"

run_step() {
  local label="$1"; shift
  printf '%q ' "$@" > "$RUN_ROOT/$label.command.txt"
  printf '\n' >> "$RUN_ROOT/$label.command.txt"
  date -u +%FT%TZ > "$RUN_ROOT/$label.started_utc.txt"
  if "$@" > "$RUN_ROOT/$label.stdout.log" 2> "$RUN_ROOT/$label.stderr.log"; then
    echo 0 > "$RUN_ROOT/$label.exit_code.txt"
    date -u +%FT%TZ > "$RUN_ROOT/$label.finished_utc.txt"
    return 0
  else
    local rc=$?
    echo "$rc" > "$RUN_ROOT/$label.exit_code.txt"
    date -u +%FT%TZ > "$RUN_ROOT/$label.finished_utc.txt"
    echo "$label failed with exit code $rc" >&2
    return "$rc"
  fi
}

if ! run_step steflow_adapter_retry1 "$ENV_BIN/python" -u "$SIM/scripts/prepare_stage4_steflow_bank.py" \
  --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" --output-root "$ADAPTER_ROOT"; then
  exit "$(cat "$RUN_ROOT/steflow_adapter_retry1.exit_code.txt")"
fi

check_idle_gpu() {
  local gpu="$1" pair used util
  pair=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) || return 1
  used=$(printf '%s' "$pair" | awk -F, '{gsub(/ /,"",$1); print $1}')
  util=$(printf '%s' "$pair" | awk -F, '{gsub(/ /,"",$2); print $2}')
  if [[ -z "$used" || -z "$util" || "$used" -gt 1024 || "$util" -gt 5 ]]; then
    echo "GPU $gpu occupied (memory=${used:-?}MiB util=${util:-?}%), refusing to launch" >&2
    return 1
  fi
}
check_idle_gpu 2 || exit 3
check_idle_gpu 4 || exit 3
if [[ -e "$TRAIN_ROOT/eventgenesis/train.pid" || -e "$TRAIN_ROOT/v2e/train.pid" ]]; then
  echo "training PID files already exist; refusing duplicate launch" >&2
  exit 2
fi
mkdir -p "$TRAIN_ROOT/eventgenesis" "$TRAIN_ROOT/v2e"
env CUDA_VISIBLE_DEVICES=2 STAGE4_PHYSICAL_GPU=2 STAGE4_REPO_COMMIT="$COMMIT" OMP_NUM_THREADS=4 \
  "$ENV_BIN/python" -u "$SIM/scripts/train_steflow_stage4.py" \
  --model-root "$MODEL_ROOT" --data-root "$ADAPTER_ROOT" --config "$CONFIG" \
  --run-root "$TRAIN_ROOT" --generator eventgenesis --seed 0 \
  > "$TRAIN_ROOT/eventgenesis/train.stdout.log" 2> "$TRAIN_ROOT/eventgenesis/train.stderr.log" &
eg_pid=$!
echo "$eg_pid" > "$TRAIN_ROOT/eventgenesis/train.pid"
env CUDA_VISIBLE_DEVICES=4 STAGE4_PHYSICAL_GPU=4 STAGE4_REPO_COMMIT="$COMMIT" OMP_NUM_THREADS=4 \
  "$ENV_BIN/python" -u "$SIM/scripts/train_steflow_stage4.py" \
  --model-root "$MODEL_ROOT" --data-root "$ADAPTER_ROOT" --config "$CONFIG" \
  --run-root "$TRAIN_ROOT" --generator v2e --seed 0 \
  > "$TRAIN_ROOT/v2e/train.stdout.log" 2> "$TRAIN_ROOT/v2e/train.stderr.log" &
v2e_pid=$!
echo "$v2e_pid" > "$TRAIN_ROOT/v2e/train.pid"
printf '%s\n' "eventgenesis=$eg_pid" "v2e=$v2e_pid" > "$TRAIN_ROOT/training_pids.txt"

eg_rc=0; v2e_train_rc=0
wait "$eg_pid" || eg_rc=$?
wait "$v2e_pid" || v2e_train_rc=$?
echo "$eg_rc" > "$TRAIN_ROOT/eventgenesis/train.exit_code.txt"
echo "$v2e_train_rc" > "$TRAIN_ROOT/v2e/train.exit_code.txt"
if [[ "$eg_rc" -ne 0 || "$v2e_train_rc" -ne 0 ]]; then exit 1; fi
exit 0
