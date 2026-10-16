#!/usr/bin/env bash
# Restart paired seed-0 STE training after a pre-update loss API failure.
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
MODEL_ROOT="$PROJECT/external/sources/downstream/STE-FlowNet"
TRAIN_ROOT="$RUN_ROOT/flow_dt1_seed0"
FAILED_ARCHIVE="$TRAIN_ROOT/failed_attempt01_upstream_loss_numpy_cuda"
CONDA_ROOT=/home/zhaoyuhan/miniconda3
ENV_BIN="$CONDA_ROOT/envs/Eventbased_WAM/bin"

if [[ ! -f "$BANK_ROOT/bank_manifest.json" || ! -f "$V2E_ROOT/v2e_complete_manifest.json" \
   || ! -f "$ADAPTER_ROOT/training_bank_manifest.json" ]]; then
  echo "complete shared bank, V2E manifest, or adapter is missing" >&2
  exit 2
fi
if [[ ! -f "$RUN_ROOT/steflow_adapter_retry1.exit_code.txt" || "$(cat "$RUN_ROOT/steflow_adapter_retry1.exit_code.txt")" != "0" \
   || ! -f "$RUN_ROOT/adapter_retry1.exit_code.txt" || "$(cat "$RUN_ROOT/adapter_retry1.exit_code.txt")" != "1" ]]; then
  echo "expected adapter retry1 success followed by training failure before retry" >&2
  exit 2
fi
for arm in eventgenesis v2e; do
  arm_root="$TRAIN_ROOT/$arm"
  [[ -f "$arm_root/train.exit_code.txt" && "$(cat "$arm_root/train.exit_code.txt")" == "1" ]] || {
    echo "$arm does not have the expected failed pre-update attempt" >&2; exit 2;
  }
  [[ -f "$arm_root/training_metrics.jsonl" && ! -s "$arm_root/training_metrics.jsonl" ]] || {
    echo "$arm has training metrics; refusing a fresh initialization" >&2; exit 2;
  }
  if [[ -e "$arm_root/last.pt" || -e "$arm_root/update_000020.pt" || -e "$arm_root/synthetic_best.pt" ]]; then
    echo "$arm has a checkpoint; refusing fresh initialization" >&2
    exit 2
  fi
  grep -Fq "TypeError: can't convert cuda:0 device type tensor to numpy" "$arm_root/train.stderr.log" || {
    echo "$arm failure does not match the recorded upstream loss API error" >&2; exit 2;
  }
  if [[ -f "$arm_root/train.pid" ]]; then
    pid=$(cat "$arm_root/train.pid")
    if kill -0 "$pid" 2>/dev/null; then
      echo "$arm process $pid is still alive; refusing to archive or relaunch" >&2
      exit 3
    fi
  fi
done
if [[ -e "$FAILED_ARCHIVE" || -e "$RUN_ROOT/training_retry2.started_utc.txt" ]]; then
  echo "failure archive or training retry already exists; inspect before retrying" >&2
  exit 2
fi

cd "$REPO" || exit 2
COMMIT=$(git rev-parse HEAD)
BRANCH=$(git branch --show-current)
CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
mkdir -p "$FAILED_ARCHIVE"
mv "$TRAIN_ROOT/eventgenesis" "$FAILED_ARCHIVE/eventgenesis"
mv "$TRAIN_ROOT/v2e" "$FAILED_ARCHIVE/v2e"
if [[ -f "$TRAIN_ROOT/training_pids.txt" ]]; then mv "$TRAIN_ROOT/training_pids.txt" "$FAILED_ARCHIVE/training_pids.txt"; fi
printf '%s\n' "$FAILED_ARCHIVE" > "$RUN_ROOT/failed_attempt01_archive_path.txt"
printf '%s\n' "$$" > "$RUN_ROOT/training_retry2.pid"
date -u +%FT%TZ > "$RUN_ROOT/training_retry2.started_utc.txt"
trap 'rc=$?; if [[ -f "$RUN_ROOT/training_retry2_watcher.pid" ]]; then watcher=$(cat "$RUN_ROOT/training_retry2_watcher.pid"); if kill -0 "$watcher" 2>/dev/null; then kill "$watcher"; fi; fi; echo "$rc" > "$RUN_ROOT/training_retry2.exit_code.txt"; date -u +%FT%TZ > "$RUN_ROOT/training_retry2.finished_utc.txt"' EXIT

{
  date -u +%FT%TZ
  hostname
  printf 'repo=%s\nbranch=%s\ncommit=%s\nconfig=%s\nconfig_sha256=%s\n' "$REPO" "$BRANCH" "$COMMIT" "$CONFIG" "$CONFIG_SHA"
  git status --short --branch
  sha256sum "$SIM/scripts/run_steflow_stage2_pilot.py" "$SIM/scripts/train_steflow_stage4.py" "$SIM/scripts/prepare_stage4_steflow_bank.py" "$SIM/scripts/stage4_v2e_artifacts.py"
  nvidia-smi
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv
  free -h
  df -h /home
} > "$RUN_ROOT/training_retry2_preflight.txt" 2>&1
if ! (cd "$SIM" && sha256sum -c "protocol/$CONFIG_CHECKSUM") \
  > "$RUN_ROOT/training_retry2_config_verification.txt" 2>&1; then
  exit 4
fi

source "$CONDA_ROOT/bin/activate" Eventbased_WAM || exit 90
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python -u "$SIM/scripts/watch_stage4_training_status.py" \
  --repo-root "$REPO" --bank-root "$BANK_ROOT" --adapter-root "$ADAPTER_ROOT" \
  --run-root "$RUN_ROOT" --config "$CONFIG" --config-sha "$CONFIG_SHA" \
  --branch "$BRANCH" --commit "$COMMIT" --poll-seconds 15 \
  > "$RUN_ROOT/training_retry2_status_watcher.log" 2>&1 &
echo $! > "$RUN_ROOT/training_retry2_watcher.pid"

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
