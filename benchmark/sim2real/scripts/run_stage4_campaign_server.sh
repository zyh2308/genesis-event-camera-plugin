#!/usr/bin/env bash
# Detached Stage-4 shared bank -> paired V2E -> adapter -> paired STE seed-0 run.
set -u

PROJECT=/home/zhaoyuhan/Eventbased_WAM
REPO="$PROJECT/repositories/gec-sim2real"
SIM="$REPO/benchmark/sim2real"
COMMIT=$(git -C "$REPO" rev-parse HEAD)
CONFIG="$SIM/configs/stage4_training_launch_v1.json"
CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
RUN_ID=${STAGE4_RUN_ID:-20261016-stage4-flowdt1-seed0-v3}
RUN_ROOT="$PROJECT/runs/sim2real/$RUN_ID"
BANK_ROOT="$PROJECT/data/sim2real/stage4/training_bank_v1_32x2s"
V2E_ROOT="$RUN_ROOT/v2e_events"
ADAPTER_ROOT="$PROJECT/data/sim2real/stage4/$RUN_ID/steflow_30ms"
MODEL_ROOT="$PROJECT/external/sources/downstream/STE-FlowNet"
CONDA_ROOT=/home/zhaoyuhan/miniconda3
ENV_BIN="$CONDA_ROOT/envs/Eventbased_WAM/bin"
LOG="$RUN_ROOT/campaign.log"

if [[ -e "$RUN_ROOT" || -e "$BANK_ROOT" || -e "$ADAPTER_ROOT" ]]; then
  echo "refusing existing Stage-4 output path" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT" "$(dirname "$BANK_ROOT")" "$(dirname "$ADAPTER_ROOT")"
printf '%s\n' "$$" > "$RUN_ROOT/campaign.pid"
date -u +%FT%TZ > "$RUN_ROOT/started_utc.txt"
trap 'rc=$?; if [[ -f "$RUN_ROOT/status_watcher.pid" ]]; then watcher=$(cat "$RUN_ROOT/status_watcher.pid"); if kill -0 "$watcher" 2>/dev/null; then kill "$watcher"; fi; fi; echo "$rc" > "$RUN_ROOT/exit_code.txt"; date -u +%FT%TZ > "$RUN_ROOT/finished_utc.txt"' EXIT
cd "$REPO" || exit 2

{
  date -u +%FT%TZ
  hostname
  printf 'repo=%s\nbranch=%s\ncommit=%s\n' "$REPO" "$(git branch --show-current)" "$COMMIT"
  printf 'config=%s\nconfig_sha256=%s\n' "$CONFIG" "$CONFIG_SHA"
  git status --short --branch
  nvidia-smi
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv
  free -h
  df -h /home
} > "$RUN_ROOT/preflight.txt" 2>&1

(cd "$SIM" && sha256sum -c protocol/training_launch_v1.sha256) \
  > "$RUN_ROOT/registered_config_verification.txt" 2>&1 || exit 4

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

for gpu in 1 2 4 5; do check_idle_gpu "$gpu" || exit 3; done

source "$CONDA_ROOT/bin/activate" Eventbased_WAM || exit 90
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg" GS_CACHE_FILE_PATH="$RUN_ROOT/cache/genesis"
export TI_OFFLINE_CACHE=1 TI_OFFLINE_CACHE_FILE_PATH="$RUN_ROOT/cache/taichi"
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PROJECT/runtime/sim2real/genesis_hdr_v2:$REPO"
mkdir -p "$XDG_CACHE_HOME" "$GS_CACHE_FILE_PATH" "$TI_OFFLINE_CACHE_FILE_PATH"

"$ENV_BIN/python" - <<'PY' > "$RUN_ROOT/environment_versions.txt" 2>&1
import importlib.metadata as md
import platform, sys
print("python", sys.version)
print("platform", platform.platform())
for name in ("torch", "genesis-world", "numpy", "h5py", "opencv-python", "scipy"):
    try: print(name, md.version(name))
    except md.PackageNotFoundError: print(name, "not-installed")
PY
{
  sha256sum "$CONFIG" "$SIM/scripts/generate_stage4_flow_bank.py" \
    "$SIM/scripts/run_stage4_v2e_shard.py" "$SIM/scripts/finalize_stage4_v2e.py" \
    "$SIM/scripts/prepare_stage4_steflow_bank.py" "$SIM/scripts/train_steflow_stage4.py" \
    "$SIM/scripts/verify_v2e_source.py" "$SIM/scripts/watch_stage4_training_status.py" \
    "$SIM/scripts/run_stage4_campaign_server.sh" "$SIM/scripts/start_stage4_campaign_tmux.sh" \
    "$SIM/manifests/v2e_source_pin.json" \
    "$PROJECT/external/sources/v2e-cac6384445/v2e-pinned/v2e.py" \
    "$PROJECT/external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt"
} > "$RUN_ROOT/source_hashes.txt" 2>&1
"$ENV_BIN/python" "$SIM/scripts/verify_source_bundle.py" \
  --manifest "$SIM/manifests/downstream_source_bundle.json" \
  --source-root "$PROJECT/external/sources/downstream" \
  > "$RUN_ROOT/downstream_source_verification.json" 2>&1 || exit 4
EXPECTED_V2E_COMMIT=$("$ENV_BIN/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["generators"]["v2e"]["commit"])' "$CONFIG")
if ! "$ENV_BIN/python" "$SIM/scripts/verify_v2e_source.py" \
  --source-root "$PROJECT/external/sources/v2e-cac6384445/v2e-pinned" \
  --manifest "$SIM/manifests/v2e_source_pin.json" --expected-commit "$EXPECTED_V2E_COMMIT" \
  --checkpoint "$PROJECT/external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt" \
  > "$RUN_ROOT/v2e_source_verification.json" 2>&1; then exit 4; fi
V2E_COMMIT="$EXPECTED_V2E_COMMIT"
printf 'STE-FlowNet commit=%s\nV2E commit=%s\n' \
  "$("$ENV_BIN/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_locks"]["STE_FlowNet_commit"])' "$CONFIG")" \
  "$V2E_COMMIT" >> "$RUN_ROOT/source_hashes.txt"

python -u "$SIM/scripts/watch_stage4_training_status.py" \
  --repo-root "$REPO" --bank-root "$BANK_ROOT" --adapter-root "$ADAPTER_ROOT" \
  --run-root "$RUN_ROOT" --config "$CONFIG" --config-sha "$CONFIG_SHA" \
  --branch exp/public-sim2real-v1 --commit "$COMMIT" --poll-seconds 15 \
  > "$RUN_ROOT/status_watcher.log" 2>&1 &
echo $! > "$RUN_ROOT/status_watcher.pid"

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

export CUDA_VISIBLE_DEVICES=1 STAGE4_PHYSICAL_GPU=1 STAGE4_REPO_COMMIT="$COMMIT"
if ! run_step shared_bank_generation "$ENV_BIN/python" -u "$SIM/scripts/generate_stage4_flow_bank.py" \
  --config "$CONFIG" --output-root "$BANK_ROOT"; then exit "$(cat "$RUN_ROOT/shared_bank_generation.exit_code.txt")"; fi

for gpu in 1 2 4 5; do check_idle_gpu "$gpu" || exit 3; done
mkdir -p "$V2E_ROOT/gpu1" "$V2E_ROOT/gpu2" "$V2E_ROOT/gpu4" "$V2E_ROOT/gpu5"
declare -a shard_pids=()
declare -a shard_gpus=(1 2 4 5)
for shard_index in 0 1 2 3; do
  gpu=${shard_gpus[$shard_index]}
  stratum=$shard_index
  episodes=()
  for local in 0 1 2 3 4 5 6 7; do episodes+=("s${stratum}_$(printf '%02d' "$local")"); done
  "$ENV_BIN/python" -u "$SIM/scripts/run_stage4_v2e_shard.py" \
    --project-root "$PROJECT" --bank-root "$BANK_ROOT" --output-root "$V2E_ROOT/gpu$gpu" \
    --gpu "$gpu" --episodes "${episodes[@]}" \
    > "$V2E_ROOT/gpu$gpu/worker.stdout.log" 2> "$V2E_ROOT/gpu$gpu/worker.stderr.log" &
  shard_pid=$!
  echo "$shard_pid" > "$V2E_ROOT/gpu$gpu/worker.pid"
  shard_pids+=("$shard_pid")
done
printf '%s\n' "${shard_pids[@]}" > "$RUN_ROOT/v2e_shard_pids.txt"
v2e_rc=0
for i in 0 1 2 3; do
  shard_rc=0
  wait "${shard_pids[$i]}" || shard_rc=$?
  echo "$shard_rc" > "$V2E_ROOT/gpu${shard_gpus[$i]}/worker.exit_code.txt"
  if [[ "$shard_rc" -ne 0 && "$v2e_rc" -eq 0 ]]; then v2e_rc=$shard_rc; fi
done
if [[ "$v2e_rc" -ne 0 ]]; then exit "$v2e_rc"; fi
if ! run_step v2e_finalize "$ENV_BIN/python" -u "$SIM/scripts/finalize_stage4_v2e.py" \
  --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT"; then exit "$(cat "$RUN_ROOT/v2e_finalize.exit_code.txt")"; fi

if ! run_step steflow_adapter "$ENV_BIN/python" -u "$SIM/scripts/prepare_stage4_steflow_bank.py" \
  --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" --output-root "$ADAPTER_ROOT"; then exit "$(cat "$RUN_ROOT/steflow_adapter.exit_code.txt")"; fi

# Training uses equal-model V100S devices 2 and 4. Recheck immediately before launch.
check_idle_gpu 2 || exit 3
check_idle_gpu 4 || exit 3
mkdir -p "$RUN_ROOT/flow_dt1_seed0/eventgenesis" "$RUN_ROOT/flow_dt1_seed0/v2e"
env CUDA_VISIBLE_DEVICES=2 STAGE4_PHYSICAL_GPU=2 STAGE4_REPO_COMMIT="$COMMIT" OMP_NUM_THREADS=4 \
  "$ENV_BIN/python" -u "$SIM/scripts/train_steflow_stage4.py" \
  --model-root "$MODEL_ROOT" --data-root "$ADAPTER_ROOT" --config "$CONFIG" \
  --run-root "$RUN_ROOT/flow_dt1_seed0" --generator eventgenesis --seed 0 \
  > "$RUN_ROOT/flow_dt1_seed0/eventgenesis/train.stdout.log" \
  2> "$RUN_ROOT/flow_dt1_seed0/eventgenesis/train.stderr.log" &
eg_pid=$!
echo "$eg_pid" > "$RUN_ROOT/flow_dt1_seed0/eventgenesis/train.pid"
env CUDA_VISIBLE_DEVICES=4 STAGE4_PHYSICAL_GPU=4 STAGE4_REPO_COMMIT="$COMMIT" OMP_NUM_THREADS=4 \
  "$ENV_BIN/python" -u "$SIM/scripts/train_steflow_stage4.py" \
  --model-root "$MODEL_ROOT" --data-root "$ADAPTER_ROOT" --config "$CONFIG" \
  --run-root "$RUN_ROOT/flow_dt1_seed0" --generator v2e --seed 0 \
  > "$RUN_ROOT/flow_dt1_seed0/v2e/train.stdout.log" \
  2> "$RUN_ROOT/flow_dt1_seed0/v2e/train.stderr.log" &
v2e_pid=$!
echo "$v2e_pid" > "$RUN_ROOT/flow_dt1_seed0/v2e/train.pid"
printf '%s\n' "eventgenesis=$eg_pid" "v2e=$v2e_pid" > "$RUN_ROOT/flow_dt1_seed0/training_pids.txt"

eg_rc=0; v2e_train_rc=0
wait "$eg_pid" || eg_rc=$?
wait "$v2e_pid" || v2e_train_rc=$?
echo "$eg_rc" > "$RUN_ROOT/flow_dt1_seed0/eventgenesis/train.exit_code.txt"
echo "$v2e_train_rc" > "$RUN_ROOT/flow_dt1_seed0/v2e/train.exit_code.txt"
if [[ "$eg_rc" -ne 0 || "$v2e_train_rc" -ne 0 ]]; then exit 1; fi
echo 0 > "$RUN_ROOT/exit_code.txt"
date -u +%FT%TZ > "$RUN_ROOT/finished_utc.txt"
