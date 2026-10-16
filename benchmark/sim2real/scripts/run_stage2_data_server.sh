#!/usr/bin/env bash
# Server-only Stage-2 shared physical bank, paired V2E, and task adapters.
set -u

PROJECT_DIR=/home/zhaoyuhan/Eventbased_WAM
REPO_DIR="$PROJECT_DIR/repositories/gec-sim2real"
RUN_ID="${1:-20261016-s2-pilot-data-v1}"
RUN_DIR="$PROJECT_DIR/runs/sim2real/$RUN_ID"
DATA_ROOT="$PROJECT_DIR/data/sim2real/stage2/20261016-s2-pilot-v1"
V2E_SOURCE="$PROJECT_DIR/external/sources/v2e-cac6384445/v2e-pinned"
SLOMO_CHECKPOINT="$PROJECT_DIR/external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt"
CONFIG="$REPO_DIR/benchmark/sim2real/configs/stage2_pilot.json"

if [[ -e "$RUN_DIR" || -e "$DATA_ROOT" ]]; then
  echo "refusing existing Stage-2 paths: $RUN_DIR or $DATA_ROOT" >&2
  exit 2
fi
mkdir -p "$RUN_DIR" "$DATA_ROOT"
printf '%s\n' "$$" > "$RUN_DIR/pid.txt"
trap 'rc=$?; echo "$rc" > "$RUN_DIR/exit_code.txt"; date -u +%FT%TZ > "$RUN_DIR/finished_utc.txt"' EXIT
date -u +%FT%TZ > "$RUN_DIR/started_utc.txt"
cd "$REPO_DIR" || exit 2
{
  date -u +%FT%TZ
  hostname
  pwd
  git rev-parse HEAD
  git status --short --branch
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
  nvidia-smi
  df -h /home
} > "$RUN_DIR/preflight.txt" 2>&1

IFS=, read -r gpu1_util gpu1_used <<< "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | sed -n '2p')"
gpu1_util=$(echo "${gpu1_util:-}" | tr -d ' ')
gpu1_used=$(echo "${gpu1_used:-}" | tr -d ' ')
if [[ -z "$gpu1_used" || -z "$gpu1_util" || "$gpu1_used" -gt 1024 || "$gpu1_util" -gt 5 ]]; then
  echo "GPU 1 is not idle (util=${gpu1_util:-unknown}%, memory=${gpu1_used:-unknown} MiB); refusing to launch" | tee "$RUN_DIR/preflight_error.txt" >&2
  echo 3 > "$RUN_DIR/exit_code.txt"
  date -u +%FT%TZ > "$RUN_DIR/finished_utc.txt"
  exit 3
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh" || exit 90
conda activate Eventbased_WAM || exit 91
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$RUN_DIR/cache/xdg"
export GS_CACHE_FILE_PATH="$RUN_DIR/cache/genesis"
export TI_OFFLINE_CACHE=1
export TI_OFFLINE_CACHE_FILE_PATH="$RUN_DIR/cache/taichi"
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PROJECT_DIR/runtime/sim2real/genesis_hdr_v2:$REPO_DIR"
mkdir -p "$XDG_CACHE_HOME" "$GS_CACHE_FILE_PATH" "$TI_OFFLINE_CACHE_FILE_PATH"

{
  python - <<'PY'
import importlib.metadata as md, sys, torch, h5py, numpy, genesis
print("python", sys.version)
for name in ("torch", "genesis-world", "h5py", "numpy", "opencv-python", "pandas"):
    try: print(name, md.version(name))
    except md.PackageNotFoundError: pass
print("torch_cuda", torch.version.cuda, torch.cuda.is_available())
print("genesis_module", genesis.__file__)
PY
  sha256sum "$CONFIG" \
    "$REPO_DIR/benchmark/sim2real/scripts/run_shared_trajectory.py" \
    "$REPO_DIR/benchmark/sim2real/scripts/prepare_e2vid_stage2.py" \
    "$REPO_DIR/benchmark/sim2real/scripts/prepare_steflow_stage2.py" \
    "$V2E_SOURCE/v2e.py" "$SLOMO_CHECKPOINT"
  git -C "$V2E_SOURCE" rev-parse HEAD
} > "$RUN_DIR/environment_and_hashes.txt" 2>&1
cp "$CONFIG" "$RUN_DIR/stage2_pilot.json"

run_step() {
  local label="$1"; shift
  printf '%q ' "$@" > "$RUN_DIR/$label.command.txt"
  printf '\n' >> "$RUN_DIR/$label.command.txt"
  if "$@" > "$RUN_DIR/$label.stdout.log" 2>&1; then
    echo 0 > "$RUN_DIR/$label.exit_code.txt"
    return 0
  else
    local status=$?
    echo "$status" > "$RUN_DIR/$label.exit_code.txt"
    return "$status"
  fi
}

if ! run_step trajectory python -u \
  "$REPO_DIR/benchmark/sim2real/scripts/run_shared_trajectory.py" \
  --config "$CONFIG" --output-dir "$DATA_ROOT/shared"; then
  status=$(cat "$RUN_DIR/trajectory.exit_code.txt")
  echo "$status" > "$RUN_DIR/exit_code.txt"
  date -u +%FT%TZ > "$RUN_DIR/finished_utc.txt"
  exit "$status"
fi

mkdir -p "$DATA_ROOT/v2e_events"
episodes=(train_00 train_01 train_02 train_03 validation_00 validation_01)
seeds=(42 43 44 45 46 47)
for i in "${!episodes[@]}"; do
  episode="${episodes[$i]}"
  input_dir="$DATA_ROOT/shared/v2e_input/$episode"
  output_h5="$DATA_ROOT/v2e_events/$episode.h5"
  output_folder="$DATA_ROOT/v2e_events/$episode"
  command=(python -u "$V2E_SOURCE/v2e.py" --input "$input_dir" \
    --input_frame_rate 100 --auto_timestamp_resolution False \
    --timestamp_resolution 0.000025 --slomo_model "$SLOMO_CHECKPOINT" \
    --output_folder "$output_folder" --unique_output_folder False \
    --skip_video_output --no_preview --dvs_h5 "$output_h5" \
    --dvs_aedat2 None --dvs_aedat4 None --dvs_text None \
    --pos_thres 0.2 --neg_thres 0.2 --sigma_thres 0.03 \
    --cutoff_hz 300 --leak_rate_hz 0.01 --leak_jitter_fraction 0.1 \
    --noise_rate_cov_decades 0.1 --shot_noise_rate_hz 0.001 \
    --refractory_period 0.0005 --dvs_emulator_seed "${seeds[$i]}")
  if ! run_step "v2e_$episode" "${command[@]}"; then
    status=$(cat "$RUN_DIR/v2e_$episode.exit_code.txt")
    echo "$status" > "$RUN_DIR/exit_code.txt"
    date -u +%FT%TZ > "$RUN_DIR/finished_utc.txt"
    exit "$status"
  fi
done

if ! run_step e2vid_conversion python -u \
  "$REPO_DIR/benchmark/sim2real/scripts/prepare_e2vid_stage2.py" \
  --trajectory-h5 "$DATA_ROOT/shared/shared_trajectory.h5" \
  --v2e-root "$DATA_ROOT/v2e_events" --output-root "$DATA_ROOT/e2vid_inputs"; then
  status=$(cat "$RUN_DIR/e2vid_conversion.exit_code.txt")
  echo "$status" > "$RUN_DIR/exit_code.txt"
  date -u +%FT%TZ > "$RUN_DIR/finished_utc.txt"
  exit "$status"
fi

if ! run_step steflow_conversion python -u \
  "$REPO_DIR/benchmark/sim2real/scripts/prepare_steflow_stage2.py" \
  --trajectory-h5 "$DATA_ROOT/shared/shared_trajectory.h5" \
  --v2e-root "$DATA_ROOT/v2e_events" --output-root "$DATA_ROOT/steflow_inputs"; then
  status=$(cat "$RUN_DIR/steflow_conversion.exit_code.txt")
  echo "$status" > "$RUN_DIR/exit_code.txt"
  date -u +%FT%TZ > "$RUN_DIR/finished_utc.txt"
  exit "$status"
fi

echo 0 > "$RUN_DIR/exit_code.txt"
date -u +%FT%TZ > "$RUN_DIR/finished_utc.txt"
echo "Stage-2 shared data and both downstream adapters completed: $DATA_ROOT"
