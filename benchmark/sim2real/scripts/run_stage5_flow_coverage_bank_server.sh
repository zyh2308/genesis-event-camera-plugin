#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 SOURCE_ROOT CONFIG_JSON BANK_ROOT RUN_ROOT PHYSICAL_GPU" >&2
  exit 2
fi

SOURCE_ROOT=$1
CONFIG_JSON=$2
BANK_ROOT=$3
RUN_ROOT=$4
PHYSICAL_GPU=$5
PYTHON_BIN=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
GENERATOR=$SOURCE_ROOT/benchmark/sim2real/scripts/generate_stage5_flow_coverage_bank.py

if [[ ! -d "$SOURCE_ROOT" || ! -f "$SOURCE_ROOT/SOURCE_FILE_HASHES.json" || ! -f "$CONFIG_JSON" \
      || ! -f "$GENERATOR" || ! -x "$PYTHON_BIN" ]]; then
  echo "registered source/config/Genesis Python inputs are missing" >&2
  exit 2
fi
if [[ -e "$BANK_ROOT" || -e "$RUN_ROOT" ]]; then
  echo "refusing to overwrite the bank or run root" >&2
  exit 2
fi
if [[ ! "$PHYSICAL_GPU" =~ ^[0-9]+$ ]]; then
  echo "physical GPU must be a numeric device index" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/cache/xdg" "$RUN_ROOT/cache/genesis" "$RUN_ROOT/cache/taichi"
STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$STARTED_UTC" > "$RUN_ROOT/started_utc.txt"
printf '%s\n' "$$" > "$RUN_ROOT/pid.txt"
printf '%q ' "$PYTHON_BIN" -u "$GENERATOR" --config "$CONFIG_JSON" --output-root "$BANK_ROOT" \
  > "$RUN_ROOT/command.txt"
printf '\n' >> "$RUN_ROOT/command.txt"
trap 'rc=$?; date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/finished_utc.txt"; printf "%s\n" "$rc" > "$RUN_ROOT/exit_code.txt"; exit "$rc"' EXIT

export CUDA_VISIBLE_DEVICES=$PHYSICAL_GPU
export STAGE5_PHYSICAL_GPU=$PHYSICAL_GPU
export PYTHONPATH=$SOURCE_ROOT
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYOPENGL_PLATFORM=egl
export XDG_CACHE_HOME=$RUN_ROOT/cache/xdg
export GS_CACHE_FILE_PATH=$RUN_ROOT/cache/genesis
export TI_OFFLINE_CACHE=1
export TI_OFFLINE_CACHE_FILE_PATH=$RUN_ROOT/cache/taichi

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  hostname
  "$PYTHON_BIN" -c 'import sys,torch,genesis,h5py,numpy,cv2; print("python",sys.version); print("torch",torch.__version__); print("genesis",genesis.__version__); print("h5py",h5py.__version__); print("numpy",numpy.__version__); print("opencv",cv2.__version__); print("cuda",torch.cuda.is_available()); print("visible_devices",torch.cuda.device_count()); print("gpu_name",torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
  sha256sum "$CONFIG_JSON" "$SOURCE_ROOT/SOURCE_FILE_HASHES.json" "$GENERATOR"
  "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("source_archive_commit",json.load(open(sys.argv[2]))["commit"]); print("config_schema",d["schema"]); print("real_scoring_enabled",d["real_scoring_enabled"])' "$CONFIG_JSON" "$SOURCE_ROOT/SOURCE_FILE_HASHES.json"
} > "$RUN_ROOT/environment_and_hashes.txt" 2>&1

"$PYTHON_BIN" -u "$GENERATOR" --config "$CONFIG_JSON" --output-root "$BANK_ROOT" \
  > "$RUN_ROOT/generation.log" 2>&1
