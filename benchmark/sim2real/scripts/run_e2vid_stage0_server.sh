#!/usr/bin/env bash
set -u

PROJECT_DIR=/home/zhaoyuhan/Eventbased_WAM
REPO_DIR="$PROJECT_DIR/repositories/gec-sim2real"
RUN_ID="${1:-20261016-s0-e2vid-stage0-v2}"
RUN_DIR="$PROJECT_DIR/runs/sim2real/$RUN_ID"
DATA_ROOT="$PROJECT_DIR/data/sim2real/stage0/e2vid_loader_20261016-v1"
MODEL_ROOT="$PROJECT_DIR/external/sources/downstream/event_cnn_minimal"

mkdir -p "$RUN_DIR"
echo "$$" > "$RUN_DIR/pid.txt"
cd "$REPO_DIR" || exit 2

{
  date -u +%FT%TZ
  hostname
  pwd
  git rev-parse HEAD
  git status --short --branch
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
  nvidia-smi
} > "$RUN_DIR/preflight.txt" 2>&1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate Eventbased_WAM
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

{
  printf 'python -u %s --model-root %s --data-root %s --output-root %s --seed 71\n' \
    "$REPO_DIR/benchmark/sim2real/scripts/run_e2vid_stage0_smoke.py" \
    "$MODEL_ROOT" "$DATA_ROOT" "$RUN_DIR"
  python - <<'PY'
import cv2, h5py, pandas, torch
print('python_environment', __import__('sys').version)
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('cv2', cv2.__version__, 'h5py', h5py.__version__, 'pandas', pandas.__version__)
PY
  sha256sum \
    "$REPO_DIR/benchmark/sim2real/scripts/run_e2vid_stage0_smoke.py" \
    "$REPO_DIR/benchmark/sim2real/scripts/prepare_e2vid_stage0.py" \
    "$REPO_DIR/benchmark/sim2real/configs/e2vid_stage0_smoke.json"
} > "$RUN_DIR/command_and_environment.txt" 2>&1

python -u "$REPO_DIR/benchmark/sim2real/scripts/run_e2vid_stage0_smoke.py" \
  --model-root "$MODEL_ROOT" \
  --data-root "$DATA_ROOT" \
  --output-root "$RUN_DIR" \
  --seed 71 > "$RUN_DIR/stdout.log" 2>&1
status=$?
echo "$status" > "$RUN_DIR/exit_code.txt"
date -u +%FT%TZ > "$RUN_DIR/finished_utc.txt"
exit "$status"
