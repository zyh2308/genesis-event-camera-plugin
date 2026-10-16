#!/usr/bin/env bash
# Detached g1 runner for the paired synthetic-only STE-FlowNet Stage-2 pilot.
set -u
PROJECT_DIR=/home/zhaoyuhan/Eventbased_WAM
REPO_DIR="$PROJECT_DIR/repositories/gec-sim2real"
RUN_ID="${1:-20261016-s2-steflow-v1}"
RUN_DIR="$PROJECT_DIR/runs/sim2real/$RUN_ID"
DATA_ROOT="$PROJECT_DIR/data/sim2real/stage2/20261016-s2-pilot-v1/steflow_inputs"
MODEL_ROOT="$PROJECT_DIR/external/sources/downstream/STE-FlowNet"

if [[ -e "$RUN_DIR" ]]; then echo "refusing existing run directory: $RUN_DIR" >&2; exit 2; fi
mkdir -p "$RUN_DIR"
printf '%s\n' "$$" > "$RUN_DIR/pid.txt"
trap 'rc=$?; echo "$rc" > "$RUN_DIR/exit_code.txt"; date -u +%FT%TZ > "$RUN_DIR/finished_utc.txt"' EXIT
date -u +%FT%TZ > "$RUN_DIR/started_utc.txt"
cd "$REPO_DIR" || exit 2
{
  date -u +%FT%TZ; hostname; pwd; git rev-parse HEAD; git status --short --branch
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
  nvidia-smi; df -h /home
} > "$RUN_DIR/preflight.txt" 2>&1
IFS=, read -r gpu1_util gpu1_used <<< "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | sed -n '2p')"
gpu1_util=$(echo "${gpu1_util:-}" | tr -d ' ')
gpu1_used=$(echo "${gpu1_used:-}" | tr -d ' ')
if [[ -z "$gpu1_used" || -z "$gpu1_util" || "$gpu1_used" -gt 1024 || "$gpu1_util" -gt 5 ]]; then
  echo "GPU 1 is not idle (util=${gpu1_util:-unknown}%, memory=${gpu1_used:-unknown} MiB); refusing to launch" > "$RUN_DIR/preflight_error.txt"
  exit 3
fi
if [[ ! -d "$DATA_ROOT/eventgenesis/train" || ! -d "$DATA_ROOT/v2e/validation" ]]; then
  echo "Stage-2 STE-FlowNet paired rollout data not found: $DATA_ROOT" > "$RUN_DIR/preflight_error.txt"
  exit 4
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh" || exit 90
conda activate Eventbased_WAM || exit 91
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
{
  python - <<'PY'
import importlib.metadata as md, sys, torch, h5py, numpy, cv2
print("python", sys.version)
for name in ("torch", "h5py", "numpy"): print(name, md.version(name))
print("opencv", cv2.__version__)
print("torch_cuda", torch.version.cuda, torch.cuda.is_available())
PY
  git -C "$MODEL_ROOT" rev-parse HEAD
  sha256sum "$REPO_DIR/benchmark/sim2real/scripts/run_steflow_stage2_pilot.py" \
    "$REPO_DIR/benchmark/sim2real/scripts/prepare_steflow_stage2.py"
} > "$RUN_DIR/environment_and_hashes.txt" 2>&1

command=(python -u "$REPO_DIR/benchmark/sim2real/scripts/run_steflow_stage2_pilot.py" \
  --model-root "$MODEL_ROOT" --data-root "$DATA_ROOT" \
  --output-root "$RUN_DIR/training" --seed 71 --epochs 2)
printf '%q ' "${command[@]}" > "$RUN_DIR/command.txt"; printf '\n' >> "$RUN_DIR/command.txt"
if "${command[@]}" > "$RUN_DIR/stdout.log" 2>&1; then status=0; else status=$?; fi
exit "$status"
