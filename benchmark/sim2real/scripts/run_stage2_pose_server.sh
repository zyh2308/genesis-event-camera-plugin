#!/usr/bin/env bash
# Run the event-only SingleShotPose pilot on the already prepared paired bank.
set -u

PROJECT_DIR=/home/zhaoyuhan/Eventbased_WAM
REPO_DIR="$PROJECT_DIR/repositories/gec-sim2real"
RUN_ID="${1:-20261016-s2-pose-pilot-v1}"
RUN_DIR="$PROJECT_DIR/runs/sim2real/$RUN_ID"
DATA_ROOT="$PROJECT_DIR/data/sim2real/stage2/20261016-s2-pilot-v1"
PILOT_CONFIG="$REPO_DIR/benchmark/sim2real/configs/pose_stage2_pilot.json"
DATA_CONFIG="$DATA_ROOT/shared/config.json"
TRAJECTORY="$DATA_ROOT/shared/shared_trajectory.h5"
POSE_SOURCE="$PROJECT_DIR/external/sources/downstream/SingleShotPose"

if [[ -e "$RUN_DIR" ]]; then
  echo "refusing existing Stage-2 pose run path: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
printf '%s\n' "$" > "$RUN_DIR/pid.txt"
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
  exit 3
fi
for required in "$TRAJECTORY" "$DATA_ROOT/v2e_events/train_00.h5" "$DATA_ROOT/v2e_events/validation_00.h5" "$PILOT_CONFIG" "$POSE_SOURCE/darknet.py"; do
  if [[ ! -e "$required" ]]; then
    echo "required pose pilot input is missing: $required" | tee "$RUN_DIR/preflight_error.txt" >&2
    exit 4
  fi
done

source "$HOME/miniconda3/etc/profile.d/conda.sh" || exit 90
conda activate Eventbased_WAM || exit 91
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$RUN_DIR/cache/xdg"
mkdir -p "$XDG_CACHE_HOME"

{
  python - <<'PY'
import importlib.metadata as md, sys, torch, h5py, numpy, cv2
print("python", sys.version)
for name in ("torch", "h5py", "numpy", "opencv-python"):
    try: print(name, md.version(name))
    except md.PackageNotFoundError: pass
print("opencv", cv2.__version__)
print("torch_cuda", torch.version.cuda, torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
  sha256sum \
    "$PILOT_CONFIG" "$DATA_CONFIG" "$TRAJECTORY" \
    "$REPO_DIR/benchmark/sim2real/scripts/run_pose_stage2_pilot.py" \
    "$POSE_SOURCE/darknet.py" "$POSE_SOURCE/region_loss.py" "$POSE_SOURCE/cfg/yolo-pose.cfg"
  git -C "$POSE_SOURCE" rev-parse HEAD
  du -sh "$DATA_ROOT"
} > "$RUN_DIR/environment_and_hashes.txt" 2>&1
cp "$PILOT_CONFIG" "$RUN_DIR/pose_stage2_pilot.json"

printf '%q ' python -u \
  "$REPO_DIR/benchmark/sim2real/scripts/run_pose_stage2_pilot.py" \
  --shared-h5 "$TRAJECTORY" \
  --v2e-root "$DATA_ROOT/v2e_events" \
  --data-config "$DATA_CONFIG" \
  --pilot-config "$PILOT_CONFIG" \
  --output-root "$RUN_DIR/output" > "$RUN_DIR/pilot.command.txt"
printf '\n' >> "$RUN_DIR/pilot.command.txt"
if python -u "$REPO_DIR/benchmark/sim2real/scripts/run_pose_stage2_pilot.py" \
  --shared-h5 "$TRAJECTORY" \
  --v2e-root "$DATA_ROOT/v2e_events" \
  --data-config "$DATA_CONFIG" \
  --pilot-config "$PILOT_CONFIG" \
  --output-root "$RUN_DIR/output" > "$RUN_DIR/pilot.stdout.log" 2>&1; then
  echo 0 > "$RUN_DIR/pilot.exit_code.txt"
  echo 0 > "$RUN_DIR/exit_code.txt"
else
  status=$?
  echo "$status" > "$RUN_DIR/pilot.exit_code.txt"
  echo "$status" > "$RUN_DIR/exit_code.txt"
  exit "$status"
fi
date -u +%FT%TZ > "$RUN_DIR/finished_utc.txt"
echo "Event-only pose pilot artifacts: $RUN_DIR/output"
