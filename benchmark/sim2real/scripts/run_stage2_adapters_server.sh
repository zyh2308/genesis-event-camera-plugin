#!/usr/bin/env bash
# Re-run only the benchmark-local task adapters over a completed Stage-2 bank.
set -u

PROJECT_DIR=/home/zhaoyuhan/Eventbased_WAM
REPO_DIR="$PROJECT_DIR/repositories/gec-sim2real"
RUN_ID="${1:-20261016-s2-pilot-adapters-v2}"
RUN_DIR="$PROJECT_DIR/runs/sim2real/$RUN_ID"
DATA_ROOT="$PROJECT_DIR/data/sim2real/stage2/20261016-s2-pilot-v1"
CONFIG="$REPO_DIR/benchmark/sim2real/configs/stage2_pilot.json"
E2VID_OUT="$DATA_ROOT/e2vid_inputs_v2"
STEFLOW_OUT="$DATA_ROOT/steflow_inputs_v2"

if [[ -e "$RUN_DIR" || -e "$E2VID_OUT" || -e "$STEFLOW_OUT" ]]; then
  echo "refusing existing Stage-2 adapter paths: $RUN_DIR, $E2VID_OUT, or $STEFLOW_OUT" >&2
  exit 2
fi
for required in \
  "$DATA_ROOT/shared/shared_trajectory.h5" \
  "$DATA_ROOT/v2e_events/train_00.h5" \
  "$DATA_ROOT/v2e_events/train_01.h5" \
  "$DATA_ROOT/v2e_events/train_02.h5" \
  "$DATA_ROOT/v2e_events/train_03.h5" \
  "$DATA_ROOT/v2e_events/validation_00.h5" \
  "$DATA_ROOT/v2e_events/validation_01.h5"; do
  if [[ ! -s "$required" ]]; then
    echo "missing completed input: $required" >&2
    exit 3
  fi
done

mkdir -p "$RUN_DIR"
printf '%s\n' "$$" > "$RUN_DIR/pid.txt"
date -u +%FT%TZ > "$RUN_DIR/started_utc.txt"
trap 'rc=$?; echo "$rc" > "$RUN_DIR/exit_code.txt"; date -u +%FT%TZ > "$RUN_DIR/finished_utc.txt"' EXIT
cd "$REPO_DIR" || exit 4
{
  date -u +%FT%TZ
  hostname
  pwd
  git rev-parse HEAD
  git status --short --branch
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
  df -h "$PROJECT_DIR"
} > "$RUN_DIR/preflight.txt" 2>&1

source "$HOME/miniconda3/etc/profile.d/conda.sh" || exit 90
conda activate Eventbased_WAM || exit 91
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$REPO_DIR/benchmark/sim2real/scripts:$REPO_DIR:$PROJECT_DIR/runtime/sim2real/genesis_hdr_v2"

{
  python - <<'PY'
import importlib.metadata as md, sys, torch, h5py, numpy
print("python", sys.version)
for name in ("torch", "h5py", "numpy", "opencv-python"):
    try: print(name, md.version(name))
    except md.PackageNotFoundError: pass
print("torch_cuda", torch.version.cuda, torch.cuda.is_available())
PY
  sha256sum \
    "$CONFIG" \
    "$REPO_DIR/benchmark/sim2real/scripts/event_window_audit.py" \
    "$REPO_DIR/benchmark/sim2real/scripts/prepare_e2vid_stage2.py" \
    "$REPO_DIR/benchmark/sim2real/scripts/prepare_steflow_stage2.py" \
    "$DATA_ROOT/shared/shared_trajectory.h5"
} > "$RUN_DIR/environment_and_hashes.txt" 2>&1

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

if ! run_step e2vid_conversion_v2 python -u \
  "$REPO_DIR/benchmark/sim2real/scripts/prepare_e2vid_stage2.py" \
  --trajectory-h5 "$DATA_ROOT/shared/shared_trajectory.h5" \
  --v2e-root "$DATA_ROOT/v2e_events" --output-root "$E2VID_OUT"; then
  echo "E2VID adapter retry failed; see $RUN_DIR/e2vid_conversion_v2.stdout.log" >&2
  exit "$(cat "$RUN_DIR/e2vid_conversion_v2.exit_code.txt")"
fi

if ! run_step steflow_conversion_v2 python -u \
  "$REPO_DIR/benchmark/sim2real/scripts/prepare_steflow_stage2.py" \
  --trajectory-h5 "$DATA_ROOT/shared/shared_trajectory.h5" \
  --v2e-root "$DATA_ROOT/v2e_events" --output-root "$STEFLOW_OUT"; then
  echo "STE-FlowNet adapter retry failed; see $RUN_DIR/steflow_conversion_v2.stdout.log" >&2
  exit "$(cat "$RUN_DIR/steflow_conversion_v2.exit_code.txt")"
fi

echo "Stage-2 adapters completed: $E2VID_OUT and $STEFLOW_OUT"
