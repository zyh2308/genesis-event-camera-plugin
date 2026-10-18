#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID="${1:-r1_native_hdr_small9_retrain_20261019a}"
WORK_ROOT="${EVENTBASED_WAM_ROOT:-$HOME/Eventbased_WAM}"
CACHE_ROOT="${R1_NATIVE_HDR_CACHE_ROOT:-$WORK_ROOT/repositories/genesis-event-camera-plugin-r1-ba137bd/benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018a/trajectory_cache}"
V2E_TRAINING_SOURCE="${R1_NATIVE_HDR_V2E_SOURCE:-$WORK_ROOT/repositories/genesis-event-camera-plugin-r1-ba137bd/benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018b/paired_r1/v2e_training}"
REFERENCE_ROOT="${R1_NATIVE_HDR_REFERENCE_ROOT:-$WORK_ROOT/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016}"
ROBOT_PYTHON="${ROBOT_PYTHON:-$WORK_ROOT/repositories/gec-robot/.venv-robot/bin/python}"
RUN_ROOT="$ROOT/benchmark/robot_closed_loop/results/$RUN_ID"
LOG_ROOT="$ROOT/benchmark/robot_closed_loop/results/runtime_logs"
LOG_PATH="$LOG_ROOT/$RUN_ID.log"
PID_PATH="$LOG_ROOT/$RUN_ID.pid"
EXIT_PATH="$LOG_ROOT/$RUN_ID.exit"
TIME_PATH="$LOG_ROOT/$RUN_ID.runtime"

mkdir -p "$LOG_ROOT"
if [[ -e "$RUN_ROOT" || -e "$LOG_PATH" || -e "$PID_PATH" || -e "$EXIT_PATH" ]]; then
  echo "refusing to overwrite existing native HDR run: $RUN_ID" >&2
  exit 2
fi
exec >"$LOG_PATH" 2>&1
printf '%s\n' "$$" > "$PID_PATH"
trap 'status=$?; printf "%s\n" "$status" > "$EXIT_PATH"; exit "$status"' EXIT

for path in "$ROBOT_PYTHON" "$CACHE_ROOT" "$V2E_TRAINING_SOURCE/SHA256SUMS" "$REFERENCE_ROOT/SHA256SUMS"; do
  [[ -e "$path" ]] || { echo "required native-HDR input missing: $path"; exit 2; }
done

echo "run_id=$RUN_ID"
echo "code_commit=$(git -C "$ROOT" rev-parse HEAD)"
echo "root=$ROOT"
echo "cache_root=$CACHE_ROOT"
echo "v2e_training_source=$V2E_TRAINING_SOURCE"
echo "direct_hdr_reference=$REFERENCE_ROOT"
echo "robot_python=$ROBOT_PYTHON"
echo "device_policy=CPU only; CUDA_VISIBLE_DEVICES is empty"

/usr/bin/time -v -o "$TIME_PATH" env CUDA_VISIBLE_DEVICES= PYTHONPATH="$ROOT" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$ROBOT_PYTHON" "$ROOT/benchmark/robot_closed_loop/diagnostics/run_r1_native_hdr_retrain.py" \
  --cache-root "$CACHE_ROOT" \
  --v2e-training-source "$V2E_TRAINING_SOURCE" \
  --reference-root "$REFERENCE_ROOT" \
  --output "$RUN_ROOT" \
  --experiment-config "$ROOT/benchmark/robot_closed_loop/configs/r1_native_hdr_small9_development_v1.json" \
  --scene-config "$ROOT/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json" \
  --control-config "$ROOT/benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json" \
  --python "$ROBOT_PYTHON"

echo "native HDR development retrain completed: $RUN_ROOT"
