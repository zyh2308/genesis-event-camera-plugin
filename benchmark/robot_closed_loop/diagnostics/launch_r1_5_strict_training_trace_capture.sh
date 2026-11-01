#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 SOURCE_ROOT CACHE_ROOT OUTPUT EXPERIMENT_CONFIG EPISODE_IDS" >&2
  exit 2
fi

SOURCE_ROOT=$(realpath "$1")
CACHE_ROOT=$(realpath "$2")
OUTPUT=$(realpath -m "$3")
EXPERIMENT_CONFIG=$(realpath "$4")
EPISODE_IDS=$5
PYTHON=${ROBOT_PYTHON:-"$HOME/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python"}
SCRIPT="$SOURCE_ROOT/benchmark/robot_closed_loop/diagnostics/capture_r1_5_strict_training_traces.py"
SCENE_CONFIG="$SOURCE_ROOT/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"
CONTROL_CONFIG="$SOURCE_ROOT/benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json"
LOG_DIR="${OUTPUT}.logs"

for path in "$SCRIPT" "$SCENE_CONFIG" "$CONTROL_CONFIG" "$EXPERIMENT_CONFIG"; do
  [[ -f "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done
[[ -d "$CACHE_ROOT" ]] || { echo "missing cache root: $CACHE_ROOT" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "missing executable Python: $PYTHON" >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite output: $OUTPUT" >&2; exit 2; }
[[ ! -e "$LOG_DIR" ]] || { echo "refusing to overwrite log directory: $LOG_DIR" >&2; exit 2; }

mkdir -p "$(dirname "$OUTPUT")" "$LOG_DIR"
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export ROBOT_EXECUTION_CODE_COMMIT=${ROBOT_EXECUTION_CODE_COMMIT:-unknown}

{
  date -Is
  hostname
  printf 'source_root=%s\ncache_root=%s\noutput=%s\npython=%s\n' \
    "$SOURCE_ROOT" "$CACHE_ROOT" "$OUTPUT" "$PYTHON"
  printf 'source_commit=%s\n' "$ROBOT_EXECUTION_CODE_COMMIT"
  "$PYTHON" -c 'import platform,sys,numpy,h5py; print("python",sys.version); print("platform",platform.platform()); print("numpy",numpy.__version__); print("h5py",h5py.__version__)'
  sha256sum "$SCRIPT" "$SCENE_CONFIG" "$CONTROL_CONFIG" "$EXPERIMENT_CONFIG"
} > "$LOG_DIR/preflight.txt" 2>&1

"$PYTHON" "$SCRIPT" \
  --cache-root "$CACHE_ROOT" \
  --scene-config "$SCENE_CONFIG" \
  --control-config "$CONTROL_CONFIG" \
  --experiment-config "$EXPERIMENT_CONFIG" \
  --output "$OUTPUT" \
  --episode-ids "$EPISODE_IDS" \
  --trace-period-s 0.0005 > "$LOG_DIR/run.log" 2>&1 &
worker_pid=$!
printf '%s\n' "$worker_pid" > "$LOG_DIR/worker_pid.txt"
printf '%s\n' "$(date -Is)" > "$LOG_DIR/worker_started.txt"
set +e
wait "$worker_pid"
status=$?
set -e
printf '%s\n' "$status" > "$LOG_DIR/exit_code.txt"
printf '%s\n' "$(date -Is)" > "$LOG_DIR/worker_finished.txt"
exit "$status"
