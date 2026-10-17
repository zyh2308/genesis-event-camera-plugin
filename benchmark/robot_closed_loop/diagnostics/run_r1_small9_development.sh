#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID="${1:?usage: run_r1_small9_development.sh RUN_ID}"
WORK_ROOT="${EVENTBASED_WAM_ROOT:-$HOME/Eventbased_WAM}"
ROBOT_PYTHON="${ROBOT_PYTHON:-$WORK_ROOT/repositories/gec-robot/.venv-robot/bin/python}"
V2E_PYTHON="${V2E_PYTHON:-$HOME/miniconda3/envs/Eventbased_WAM/bin/python}"
V2E_SOURCE="${V2E_SOURCE:-$WORK_ROOT/code/v2e_source}"
SLOMO_MODEL="${SLOMO_MODEL:-$V2E_SOURCE/input/SuperSloMo39.ckpt}"
OLD_CACHE_ROOT="${R1_OLD_CACHE_ROOT:-$WORK_ROOT/repositories/genesis-event-camera-plugin-r0_8-final-oracle-20261017/benchmark/robot_closed_loop/results/r0_8_paired_pipeline_smoke_20261017c/trajectory_cache}"
REFERENCE_ROOT="${R1_REFERENCE_ROOT:-$WORK_ROOT/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016}"

RUN_ROOT="$ROOT/benchmark/robot_closed_loop/results/$RUN_ID"
CACHE_ROOT="$RUN_ROOT/trajectory_cache"
PAIRED_ROOT="$RUN_ROOT/paired_r1"
EVAL_ROOT="$RUN_ROOT/common_reference_eval"
LOG_ROOT="$ROOT/benchmark/robot_closed_loop/results/runtime_logs"
LOG_PATH="$LOG_ROOT/$RUN_ID.log"
PID_PATH="$LOG_ROOT/$RUN_ID.pid"
EXIT_PATH="$LOG_ROOT/$RUN_ID.exit"
RUNTIME_PATH="$LOG_ROOT/$RUN_ID.runtime"

mkdir -p "$LOG_ROOT"
if [[ -e "$RUN_ROOT" || -e "$LOG_PATH" || -e "$PID_PATH" || -e "$EXIT_PATH" ]]; then
  echo "refusing to overwrite existing R1 run path: $RUN_ID" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"
exec >"$LOG_PATH" 2>&1
printf '%s\n' "$$" > "$PID_PATH"
date -u +%FT%TZ > "$RUNTIME_PATH"
trap 'status=$?; printf "%s\n" "$status" > "$EXIT_PATH"; date -u +%FT%TZ >> "$RUNTIME_PATH"; exit "$status"' EXIT

for path in "$ROBOT_PYTHON" "$V2E_PYTHON" "$V2E_SOURCE/v2e.py" "$SLOMO_MODEL" "$OLD_CACHE_ROOT" "$REFERENCE_ROOT/SHA256SUMS"; do
  [[ -e "$path" ]] || { echo "required R1 input missing: $path"; exit 2; }
done

echo "R1 small9 development run: $RUN_ID"
echo "code_commit=$(git -C "$ROOT" rev-parse HEAD)"
echo "root=$ROOT"
echo "old_cache_root=$OLD_CACHE_ROOT"
echo "direct_hdr_reference=$REFERENCE_ROOT"
echo "robot_python=$ROBOT_PYTHON"
echo "v2e_python=$V2E_PYTHON"
echo "v2e_source=$V2E_SOURCE"
echo "slomo_model=$SLOMO_MODEL"
echo "render_device=EGL_DEVICE_ID 3 (physical GPU 3); V2E=physical GPU 3; model fits=CPU"

/usr/bin/time -v -o "$RUN_ROOT/capture.time" env \
  CUDA_VISIBLE_DEVICES=3 EGL_DEVICE_ID=3 PYOPENGL_PLATFORM=egl PYTHONPATH="$ROOT" \
  "$ROBOT_PYTHON" "$ROOT/benchmark/robot_closed_loop/diagnostics/capture_r1_supplementary_rollouts.py" \
  --output "$CACHE_ROOT"

for episode_id in 000 001; do
  [[ -d "$OLD_CACHE_ROOT/episode_$episode_id" ]] || { echo "missing old shared cache episode_$episode_id"; exit 2; }
  ln -s "$OLD_CACHE_ROOT/episode_$episode_id" "$CACHE_ROOT/episode_$episode_id"
done

/usr/bin/time -v -o "$RUN_ROOT/paired_training.time" env \
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$ROOT" \
  "$ROBOT_PYTHON" "$ROOT/benchmark/robot_closed_loop/diagnostics/run_r1_from_cached_rollouts.py" \
  --cache-root "$CACHE_ROOT" \
  --output "$PAIRED_ROOT" \
  --experiment-config "$ROOT/benchmark/robot_closed_loop/configs/r1_shared_ideal_small9_v1.json" \
  --v2e-config "$ROOT/benchmark/robot_closed_loop/configs/r1_shared_ideal_response_v1_v2e.yaml" \
  --scene-config "$ROOT/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json" \
  --v2e-source "$V2E_SOURCE" \
  --slomo-model "$SLOMO_MODEL" \
  --python "$ROBOT_PYTHON" \
  --v2e-python "$V2E_PYTHON" \
  --episode-ids 0,1,2,3,4,5,6,7,8

DECCELERATION="$("$ROBOT_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["controller"]["initial_deceleration_assumption_m_s2"])' \
  "$ROOT/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json")"
/usr/bin/time -v -o "$RUN_ROOT/common_reference_eval.time" env \
  CUDA_VISIBLE_DEVICES= PYTHONPATH="$ROOT" \
  "$ROBOT_PYTHON" -m benchmark.robot_closed_loop.diagnostics.evaluate_r1_common_direct_hdr \
  --scene-config "$ROOT/benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json" \
  --control-config "$ROOT/benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json" \
  --reference-root "$REFERENCE_ROOT" \
  --ours-checkpoint "$PAIRED_ROOT/ours_fit/spatial_event_state.pt" \
  --v2e-checkpoint "$PAIRED_ROOT/v2e_fit/spatial_event_state.pt" \
  --output "$EVAL_ROOT" \
  --deceleration "$DECCELERATION"

printf 'capture='; cat "$RUN_ROOT/capture.time" | rg '^\s*Elapsed \(wall clock\)|Maximum resident set size|Exit status'
printf 'paired='; cat "$RUN_ROOT/paired_training.time" | rg '^\s*Elapsed \(wall clock\)|Maximum resident set size|Exit status'
printf 'evaluation='; cat "$RUN_ROOT/common_reference_eval.time" | rg '^\s*Elapsed \(wall clock\)|Maximum resident set size|Exit status'
echo "R1 small9 pipeline completed: $RUN_ROOT"
