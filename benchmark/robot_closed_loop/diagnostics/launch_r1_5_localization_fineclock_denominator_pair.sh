#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_ROOT GPU_UUID RESERVATION_JSON" >&2
  exit 64
fi

RUN_ROOT=$(realpath -m "$1")
GPU_UUID=$2
RESERVATION_JSON=$(realpath -m "$3")
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RUN_ID=$(basename "$RUN_ROOT")
CORE_ROOT=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/robot-r1-localization-20261101a-914ffff/frozen_core
CACHE_ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-ba137bd/benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018a/trajectory_cache
PY=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
EXPERIMENT_CONFIG="$ROOT/benchmark/robot_closed_loop/configs/r1_5_localization_recovery_fineclock_20261101c.json"
SCENE_CONFIG="$ROOT/benchmark/robot_closed_loop/configs/scene_r1_5_localization_fineclock_20261101c.json"
CONTROL_CONFIG="$ROOT/benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json"
TRACE_ROOT="$RUN_ROOT/fineclock_traces"
RENDER_ROOT="$RUN_ROOT/strict_direct_events"
EVAL_ROOT="$RUN_ROOT/localization_eval"
OURS_CHECKPOINT="$ROOT/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/ours_fit/spatial_event_state.pt"
V2E_CHECKPOINT="$ROOT/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/v2e_fit/spatial_event_state.pt"
TRACE_NAME=state_trace_dt_23810ns.jsonl
DT=0.00002380952380952381
END_S=0.625

if [[ -e "$RUN_ROOT" ]]; then
  echo "refusing to overwrite run root: $RUN_ROOT" >&2
  exit 73
fi
mkdir -p "$RUN_ROOT/logs"
exec > >(tee -a "$RUN_ROOT/logs/launcher.log") 2>&1
echo "run_id=$RUN_ID"
echo "host=$(hostname)"
echo "launcher_pid=$$"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "code_commit=$(git -C "$ROOT" rev-parse HEAD)"
echo "gpu_uuid=$GPU_UUID"
echo "physics_dt_s=$DT"
echo "capture_horizon_after_launch_s=$END_S"

finish_reservation() {
  local exit_code=$?
  python3 - "$RESERVATION_JSON" "$exit_code" "$RUN_ID" <<'PY'
import datetime, json, pathlib, sys
p=pathlib.Path(sys.argv[1])
code=int(sys.argv[2])
run_id=sys.argv[3]
try:
    d=json.loads(p.read_text())
    if d.get("run_id") == run_id:
        d["status"] = "released_complete" if code == 0 else "released_failed"
        d["exit_code"] = code
        d["ended_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        p.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
except FileNotFoundError:
    pass
PY
  echo "exit_code=$exit_code"
  echo "ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$exit_code"
}
trap finish_reservation EXIT

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export ROBOT_EXECUTION_CODE_COMMIT=$(git -C "$ROOT" rev-parse HEAD)

echo "phase=exact_physics_trace_capture_start utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CUDA_VISIBLE_DEVICES="" "$PY" -u "$ROOT/benchmark/robot_closed_loop/diagnostics/capture_r1_5_strict_training_traces.py" \
  --cache-root "$CACHE_ROOT" \
  --scene-config "$SCENE_CONFIG" \
  --control-config "$CONTROL_CONFIG" \
  --output "$TRACE_ROOT" \
  --episode-ids 4,5 \
  --trace-period-s "$DT" \
  --stop-after-s "$END_S" \
  --allow-physics-clock-change \
  --experiment-config "$EXPERIMENT_CONFIG"

for EP in 4 5; do
  current=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits | awk -F, -v u="$GPU_UUID" 'BEGIN{n=0} {gsub(/^[ \t]+|[ \t]+$/, "", $1); if($1==u){print; n++}} END{if(n) exit 1}') || {
    echo "assigned GPU already has a compute process; refusing launch on episode $EP"
    exit 90
  }
  echo "phase=render_episode_start episode=$EP utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  CUDA_VISIBLE_DEVICES="$GPU_UUID" EGL_DEVICE_ID=0 "$PY" -u \
    "$ROOT/benchmark/robot_closed_loop/diagnostics/render_r1_5_strict_direct_episode.py" \
    --frozen-core-root "$CORE_ROOT" \
    --profile-config "$CORE_ROOT/benchmark/robot_closed_loop/configs/r1_5_core_strict_sampled_20261102.json" \
    --source-root "$ROOT" \
    --support-source-commit "$ROBOT_EXECUTION_CODE_COMMIT" \
    --trace-root "$TRACE_ROOT" \
    --trace-filename "$TRACE_NAME" \
    --trace-end-s "$END_S" \
    --trajectory-mode paired-new-physics \
    --legacy-cache-root "$CACHE_ROOT" \
    --scene-config "$SCENE_CONFIG" \
    --control-config "$CONTROL_CONFIG" \
    --experiment-config "$EXPERIMENT_CONFIG" \
    --episode-id "$EP" \
    --output "$RENDER_ROOT" \
    --genesis-backend gpu > "$RUN_ROOT/logs/render_episode_${EP}.log" 2>&1
  echo "phase=render_episode_end episode=$EP utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "phase=paired_model_localization_eval_start utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CUDA_VISIBLE_DEVICES="" "$PY" -u "$ROOT/benchmark/robot_closed_loop/diagnostics/evaluate_r1_5_fineclock_localization.py" \
  --source-root "$RENDER_ROOT" \
  --config "$EXPERIMENT_CONFIG" \
  --scene-config "$SCENE_CONFIG" \
  --ours-checkpoint "$OURS_CHECKPOINT" \
  --v2e-checkpoint "$V2E_CHECKPOINT" \
  --output "$EVAL_ROOT"

find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/SHA256SUMS"
echo "phase=complete run_root=$RUN_ROOT"
