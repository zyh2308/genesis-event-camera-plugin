#!/usr/bin/env bash
set -uo pipefail
RUN_ID=${1:?run id}
HOST=${2:?host}
GPU_INDEX=${3:?GPU index}
GPU_UUID=${4:?GPU UUID}
SOURCE_COMMIT=${5:?source commit}
MODE=${6:?smoke or full}
RUN_ROOT="/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/$RUN_ID"
ROOT="$RUN_ROOT/source"
CACHE_ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-diag-20261019/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059
PYTHON=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
LOCK_PARENT=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/.gpu_reservations
GPU_LOCK="$LOCK_PARENT/$GPU_UUID"
SLOT="$LOCK_PARENT/.r1_5_core_candidate_cuda_slot"
OUT="$RUN_ROOT/output"
LOGDIR="$RUN_ROOT/logs"
SCRIPT=benchmark/robot_closed_loop/diagnostics/measure_r1_5_strict_direct_rerender.py
SCENE_CONFIG=benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json
if [[ "$MODE" == smoke ]]; then
  CONFIG=benchmark/robot_closed_loop/configs/r1_5_strict_hdr_runtime_smoke_20261031h.json
  MODE_ARGS=(--smoke-only)
elif [[ "$MODE" == full ]]; then
  CONFIG=benchmark/robot_closed_loop/configs/r1_5_strict_direct_rerender_20261031h.json
  MODE_ARGS=()
else
  echo "invalid mode=$MODE" >&2
  exit 64
fi

printf '%s\n' "$$" > "$LOGDIR/worker_pid.txt"
printf 'started_utc=%s\nsource_commit=%s\nmode=%s\n' "$(date -u +%FT%TZ)" "$SOURCE_COMMIT" "$MODE" > "$LOGDIR/worker_started.txt"
python3 - "$GPU_LOCK/allocation.json" "$RUN_ID" "$GPU_UUID" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d.get('run_id') == sys.argv[2] and d.get('gpu_uuid') == sys.argv[3]
assert d.get('status') == 'running'
PY

set +e
/usr/bin/time -v -o "$LOGDIR/time_v.txt" \
  env EGL_DEVICE_ID="$GPU_INDEX" CUDA_VISIBLE_DEVICES="$GPU_UUID" PYOPENGL_PLATFORM=egl \
  GENESIS_EVENT_PLUGIN_SOURCE_COMMIT="$SOURCE_COMMIT" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="$ROOT" \
  timeout 1800s "$PYTHON" -u "$ROOT/$SCRIPT" \
    --cache-root "$CACHE_ROOT" --scene-config "$ROOT/$SCENE_CONFIG" \
    --experiment-config "$ROOT/$CONFIG" --output "$OUT" "${MODE_ARGS[@]}" 2>&1 | tee "$LOGDIR/run.log"
RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "$RC" > "$LOGDIR/exit_code.txt"
printf 'finished_utc=%s\nexit_code=%s\nmode=%s\n' "$(date -u +%FT%TZ)" "$RC" "$MODE" > "$LOGDIR/worker_finished.txt"

if [[ -d "$OUT" ]]; then
  cp "$ROOT/$CONFIG" "$OUT/config.json"
  (cd "$OUT" && sha256sum -c SHA256SUMS > "$LOGDIR/output_checksum_verify.txt")
fi
cp "$GPU_LOCK/allocation.json" "$LOGDIR/resource_allocation.json"
python3 - "$LOGDIR/status.json" "$RUN_ID" "$RC" "$OUT" "$MODE" "$LOGDIR/worker_pid.txt" <<'PY'
import datetime,json,os,sys
p,run_id,rc,out,mode,pidfile=sys.argv[1:]
json.dump({'run_id':run_id,'mode':mode,'status':'completed' if int(rc)==0 else 'failed',
           'exit_code':int(rc),'worker_pid':int(open(pidfile).read().strip()),
           'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'remote_output':out,'output_exists':os.path.isdir(out)},open(p,'w'),indent=2)
PY

if python3 -c 'import json,sys; assert json.load(open(sys.argv[1])).get("run_id")==sys.argv[2]' "$GPU_LOCK/allocation.json" "$RUN_ID"; then
  rm -f "$GPU_LOCK/allocation.json" "$GPU_LOCK/created_utc.txt"
  rmdir "$GPU_LOCK" 2>/dev/null || true
fi
if [[ "$(cat "$SLOT/owner.txt" 2>/dev/null)" == "$RUN_ID $HOST $GPU_UUID" ]]; then
  rm -f "$SLOT/owner.txt"
  rmdir "$SLOT" 2>/dev/null || true
fi
exit "$RC"
