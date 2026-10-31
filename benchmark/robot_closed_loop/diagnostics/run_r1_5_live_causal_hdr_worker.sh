#!/usr/bin/env bash
set -uo pipefail
RUN_ID=${1:?run id}
HOST=${2:?host}
GPU_INDEX=${3:?GPU index}
GPU_UUID=${4:?GPU UUID}
SOURCE_COMMIT=${5:?source commit}
RUN_ROOT="/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/$RUN_ID"
ROOT="$RUN_ROOT/source"
PYTHON=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
LOCK_PARENT=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/.gpu_reservations
GPU_LOCK="$LOCK_PARENT/$GPU_UUID"
SLOT="$LOCK_PARENT/.r1_5_core_candidate_cuda_slot"
OUT="$RUN_ROOT/output"
LOGDIR="$RUN_ROOT/logs"
CONFIG="benchmark/robot_closed_loop/configs/${RUN_ID}.json"
SCENE_CONFIG=benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json
SCRIPT=benchmark/robot_closed_loop/diagnostics/run_r1_5_live_causal_hdr_plugin.py

release_owned_reservation() {
  if [[ -f "$GPU_LOCK/allocation.json" ]] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("run_id")==sys.argv[2] and d.get("gpu_uuid")==sys.argv[3] else 1)' "$GPU_LOCK/allocation.json" "$RUN_ID" "$GPU_UUID"; then
    cp "$GPU_LOCK/allocation.json" "$LOGDIR/resource_allocation.json" 2>/dev/null || true
    rm -f "$GPU_LOCK/allocation.json" "$GPU_LOCK/created_utc.txt"
    rmdir "$GPU_LOCK" 2>/dev/null || true
  fi
  if [[ "$(cat "$SLOT/owner.txt" 2>/dev/null || true)" == "$RUN_ID $HOST $GPU_UUID" ]]; then
    rm -f "$SLOT/owner.txt"
    rmdir "$SLOT" 2>/dev/null || true
  fi
}
trap release_owned_reservation EXIT

printf '%s\n' "$$" > "$LOGDIR/worker_pid.txt"
printf 'started_utc=%s\nsource_commit=%s\n' "$(date -u +%FT%TZ)" "$SOURCE_COMMIT" > "$LOGDIR/worker_started.txt"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("run_id")==sys.argv[2] and d.get("gpu_uuid")==sys.argv[3] and d.get("status")=="running"' "$GPU_LOCK/allocation.json" "$RUN_ID" "$GPU_UUID"

set +e
/usr/bin/time -v -o "$LOGDIR/time_v.txt" \
  env EGL_DEVICE_ID="$GPU_INDEX" CUDA_VISIBLE_DEVICES="$GPU_UUID" PYOPENGL_PLATFORM=egl \
  GENESIS_EVENT_PLUGIN_SOURCE_COMMIT="$SOURCE_COMMIT" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="$ROOT" \
  timeout 1800s "$PYTHON" -u "$ROOT/$SCRIPT" \
    --config "$ROOT/$CONFIG" --scene-config "$ROOT/$SCENE_CONFIG" --output "$OUT" 2>&1 | tee "$LOGDIR/run.log"
RC=${PIPESTATUS[0]}
set -e
CHECKSUM_RC=0
if [[ -d "$OUT" && -f "$OUT/SHA256SUMS" ]]; then
  set +e
  (cd "$OUT" && sha256sum -c SHA256SUMS) > "$LOGDIR/output_checksum_verify.txt" 2>&1
  CHECKSUM_RC=$?
  set -e
elif [[ -d "$OUT" ]]; then
  printf 'SKIPPED: incomplete run did not produce SHA256SUMS\n' > "$LOGDIR/output_checksum_verify.txt"
fi
FINAL_RC=$RC
(( CHECKSUM_RC == 0 )) || FINAL_RC=$CHECKSUM_RC
printf '%s\n' "$FINAL_RC" > "$LOGDIR/exit_code.txt"
printf '{"run_id":"%s","status":"%s","exit_code":%d,"runner_exit_code":%d,"checksum_exit_code":%d,"worker_pid":%d,"finished_utc":"%s","remote_output":"%s","output_exists":%s}\n' \
  "$RUN_ID" "$([[ $FINAL_RC -eq 0 ]] && echo completed || echo failed)" "$FINAL_RC" "$RC" "$CHECKSUM_RC" "$$" \
  "$(date -u +%FT%TZ)" "$OUT" "$([[ -d $OUT ]] && echo true || echo false)" > "$LOGDIR/status.json"
exit "$FINAL_RC"
