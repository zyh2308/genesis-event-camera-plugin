#!/usr/bin/env bash
set -uo pipefail
RUN_ID=${1:?run id}
HOST=${2:?host}
GPU_INDEX=${3:?GPU index}
GPU_UUID=${4:?GPU UUID}
SOURCE_COMMIT=${5:?source commit}
RUN_ROOT="/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/$RUN_ID"
ROOT="$RUN_ROOT/source"
CACHE_ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-diag-20261019/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059
PYTHON=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
LOCK_PARENT=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/.gpu_reservations
GPU_LOCK="$LOCK_PARENT/$GPU_UUID"
SLOT="$LOCK_PARENT/.r1_5_core_candidate_cuda_slot"
OUT="$RUN_ROOT/output"
LOGDIR="$RUN_ROOT/logs"
SCRIPT=benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_core_candidate_cuda_parity.py
CONFIG=benchmark/robot_closed_loop/configs/r1_5_core_cuda_parity_20261030c.json

printf '%s\n' "$$" > "$LOGDIR/worker_pid.txt"
printf 'started_utc=%s\ncode_commit=%s\n' "$(date -u +%FT%TZ)" "$SOURCE_COMMIT" > "$LOGDIR/worker_started.txt"
set +e
env CUDA_VISIBLE_DEVICES="$GPU_UUID" GENESIS_EVENT_PLUGIN_SOURCE_COMMIT="$SOURCE_COMMIT" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="$ROOT" "$PYTHON" -u "$ROOT/$SCRIPT" --cache-root "$CACHE_ROOT" --config "$ROOT/$CONFIG" --output "$OUT" 2>&1 | tee "$LOGDIR/run.log"
RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "$RC" > "$LOGDIR/exit_code.txt"
printf 'finished_utc=%s\nexit_code=%s\n' "$(date -u +%FT%TZ)" "$RC" > "$LOGDIR/worker_finished.txt"
if [[ -d "$OUT" ]]; then
  cp "$ROOT/$CONFIG" "$OUT/config.json"
  python3 - "$OUT/run_manifest.json" "$RUN_ID" "$HOST" "$GPU_INDEX" "$GPU_UUID" "$SOURCE_COMMIT" "$LOGDIR" <<'PY'
import json,sys,datetime
p,run_id,host,index,uuid,commit,logdir=sys.argv[1:]
json.dump({'run_id':run_id,'host':host,'gpu_index':int(index),'gpu_uuid':uuid,'source_commit':commit,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runtime_log_path':logdir},open(p,'w'),indent=2)
PY
  (cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
  (cd "$OUT" && sha256sum -c SHA256SUMS > "$LOGDIR/output_checksum_verify.txt")
fi
python3 - "$GPU_LOCK/allocation.json" "$RUN_ID" "$RC" <<'PY'
import json,sys,datetime
p,run_id,rc=sys.argv[1],sys.argv[2],int(sys.argv[3])
data=json.load(open(p)); assert data.get('run_id')==run_id
data.update(status=('completed' if rc==0 else 'failed'),exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
json.dump(data,open(p,'w'),indent=2)
PY
cp "$GPU_LOCK/allocation.json" "$LOGDIR/resource_allocation.json"
if python3 -c 'import json,sys; assert json.load(open(sys.argv[1])).get("run_id")==sys.argv[2]' "$GPU_LOCK/allocation.json" "$RUN_ID"; then
  rm -f "$GPU_LOCK/allocation.json" "$GPU_LOCK/created_utc.txt"
  rmdir "$GPU_LOCK" 2>/dev/null || true
fi
if [[ "$(cat "$SLOT/owner.txt" 2>/dev/null)" == "$RUN_ID $HOST $GPU_UUID" ]]; then
  rm -f "$SLOT/owner.txt"
  rmdir "$SLOT" 2>/dev/null || true
fi
python3 - "$LOGDIR/status.json" "$RUN_ID" "$RC" "$OUT" <<'PY'
import json,sys,datetime
p,run_id,rc,out=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
import os
json.dump({'run_id':run_id,'status':'completed' if rc==0 else 'failed','exit_code':rc,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_output':out,'output_exists':os.path.isdir(out)},open(p,'w'),indent=2)
PY
exit "$RC"
