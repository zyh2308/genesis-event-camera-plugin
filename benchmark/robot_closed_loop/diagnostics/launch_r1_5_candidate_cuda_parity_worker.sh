#!/usr/bin/env bash
set -uo pipefail
RUN_ID=${1:?run id}
HOST=${2:?host}
GPU_INDEX=${3:?GPU index}
GPU_UUID=${4:?GPU UUID}
SOURCE_COMMIT=${5:?source commit}
ROOT=${6:?candidate checkout}
BRANCH=${7:?candidate branch}
BASE=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin
CACHE_ROOT="$BASE/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059"
PYTHON=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
LOCK_PARENT=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/.gpu_reservations
GPU_LOCK="$LOCK_PARENT/$GPU_UUID"
SLOT="$LOCK_PARENT/.r1_5_core_candidate_cuda_slot"
OUT="$ROOT/benchmark/robot_closed_loop/results/$RUN_ID"
LOGDIR="$ROOT/benchmark/robot_closed_loop/results/runtime_logs/$RUN_ID"
SCRIPT=benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_core_candidate_cuda_parity.py
CONFIG=benchmark/robot_closed_loop/configs/r1_5_core_cuda_parity_20261030a.json

printf '%s\n' "$$" > "$LOGDIR/worker_pid.txt"
printf 'started_utc=%s\ncode_commit=%s\n' "$(date -u +%FT%TZ)" "$SOURCE_COMMIT" > "$LOGDIR/worker_started.txt"
set +e
env CUDA_VISIBLE_DEVICES="$GPU_UUID" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="$ROOT" "$PYTHON" -u "$ROOT/$SCRIPT" --cache-root "$CACHE_ROOT" --config "$ROOT/$CONFIG" --output "$OUT" 2>&1 | tee "$LOGDIR/run.log"
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
git -C "$ROOT" add -f "benchmark/robot_closed_loop/results/$RUN_ID" "benchmark/robot_closed_loop/results/runtime_logs/$RUN_ID"
if ! git -C "$ROOT" diff --cached --quiet; then
  if git -C "$ROOT" -c user.name='Robot R1.5 diagnostic' -c user.email='robot-r1-5-diagnostic@users.noreply.github.com' commit -m "Record R1.5 candidate CUDA parity run" > "$LOGDIR/git_publish.txt" 2>&1; then
    if git -C "$ROOT" push origin "HEAD:refs/heads/$BRANCH" >> "$LOGDIR/git_publish.txt" 2>&1; then
      echo pushed >> "$LOGDIR/git_publish.txt"
    else
      echo push_failed_preserve_local_commit >> "$LOGDIR/git_publish.txt"
    fi
  fi
fi
if python3 -c 'import json,sys; assert json.load(open(sys.argv[1])).get("run_id")==sys.argv[2]' "$GPU_LOCK/allocation.json" "$RUN_ID"; then
  rm -f "$GPU_LOCK/allocation.json" "$GPU_LOCK/created_utc.txt"
  rmdir "$GPU_LOCK" 2>/dev/null || true
fi
if [[ "$(cat "$SLOT/owner.txt" 2>/dev/null)" == "$RUN_ID $HOST $GPU_UUID" ]]; then
  rm -f "$SLOT/owner.txt"
  rmdir "$SLOT" 2>/dev/null || true
fi
exit "$RC"
