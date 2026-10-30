#!/usr/bin/env bash
set -euo pipefail
RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-replicad-v3-sc2-v2e-gpu-run21
PROJECT=/home/zhaoyuhan/Eventbased_WAM
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
LOCK=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/resource_reservations/GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23
GPU_UUID=GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23
release_reservation() {
  if [[ -f "$LOCK/owner.txt" ]] && [[ "$(cat "$LOCK/owner.txt")" == "$RUN" ]]; then
    "$PY" - "$RUN/resource_reservation.json" "$LOCK/resource_reservation.json" <<'PY'
import json, sys
from datetime import datetime, timezone
for name in sys.argv[1:]:
    try:
        d=json.load(open(name))
        d['status']='released_after_run_exit'
        d['released_utc']=datetime.now(timezone.utc).isoformat()
        open(name,'w').write(json.dumps(d,indent=2,sort_keys=True)+'\n')
    except FileNotFoundError:
        pass
PY
    rm -f -- "$LOCK/owner.txt" "$LOCK/resource_reservation.json"
    rmdir -- "$LOCK" 2>/dev/null || true
  fi
}
trap release_reservation EXIT
INPUT_RAD=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-replicad-v3_sc2_staging_00-fast-run16/render_output/camera_motion_30ms/radiance_frames.npy
CONVERSION_SOURCE=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030/benchmark/sim2real/scripts/generate_stage5_flow_coverage_bank.py
exec > >(tee -a "$RUN/driver.log") 2>&1
set +e
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_utc.txt"
timeout 900s env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  "$PY" -u "$RUN/run_v2e_same_radiance_camera_motion.py" \
  --radiance-frames "$INPUT_RAD" --conversion-source "$CONVERSION_SOURCE" \
  --project-root "$PROJECT" --output-root "$RUN/artifacts" --cuda-visible-devices 0
RC=$?
printf '%s\n' "$RC" > "$RUN/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
echo v2e_driver_exit_code=$RC
exit "$RC"
