#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT=$PROJECT/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1
OLD=$PROJECT/runs/sim2real
PY=$PROJECT/../miniconda3/envs/Eventbased_WAM_torch28/bin/python
MINI=$RUN_ROOT/source_v2/mini_repo
AUDIT=$MINI/benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28/audit_camera_motion_coverage.py
MONTAGE=$MINI/benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28/render_replicad_flow_event_montage.py
CRITERIA=$RUN_ROOT/source_v2/genesis_env_torch28/camera_motion_coverage_criteria_v1.json
OUT=$RUN_ROOT/coverage_audits_retry01
test "$(cat "$RUN_ROOT/v2e_retry01/exit_code.txt")" = 0
test ! -e "$OUT"
mkdir -p "$OUT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/started_utc.txt"
{
  printf 'host=%s\n' "$(hostname)"
  printf 'execution_checkout_commit='; git -C "$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030" rev-parse HEAD
  printf 'audit_sha256='; sha256sum "$AUDIT"
  printf 'montage_sha256='; sha256sum "$MONTAGE"
  printf 'criteria_sha256='; sha256sum "$CRITERIA"
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader
  uptime; free -h; df -h "$RUN_ROOT" /tmp
} > "$OUT/environment.txt" 2>&1

for scene in v3_sc0_staging_00 v3_sc1_staging_00 v3_sc2_staging_00; do
  case "$scene" in
    v3_sc0_staging_00) old_g=20261030-stage7-replicad-v3_sc0_staging_00-fast-run14; old_v=20261030-stage7-replicad-v3-sc0-v2e-cpu-run17 ;;
    v3_sc1_staging_00) old_g=20261030-stage7-replicad-v3_sc1_staging_00-fast-run15; old_v=20261030-stage7-replicad-v3-sc1-v2e-cpu-run20 ;;
    v3_sc2_staging_00) old_g=20261030-stage7-replicad-v3_sc2_staging_00-fast-run16; old_v=20261030-stage7-replicad-v3-sc2-v2e-gpu-run21 ;;
  esac
  for path in x_060mm y_040mm; do
    if [[ "$path" == x_060mm ]]; then
      MOTION=$OLD/$old_g/render_output/camera_motion_30ms
      V2E=$OLD/$old_v/artifacts/v2e_events.h5
    else
      MOTION=$RUN_ROOT/$scene/camera_motion_30ms
      V2E=$RUN_ROOT/v2e_retry01/$scene/v2e_gpu/events/v2e_events.h5
    fi
    ID=${scene%_staging_00}_$path
    mkdir -p "$OUT/$ID"
    test -d "$MOTION" && test -f "$V2E"
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$MINI" \
      "$PY" -u "$AUDIT" \
        --camera-motion-dir "$MOTION" --v2e-h5 "$V2E" --criteria "$CRITERIA" \
        --output "$OUT/$ID/observability.json" > "$OUT/$ID/audit.stdout.log" 2> "$OUT/$ID/audit.stderr.log"
    if [[ "$path" == y_040mm ]]; then
      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$MINI" \
        "$PY" -u "$MONTAGE" --motion-dir "$MOTION" --v2e-h5 "$V2E" \
        --output "$OUT/$ID/paired_flow_event_montage.png" --name "$ID"
    fi
  done
done

"$PY" - "$OUT" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
root=Path(sys.argv[1])
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
rows=[]
for p in sorted(root.glob('*/observability.json')):
 d=json.loads(p.read_text())
 rows.append({'rollout_id':p.parent.name,'path':str(p),'sha256':digest(p),'status':d['status'],'intervals':d['intervals']})
for p in sorted(root.glob('*/paired_flow_event_montage.png')):
 rows.append({'rollout_id':p.parent.name,'type':'paired_flow_event_montage','path':str(p),'sha256':digest(p),'size_bytes':p.stat().st_size})
index={'schema':'stage7_multiaxis_coverage_evidence_index_v1','created_utc':datetime.now(timezone.utc).isoformat(),'entries':rows}
(root/'coverage_index.json').write_text(json.dumps(index,indent=2,sort_keys=True)+'\n')
print(json.dumps({'entries':len(rows),'audits':sum('intervals' in x for x in rows),'montages':sum(x.get('type')=='paired_flow_event_montage' for x in rows)},indent=2))
PY
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/finished_utc.txt"
printf '0\n' > "$OUT/exit_code.txt"
