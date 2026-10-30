#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT=$PROJECT/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1
EXEC=$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
MINI_REPO=$RUN_ROOT/source_v2/mini_repo
PY=$PROJECT/../miniconda3/envs/Eventbased_WAM_torch28/bin/python
SCRIPT=$RUN_ROOT/v2e_retry01/prepare_stage7_replicad_flow_bank_retry01.py
CONFIG=$RUN_ROOT/stage7_replicad_multiaxis_gtfit_v1.json
REGISTRATION=$RUN_ROOT/replicad_layout_multiaxis_paths_registration_v1.json
OUT=$RUN_ROOT/flow_bank_retry01
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/v2e_retry01/bank_builder_started_utc.txt"
test -x "$PY" && test -f "$SCRIPT" && test -f "$CONFIG" && test -f "$REGISTRATION"
test ! -e "$OUT"
{
  printf 'host=%s\nproject=%s\nexecution_checkout=' "$(hostname)" "$PROJECT"
  git -C "$EXEC" rev-parse HEAD
  printf 'builder_sha256='; sha256sum "$SCRIPT"
  printf 'registration_sha256='; sha256sum "$REGISTRATION"
  printf 'config_sha256='; sha256sum "$CONFIG"
  printf 'mini_repo_root=%s\nphysics_warp_sha256=' "$MINI_REPO"; sha256sum "$MINI_REPO/genesis_event_plugin/physics_warp.py"
  printf 'python=%s\n' "$PY"
  "$PY" --version
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader
  uptime; free -h; df -h "$RUN_ROOT" /tmp
} > "$RUN_ROOT/v2e_retry01/bank_builder_environment.txt" 2>&1
set +e
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$EXEC:$PROJECT" "$PY" -u "$SCRIPT" \
    --run-root "$RUN_ROOT" \
    --old-run-root "$PROJECT/runs/sim2real" \
    --repo-root "$MINI_REPO" \
    --registration "$REGISTRATION" \
    --output-root "$OUT" > "$RUN_ROOT/v2e_retry01/bank_builder.stdout.log" 2> "$RUN_ROOT/v2e_retry01/bank_builder.stderr.log"
RC=$?
set -e
printf '%s\n' "$RC" > "$RUN_ROOT/v2e_retry01/bank_builder_exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/v2e_retry01/bank_builder_finished_utc.txt"
if [[ "$RC" -ne 0 ]]; then
  exit "$RC"
fi
sha256sum "$OUT/training_bank_manifest.json" > "$RUN_ROOT/v2e_retry01/bank_manifest.sha256"
printf 'bank manifest: %s\n' "$OUT/training_bank_manifest.json"
