#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM2REAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SIM2REAL_DIR/../.." && pwd)"

python -m py_compile \
  "$SIM2REAL_DIR/scripts/audit_mvsec_real_fragment.py" \
  "$SIM2REAL_DIR/scripts/run_mvsec_candidate_inference.py" \
  "$SIM2REAL_DIR/scripts/run_pose_stage2_pilot.py" \
  "$SIM2REAL_DIR/scripts/run_pose_overfit_diagnostic.py" \
  "$SIM2REAL_DIR/scripts/rescore_steflow_checkpoints.py" \
  "$SIM2REAL_DIR/scripts/infer_hqf_e2vid_candidate.py" \
  "$SIM2REAL_DIR/tests/test_flow_epe.py" \
  "$SIM2REAL_DIR/tests/test_mvsec_adapter.py"
python -m unittest discover -s "$SIM2REAL_DIR/tests" -p 'test_*.py' -v
bash -n "$SIM2REAL_DIR/scripts/run_stage3_server_job.sh"

printf 'controlled_checks=pass\nrepository=%s\n' "$REPO_ROOT"
