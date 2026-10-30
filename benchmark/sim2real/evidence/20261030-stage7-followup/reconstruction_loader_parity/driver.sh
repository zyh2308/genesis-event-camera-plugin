#!/usr/bin/env bash
set -u -o pipefail

PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-recon-loader-parity-run03
PYTHON=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
REPO=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
SCRIPT="$REPO/benchmark/sim2real/scripts/audit_e2vid_train_eval_loader_parity.py"
MODEL_ROOT="$PROJECT/external/sources/downstream/event_cnn_minimal"
EVREAL_ROOT="$PROJECT/external/sources/downstream/EVREAL"
DATA_ROOT="$PROJECT/data/sim2real/stage4/e2vid_corrected_10ms_20261019_v1"

mkdir -m 755 "$RUN_ROOT"
cd "$REPO"
exec > >(tee -a "$RUN_ROOT/driver.log") 2>&1
printf '%s\n' "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" "host=$(hostname -f)" "pid=$$"
printf '%s\n' "repo_commit=$(git rev-parse HEAD)" "script_sha256=$(sha256sum "$SCRIPT" | awk '{print $1}')"
printf '%s\n' "python=$($PYTHON --version 2>&1)" "threads=2" "CUDA_VISIBLE_DEVICES=''"
sha256sum "$DATA_ROOT/eventgenesis/train/s0_00.h5" "$DATA_ROOT/v2e/train/s0_00.h5"
set +e
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PYTHON" -u "$SCRIPT" \
    --model-root "$MODEL_ROOT" --evreal-root "$EVREAL_ROOT" \
    --eventgenesis-h5 "$DATA_ROOT/eventgenesis/train/s0_00.h5" \
    --v2e-h5 "$DATA_ROOT/v2e/train/s0_00.h5" \
    --output "$RUN_ROOT/parity.json"
rc=$?
set -e
printf '%s\n' "exit_code=$rc" "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$rc" > "$RUN_ROOT/exit_code.txt"
exit "$rc"
