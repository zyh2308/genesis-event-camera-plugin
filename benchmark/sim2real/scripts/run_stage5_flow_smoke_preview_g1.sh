#!/usr/bin/env bash
set -u

REMOTE_SCRIPT="/tmp/sim2real_render_stage5_flow_smoke_preview_20261024.py"
EXPECTED_SCRIPT_SHA256="7798dc51ed015b9dd09b7c3760f3f259b51c75f7b843d4e5a093993170508452"
RUN_DIR="/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261024-stage6-historical-smoke-preview/repro_run1"
BANK="/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-stage5-flow-coverage-v3-smoke-r3/shared_trajectory.h5"
BANK_DIR="/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-stage5-flow-coverage-v3-smoke-r3"
V2E_DIR="/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-stage5-flow-coverage-v3-smoke-v2e"
PYTHON="/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python"

mkdir -p "${RUN_DIR}"
date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_DIR}/started_utc.txt"
printf '%s\n' "${PYTHON} ${REMOTE_SCRIPT} --bank ${BANK} --bank-manifest ${BANK_DIR}/bank_manifest.json --split ${BANK_DIR}/registered_split.json --v2e-events ${V2E_DIR}/s5q_02_train.h5 --v2e-manifest ${V2E_DIR}/v2e_complete_manifest.json --output ${RUN_DIR}/s5q_02_train_dt1_000.png --manifest-output ${RUN_DIR}/preview_manifest.json" > "${RUN_DIR}/command.txt"
printf '%s  %s\n' "${EXPECTED_SCRIPT_SHA256}" "${REMOTE_SCRIPT}" > "${RUN_DIR}/script_sha256_expected.txt"
sha256sum "${REMOTE_SCRIPT}" > "${RUN_DIR}/script_sha256_actual.txt"
if ! grep -q "^${EXPECTED_SCRIPT_SHA256}  ${REMOTE_SCRIPT}$" "${RUN_DIR}/script_sha256_actual.txt"; then
  printf '%s\n' "script_sha256_mismatch" > "${RUN_DIR}/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_DIR}/finished_utc.txt"
  exit 2
fi
"${PYTHON}" --version > "${RUN_DIR}/python_version.txt" 2>&1
"${PYTHON}" - <<'PY' > "${RUN_DIR}/environment.json" 2>&1
import json, h5py, numpy, PIL
print(json.dumps({"h5py": h5py.__version__, "numpy": numpy.__version__, "pillow": PIL.__version__}, sort_keys=True))
PY

"${PYTHON}" "${REMOTE_SCRIPT}" \
  --bank "${BANK}" \
  --bank-manifest "${BANK_DIR}/bank_manifest.json" \
  --split "${BANK_DIR}/registered_split.json" \
  --v2e-events "${V2E_DIR}/s5q_02_train.h5" \
  --v2e-manifest "${V2E_DIR}/v2e_complete_manifest.json" \
  --output "${RUN_DIR}/s5q_02_train_dt1_000.png" \
  --manifest-output "${RUN_DIR}/preview_manifest.json" \
  > "${RUN_DIR}/stdout.log" 2> "${RUN_DIR}/stderr.log"
EXIT_CODE=$?
printf '%s\n' "${EXIT_CODE}" > "${RUN_DIR}/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_DIR}/finished_utc.txt"
exit "${EXIT_CODE}"
