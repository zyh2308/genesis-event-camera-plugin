#!/usr/bin/env bash
set -u

ROOT_DIR="/home/科研/Eventbased_WAM/repositories/gec-sim2real"
MODEL_ROOT="/home/科研/Eventbased_WAM/external_sources/STE-FlowNet"
OUTPUT_DIR="${ROOT_DIR}/benchmark/sim2real/evidence/20261024-flow-entrypoint/recheck"

date -u +%Y-%m-%dT%H:%M:%SZ > "${OUTPUT_DIR}/started_utc.txt"
python3 --version > "${OUTPUT_DIR}/python_version.txt" 2>&1
python3 - <<'PY' > "${OUTPUT_DIR}/torch_environment.txt" 2>&1
import json
import torch
print(json.dumps({
    "torch": torch.__version__,
    "cuda_available_to_process": torch.cuda.is_available(),
    "cuda_build": torch.version.cuda,
}, sort_keys=True))
PY

python3 "${ROOT_DIR}/benchmark/sim2real/scripts/validate_stage5_flow_entrypoint_cpu_shim.py" \
  --repo-root "${ROOT_DIR}" \
  --model-root "${MODEL_ROOT}" \
  --output "${OUTPUT_DIR}/fingerprint_cpu_shim.json" \
  > "${OUTPUT_DIR}/stdout.log" 2> "${OUTPUT_DIR}/stderr.log"
EXIT_CODE=$?
printf '%s\n' "${EXIT_CODE}" > "${OUTPUT_DIR}/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${OUTPUT_DIR}/finished_utc.txt"
exit "${EXIT_CODE}"
