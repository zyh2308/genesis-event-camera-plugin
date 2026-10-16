#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$1"
GPU_PHYSICAL="$2"
shift 2
if [[ -e "$RUN_DIR/started_utc.txt" ]]; then
  echo "refusing to reuse an existing Stage 3 run directory: $RUN_DIR" >&2
  exit 90
fi
mkdir -p "$RUN_DIR"
exec >"$RUN_DIR/stdout.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/started_utc.txt"
printf '%s\n' "$$" > "$RUN_DIR/pid.txt"
printf '%q ' "$@" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"
record_exit() {
  local rc=$?
  printf '%s\n' "$rc" > "$RUN_DIR/exit_code.txt" || true
  date -u +%FT%TZ > "$RUN_DIR/finished_utc.txt" || true
}
trap record_exit EXIT
source /home/zhaoyuhan/miniconda3/bin/activate Eventbased_WAM
{
  printf 'hostname='; hostname
  printf 'python='; python --version
  python - <<'PY'
import importlib.metadata
import torch
for name in ("numpy", "h5py", "opencv-python", "rosbags", "PyYAML"):
    try:
        print(f"{name}={importlib.metadata.version(name)}")
    except importlib.metadata.PackageNotFoundError:
        pass
print(f"torch={torch.__version__}")
print(f"torch_cuda_available={torch.cuda.is_available()}")
PY
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
} > "$RUN_DIR/environment.txt" 2>&1

if [[ "$GPU_PHYSICAL" != "none" ]]; then
  PREFLIGHT_FILE="$RUN_DIR/gpu${GPU_PHYSICAL}_preflight.txt"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader -i "$GPU_PHYSICAL" > "$PREFLIGHT_FILE"
  python - "$PREFLIGHT_FILE" "$GPU_PHYSICAL" <<'PY'
import sys
row = open(sys.argv[1]).read().strip().splitlines()[0]
fields = [field.strip() for field in row.split(",")]
if fields[0] != sys.argv[2]:
    raise SystemExit("GPU preflight returned a different physical GPU")
used_mib = int(fields[2].split()[0])
util_pct = int(fields[4].split()[0])
if used_mib > 1024 or util_pct > 10:
    raise SystemExit(f"GPU{sys.argv[2]} is busy (used={used_mib} MiB, utilization={util_pct}%)")
print(f"assigned physical GPU {sys.argv[2]} accepted: {used_mib} MiB used, {util_pct}% utilization")
PY
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL"
fi

set +e
"$@"
RC=$?
set -e
exit "$RC"
