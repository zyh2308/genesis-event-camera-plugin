#!/usr/bin/env bash
set -euo pipefail

ENV=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28
PY=$ENV/bin/python
PIP=$ENV/bin/pip
RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-genesis-torch28-setup-run03
WHEEL_DIR=/home/zhaoyuhan/Eventbased_WAM/external_sources/pytorch_wheels/stage7-20261030
TORCH=$WHEEL_DIR/torch-2.8.0+cu126-cp311-cp311-manylinux_2_28_x86_64.whl
VISION=$WHEEL_DIR/torchvision-0.23.0+cu126-cp311-cp311-manylinux_2_28_x86_64.whl
AUDIO=$WHEEL_DIR/torchaudio-2.8.0+cu126-cp311-cp311-manylinux_2_28_x86_64.whl
TRITON=$WHEEL_DIR/triton-3.4.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
EXPECTED_TORCH_SHA=542cdbf042aaf5d6ddbed43cb8cd8c4df1e586bebb1338f5dcba14fa52830d3c
EXPECTED_VISION_SHA=b697f5ca72399822d6454929cfb8835298527881e0ba58982f80ff0401331a92
EXPECTED_AUDIO_SHA=77859ec6c6a92ca3b666a3f02363a5631e9676c027fb4dce99a461a089c9962c
EXPECTED_TRITON_SHA=7b70f5e6a41e52e48cfc087436c8a28c17ff98db369447bcaff3b887a3ab4467
SESSION=sim2real_genesis_torch28_setup_20261030_03

if [[ -e "$RUN/started_utc.txt" ]]; then
  echo "refusing to overwrite existing setup run: $RUN" >&2
  exit 3
fi
if [[ ! -x "$PY" || ! -x "$PIP" || ! -f "$TORCH" || ! -f "$VISION" || ! -f "$AUDIO" || ! -f "$TRITON" ]]; then
  echo "verified local PyTorch wheels or isolated environment are missing" >&2
  exit 4
fi
ACTUAL_TORCH_SHA=$(sha256sum "$TORCH" | awk '{print $1}')
for item in \
  "$TORCH:$EXPECTED_TORCH_SHA" \
  "$VISION:$EXPECTED_VISION_SHA" \
  "$AUDIO:$EXPECTED_AUDIO_SHA" \
  "$TRITON:$EXPECTED_TRITON_SHA"; do
  wheel=${item%%:*}; expected=${item#*:}; actual=$(sha256sum "$wheel" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "wheel SHA256 mismatch for $wheel: $actual" >&2
    exit 7
  fi
done
LOAD1=$(awk '{print $1}' /proc/loadavg)
NCPU=$(getconf _NPROCESSORS_ONLN)
MEM_AVAILABLE_KB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if ! awk -v l1="$LOAD1" -v cpus="$NCPU" 'BEGIN { exit !(l1 < cpus * 0.8) }'; then
  echo "CPU/resource gate refused isolated environment update: load=$LOAD1 cpus=$NCPU" >&2
  exit 5
fi
if [[ "$MEM_AVAILABLE_KB" -lt 8388608 ]]; then
  echo "memory gate refused environment update: available_kb=$MEM_AVAILABLE_KB" >&2
  exit 5
fi
if ps -eo args | awk -v env="$ENV/" 'index($0, env) && !/awk/ {found=1} END {exit !found}'; then
  echo "another process is using the isolated environment; refusing package changes" >&2
  exit 6
fi

mkdir -p "$RUN"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_utc.txt"
{
  printf 'host='; hostname -f
  printf 'python='; "$PY" --version
  printf 'pip='; "$PIP" --version
  printf 'device=CPU-only package update; no CUDA context or Genesis renderer initialized\n'
  printf 'cpu_load1=%s\ncpu_count=%s\nmem_available_kb=%s\n' "$LOAD1" "$NCPU" "$MEM_AVAILABLE_KB"
  for wheel in "$TORCH" "$VISION" "$AUDIO" "$TRITON"; do sha256sum "$wheel"; done
  df -h /home/zhaoyuhan
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  ls -1 "$ENV/lib/python3.11/site-packages" | awk '/^(torch|torchvision|torchaudio)-.*dist-info$/ {print}'
} > "$RUN/environment_before.txt" 2>&1
"$PIP" freeze > "$RUN/packages_before.txt" 2> "$RUN/pip_freeze_before.stderr"

set +e
"$PIP" install --no-cache-dir "$TORCH" "$VISION" "$AUDIO" "$TRITON" \
  --index-url https://download.pytorch.org/whl/cu126 \
  > "$RUN/pip_install.log" 2>&1
INSTALL_RC=$?
printf '%s\n' "$INSTALL_RC" > "$RUN/pip_install_exit_code.txt"
if [[ "$INSTALL_RC" -ne 0 ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
  printf '%s\n' "$INSTALL_RC" > "$RUN/exit_code.txt"
  exit "$INSTALL_RC"
fi

"$PIP" freeze > "$RUN/packages_after.txt" 2> "$RUN/pip_freeze_after.stderr"
set +e
"$PY" -u -c 'import torch; print("torch", torch.__version__, "built_cuda", torch.version.cuda); import torchvision; print("torchvision", torchvision.__version__); import torchaudio; print("torchaudio", torchaudio.__version__); import genesis; from genesis.ext import pyrender; print("genesis", genesis.__version__); print("genesis_pyrender", pyrender.__file__)' \
  > "$RUN/cpu_import_check.log" 2>&1
IMPORT_RC=$?
printf '%s\n' "$IMPORT_RC" > "$RUN/cpu_import_check_exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
printf '%s\n' "$IMPORT_RC" > "$RUN/exit_code.txt"
exit "$IMPORT_RC"
