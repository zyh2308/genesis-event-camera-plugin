#!/usr/bin/env bash
set -euo pipefail

RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-g3-torch28-slow-wheels-run02
DEST=/home/zhaoyuhan/Eventbased_WAM/external_sources/pytorch_wheels/stage7-20261030
SESSION=sim2real_g3_torch28_deps_20261030_02
TORCHVISION=torchvision-0.23.0+cu126-cp311-cp311-manylinux_2_28_x86_64.whl
TORCHAUDIO=torchaudio-2.8.0+cu126-cp311-cp311-manylinux_2_28_x86_64.whl
TRITON=triton-3.4.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
URL_TORCHVISION=https://download-r2.pytorch.org/whl/cu126/torchvision-0.23.0%2Bcu126-cp311-cp311-manylinux_2_28_x86_64.whl
URL_TORCHAUDIO=https://download-r2.pytorch.org/whl/cu126/torchaudio-2.8.0%2Bcu126-cp311-cp311-manylinux_2_28_x86_64.whl
URL_TRITON=https://files.pythonhosted.org/packages/7d/39/43325b3b651d50187e591eefa22e236b2981afcebaefd4f2fc0ea99df191/triton-3.4.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
SHA_TORCHVISION=b697f5ca72399822d6454929cfb8835298527881e0ba58982f80ff0401331a92
SHA_TORCHAUDIO=77859ec6c6a92ca3b666a3f02363a5631e9676c027fb4dce99a461a089c9962c
SHA_TRITON=7b70f5e6a41e52e48cfc087436c8a28c17ff98db369447bcaff3b887a3ab4467

if [[ -e "$RUN/started_utc.txt" ]]; then
  echo "refusing to overwrite run: $RUN" >&2
  exit 3
fi
for name in "$TORCHVISION" "$TORCHAUDIO" "$TRITON"; do
  if [[ -e "$DEST/$name" ]]; then
    echo "refusing to overwrite existing wheel: $DEST/$name" >&2
    exit 4
  fi
done

mkdir -p "$RUN" "$DEST"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_utc.txt"
{
  printf 'host='; hostname -f
  printf 'activity=direct official wheel downloads only; no environment installation and no GPU use\n'
  printf 'load='; cat /proc/loadavg
  free -h | head -2
  df -h /home/zhaoyuhan /tmp
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
} > "$RUN/environment_before.txt" 2>&1

download_and_verify() {
  local name=$1 url=$2 expected=$3 part="$DEST/$1.g3-download-part"
  curl --location --fail --show-error --retry 2 --retry-delay 3 --connect-timeout 15 \
    --max-time 1800 --output "$part" "$url" >> "$RUN/curl.stdout.log" 2>> "$RUN/curl.stderr.log"
  local actual
  actual=$(sha256sum "$part" | awk '{print $1}')
  printf '%s  %s\n' "$actual" "$name" >> "$RUN/slow_wheel_sha256.txt"
  if [[ "$actual" != "$expected" ]]; then
    echo "wheel SHA256 mismatch for $name: $actual" >&2
    return 8
  fi
  mv -- "$part" "$DEST/$name"
  stat -c '%n %s bytes %y' "$DEST/$name" >> "$RUN/slow_wheel_stat.txt"
}

set +e
download_and_verify "$TORCHVISION" "$URL_TORCHVISION" "$SHA_TORCHVISION" || RC=$?
if [[ ${RC:-0} -eq 0 ]]; then download_and_verify "$TORCHAUDIO" "$URL_TORCHAUDIO" "$SHA_TORCHAUDIO" || RC=$?; fi
if [[ ${RC:-0} -eq 0 ]]; then download_and_verify "$TRITON" "$URL_TRITON" "$SHA_TRITON" || RC=$?; fi
RC=${RC:-0}
printf '%s\n' "$RC" > "$RUN/download_exit_code.txt"
if [[ "$RC" -ne 0 ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
  printf '%s\n' "$RC" > "$RUN/exit_code.txt"
  exit "$RC"
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
printf '0\n' > "$RUN/exit_code.txt"
