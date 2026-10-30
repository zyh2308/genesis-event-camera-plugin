#!/usr/bin/env bash
set -euo pipefail

RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-g3-torch28-wheel-download-run01
DEST_DIR=/home/zhaoyuhan/Eventbased_WAM/external_sources/pytorch_wheels/stage7-20261030
WHEEL=$DEST_DIR/torch-2.8.0+cu126-cp311-cp311-manylinux_2_28_x86_64.whl
PART=$WHEEL.g3-download-part
URL=https://download-r2.pytorch.org/whl/cu126/torch-2.8.0%2Bcu126-cp311-cp311-manylinux_2_28_x86_64.whl
EXPECTED_SHA=542cdbf042aaf5d6ddbed43cb8cd8c4df1e586bebb1338f5dcba14fa52830d3c

if [[ -e "$RUN/started_utc.txt" || -e "$WHEEL" || -e "$PART" ]]; then
  echo "refusing to overwrite existing run or wheel: $RUN / $WHEEL" >&2
  exit 3
fi
if [[ ! -d /home/zhaoyuhan/Eventbased_WAM || ! -d /home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030 ]]; then
  echo "expected shared project and isolated checkout paths are unavailable" >&2
  exit 4
fi

mkdir -p "$RUN" "$DEST_DIR"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_utc.txt"
{
  printf 'host='; hostname -f
  printf 'url=%s\npart=%s\nexpected_sha256=%s\n' "$URL" "$PART" "$EXPECTED_SHA"
  printf 'activity=official PyTorch wheel download only; no training or GPU context\n'
  printf 'load='; cat /proc/loadavg
  free -h | head -2
  df -h /home/zhaoyuhan /tmp
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
} > "$RUN/environment_before.txt" 2>&1

set +e
curl --location --fail --show-error --retry 2 --retry-delay 3 --connect-timeout 15 \
  --max-time 3600 --output "$PART" "$URL" > "$RUN/curl.stdout.log" 2> "$RUN/curl.stderr.log"
CURL_RC=$?
printf '%s\n' "$CURL_RC" > "$RUN/curl_exit_code.txt"
if [[ "$CURL_RC" -ne 0 ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
  printf '%s\n' "$CURL_RC" > "$RUN/exit_code.txt"
  exit "$CURL_RC"
fi
ACTUAL_SHA=$(sha256sum "$PART" | awk '{print $1}')
printf '%s\n' "$ACTUAL_SHA" > "$RUN/wheel_sha256.txt"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "downloaded wheel SHA256 mismatch: $ACTUAL_SHA" >&2
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
  printf '8\n' > "$RUN/exit_code.txt"
  exit 8
fi
mv -- "$PART" "$WHEEL"
stat -c '%n %s bytes %y' "$WHEEL" > "$RUN/wheel_stat.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
printf '0\n' > "$RUN/exit_code.txt"
