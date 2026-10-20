#!/usr/bin/env bash
set -uo pipefail
REPO=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real-exec-bb71f7b1
RAW=/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-hqf14-raw
MEMMAP=/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-hqf14-memmap-v3
RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-hqf14-conversion-v3
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
DEPS=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-hqf-convert-deps/pydeps
if [[ -e "$RUN" || -e "$MEMMAP" ]]; then
  echo "refusing existing run/output path" >&2
  exit 4
fi
mkdir -p "$RUN"
cp "$0" "$RUN/driver.sh"
printf '%s\n' "$$" > "$RUN/driver_pid.txt"
date -u +%FT%TZ > "$RUN/started_utc.txt"
printf '%s\n' "bash $REPO/benchmark/sim2real/scripts/convert_hqf_14_dataset.sh $RAW $MEMMAP $RUN/manifest.json" > "$RUN/command.txt"
"$PY" --version > "$RUN/python_version.txt" 2>&1
printf '%s\n' "rosbags isolated via PYTHONPATH=$DEPS" > "$RUN/environment.txt"
printf '%s\n' "bb71f7b1ddc586269352cbdba1b0704e8364d4df" > "$RUN/source_revision.txt"
sha256sum "$REPO/benchmark/sim2real/scripts/convert_hqf_bag_rosbags.py" "$REPO/benchmark/sim2real/scripts/build_hqf_14_file_manifest.py" "$REPO/benchmark/sim2real/scripts/convert_hqf_14_dataset.sh" > "$RUN/source_sha256.txt"
export PYTHONPATH="$DEPS"
set +e
PYTHON_BIN="$PY" bash "$REPO/benchmark/sim2real/scripts/convert_hqf_14_dataset.sh" "$RAW" "$MEMMAP" "$RUN/manifest.json" > "$RUN/conversion.stdout.log" 2> "$RUN/conversion.stderr.log"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN/exit_code.txt"
date -u +%FT%TZ > "$RUN/finished_utc.txt"
exit "$rc"
