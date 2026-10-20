#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: download_hqf_14_bags.sh OUTPUT_DIR RUN_RECORD_DIR" >&2
  exit 2
fi

output_dir="$(realpath -m "$1")"
run_dir="$(realpath -m "$2")"
official_folder="https://drive.google.com/drive/folders/18Xdr6pxJX0ZXTrXW9tK0hC3ZpmKDIt6_"
python="/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python"

if [[ -e "$output_dir" ]] && find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty dataset directory: $output_dir" >&2
  exit 3
fi
mkdir -p "$output_dir" "$run_dir"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$run_dir/started_utc.txt"
printf '%s\n' "$official_folder" > "$run_dir/source.txt"
set +e
"$python" -m gdown --folder "$official_folder" --output "$output_dir" --remaining-ok
status=$?
set -e
printf '%s\n' "$status" > "$run_dir/exit_code.txt"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$run_dir/finished_utc.txt"
exit "$status"
