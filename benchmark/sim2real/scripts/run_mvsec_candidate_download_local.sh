#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <repo-root> <run-root> <data-root> <verified-stage3-root>" >&2
  exit 2
fi

repo_root="$1"
run_root="$2"
data_root="$3"
seed_from="$4"
expected_data_root="/home/科研/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-download-v2"

if [[ "$data_root" != "$expected_data_root" ]]; then
  echo "refusing unexpected WSL staging root: $data_root" >&2
  exit 2
fi
if [[ ! -f "$repo_root/benchmark/sim2real/scripts/download_mvsec_candidate_hdf5.py" ]]; then
  echo "download entrypoint missing from repository: $repo_root" >&2
  exit 2
fi
if [[ ! -d "$seed_from/indoor_flying" ]]; then
  echo "verified Stage-3 seed source is missing: $seed_from" >&2
  exit 2
fi

python_bin="$(command -v python3)"
mkdir -p "$run_root" "$data_root"
printf '%s\n' "$$" > "$run_root/pid.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_root/started_utc.txt"
printf '%s\n' "$python_bin -u $repo_root/benchmark/sim2real/scripts/download_mvsec_candidate_hdf5.py --root $data_root --seed-from $seed_from" > "$run_root/command.txt"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  hostname
  git -C "$repo_root" rev-parse HEAD
  git -C "$repo_root" status --porcelain
  "$python_bin" --version
  "$python_bin" -c 'import gdown, h5py, numpy; print("gdown", gdown.__version__); print("h5py", h5py.__version__); print("numpy", numpy.__version__)'
  df -h "$data_root"
} > "$run_root/environment.txt" 2>&1

set +e
"$python_bin" -u "$repo_root/benchmark/sim2real/scripts/download_mvsec_candidate_hdf5.py" \
  --root "$data_root" --seed-from "$seed_from" \
  > "$run_root/stdout.log" 2> "$run_root/stderr.log"
exit_code=$?
set -e
printf '%s\n' "$exit_code" > "$run_root/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_root/finished_utc.txt"
exit "$exit_code"
