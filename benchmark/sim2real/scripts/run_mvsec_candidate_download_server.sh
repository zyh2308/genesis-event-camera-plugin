#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <clean-repo-root> <run-root> <data-root>" >&2
  exit 2
fi

repo_root="$1"
run_root="$2"
data_root="$3"
expected_data_root="/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC"
python_bin="/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python"

if [[ "$data_root" != "$expected_data_root" ]]; then
  echo "refusing unexpected data root: $data_root" >&2
  exit 2
fi
if [[ ! -f "$repo_root/benchmark/sim2real/scripts/download_mvsec_candidate_hdf5.py" ]]; then
  echo "download entrypoint missing from repo: $repo_root" >&2
  exit 2
fi

mkdir -p "$run_root" "$data_root"
printf '%s\n' "$$" > "$run_root/pid.txt"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$run_root/started_utc.txt"
printf '%s\n' "$python_bin -u $repo_root/benchmark/sim2real/scripts/download_mvsec_candidate_hdf5.py --root $data_root" > "$run_root/command.txt"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  hostname -f
  git -C "$repo_root" rev-parse HEAD
  git -C "$repo_root" status --porcelain
  "$python_bin" --version
  "$python_bin" -c 'import gdown, h5py, numpy; print("gdown", gdown.__version__); print("h5py", h5py.__version__); print("numpy", numpy.__version__)'
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
} > "$run_root/environment_and_prelaunch_resources.txt" 2>&1

set +e
"$python_bin" -u "$repo_root/benchmark/sim2real/scripts/download_mvsec_candidate_hdf5.py" \
  --root "$data_root" > "$run_root/stdout.log" 2> "$run_root/stderr.log"
exit_code=$?
set -e
printf '%s\n' "$exit_code" > "$run_root/exit_code.txt"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$run_root/finished_utc.txt"
exit "$exit_code"
