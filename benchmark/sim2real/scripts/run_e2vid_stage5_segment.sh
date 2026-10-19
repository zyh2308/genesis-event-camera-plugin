#!/usr/bin/env bash
# User-level wrapper for one bounded or full E2VID Stage-5 segment.
set -u
if [ "$#" -lt 10 ]; then
  echo "usage: $0 GENERATOR WORKTREE MODEL_ROOT DATA_ROOT CONFIG GROUP_MANIFEST RUN_ROOT LOG_FILE EXIT_FILE MAX_UPDATES [RESUME_CHECKPOINT]" >&2
  exit 64
fi
generator=$1
worktree=$2
model_root=$3
data_root=$4
config=$5
group_manifest=$6
run_root=$7
log_file=$8
exit_file=$9
max_updates=${10}
resume_checkpoint=${11:-}
if [ "$generator" != eventgenesis ] && [ "$generator" != v2e ]; then
  echo "invalid generator: $generator" >&2
  exit 64
fi
mkdir -p "$run_root" "$(dirname "$log_file")" "$(dirname "$exit_file")"
source /home/zhaoyuhan/miniconda3/bin/activate Eventbased_WAM || exit 70
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2
cd "$worktree" || exit 72
python - <<'PY'
import json, os, platform, torch
print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                  "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                  "visible_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "pid": os.getpid()}))
PY
pid=$$
printf '%s\n' "$pid" > "$run_root/segment.pid"
printf 'started_utc=%s\n' "$(date -u -Is)" >> "$run_root/segment_times.txt"
printf 'pid=%s\n' "$pid" >> "$run_root/segment_times.txt"
cmd=(python -u benchmark/sim2real/scripts/train_e2vid_stage5.py
  --model-root "$model_root" --data-root "$data_root" --config "$config"
  --output-root "$run_root" --generator "$generator" --seed 0 --device cuda:0
  --group-manifest "$group_manifest")
if [ "$max_updates" -gt 0 ]; then cmd+=(--max-updates "$max_updates"); fi
if [ -n "$resume_checkpoint" ]; then cmd+=(--resume "$resume_checkpoint"); fi
printf '%q ' "${cmd[@]}" > "$run_root/segment_command.txt"
printf '\n' >> "$run_root/segment_command.txt"
"${cmd[@]}" > "$log_file" 2>&1
rc=$?
printf '%s\n' "$rc" > "$exit_file"
printf 'finished_utc=%s\n' "$(date -u -Is)" >> "$run_root/segment_times.txt"
printf 'exit_code=%s\n' "$rc" >> "$run_root/segment_times.txt"
exit "$rc"
