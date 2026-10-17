#!/usr/bin/env bash
# Revalidate the existing Stage-4 pilot after a failed strict replay check,
# prove one actual process-exit resume with a source-checkpoint forward oracle,
# then resume the paired 45-epoch runs into a new output directory.
set -Eeuo pipefail

PROJECT=/home/zhaoyuhan/Eventbased_WAM
REPO="$PROJECT/repositories/gec-sim2real"
SIM="$REPO/benchmark/sim2real"
CONFIG="$SIM/configs/stage4_corrected_v2_flowdt1_seed0.json"
RUN_ID=20261017-stage4-flowdt1-seed0-corrected-v5
RUN_ROOT="$PROJECT/runs/sim2real/$RUN_ID"
AUDIT_ROOT="$RUN_ROOT/resume_retry1"
FULL_ROOT="$RUN_ROOT/full_training_retry1"
BANK_ROOT="$PROJECT/data/sim2real/stage4/training_bank_v3_geometry_corrected_20261017_32x2s"
V2E_ROOT="$RUN_ROOT/v2e_events"
DATA_ROOT="$PROJECT/data/sim2real/stage4/$RUN_ID/steflow_30ms"
MODEL_ROOT="$PROJECT/external/sources/downstream/STE-FlowNet"
GENESIS_ROOT="$PROJECT/runtime/sim2real/genesis_hdr_v2"
ENV_ROOT="/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM"
PY="$ENV_ROOT/bin/python"
PREFLIGHT_REPORT="$AUDIT_ROOT/preflight_result_resume_retry1.json"
SOURCE_CHECKPOINT="$RUN_ROOT/pilot/eventgenesis/last.pt"
RESUMED_RUN_ROOT="$RUN_ROOT/resume_equivalence/runs/process_exit_resume_from_update198"
COMMIT=$(git -C "$REPO" rev-parse HEAD)

[[ "$(whoami)" == zhaoyuhan ]] || { echo "wrong server user" >&2; exit 2; }
[[ "$(git -C "$REPO" branch --show-current)" == exp/public-sim2real-v1 ]] || { echo "wrong branch" >&2; exit 2; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "repo working tree must be clean" >&2; exit 2; }
[[ -f "$RUN_ROOT/continue_after_data.exit_code.txt" && "$(cat "$RUN_ROOT/continue_after_data.exit_code.txt")" == 1 ]] || {
  echo "expected preserved strict-replay failure before this retry" >&2; exit 2;
}
[[ -f "$RUN_ROOT/pilot_validation.exit_code.txt" && "$(cat "$RUN_ROOT/pilot_validation.exit_code.txt")" == 0 ]] || {
  echo "existing paired pilot did not pass validation" >&2; exit 2;
}
[[ -s "$RUN_ROOT/paired_pilot_validation.json" && -s "$SOURCE_CHECKPOINT" ]] || {
  echo "validated pilot receipt/checkpoint is missing" >&2; exit 2;
}
[[ -s "$RESUMED_RUN_ROOT/eventgenesis/last.pt" && -s "$RESUMED_RUN_ROOT/eventgenesis/training_metrics.jsonl" ]] || {
  echo "actual process-exit/resumed update-199 artifacts are missing" >&2; exit 2;
}
[[ ! -e "$AUDIT_ROOT" && ! -e "$FULL_ROOT" ]] || {
  echo "refusing to overwrite retry evidence or full-run output" >&2; exit 2;
}

ensure_idle_gpu() {
  local gpu="$1" stats used util uuid pid process_uuid owner
  uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
  stats=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r used util <<< "$(awk -F, '{gsub(/ /,"",$1);gsub(/ /,"",$2);print $1,$2}' <<< "$stats")"
  [[ "$used" -le 1024 && "$util" -le 5 ]] || { echo "GPU $gpu occupied: $stats" >&2; return 1; }
  while IFS=, read -r pid _ process_uuid; do
    pid=${pid// /}; process_uuid=${process_uuid// /}
    [[ "$process_uuid" == "$uuid" ]] || continue
    if [[ -n "$pid" ]]; then
      owner=$(ps -o user= -p "$pid" | tr -d ' ' || true)
      echo "GPU $gpu has active PID $pid owner=${owner:-unknown}" >&2
      return 1
    fi
  done < <(nvidia-smi --query-compute-apps=pid,used_gpu_memory,gpu_uuid --format=csv,noheader)
}
ensure_idle_gpu 2 || exit 3
ensure_idle_gpu 4 || exit 3
ensure_idle_gpu 1 || exit 3

mkdir -p "$AUDIT_ROOT"
date -u +%FT%TZ > "$AUDIT_ROOT/started_utc.txt"
printf '%s\n' "$$" > "$AUDIT_ROOT/supervisor.pid"
trap 'rc=$?; echo "$rc" > "$AUDIT_ROOT/exit_code.txt"; date -u +%FT%TZ > "$AUDIT_ROOT/finished_utc.txt"' EXIT

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYOPENGL_PLATFORM=egl
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export GS_CACHE_FILE_PATH="$RUN_ROOT/cache/genesis"
export TI_OFFLINE_CACHE=1 TI_OFFLINE_CACHE_FILE_PATH="$RUN_ROOT/cache/taichi"
export PYTHONPATH="$PROJECT/runtime/sim2real/genesis_hdr_v2:$REPO"
mkdir -p "$XDG_CACHE_HOME" "$GS_CACHE_FILE_PATH" "$TI_OFFLINE_CACHE_FILE_PATH"

{
  date -u +%FT%TZ
  hostname
  printf 'repo=%s\nbranch=%s\ncommit=%s\nrun_root=%s\n' "$REPO" exp/public-sim2real-v1 "$COMMIT" "$RUN_ROOT"
  git -C "$REPO" status --short --branch
  nvidia-smi
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv
  free -h
  df -h /home
} > "$AUDIT_ROOT/environment_preflight.txt" 2>&1

run_step() {
  local label="$1"; shift
  printf '%q ' "$@" > "$AUDIT_ROOT/$label.command.txt"; printf '\n' >> "$AUDIT_ROOT/$label.command.txt"
  date -u +%FT%TZ > "$AUDIT_ROOT/$label.started_utc.txt"
  if "$@" > "$AUDIT_ROOT/$label.stdout.log" 2> "$AUDIT_ROOT/$label.stderr.log"; then
    echo 0 > "$AUDIT_ROOT/$label.exit_code.txt"; date -u +%FT%TZ > "$AUDIT_ROOT/$label.finished_utc.txt"
  else
    rc=$?; echo "$rc" > "$AUDIT_ROOT/$label.exit_code.txt"; date -u +%FT%TZ > "$AUDIT_ROOT/$label.finished_utc.txt"
    echo "$label failed with exit code $rc" >&2; return "$rc"
  fi
}

export CUDA_VISIBLE_DEVICES=1 STAGE4_PHYSICAL_GPU=1 STAGE4_REPO_COMMIT="$COMMIT"
run_step full_preflight_retry1 "$PY" -u "$SIM/scripts/stage4_correctness_preflight.py" \
  --stage full --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" \
  --v2e-root "$V2E_ROOT" --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" \
  --genesis-root "$GENESIS_ROOT" --output "$PREFLIGHT_REPORT"

PREFLIGHT_ARGS=(--report "$PREFLIGHT_REPORT" --repo-root "$REPO" --config "$CONFIG" \
  --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" --data-root "$DATA_ROOT" \
  --model-root "$MODEL_ROOT" --genesis-root "$GENESIS_ROOT")
run_step attach_existing_pilot_validation "$PY" -u "$SIM/scripts/stage4_update_preflight_result.py" \
  "${PREFLIGHT_ARGS[@]}" --name paired_formal_epoch_pilot --status PASS \
  --evidence "$RUN_ROOT/paired_pilot_validation.json" \
  --command "reuse previously validated 198-update paired pilot; training source files/config/data unchanged by resume-oracle fix"

export CUDA_VISIBLE_DEVICES=2 STAGE4_PHYSICAL_GPU=2
run_step resume_checkpoint_oracle "$PY" -u "$SIM/scripts/stage4_resume_checkpoint_oracle.py" \
  --repo-root "$REPO" --config "$CONFIG" --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" \
  --source-checkpoint "$SOURCE_CHECKPOINT" --resumed-run-root "$RESUMED_RUN_ROOT" \
  --generator eventgenesis --seed 0 --output "$AUDIT_ROOT/resume_checkpoint_oracle.json"
run_step attach_resume_checkpoint_oracle "$PY" -u "$SIM/scripts/stage4_update_preflight_result.py" \
  "${PREFLIGHT_ARGS[@]}" --name resume_equivalence_epoch_boundary --status PASS \
  --evidence "$AUDIT_ROOT/resume_checkpoint_oracle.json" \
  --command "actual fresh process resumed update198 checkpoint; independent sampler and forward-loss oracle verified update199"

run_step launcher_negative_controls_retry1 "$PY" -u "$SIM/scripts/stage4_gate_negative_controls.py" \
  --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
  --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$GENESIS_ROOT" \
  --receipt "$PREFLIGHT_REPORT" --output "$AUDIT_ROOT/launcher_negative_controls"
run_step attach_negative_controls_retry1 "$PY" -u "$SIM/scripts/stage4_update_preflight_result.py" \
  "${PREFLIGHT_ARGS[@]}" --name launcher_negative_controls --status PASS \
  --evidence "$AUDIT_ROOT/launcher_negative_controls/launcher_negative_controls.json" \
  --command "production stage4_launch_gate negative controls blocked each invalid receipt/config/data case before sentinel and run root"

run_step final_full_gate_verify_retry1 "$PY" -u "$SIM/scripts/stage4_launch_gate.py" --phase full \
  --receipt "$PREFLIGHT_REPORT" --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" \
  --v2e-root "$V2E_ROOT" --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" \
  --genesis-root "$GENESIS_ROOT" --run-root "$FULL_ROOT" --generator eventgenesis \
  --resume "$SOURCE_CHECKPOINT" --verify-only

for gpu in 2 4; do ensure_idle_gpu "$gpu" || { echo "GPU $gpu occupied before paired full resume" >&2; exit 3; }; done
mkdir -p "$FULL_ROOT"
for spec in "eventgenesis 2" "v2e 4"; do
  read -r generator gpu <<< "$spec"
  run_dir="$FULL_ROOT/$generator"
  date -u +%FT%TZ > "$AUDIT_ROOT/full_resume_${generator}.started_utc.txt"
  printf 'CUDA_VISIBLE_DEVICES=%s STAGE4_PHYSICAL_GPU=%s STAGE4_REPO_COMMIT=%s OMP_NUM_THREADS=4 ' "$gpu" "$gpu" "$COMMIT" > "$AUDIT_ROOT/full_resume_${generator}.command.txt"
  printf '%q ' "$PY" -u "$SIM/scripts/stage4_launch_gate.py" --phase full --receipt "$PREFLIGHT_REPORT" \
    --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
    --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$GENESIS_ROOT" \
    --run-root "$FULL_ROOT" --generator "$generator" --resume "$RUN_ROOT/pilot/$generator/last.pt" \
    >> "$AUDIT_ROOT/full_resume_${generator}.command.txt"
  printf '\n' >> "$AUDIT_ROOT/full_resume_${generator}.command.txt"
  CUDA_VISIBLE_DEVICES="$gpu" STAGE4_PHYSICAL_GPU="$gpu" STAGE4_REPO_COMMIT="$COMMIT" OMP_NUM_THREADS=4 \
    "$PY" -u "$SIM/scripts/stage4_launch_gate.py" --phase full --receipt "$PREFLIGHT_REPORT" \
    --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
    --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$GENESIS_ROOT" \
    --run-root "$FULL_ROOT" --generator "$generator" --resume "$RUN_ROOT/pilot/$generator/last.pt" \
    > "$AUDIT_ROOT/full_resume_${generator}.stdout.log" 2> "$AUDIT_ROOT/full_resume_${generator}.stderr.log" &
  pid=$!; echo "$pid" > "$AUDIT_ROOT/full_resume_${generator}.pid"
done
pid_eg=$(cat "$AUDIT_ROOT/full_resume_eventgenesis.pid")
pid_v2e=$(cat "$AUDIT_ROOT/full_resume_v2e.pid")
printf 'eventgenesis=%s\nv2e=%s\n' "$pid_eg" "$pid_v2e" > "$AUDIT_ROOT/full_training_pids.txt"
printf 'eventgenesis=%s v2e=%s\n' "$pid_eg" "$pid_v2e" > "$AUDIT_ROOT/full_training_started.txt"
echo "paired full-budget resume started; outputs are isolated under $FULL_ROOT" > "$AUDIT_ROOT/campaign_started.txt"

eg_rc=0; wait "$pid_eg" || eg_rc=$?
v2e_rc=0; wait "$pid_v2e" || v2e_rc=$?
echo "$eg_rc" > "$AUDIT_ROOT/full_resume_eventgenesis.exit_code.txt"
echo "$v2e_rc" > "$AUDIT_ROOT/full_resume_v2e.exit_code.txt"
date -u +%FT%TZ > "$AUDIT_ROOT/full_resume_eventgenesis.finished_utc.txt"
date -u +%FT%TZ > "$AUDIT_ROOT/full_resume_v2e.finished_utc.txt"
[[ "$eg_rc" -eq 0 && "$v2e_rc" -eq 0 ]] || { echo "full training failed: eventgenesis=$eg_rc v2e=$v2e_rc" >&2; exit 5; }
echo "both paired 45-epoch runs completed successfully" > "$AUDIT_ROOT/campaign_complete.txt"
