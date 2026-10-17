#!/usr/bin/env bash
# Resume the corrected Stage-4 campaign after complete immutable data already exists.
set -Eeuo pipefail

[[ "${1:-}" == "--continue-after-data" ]] || {
  echo "usage: bash $0 --continue-after-data" >&2; exit 2;
}

PROJECT=/home/zhaoyuhan/Eventbased_WAM
REPO="$PROJECT/repositories/gec-sim2real"
SIM="$REPO/benchmark/sim2real"
CONFIG="$SIM/configs/stage4_corrected_v2_flowdt1_seed0.json"
RUN_ID=20261017-stage4-flowdt1-seed0-corrected-v5
RUN_ROOT="$PROJECT/runs/sim2real/$RUN_ID"
BANK_ROOT="$PROJECT/data/sim2real/stage4/training_bank_v3_geometry_corrected_20261017_32x2s"
V2E_ROOT="$RUN_ROOT/v2e_events"
DATA_ROOT="$PROJECT/data/sim2real/stage4/$RUN_ID/steflow_30ms"
MODEL_ROOT="$PROJECT/external/sources/downstream/STE-FlowNet"
ENV_ROOT="/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM"
PY="$ENV_ROOT/bin/python"
PREFLIGHT_REPORT="$RUN_ROOT/preflight_result_corrected_v2.json"
COMMIT=$(git -C "$REPO" rev-parse HEAD)

[[ "$(whoami)" == zhaoyuhan ]] || { echo "wrong server user" >&2; exit 2; }
[[ "$(git -C "$REPO" branch --show-current)" == exp/public-sim2real-v1 ]] || { echo "wrong branch" >&2; exit 2; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "repo working tree must be clean" >&2; exit 2; }
[[ -f "$RUN_ROOT/exit_code.txt" && "$(cat "$RUN_ROOT/exit_code.txt")" == 2 ]] || {
  echo "expected original campaign to have stopped at the failed full preflight" >&2; exit 2;
}
[[ ! -e "$RUN_ROOT/pilot" && ! -e "$RUN_ROOT/full_training_pids.txt" ]] || {
  echo "refusing to overwrite a prior pilot or training attempt" >&2; exit 2;
}
[[ -s "$BANK_ROOT/shared_trajectory.h5" && -s "$BANK_ROOT/bank_manifest.json" && -s "$BANK_ROOT/registered_split.json" ]] || {
  echo "corrected shared bank is incomplete" >&2; exit 2;
}
[[ -s "$V2E_ROOT/v2e_complete_manifest.json" && -s "$DATA_ROOT/training_bank_manifest.json" ]] || {
  echo "complete V2E or paired adapter manifest is missing" >&2; exit 2;
}
[[ -f "$RUN_ROOT/v2e_finalize.exit_code.txt" && "$(cat "$RUN_ROOT/v2e_finalize.exit_code.txt")" == 0 ]] || exit 2
[[ -f "$RUN_ROOT/paired_steflow_adapter.exit_code.txt" && "$(cat "$RUN_ROOT/paired_steflow_adapter.exit_code.txt")" == 0 ]] || exit 2
[[ -f "$V2E_ROOT/gpu2/worker.exit_code.txt" && "$(cat "$V2E_ROOT/gpu2/worker.exit_code.txt")" == 0 ]] || exit 2
[[ -f "$V2E_ROOT/gpu4/worker.exit_code.txt" && "$(cat "$V2E_ROOT/gpu4/worker.exit_code.txt")" == 0 ]] || exit 2
[[ ! -e "$PREFLIGHT_REPORT" ]] || { echo "corrected report path already exists" >&2; exit 2; }

# Permit this continuation only for the exact known reduction-axis false reject.
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); f=[x for x in d["checks"] if x["status"]!="PASS"]; assert len(f)==1 and f[0]["check"]=="bank_manifest_and_geometry" and "max=1.7672651665634476e-05" in f[0]["actual"], f' "$RUN_ROOT/preflight_result.json"

ensure_idle_gpu() {
  local gpu="$1" uuid stats used util pid process_uuid owner
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

ensure_idle_gpu 1 || exit 3
for gpu in 2 4; do ensure_idle_gpu "$gpu" || exit 3; done

printf '%s\n' "$$" > "$RUN_ROOT/continue_after_data.pid"
date -u +%FT%TZ > "$RUN_ROOT/continue_after_data.started_utc.txt"
trap 'rc=$?; echo "$rc" > "$RUN_ROOT/continue_after_data.exit_code.txt"; date -u +%FT%TZ > "$RUN_ROOT/continue_after_data.finished_utc.txt"' EXIT

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYOPENGL_PLATFORM=egl
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export GS_CACHE_FILE_PATH="$RUN_ROOT/cache/genesis"
export TI_OFFLINE_CACHE=1 TI_OFFLINE_CACHE_FILE_PATH="$RUN_ROOT/cache/taichi"
export PYTHONPATH="$PROJECT/runtime/sim2real/genesis_hdr_v2:$REPO"
mkdir -p "$XDG_CACHE_HOME" "$GS_CACHE_FILE_PATH" "$TI_OFFLINE_CACHE_FILE_PATH"

{
  date -u +%FT%TZ
  hostname
  printf 'project=%s\nrepo=%s\nbranch=%s\ncommit=%s\nconfig=%s\n' "$PROJECT" "$REPO" exp/public-sim2real-v1 "$COMMIT" "$CONFIG"
  git -C "$REPO" status --short --branch
  nvidia-smi
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv
  free -h
  df -h /home
} > "$RUN_ROOT/continuation_environment_preflight.txt" 2>&1

run_step() {
  local label="$1"; shift
  printf '%q ' "$@" > "$RUN_ROOT/$label.command.txt"; printf '\n' >> "$RUN_ROOT/$label.command.txt"
  date -u +%FT%TZ > "$RUN_ROOT/$label.started_utc.txt"
  if "$@" > "$RUN_ROOT/$label.stdout.log" 2> "$RUN_ROOT/$label.stderr.log"; then
    echo 0 > "$RUN_ROOT/$label.exit_code.txt"; date -u +%FT%TZ > "$RUN_ROOT/$label.finished_utc.txt"
  else
    rc=$?; echo "$rc" > "$RUN_ROOT/$label.exit_code.txt"; date -u +%FT%TZ > "$RUN_ROOT/$label.finished_utc.txt"
    echo "$label failed with exit code $rc" >&2; return "$rc"
  fi
}

export CUDA_VISIBLE_DEVICES=1 STAGE4_PHYSICAL_GPU=1 STAGE4_REPO_COMMIT="$COMMIT"
run_step full_correctness_preflight_corrected_v2 "$PY" -u "$SIM/scripts/stage4_correctness_preflight.py" \
  --stage full --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" \
  --v2e-root "$V2E_ROOT" --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" \
  --genesis-root "$PROJECT/runtime/sim2real/genesis_hdr_v2" --output "$PREFLIGHT_REPORT"

for gpu in 2 4; do ensure_idle_gpu "$gpu" || { echo "GPU $gpu occupied before paired pilot" >&2; exit 3; }; done

PRECEIPT_ARGS=(--report "$PREFLIGHT_REPORT" --repo-root "$REPO" --config "$CONFIG" \
  --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" --data-root "$DATA_ROOT" \
  --model-root "$MODEL_ROOT" --genesis-root "$PROJECT/runtime/sim2real/genesis_hdr_v2")

pilot_gate="$SIM/scripts/stage4_launch_gate.py"
date -u +%FT%TZ > "$RUN_ROOT/paired_pilot.started_utc.txt"
for spec in "eventgenesis 2" "v2e 4"; do
  read -r generator gpu <<< "$spec"
  printf 'CUDA_VISIBLE_DEVICES=%s STAGE4_PHYSICAL_GPU=%s STAGE4_REPO_COMMIT=%s OMP_NUM_THREADS=4 ' "$gpu" "$gpu" "$COMMIT" > "$RUN_ROOT/pilot_${generator}.command.txt"
  printf '%q ' "$PY" -u "$pilot_gate" --phase pilot --receipt "$PREFLIGHT_REPORT" \
    --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
    --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$PROJECT/runtime/sim2real/genesis_hdr_v2" \
    --run-root "$RUN_ROOT/pilot" --generator "$generator" --max-updates 198 >> "$RUN_ROOT/pilot_${generator}.command.txt"
  printf '\n' >> "$RUN_ROOT/pilot_${generator}.command.txt"
  CUDA_VISIBLE_DEVICES="$gpu" STAGE4_PHYSICAL_GPU="$gpu" STAGE4_REPO_COMMIT="$COMMIT" OMP_NUM_THREADS=4 \
    "$PY" -u "$pilot_gate" --phase pilot --receipt "$PREFLIGHT_REPORT" \
    --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
    --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$PROJECT/runtime/sim2real/genesis_hdr_v2" \
    --run-root "$RUN_ROOT/pilot" --generator "$generator" --max-updates 198 \
    > "$RUN_ROOT/pilot_${generator}.stdout.log" 2> "$RUN_ROOT/pilot_${generator}.stderr.log" &
  pid=$!; echo "$pid" > "$RUN_ROOT/pilot_${generator}.pid"
done
pilot_eg=$(cat "$RUN_ROOT/pilot_eventgenesis.pid"); pilot_v2e=$(cat "$RUN_ROOT/pilot_v2e.pid")
printf 'eventgenesis=%s\nv2e=%s\n' "$pilot_eg" "$pilot_v2e" > "$RUN_ROOT/pilot_pids.txt"
pilot_eg_rc=0; wait "$pilot_eg" || pilot_eg_rc=$?
pilot_v2e_rc=0; wait "$pilot_v2e" || pilot_v2e_rc=$?
echo "$pilot_eg_rc" > "$RUN_ROOT/pilot_eventgenesis.exit_code.txt"
echo "$pilot_v2e_rc" > "$RUN_ROOT/pilot_v2e.exit_code.txt"
date -u +%FT%TZ > "$RUN_ROOT/pilot_eventgenesis.finished_utc.txt"
date -u +%FT%TZ > "$RUN_ROOT/pilot_v2e.finished_utc.txt"
date -u +%FT%TZ > "$RUN_ROOT/paired_pilot.finished_utc.txt"
[[ "$pilot_eg_rc" -eq 0 && "$pilot_v2e_rc" -eq 0 ]] || { echo "paired pilot failed: eventgenesis=$pilot_eg_rc v2e=$pilot_v2e_rc" >&2; exit 5; }

run_step pilot_validation "$PY" -u "$SIM/scripts/stage4_validate_pilot.py" \
  --run-root "$RUN_ROOT/pilot" --output "$RUN_ROOT/paired_pilot_validation.json"
run_step receipt_add_pilot "$PY" -u "$SIM/scripts/stage4_update_preflight_result.py" \
  "${PRECEIPT_ARGS[@]}" --name paired_formal_epoch_pilot --status PASS \
  --evidence "$RUN_ROOT/paired_pilot_validation.json" --command "two stage4_launch_gate.py --phase pilot --max-updates 198 runs"

ensure_idle_gpu 2 || { echo "GPU 2 became occupied before resume check" >&2; exit 3; }
run_step formal_resume_equivalence "$PY" -u "$SIM/scripts/stage4_resume_equivalence.py" \
  --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
  --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$PROJECT/runtime/sim2real/genesis_hdr_v2" \
  --receipt "$PREFLIGHT_REPORT" --pilot-run-root "$RUN_ROOT/pilot" \
  --output-root "$RUN_ROOT/resume_equivalence" --gpu 2
run_step receipt_add_resume "$PY" -u "$SIM/scripts/stage4_update_preflight_result.py" \
  "${PRECEIPT_ARGS[@]}" --name resume_equivalence_epoch_boundary --status PASS \
  --evidence "$RUN_ROOT/resume_equivalence/resume_equivalence.json" \
  --command "python scripts/stage4_resume_equivalence.py; process exit then gate --resume across update 198/199"

run_step launcher_negative_controls "$PY" -u "$SIM/scripts/stage4_gate_negative_controls.py" \
  --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
  --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$PROJECT/runtime/sim2real/genesis_hdr_v2" \
  --receipt "$PREFLIGHT_REPORT" --output "$RUN_ROOT/launcher_negative_controls"
run_step receipt_add_negative_controls "$PY" -u "$SIM/scripts/stage4_update_preflight_result.py" \
  "${PRECEIPT_ARGS[@]}" --name launcher_negative_controls --status PASS \
  --evidence "$RUN_ROOT/launcher_negative_controls/launcher_negative_controls.json" \
  --command "python scripts/stage4_gate_negative_controls.py; production launcher blocks before sentinel process"

run_step final_full_gate_verify "$PY" -u "$pilot_gate" --phase full --receipt "$PREFLIGHT_REPORT" \
  --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
  --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$PROJECT/runtime/sim2real/genesis_hdr_v2" \
  --run-root "$RUN_ROOT/pilot" --generator eventgenesis --resume "$RUN_ROOT/pilot/eventgenesis/last.pt" --verify-only

for gpu in 2 4; do ensure_idle_gpu "$gpu" || { echo "GPU $gpu occupied before full training" >&2; exit 3; }; done
for spec in "eventgenesis 2" "v2e 4"; do
  read -r generator gpu <<< "$spec"
  run_dir="$RUN_ROOT/pilot/$generator"
  date -u +%FT%TZ > "$run_dir/full_resume.started_utc.txt"
  printf 'CUDA_VISIBLE_DEVICES=%s STAGE4_PHYSICAL_GPU=%s STAGE4_REPO_COMMIT=%s OMP_NUM_THREADS=4 ' "$gpu" "$gpu" "$COMMIT" > "$run_dir/full_resume.command.txt"
  printf '%q ' "$PY" -u "$pilot_gate" --phase full --receipt "$PREFLIGHT_REPORT" \
    --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
    --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$PROJECT/runtime/sim2real/genesis_hdr_v2" \
    --run-root "$RUN_ROOT/pilot" --generator "$generator" --resume "$run_dir/last.pt" >> "$run_dir/full_resume.command.txt"
  printf '\n' >> "$run_dir/full_resume.command.txt"
  CUDA_VISIBLE_DEVICES="$gpu" STAGE4_PHYSICAL_GPU="$gpu" STAGE4_REPO_COMMIT="$COMMIT" OMP_NUM_THREADS=4 \
    "$PY" -u "$pilot_gate" --phase full --receipt "$PREFLIGHT_REPORT" \
    --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
    --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$PROJECT/runtime/sim2real/genesis_hdr_v2" \
    --run-root "$RUN_ROOT/pilot" --generator "$generator" --resume "$run_dir/last.pt" \
    > "$run_dir/full_resume.stdout.log" 2> "$run_dir/full_resume.stderr.log" &
  pid=$!; echo "$pid" > "$run_dir/full_resume.pid"
done
pid_eg=$(cat "$RUN_ROOT/pilot/eventgenesis/full_resume.pid")
pid_v2e=$(cat "$RUN_ROOT/pilot/v2e/full_resume.pid")
echo "eventgenesis=$pid_eg v2e=$pid_v2e" > "$RUN_ROOT/full_training_pids.txt"
eg_rc=0; wait "$pid_eg" || eg_rc=$?
v2e_rc=0; wait "$pid_v2e" || v2e_rc=$?
echo "$eg_rc" > "$RUN_ROOT/pilot/eventgenesis/full_resume.exit_code.txt"
echo "$v2e_rc" > "$RUN_ROOT/pilot/v2e/full_resume.exit_code.txt"
date -u +%FT%TZ > "$RUN_ROOT/pilot/eventgenesis/full_resume.finished_utc.txt"
date -u +%FT%TZ > "$RUN_ROOT/pilot/v2e/full_resume.finished_utc.txt"
[[ "$eg_rc" -eq 0 && "$v2e_rc" -eq 0 ]] || { echo "full training failed: eventgenesis=$eg_rc v2e=$v2e_rc" >&2; exit 5; }
echo "both paired 45-epoch runs completed successfully" > "$RUN_ROOT/campaign_complete.txt"
