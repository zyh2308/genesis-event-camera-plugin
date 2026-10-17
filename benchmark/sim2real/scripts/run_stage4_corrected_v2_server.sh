#!/usr/bin/env bash
# Corrected Stage-4 data -> fail-closed pilot/resume controls -> paired seed-0 run.
set -Eeuo pipefail

PROJECT=/home/zhaoyuhan/Eventbased_WAM
REPO="$PROJECT/repositories/gec-sim2real"
SIM="$REPO/benchmark/sim2real"
CONFIG="$SIM/configs/stage4_corrected_v2_flowdt1_seed0.json"
RUN_ID=20261017-stage4-flowdt1-seed0-corrected-v2
RUN_ROOT="$PROJECT/runs/sim2real/$RUN_ID"
BANK_ROOT="$PROJECT/data/sim2real/stage4/training_bank_v2_geometry_corrected_20261017_32x2s"
V2E_ROOT="$RUN_ROOT/v2e_events"
DATA_ROOT="$PROJECT/data/sim2real/stage4/$RUN_ID/steflow_30ms"
MODEL_ROOT="$PROJECT/external/sources/downstream/STE-FlowNet"
V2E_SOURCE="$PROJECT/external/sources/v2e-cac6384445/v2e-pinned"
GENESIS_ROOT="$PROJECT/runtime/sim2real/genesis_hdr_v2"
ENV_ROOT="/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM"
PY="$ENV_ROOT/bin/python"
COMMIT=$(git -C "$REPO" rev-parse HEAD)

[[ "$(whoami)" == zhaoyuhan ]] || { echo "wrong server user" >&2; exit 2; }
[[ "$REPO" == /home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real ]] || exit 2
[[ "$(git -C "$REPO" branch --show-current)" == exp/public-sim2real-v1 ]] || { echo "wrong branch" >&2; exit 2; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "repo working tree must be clean before campaign" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" && ! -e "$BANK_ROOT" && ! -e "$DATA_ROOT" ]] || { echo "refusing an existing output directory" >&2; exit 2; }
[[ -x "$PY" && -d "$V2E_SOURCE" && -d "$MODEL_ROOT" && -d "$GENESIS_ROOT" ]] || { echo "required pinned environment/source path missing" >&2; exit 2; }

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

for gpu in 1 2 4; do
  ensure_idle_gpu "$gpu" || exit 3
done

mkdir -p "$RUN_ROOT"
printf '%s\n' "$$" > "$RUN_ROOT/campaign.pid"
date -u +%FT%TZ > "$RUN_ROOT/started_utc.txt"
trap 'rc=$?; echo "$rc" > "$RUN_ROOT/exit_code.txt"; date -u +%FT%TZ > "$RUN_ROOT/finished_utc.txt"' EXIT
cd "$REPO"
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
  git status --short --branch
  nvidia-smi
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv
  free -h
  df -h /home
} > "$RUN_ROOT/environment_preflight.txt" 2>&1

{
  nvidia-smi
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv
  for gpu in 1 2 4; do
    printf 'prelaunch_gpu_%s=' "$gpu"
    nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
  done
} >> "$RUN_ROOT/environment_preflight.txt" 2>&1

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
run_step predata_correctness "$PY" -u "$SIM/scripts/stage4_correctness_preflight.py" \
  --stage predata --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" \
  --v2e-root "$V2E_ROOT" --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" \
  --genesis-root "$GENESIS_ROOT" --output "$RUN_ROOT/predata_correctness.json"

ensure_idle_gpu 1 || { echo "GPU1 became occupied before shared bank rendering" >&2; exit 3; }
mkdir -p "$(dirname "$BANK_ROOT")" "$(dirname "$DATA_ROOT")"
run_step corrected_shared_bank_generation "$PY" -u "$SIM/scripts/generate_stage4_flow_bank.py" \
  --config "$CONFIG" --output-root "$BANK_ROOT"

# V2E 25-us processing is split by disjoint strata across two checked GPUs.
for gpu in 2 4; do ensure_idle_gpu "$gpu" || { echo "GPU $gpu became occupied before V2E" >&2; exit 3; }; done
mkdir -p "$V2E_ROOT/gpu2" "$V2E_ROOT/gpu4"
"$PY" -u "$SIM/scripts/run_stage4_v2e_shard.py" --project-root "$PROJECT" --bank-root "$BANK_ROOT" \
  --output-root "$V2E_ROOT/gpu2" --gpu 2 --episodes s0_00 s0_01 s0_02 s0_03 s0_04 s0_05 s0_06 s0_07 s2_00 s2_01 s2_02 s2_03 s2_04 s2_05 s2_06 s2_07 \
  > "$V2E_ROOT/gpu2/worker.stdout.log" 2> "$V2E_ROOT/gpu2/worker.stderr.log" &
pid2=$!; echo "$pid2" > "$V2E_ROOT/gpu2/worker.pid"
"$PY" -u "$SIM/scripts/run_stage4_v2e_shard.py" --project-root "$PROJECT" --bank-root "$BANK_ROOT" \
  --output-root "$V2E_ROOT/gpu4" --gpu 4 --episodes s1_00 s1_01 s1_02 s1_03 s1_04 s1_05 s1_06 s1_07 s3_00 s3_01 s3_02 s3_03 s3_04 s3_05 s3_06 s3_07 \
  > "$V2E_ROOT/gpu4/worker.stdout.log" 2> "$V2E_ROOT/gpu4/worker.stderr.log" &
pid4=$!; echo "$pid4" > "$V2E_ROOT/gpu4/worker.pid"
v2e_rc=0; wait "$pid2" || v2e_rc=$?; echo "$v2e_rc" > "$V2E_ROOT/gpu2/worker.exit_code.txt"
rc4=0; wait "$pid4" || rc4=$?; echo "$rc4" > "$V2E_ROOT/gpu4/worker.exit_code.txt"
[[ "$v2e_rc" -eq 0 && "$rc4" -eq 0 ]] || { echo "V2E shard failure: gpu2=$v2e_rc gpu4=$rc4" >&2; exit 4; }
run_step v2e_finalize "$PY" -u "$SIM/scripts/finalize_stage4_v2e.py" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT"
run_step paired_steflow_adapter "$PY" -u "$SIM/scripts/prepare_stage4_steflow_bank.py" \
  --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" --output-root "$DATA_ROOT"

run_step full_correctness_preflight "$PY" -u "$SIM/scripts/stage4_correctness_preflight.py" \
  --stage full --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" \
  --v2e-root "$V2E_ROOT" --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" \
  --genesis-root "$GENESIS_ROOT" --output "$RUN_ROOT/preflight_result.json"

PRECEIPT_ARGS=(--report "$RUN_ROOT/preflight_result.json" --repo-root "$REPO" --config "$CONFIG" \
  --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" --data-root "$DATA_ROOT" \
  --model-root "$MODEL_ROOT" --genesis-root "$GENESIS_ROOT")

for gpu in 2 4; do ensure_idle_gpu "$gpu" || { echo "GPU $gpu occupied before paired pilot" >&2; exit 3; }; done
pilot_gate="$SIM/scripts/stage4_launch_gate.py"
date -u +%FT%TZ > "$RUN_ROOT/paired_pilot.started_utc.txt"
printf '%q ' "$PY" -u "$pilot_gate" --phase pilot --receipt "$RUN_ROOT/preflight_result.json" \
  --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
  --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$GENESIS_ROOT" \
  --run-root "$RUN_ROOT/pilot" --generator eventgenesis --max-updates 198 > "$RUN_ROOT/pilot_eventgenesis.command.txt"
printf '\n' >> "$RUN_ROOT/pilot_eventgenesis.command.txt"
CUDA_VISIBLE_DEVICES=2 STAGE4_PHYSICAL_GPU=2 STAGE4_REPO_COMMIT="$COMMIT" OMP_NUM_THREADS=4 \
  "$PY" -u "$pilot_gate" --phase pilot --receipt "$RUN_ROOT/preflight_result.json" \
  --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
  --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$GENESIS_ROOT" \
  --run-root "$RUN_ROOT/pilot" --generator eventgenesis --max-updates 198 \
  > "$RUN_ROOT/pilot_eventgenesis.stdout.log" 2> "$RUN_ROOT/pilot_eventgenesis.stderr.log" &
pilot_eg=$!; echo "$pilot_eg" > "$RUN_ROOT/pilot_eventgenesis.pid"
printf '%q ' "$PY" -u "$pilot_gate" --phase pilot --receipt "$RUN_ROOT/preflight_result.json" \
  --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
  --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$GENESIS_ROOT" \
  --run-root "$RUN_ROOT/pilot" --generator v2e --max-updates 198 > "$RUN_ROOT/pilot_v2e.command.txt"
printf '\n' >> "$RUN_ROOT/pilot_v2e.command.txt"
CUDA_VISIBLE_DEVICES=4 STAGE4_PHYSICAL_GPU=4 STAGE4_REPO_COMMIT="$COMMIT" OMP_NUM_THREADS=4 \
  "$PY" -u "$pilot_gate" --phase pilot --receipt "$RUN_ROOT/preflight_result.json" \
  --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
  --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$GENESIS_ROOT" \
  --run-root "$RUN_ROOT/pilot" --generator v2e --max-updates 198 \
  > "$RUN_ROOT/pilot_v2e.stdout.log" 2> "$RUN_ROOT/pilot_v2e.stderr.log" &
pilot_v2e=$!; echo "$pilot_v2e" > "$RUN_ROOT/pilot_v2e.pid"
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
  --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$GENESIS_ROOT" \
  --receipt "$RUN_ROOT/preflight_result.json" --pilot-run-root "$RUN_ROOT/pilot" \
  --output-root "$RUN_ROOT/resume_equivalence" --gpu 2
run_step receipt_add_resume "$PY" -u "$SIM/scripts/stage4_update_preflight_result.py" \
  "${PRECEIPT_ARGS[@]}" --name resume_equivalence_epoch_boundary --status PASS \
  --evidence "$RUN_ROOT/resume_equivalence/resume_equivalence.json" \
  --command "python scripts/stage4_resume_equivalence.py; process exit then gate --resume across update 198/199"

run_step launcher_negative_controls "$PY" -u "$SIM/scripts/stage4_gate_negative_controls.py" \
  --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
  --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$GENESIS_ROOT" \
  --receipt "$RUN_ROOT/preflight_result.json" --output "$RUN_ROOT/launcher_negative_controls"
run_step receipt_add_negative_controls "$PY" -u "$SIM/scripts/stage4_update_preflight_result.py" \
  "${PRECEIPT_ARGS[@]}" --name launcher_negative_controls --status PASS \
  --evidence "$RUN_ROOT/launcher_negative_controls/launcher_negative_controls.json" \
  --command "python scripts/stage4_gate_negative_controls.py; production launcher blocks before sentinel process"

run_step final_full_gate_verify "$PY" -u "$SIM/scripts/stage4_launch_gate.py" \
  --phase full --receipt "$RUN_ROOT/preflight_result.json" --repo-root "$REPO" --config "$CONFIG" \
  --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" --data-root "$DATA_ROOT" \
  --model-root "$MODEL_ROOT" --genesis-root "$GENESIS_ROOT" --run-root "$RUN_ROOT/pilot" \
  --generator eventgenesis --resume "$RUN_ROOT/pilot/eventgenesis/last.pt" --verify-only

# Continue the validated first epoch in the same run directories. Both arms
# use the official launcher, identical data/config/sampler/seed/optimizer.
for gpu in 2 4; do ensure_idle_gpu "$gpu" || { echo "GPU $gpu occupied before full training" >&2; exit 3; }; done
for spec in "eventgenesis 2" "v2e 4"; do
  read -r generator gpu <<< "$spec"
  run_dir="$RUN_ROOT/pilot/$generator"
  date -u +%FT%TZ > "$run_dir/full_resume.started_utc.txt"
  printf 'CUDA_VISIBLE_DEVICES=%s STAGE4_PHYSICAL_GPU=%s STAGE4_REPO_COMMIT=%s OMP_NUM_THREADS=4 ' "$gpu" "$gpu" "$COMMIT" > "$run_dir/full_resume.command.txt"
  printf '%q ' "$PY" -u "$SIM/scripts/stage4_launch_gate.py" --phase full --receipt "$RUN_ROOT/preflight_result.json" \
    --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
    --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$GENESIS_ROOT" \
    --run-root "$RUN_ROOT/pilot" --generator "$generator" --resume "$run_dir/last.pt" >> "$run_dir/full_resume.command.txt"
  printf '\n' >> "$run_dir/full_resume.command.txt"
  CUDA_VISIBLE_DEVICES="$gpu" STAGE4_PHYSICAL_GPU="$gpu" STAGE4_REPO_COMMIT="$COMMIT" OMP_NUM_THREADS=4 \
    "$PY" -u "$SIM/scripts/stage4_launch_gate.py" --phase full --receipt "$RUN_ROOT/preflight_result.json" \
    --repo-root "$REPO" --config "$CONFIG" --bank-root "$BANK_ROOT" --v2e-root "$V2E_ROOT" \
    --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --genesis-root "$GENESIS_ROOT" \
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
