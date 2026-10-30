#!/usr/bin/env bash
set -u

CHECKOUT=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-recon-state-pad-cpu-run04
PYTHON=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
SCRIPT=$CHECKOUT/benchmark/sim2real/scripts/audit_e2vid_checkpoint_state_padding.py
EVREAL=/home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/EVREAL
MODEL=/home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/event_cnn_minimal
CKPT_ROOT=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-e2vid-tc-aligntrue-seed0
MODEL_CONFIG=$CHECKOUT/benchmark/sim2real/configs/stage4_corrected_v2_flowdt1_seed0.json
FIXTURE_EG=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-stage5-recon-synth-fixture-eventgenesis-v2/fixture/eventgenesis/synthetic_fixture_manifest.json
FIXTURE_V2E=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-stage5-recon-synth-fixture-v2e-v1/fixture/v2e/synthetic_fixture_manifest.json

if [[ -e "$RUN" ]]; then
  echo "refusing to overwrite existing run: $RUN" >&2
  exit 3
fi
if [[ ! -x "$PYTHON" || ! -f "$SCRIPT" || ! -d "$EVREAL" || ! -d "$MODEL" ]]; then
  echo "registered Python or source path missing" >&2
  exit 4
fi
LOAD1=$(awk '{print $1}' /proc/loadavg)
NCPU=$(getconf _NPROCESSORS_ONLN)
MEM_AVAILABLE_KB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if ! awk -v l1="$LOAD1" -v cpus="$NCPU" 'BEGIN { exit !(l1 < cpus * 0.8) }'; then
  echo "CPU load gate refused this bounded two-thread audit: load=$LOAD1 cpus=$NCPU" >&2
  exit 5
fi
if [[ "$MEM_AVAILABLE_KB" -lt 8388608 ]]; then
  echo "memory gate refused this bounded audit: available_kb=$MEM_AVAILABLE_KB" >&2
  exit 5
fi
mkdir -p "$RUN/eventgenesis" "$RUN/v2e"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_utc.txt"
git -C "$CHECKOUT" rev-parse HEAD > "$RUN/checkout_commit.txt"
sha256sum "$SCRIPT" > "$RUN/audit_script.sha256"
{
  "$PYTHON" -c 'import sys,torch,numpy; print(sys.version); print("torch",torch.__version__,"numpy",numpy.__version__)'
  printf 'CUDA_VISIBLE_DEVICES=%s\nOMP_NUM_THREADS=2\n' "${CUDA_VISIBLE_DEVICES-}"
  printf 'threads_per_arm=2\ndevice=cpu\n'
  printf 'cpu_load1=%s\ncpu_count=%s\nmem_available_kb=%s\n' "$LOAD1" "$NCPU" "$MEM_AVAILABLE_KB"
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  sha256sum "$FIXTURE_EG" "$FIXTURE_V2E" \
    "$CKPT_ROOT/eventgenesis/synthetic_best.pt" "$CKPT_ROOT/v2e/synthetic_best.pt" "$MODEL_CONFIG"
} > "$RUN/environment_and_inputs.txt" 2>&1

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

set +e
"$PYTHON" -u "$SCRIPT" --generator eventgenesis --fixture-manifest "$FIXTURE_EG" \
  --evreal-root "$EVREAL" --model-root "$MODEL" \
  --model-config "$MODEL_CONFIG" \
  --checkpoint "$CKPT_ROOT/eventgenesis/synthetic_best.pt" \
  --output "$RUN/eventgenesis/audit.json" > "$RUN/eventgenesis/driver.log" 2>&1
EG_RC=$?
printf '%s\n' "$EG_RC" > "$RUN/eventgenesis/exit_code.txt"

"$PYTHON" -u "$SCRIPT" --generator v2e --fixture-manifest "$FIXTURE_V2E" \
  --evreal-root "$EVREAL" --model-root "$MODEL" \
  --model-config "$MODEL_CONFIG" \
  --checkpoint "$CKPT_ROOT/v2e/synthetic_best.pt" \
  --output "$RUN/v2e/audit.json" > "$RUN/v2e/driver.log" 2>&1
V2E_RC=$?
printf '%s\n' "$V2E_RC" > "$RUN/v2e/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
if [[ "$EG_RC" -eq 0 && "$V2E_RC" -eq 0 ]]; then
  printf '0\n' > "$RUN/exit_code.txt"
else
  printf '1\n' > "$RUN/exit_code.txt"
fi
cat "$RUN/eventgenesis/driver.log"
cat "$RUN/v2e/driver.log"
exit "$((EG_RC || V2E_RC))"
