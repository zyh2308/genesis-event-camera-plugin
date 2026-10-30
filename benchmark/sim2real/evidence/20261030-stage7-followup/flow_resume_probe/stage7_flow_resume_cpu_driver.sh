#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/home/zhaoyuhan/Eventbased_WAM
REPO=$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
SCRIPT=$REPO/benchmark/sim2real/scripts/resume_stage5_flow_fit_one_update.py
RUN_ROOT=$PROJECT/runs/sim2real/20261030-stage7-flow-resume-cpu-200to201-g1-run01
SOURCE_RUN=$PROJECT/runs/sim2real/20261029-stage5-hightexture-fixed-gtfit-g3-v2
DATA_ROOT=$SOURCE_RUN/paired_adapter
CONFIG=$REPO/benchmark/sim2real/configs/stage5_genesis_high_texture_fixed_sample_gt_fit_v1.json
MODEL=$PROJECT/external/sources/downstream/STE-FlowNet
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
RUN_ID=20261030-stage7-flow-resume-cpu-200to201-g1-run01
EXPECTED_SCRIPT_SHA256=6197f246629303343ffdb2c0512661622af72fd15951e707b79b2aad3c188e50
EXPECTED_CONFIG_SHA256=ce7c7e6c7670f2514e75db821eb7ef93939ff6794bcb5af8bc90e024b4597665
EXPECTED_DATA_MANIFEST_SHA256=02f8eab65559a9b8442fe04b46ae202fe23a15116ae62fe7b3385e7e6c3cf50d
EXPECTED_EVENTGENESIS_CHECKPOINT_SHA256=4da12339e52e29be1425287e3e4af5056cc3c7a125f7cc12f3ba326ad69f2a9b
EXPECTED_V2E_CHECKPOINT_SHA256=07bae17f1c1d5c635e863b73349a13ea04c635f1a079c31f466b254142bf143c

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/driver.log") 2>&1
echo "driver_pid=$$" > "$RUN_ROOT/driver.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/started_utc.txt"
echo running > "$RUN_ROOT/run_state.txt"
finish() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$RUN_ROOT/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/finished_utc.txt"
  if [ "$rc" -eq 0 ]; then echo complete > "$RUN_ROOT/run_state.txt"; else echo failed > "$RUN_ROOT/run_state.txt"; fi
  exit "$rc"
}
trap finish EXIT

echo "run_id=$RUN_ID host=$(hostname -s) pid=$$ started_utc=$(cat "$RUN_ROOT/started_utc.txt")"
echo "execution_base_commit=$(git -C "$REPO" rev-parse HEAD)"
echo "execution_worktree_status=$(git -C "$REPO" status --porcelain | tr '\n' ' ')"
echo "probe_script_sha256=$(sha256sum "$SCRIPT" | awk '{print $1}')"
echo "config_sha256=$(sha256sum "$CONFIG" | awk '{print $1}')"
echo "data_manifest_sha256=$(sha256sum "$DATA_ROOT/training_bank_manifest.json" | awk '{print $1}')"
echo "eventgenesis_source_checkpoint_sha256=$(sha256sum "$SOURCE_RUN/fixed_sample_fit/eventgenesis_last.pt" | awk '{print $1}')"
echo "v2e_source_checkpoint_sha256=$(sha256sum "$SOURCE_RUN/fixed_sample_fit/v2e_last.pt" | awk '{print $1}')"
test "$(sha256sum "$SCRIPT" | awk '{print $1}')" = "$EXPECTED_SCRIPT_SHA256"
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA256"
test "$(sha256sum "$DATA_ROOT/training_bank_manifest.json" | awk '{print $1}')" = "$EXPECTED_DATA_MANIFEST_SHA256"
test "$(sha256sum "$SOURCE_RUN/fixed_sample_fit/eventgenesis_last.pt" | awk '{print $1}')" = "$EXPECTED_EVENTGENESIS_CHECKPOINT_SHA256"
test "$(sha256sum "$SOURCE_RUN/fixed_sample_fit/v2e_last.pt" | awk '{print $1}')" = "$EXPECTED_V2E_CHECKPOINT_SHA256"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$PY" -c 'import os; load=float(open("/proc/loadavg").read().split()[0]); mem={line.split(":")[0]:int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith(("MemAvailable:","MemTotal:"))}; ncpu=os.cpu_count(); print("cpu_count",ncpu,"load1",load,"MemAvailable_GiB",round(mem["MemAvailable"]/1048576,1)); assert mem["MemAvailable"] >= 16*1048576, "less than 16 GiB available memory"; assert load < max(32,ncpu*0.8), "CPU load too high for small probe"'
free -h
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
"$PY" -c 'import sys,torch,h5py,numpy,cv2; print("python",sys.version); print("torch",torch.__version__,"cuda",torch.version.cuda,"available",torch.cuda.is_available()); print("h5py",h5py.__version__,"numpy",numpy.__version__,"opencv",cv2.__version__)'

for arm in eventgenesis v2e; do
  "$PY" -u "$SCRIPT" \
    --checkpoint "$SOURCE_RUN/fixed_sample_fit/${arm}_last.pt" \
    --model-root "$MODEL" --data-root "$DATA_ROOT" --config "$CONFIG" \
    --device cpu --cpu-shim --cpu-threads 2 --output-dir "$RUN_ROOT/$arm"
done

sha256sum "$RUN_ROOT"/eventgenesis/resume_update_201.pt "$RUN_ROOT"/eventgenesis/resume_probe.json \
  "$RUN_ROOT"/v2e/resume_update_201.pt "$RUN_ROOT"/v2e/resume_probe.json
