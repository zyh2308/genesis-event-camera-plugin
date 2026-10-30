#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/home/zhaoyuhan/Eventbased_WAM
REPO=$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
RUN_ROOT=$PROJECT/runs/sim2real/20261030-stage7-flow-resume-200to201-g1-run01
SOURCE_RUN=$PROJECT/runs/sim2real/20261029-stage5-hightexture-fixed-gtfit-g3-v2
DATA_ROOT=$SOURCE_RUN/paired_adapter
CONFIG=$REPO/benchmark/sim2real/configs/stage5_genesis_high_texture_fixed_sample_gt_fit_v1.json
SCRIPT=$REPO/benchmark/sim2real/scripts/resume_stage5_flow_fit_one_update.py
MODEL=$PROJECT/external/sources/downstream/STE-FlowNet
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
GPU=2
GPU_UUID=GPU-a02affe0-7598-579e-ca39-f551f35ad6e9
RESERVATION_ROOT=$PROJECT/runs/sim2real/.gpu_reservations
LOCK=$RESERVATION_ROOT/g1_$GPU_UUID
RUN_ID=20261030-stage7-flow-resume-200to201-g1-run01
EXPECTED_SCRIPT_SHA256=2184f94d120ca9447fc3c2e4b7280c4aed0acda12176e1bbc285ddfdd4ab3478
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
  if [ -f "$LOCK/reservation.txt" ] && grep -Fqx "run_id=$RUN_ID" "$LOCK/reservation.txt"; then
    cp "$LOCK/reservation.txt" "$RUN_ROOT/reservation_final.txt"
    rm -f "$LOCK/reservation.txt"
    rmdir "$LOCK" 2>/dev/null || true
  fi
  exit "$rc"
}
trap finish EXIT

grep -Fqx "run_id=$RUN_ID" "$LOCK/reservation.txt"
grep -Fqx 'status=reserved' "$LOCK/reservation.txt"
sed -i 's/^status=reserved$/status=running/' "$LOCK/reservation.txt"
printf 'driver_pid=%s\nstarted_utc=%s\n' "$$" "$(cat "$RUN_ROOT/started_utc.txt")" >> "$LOCK/reservation.txt"

echo "run_id=$RUN_ID host=$(hostname -s) pid=$$ gpu_index=$GPU gpu_uuid=$GPU_UUID"
echo "repository_head=$(git -C "$REPO" rev-parse HEAD)"
echo "repository_status=$(git -C "$REPO" status --porcelain | tr '\n' ' ')"
echo "probe_script_sha256=$(sha256sum "$SCRIPT" | awk '{print $1}')"
echo "config_sha256=$(sha256sum "$CONFIG" | awk '{print $1}')"
echo "data_manifest_sha256=$(sha256sum "$DATA_ROOT/training_bank_manifest.json" | awk '{print $1}')"
echo "model_root=$MODEL"
echo "data_root=$DATA_ROOT"
echo "python=$PY"
echo "expected_script_sha256=$EXPECTED_SCRIPT_SHA256"
echo "expected_config_sha256=$EXPECTED_CONFIG_SHA256"
echo "expected_data_manifest_sha256=$EXPECTED_DATA_MANIFEST_SHA256"
echo "eventgenesis_source_checkpoint_sha256=$(sha256sum "$SOURCE_RUN/fixed_sample_fit/eventgenesis_last.pt" | awk '{print $1}')"
echo "v2e_source_checkpoint_sha256=$(sha256sum "$SOURCE_RUN/fixed_sample_fit/v2e_last.pt" | awk '{print $1}')"
test "$(sha256sum "$SCRIPT" | awk '{print $1}')" = "$EXPECTED_SCRIPT_SHA256"
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA256"
test "$(sha256sum "$DATA_ROOT/training_bank_manifest.json" | awk '{print $1}')" = "$EXPECTED_DATA_MANIFEST_SHA256"
test "$(sha256sum "$SOURCE_RUN/fixed_sample_fit/eventgenesis_last.pt" | awk '{print $1}')" = "$EXPECTED_EVENTGENESIS_CHECKPOINT_SHA256"
test "$(sha256sum "$SOURCE_RUN/fixed_sample_fit/v2e_last.pt" | awk '{print $1}')" = "$EXPECTED_V2E_CHECKPOINT_SHA256"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
nvidia-smi
"$PY" -c 'import sys,torch,h5py,numpy,cv2; print("python",sys.version); print("torch",torch.__version__,"cuda",torch.version.cuda,"available",torch.cuda.is_available(),"device",torch.cuda.get_device_name(0)); print("h5py",h5py.__version__,"numpy",numpy.__version__,"opencv",cv2.__version__)'

for arm in eventgenesis v2e; do
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -u "$SCRIPT" \
    --checkpoint "$SOURCE_RUN/fixed_sample_fit/${arm}_last.pt" \
    --model-root "$MODEL" --data-root "$DATA_ROOT" --config "$CONFIG" \
    --device cuda:0 --output-dir "$RUN_ROOT/$arm"
done

sha256sum "$RUN_ROOT"/eventgenesis/resume_update_201.pt "$RUN_ROOT"/eventgenesis/resume_probe.json \
  "$RUN_ROOT"/v2e/resume_update_201.pt "$RUN_ROOT"/v2e/resume_probe.json
