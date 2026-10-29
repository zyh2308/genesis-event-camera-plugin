#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=/home/zhaoyuhan/Eventbased_WAM
REPO=$PROJECT/repositories/exec-stage5-gtfit-20261028-1377929
BANK=$PROJECT/data/sim2real/stage5/20261021-stage5-flow-coverage-v3-eight-rollouts
RUN_ROOT=$PROJECT/runs/sim2real/20261029-stage5-hightexture-fixed-gtfit-g3-v2
CONFIG=$REPO/benchmark/sim2real/configs/stage5_genesis_high_texture_fixed_sample_gt_fit_v1.json
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
MODEL=$PROJECT/external/sources/downstream/STE-FlowNet
GPU=5
GPU_UUID=GPU-dd93b58d-df3d-9d33-bffc-2203bb503e87
LOCK=$PROJECT/runs/sim2real/.gpu_reservations/g3_$GPU_UUID
RUN_ID=20261029-stage5-hightexture-fixed-gtfit-g3-v2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/driver.log") 2>&1
echo "$$" > "$RUN_ROOT/driver.pid"
echo running > "$RUN_ROOT/run_state.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/started_utc.txt"
finish() {
  rc=$?
  trap - EXIT
  set +e
  printf '%s\n' "$rc" > "$RUN_ROOT/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/finished_utc.txt"
  if [ "$rc" -eq 0 ]; then echo complete > "$RUN_ROOT/run_state.txt"; else echo failed > "$RUN_ROOT/run_state.txt"; fi
  if [ -f "$LOCK/reservation.txt" ] && grep -Fqx "run_id=$RUN_ID" "$LOCK/reservation.txt"; then
    cp "$LOCK/reservation.txt" "$RUN_ROOT/reservation_final.txt"
    rm -f "$LOCK/reservation.txt"
    rmdir "$LOCK" 2>/dev/null
  fi
  exit "$rc"
}
trap finish EXIT
echo "run_id=$RUN_ID host=$(hostname -s) pid=$$ started_utc=$(cat "$RUN_ROOT/started_utc.txt") gpu_uuid=$GPU_UUID"
grep -Fqx "run_id=$RUN_ID" "$LOCK/reservation.txt"
sed -i 's/^status=preflight$/status=running/' "$LOCK/reservation.txt"
printf 'driver_pid=%s\n' "$$" >> "$LOCK/reservation.txt"
echo ENVIRONMENT
printf 'commit='; git -C "$REPO" rev-parse HEAD
printf 'status='; git -C "$REPO" status --porcelain
printf 'config_sha='; sha256sum "$CONFIG"
printf 'bank_sha='; sha256sum "$BANK/shared_trajectory.h5"
printf 'bank_manifest_sha='; sha256sum "$BANK/bank_manifest.json"
printf 'split_sha='; sha256sum "$BANK/registered_split.json"
printf 'v2e_sha='; sha256sum "$PROJECT/external/sources/v2e-cac6384445/v2e-pinned/v2e.py"
printf 'slomo_sha='; sha256sum "$PROJECT/external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt"
"$PY" -c 'import sys,torch,h5py,numpy,cv2; print("python",sys.version); print("torch",torch.__version__,"cuda",torch.version.cuda,"available",torch.cuda.is_available(),"device",torch.cuda.get_device_name(0)); print("h5py",h5py.__version__,"numpy",numpy.__version__,"opencv",cv2.__version__)'
nvidia-smi
echo V2E_BEGIN
"$PY" -u "$REPO/benchmark/sim2real/scripts/run_stage4_v2e_shard.py" \
  --project-root "$PROJECT" --bank-root "$BANK" --output-root "$RUN_ROOT/v2e" \
  --gpu "$GPU" --episodes s5q_02_train s5q_02_val
echo V2E_COMPLETE
echo PAIR_ADAPTER_BEGIN
"$PY" -u "$REPO/benchmark/sim2real/scripts/prepare_stage5_genesis_gtfit_pair_bank.py" \
  --bank-root "$BANK" --v2e-root "$RUN_ROOT/v2e" --config "$CONFIG" \
  --output-root "$RUN_ROOT/paired_adapter"
echo PAIR_ADAPTER_COMPLETE
echo FIXED_SAMPLE_FIT_BEGIN
CUDA_VISIBLE_DEVICES=$GPU "$PY" -u "$REPO/benchmark/sim2real/scripts/stage5_flow_fixed_sample_fit.py" \
  --model-root "$MODEL" --data-root "$RUN_ROOT/paired_adapter" --config "$CONFIG" \
  --output-root "$RUN_ROOT/fixed_sample_fit" --device cuda:0
echo FIXED_SAMPLE_FIT_COMPLETE
echo INDEPENDENT_ROLLOUT_EVAL_BEGIN
CUDA_VISIBLE_DEVICES=$GPU "$PY" -u "$REPO/benchmark/sim2real/scripts/evaluate_stage5_genesis_gtfit_rollout.py" \
  --model-root "$MODEL" --data-root "$RUN_ROOT/paired_adapter" --config "$CONFIG" \
  --eventgenesis-checkpoint "$RUN_ROOT/fixed_sample_fit/eventgenesis_last.pt" \
  --v2e-checkpoint "$RUN_ROOT/fixed_sample_fit/v2e_last.pt" \
  --output "$RUN_ROOT/independent_rollout_eval.json" --device cuda:0
echo INDEPENDENT_ROLLOUT_EVAL_COMPLETE
for arm in eventgenesis v2e; do
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -u "$REPO/benchmark/sim2real/scripts/verify_stage5_flow_fit_checkpoint.py" \
    --checkpoint "$RUN_ROOT/fixed_sample_fit/\${arm}_last.pt" --model-root "$MODEL" \
    --data-root "$RUN_ROOT/paired_adapter" --config "$CONFIG" --device cuda:0 \
    --output "$RUN_ROOT/\${arm}_restore_check.json"
done
echo CHECKPOINT_RESTORE_COMPLETE
sha256sum "$RUN_ROOT"/fixed_sample_fit/*.pt "$RUN_ROOT"/fixed_sample_fit/summary.json "$RUN_ROOT"/independent_rollout_eval.json
