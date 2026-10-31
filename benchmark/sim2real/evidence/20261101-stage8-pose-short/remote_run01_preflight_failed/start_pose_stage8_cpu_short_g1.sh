#!/usr/bin/env bash
set -uo pipefail

PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN="$PROJECT/runs/sim2real/20261101-stage8-pose-cpu-short-v1"
CHECKOUT="$PROJECT/execution_checkouts/sim2real-stage8-pose-cpu-short-20261101"
DATA="$PROJECT/data/sim2real/stage2/20261016-s2-pilot-v1"
PYTHON="$PROJECT/miniconda3/envs/Eventbased_WAM_torch28/bin/python"
SCRIPT="$CHECKOUT/benchmark/sim2real/scripts/run_pose_stage2_pilot.py"
CONFIG="$CHECKOUT/benchmark/sim2real/configs/pose_stage4_seed0_eval_stable.json"
POSE_SOURCE="$PROJECT/external/sources/downstream/SingleShotPose"

if [[ -e "$RUN/training_output" ]]; then
  echo "Refusing to overwrite existing pose run output: $RUN/training_output" >&2
  exit 2
fi

mkdir -p "$RUN/logs"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/training_started_utc.txt"
printf '%s\n' "$(hostname -f)" > "$RUN/host.txt"
printf '%s\n' "$$" > "$RUN/launcher_pid.txt"
printf '%s\n' "$PROJECT" > "$RUN/project_root.txt"
printf '%s\n' "CUDA_VISIBLE_DEVICES= (hidden; CPU-only)" > "$RUN/device_allocation.txt"

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{
  printf '%q ' "$PYTHON" -u "$SCRIPT" \
    --shared-h5 "$DATA/shared/shared_trajectory.h5" \
    --v2e-root "$DATA/v2e_events" \
    --data-config "$DATA/shared/config.json" \
    --pilot-config "$CONFIG" \
    --output-root "$RUN/training_output" \
    --device cpu --cpu-threads 1 \
    --pose-source "$POSE_SOURCE"
  printf '\n'
} > "$RUN/command.txt"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  hostname -f
  "$PYTHON" -c 'import torch, numpy, cv2, h5py; print("python-pytorch",torch.__version__,"cuda_available",torch.cuda.is_available(),"threads",torch.get_num_threads(),"numpy",numpy.__version__,"opencv",cv2.__version__,"h5py",h5py.__version__)'
  nproc
  uptime
  free -h
} > "$RUN/environment.txt" 2>&1

"$PYTHON" -u "$SCRIPT" \
  --shared-h5 "$DATA/shared/shared_trajectory.h5" \
  --v2e-root "$DATA/v2e_events" \
  --data-config "$DATA/shared/config.json" \
  --pilot-config "$CONFIG" \
  --output-root "$RUN/training_output" \
  --device cpu --cpu-threads 1 \
  --pose-source "$POSE_SOURCE" \
  > "$RUN/logs/stdout.log" 2> "$RUN/logs/stderr.log" &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$RUN/trainer_pid.txt"
wait "$TRAIN_PID"
STATUS=$?
printf '%s\n' "$STATUS" > "$RUN/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/training_finished_utc.txt"
exit "$STATUS"
