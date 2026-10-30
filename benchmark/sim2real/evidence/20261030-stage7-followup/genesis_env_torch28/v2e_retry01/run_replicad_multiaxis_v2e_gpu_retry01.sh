#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT=$PROJECT/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1
EXEC=$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
PY=$PROJECT/../miniconda3/envs/Eventbased_WAM_torch28/bin/python
RETRY_ROOT=$RUN_ROOT/v2e_retry01
CONVERTER=$RETRY_ROOT/run_v2e_same_radiance_camera_motion_retry01.py
CONVERSION_SOURCE=$EXEC/benchmark/sim2real/scripts/generate_stage5_flow_coverage_bank.py
RESERVATIONS=$PROJECT/runs/sim2real/resource_reservations
SCENES=(v3_sc0_staging_00 v3_sc1_staging_00 v3_sc2_staging_00)

mkdir -p "$RETRY_ROOT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RETRY_ROOT/started_utc.txt"
for scene in "${SCENES[@]}"; do
  RUN=$RETRY_ROOT/$scene/v2e_gpu
  RADIANCE=$RUN_ROOT/$scene/camera_motion_30ms/radiance_frames.npy
  test -f "$RADIANCE" && test -f "$CONVERTER" && test -f "$CONVERSION_SOURCE"
  mkdir -p "$RUN"
  test ! -e "$RUN/started_utc.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/preflight_utc.txt"
  APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>&1 || true)
  mkdir -p "$RESERVATIONS"
  GPU_INDEX=""; GPU_UUID=""; LOCK=""; UTIL=""; USED=""
  while IFS=',' read -r idx _name uuid util used _total; do
    idx=${idx//[[:space:]]/}; uuid=${uuid//[[:space:]]/}
    util=${util//[[:space:]]/}; util=${util%%%}
    used=${used//[[:space:]]/}
    if [[ "$util" != 0 || ! "$used" =~ ^[0-9]+$ || "$used" -gt 64 ]] || grep -Fq "$uuid" <<< "$APPS"; then
      continue
    fi
    candidate="$RESERVATIONS/$uuid"
    if mkdir "$candidate" 2>/dev/null; then
      GPU_INDEX=$idx; GPU_UUID=$uuid; LOCK=$candidate; UTIL=$util; USED=$used
      break
    fi
  done < <(nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
  if [[ -z "$GPU_INDEX" ]]; then
    echo "GPU gate refused: no unowned idle GPU available at this live check" >&2
    printf '5\n' > "$RETRY_ROOT/exit_code.txt"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$RETRY_ROOT/finished_utc.txt"
    exit 5
  fi
  {
    printf 'host=%s\nproject=%s\nrun_id=20261031-stage7-replicad-multiaxis-pathpilot-v1/v2e_retry01/%s/v2e_gpu\n' "$(hostname)" "$PROJECT" "$scene"
    printf 'gpu_index=%s\ngpu_uuid=%s\nutilization_percent=%s\nmemory_used_mib=%s\n' "$GPU_INDEX" "$GPU_UUID" "$UTIL" "$USED"
    printf 'compute_app_gpu_uuid_rows:\n%s\n' "$APPS"
    printf 'v2e_python=%s\n' "$PY"
    printf 'radiance_sha256='; sha256sum "$RADIANCE"
    printf 'converter_sha256='; sha256sum "$CONVERTER"
    printf 'conversion_source_sha256='; sha256sum "$CONVERSION_SOURCE"
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    free -h
    df -h "$RUN" /tmp
  } > "$RUN/resource_preflight.txt" 2>&1
  RUN_ID="20261031-stage7-replicad-multiaxis-pathpilot-v1/v2e_retry01/$scene/v2e_gpu"
  printf '%s\n' "$RUN_ID" > "$LOCK/run_id.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$LOCK/acquired_utc.txt"
  cp "$LOCK/run_id.txt" "$RUN/reservation_run_id.txt"
  cp "$LOCK/acquired_utc.txt" "$RUN/reservation_acquired_utc.txt"
  release_lock() {
    if [[ -f "$LOCK/run_id.txt" ]] && [[ "$(cat "$LOCK/run_id.txt")" == "$RUN_ID" ]]; then
      date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/reservation_released_utc.txt"
      rm -f "$LOCK/run_id.txt" "$LOCK/acquired_utc.txt"
      rmdir "$LOCK"
    fi
  }
  trap release_lock EXIT
  UTIL2=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' %')
  USED2=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' MiB')
  APPS2=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>&1 || true)
  printf 'post_reservation_utilization_percent=%s\npost_reservation_memory_used_mib=%s\npost_reservation_compute_app_rows:\n%s\n' "$UTIL2" "$USED2" "$APPS2" >> "$RUN/resource_preflight.txt"
  if [[ "$UTIL2" != 0 || "$USED2" -gt 64 ]] || grep -Fq "$GPU_UUID" <<< "$APPS2"; then
    echo "GPU changed after reservation; preserving run and releasing only our reservation" >&2
    printf '5\n' > "$RETRY_ROOT/exit_code.txt"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$RETRY_ROOT/finished_utc.txt"
    exit 5
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started_utc.txt"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    "$PY" -u "$CONVERTER" \
      --radiance-frames "$RADIANCE" \
      --conversion-source "$CONVERSION_SOURCE" \
      --project-root "$PROJECT" \
      --output-root "$RUN/events" \
      --cuda-visible-devices "$GPU_INDEX" > "$RUN/converter_launcher.log" 2>&1
  RC=$?
  set -e
  printf '%s\n' "$RC" > "$RUN/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/finished_utc.txt"
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader > "$RUN/resource_after.txt"
  if [[ "$RC" -ne 0 ]]; then
    echo "V2E conversion failed for $scene with exit $RC" >&2
    printf '%s\n' "$RC" > "$RETRY_ROOT/exit_code.txt"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$RETRY_ROOT/finished_utc.txt"
    exit "$RC"
  fi
  release_lock
  trap - EXIT
done
date -u +%Y-%m-%dT%H:%M:%SZ > "$RETRY_ROOT/finished_utc.txt"
printf '0\n' > "$RETRY_ROOT/exit_code.txt"
echo 'all registered V2E conversions completed on the reserved RTX 3090'
