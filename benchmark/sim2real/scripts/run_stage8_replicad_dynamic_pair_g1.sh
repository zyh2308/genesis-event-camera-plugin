#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/zhaoyuhan/Eventbased_WAM
EXEC=${SIM2REAL_EXEC_CHECKOUT:?set SIM2REAL_EXEC_CHECKOUT to the immutable isolated checkout}
PY=$PROJECT/../miniconda3/envs/Eventbased_WAM_torch28/bin/python
ASSET_ROOT=$PROJECT/external/sim2real_assets/replicacad_interactive_v1_5/extracted_v1_5
GENESIS_ROOT=$PROJECT/runtime/sim2real/genesis_hdr_v2
MODEL_ROOT=$PROJECT/external/sources/downstream/STE-FlowNet
RUN_ROOT=$PROJECT/runs/sim2real/20261101-stage8-replicad-dynamic-v2
TMP_ROOT=/tmp/sim2real-stage8-replicad-dynamic-v1
CONFIG=$EXEC/benchmark/sim2real/configs/stage8_replicad_dynamic_gtfit_v2.json
GENERATOR=$EXEC/benchmark/sim2real/scripts/generate_stage8_replicad_dynamic_flow_pilot.py
V2E=$EXEC/benchmark/sim2real/scripts/run_stage8_v2e_same_radiance.py
BANK_BUILDER=$EXEC/benchmark/sim2real/scripts/build_stage8_dynamic_flow_bank.py
TRAINER=$EXEC/benchmark/sim2real/scripts/run_stage8_replicad_dynamic_gtfit.py
CONVERTER_SOURCE=$EXEC/benchmark/sim2real/scripts/generate_stage5_flow_coverage_bank.py
RESERVATIONS=$PROJECT/runs/sim2real/resource_reservations

mkdir -p "$RUN_ROOT/driver_logs"
case "$(hostname -s)" in g1|g3) ;; *) echo "refusing unapproved compute host: $(hostname -s)" >&2; exit 2 ;; esac
test -x "$PY" && test -d "$ASSET_ROOT" && test -d "$GENESIS_ROOT" && test -d "$MODEL_ROOT"
test -f "$CONFIG" && test -f "$GENERATOR" && test -f "$V2E" && test -f "$BANK_BUILDER" && test -f "$TRAINER"
test ! -e "$RUN_ROOT/stage8_finished_utc.txt"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  printf 'host=%s\nproject=%s\nexec=%s\n' "$(hostname -f)" "$PROJECT" "$EXEC"
  printf 'commit='; git -C "$EXEC" rev-parse HEAD
  printf 'python=%s\n' "$PY"; "$PY" --version
  printf '\n-- source and config hashes --\n'
  sha256sum "$CONFIG" "$GENERATOR" "$V2E" "$BANK_BUILDER" "$TRAINER" "$CONVERTER_SOURCE"
  printf '\n-- environment and disk --\n'; uptime; free -h; df -h "$RUN_ROOT" /tmp
} > "$RUN_ROOT/preflight.txt" 2>&1

for spec in 'stage8-sc0-dynamic-pilot-v2 v3_sc0_staging_00' 'stage8-sc2-dynamic-pilot-v2 v3_sc2_staging_00'; do
  read -r episode scene <<< "$spec"
  EROOT=$RUN_ROOT/$episode
  TMP=$TMP_ROOT/$episode
  OUT=$TMP/scene
  CACHE=$TMP/cache
  mkdir -p "$EROOT" "$CACHE"
  test ! -e "$TMP/scene"
  test ! -e "$EROOT/scene_exit_code.txt"
  PREV_EPISODE=${episode/-v2/-v1}
  PREV_RUN=$PROJECT/runs/sim2real/20261101-stage8-replicad-dynamic-v1/$PREV_EPISODE
  PREV_OUT=/tmp/sim2real-stage8-replicad-dynamic-v1/$PREV_EPISODE/scene/scene
  GENERATOR_SHA=$(sha256sum "$GENERATOR" | awk '{print $1}')
  REUSE_PREVIOUS=false
  if [[ -f "$PREV_OUT/result.json" && -f "$PREV_RUN/preflight.txt" \
      && -f "$PREV_OUT/camera_motion_30ms/gt_flow_px_t0_to_t30ms.npy" \
      && -f "$PREV_OUT/camera_motion_30ms/eventgenesis_events_seconds_xy_p.npy" \
      && -f "$PREV_OUT/layout_rgb.png" && -f "$PREV_OUT/layout_depth.npy" \
      && -f "$PREV_OUT/layout_segmentation.npy" && -f "$PREV_OUT/layout_radiance.npy" \
      && "$(grep -F '"status": "passed_replica_scene_camera_and_sofa_motion_probe"' "$PREV_OUT/result.json" >/dev/null && echo yes || echo no)" = yes \
      && "$(grep -F '"scene_id": "'"$scene"'"' "$PREV_OUT/result.json" >/dev/null && echo yes || echo no)" = yes \
      && "$(grep -F "$GENERATOR_SHA  " "$PREV_RUN/preflight.txt" >/dev/null && echo yes || echo no)" = yes ]]; then
    REUSE_PREVIOUS=true
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$EROOT/scene_started_utc.txt"
  if [[ "$REUSE_PREVIOUS" = true ]]; then
    cp -a "$PREV_OUT/camera_motion_30ms" "$EROOT/camera_motion_30ms"
    cp "$PREV_OUT/result.json" "$EROOT/genesis_result.json"
    cp "$PREV_OUT/layout_rgb.png" "$PREV_OUT/layout_depth.npy" "$PREV_OUT/layout_segmentation.npy" "$PREV_OUT/layout_radiance.npy" "$EROOT/"
    printf '{"schema":"stage8_scene_artifact_reuse_v1","status":"reused_verified_genesis_scene_output","previous_run":"%s","previous_output":"%s","generator_sha256":"%s","reason":"previous driver had an output-archive path defect after successful scene rendering; preserve and reuse only hash-matched scene output"}\n' "$PREV_EPISODE" "$PREV_OUT" "$GENERATOR_SHA" > "$EROOT/scene_reuse.json"
    printf '0\n' > "$EROOT/scene_exit_code.txt"
  else
    set +e
    timeout 1200s env CUDA_VISIBLE_DEVICES= LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless QD_OFFLINE_CACHE=1 QD_OFFLINE_CACHE_FILE_PATH="$CACHE/quadrants" QD_OFFLINE_CACHE_CLEANING_POLICY=never QD_NUM_THREADS=2 GS_CACHE_FILE_PATH="$CACHE/genesis" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$GENESIS_ROOT:$EXEC:$PROJECT" \
      "$PY" -u "$GENERATOR" --asset-root "$ASSET_ROOT" --scene-config "${scene}.scene_instance.json" --output-dir "$OUT" \
      > "$EROOT/genesis.stdout.log" 2> "$EROOT/genesis.stderr.log"
    RC=$?
    set -e
    printf '%s\n' "$RC" > "$EROOT/scene_exit_code.txt"
    if [[ "$RC" -ne 0 ]]; then printf '%s\n' "$RC" > "$RUN_ROOT/exit_code.txt"; exit "$RC"; fi
    cp -a "$OUT/camera_motion_30ms" "$EROOT/camera_motion_30ms"
    cp "$OUT/result.json" "$EROOT/genesis_result.json"
    cp "$OUT/layout_rgb.png" "$OUT/layout_depth.npy" "$OUT/layout_segmentation.npy" "$OUT/layout_radiance.npy" "$EROOT/"
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$EROOT/scene_finished_utc.txt"
  sha256sum "$EROOT/genesis_result.json" "$EROOT/layout_rgb.png" "$EROOT/layout_depth.npy" "$EROOT/layout_segmentation.npy" "$EROOT/layout_radiance.npy" "$EROOT/camera_motion_30ms/radiance_frames.npy" "$EROOT/camera_motion_30ms/gt_flow_px_t0_to_t30ms.npy" > "$EROOT/scene_artifacts.sha256"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$EROOT/v2e_started_utc.txt"
  set +e
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$EXEC:$PROJECT" \
    "$PY" -u "$V2E" --radiance-frames "$EROOT/camera_motion_30ms/radiance_frames.npy" \
      --conversion-source "$CONVERTER_SOURCE" --project-root "$PROJECT" \
      --output-root "$EROOT/v2e" --cuda-visible-devices "" \
      > "$EROOT/v2e_driver.stdout.log" 2> "$EROOT/v2e_driver.stderr.log"
  RC=$?
  set -e
  printf '%s\n' "$RC" > "$EROOT/v2e_exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$EROOT/v2e_finished_utc.txt"
  if [[ "$RC" -ne 0 ]]; then printf '%s\n' "$RC" > "$RUN_ROOT/exit_code.txt"; exit "$RC"; fi
  rm -rf -- "$TMP"
done

FLOW_BANK=$RUN_ROOT/flow_bank
set +e
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$EXEC:$PROJECT" \
  "$PY" -u "$BANK_BUILDER" --repo-root "$EXEC" --config "$CONFIG" --output-root "$FLOW_BANK" \
  > "$RUN_ROOT/bank_builder.stdout.log" 2> "$RUN_ROOT/bank_builder.stderr.log"
RC=$?
set -e
printf '%s\n' "$RC" > "$RUN_ROOT/bank_builder_exit_code.txt"
if [[ "$RC" -ne 0 ]]; then printf '%s\n' "$RC" > "$RUN_ROOT/exit_code.txt"; exit "$RC"; fi

# Queue politely if other jobs currently use all cards; reserve atomically and verify again.
mkdir -p "$RESERVATIONS"
GPU_INDEX= GPU_UUID= GPU_NAME= LOCK=
for attempt in $(seq 1 60); do
  APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>&1 || true)
  while IFS=',' read -r idx name uuid util used total; do
    idx=${idx//[[:space:]]/}; name=${name# }; name=${name% }; uuid=${uuid//[[:space:]]/}
    util=${util//[[:space:]]/}; used=${used//[[:space:]]/}
    if [[ "$util" != 0 || ! "$used" =~ ^[0-9]+$ || "$used" -gt 64 ]] || grep -Fq "$uuid" <<< "$APPS"; then continue; fi
    candidate=$RESERVATIONS/$uuid
    if mkdir "$candidate" 2>/dev/null; then GPU_INDEX=$idx; GPU_UUID=$uuid; GPU_NAME=$name; LOCK=$candidate; break; fi
  done < <(nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
  [[ -n "$GPU_INDEX" ]] && break
  printf 'attempt=%s utc=%s state=waiting_no_idle_unowned_gpu\n' "$attempt" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_ROOT/gpu_wait_log.txt"
  sleep 60
done
if [[ -z "$GPU_INDEX" ]]; then echo 'GPU gate timed out after 60 one-minute checks; no idle unowned GPU' > "$RUN_ROOT/gpu_gate.txt"; printf '5\n' > "$RUN_ROOT/exit_code.txt"; exit 5; fi
RUN_ID=20261101-stage8-replicad-dynamic-v2
printf '%s\n' "$RUN_ID" > "$LOCK/run_id.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOCK/acquired_utc.txt"
printf '{"schema":"sim2real_gpu_reservation_v1","status":"reserved","host":"%s","gpu_uuid":"%s","run_id":"%s","run_path":"%s","purpose":"paired 100-update synthetic-only dynamic ReplicaCAD GT-flow diagnostic","cuda_visible_devices":"%s"}\n' "$(hostname -f)" "$GPU_UUID" "$RUN_ID" "$RUN_ROOT" "$GPU_INDEX" > "$LOCK/resource_reservation.json"
cp "$LOCK/resource_reservation.json" "$RUN_ROOT/resource_reservation.json"
printf 'gpu_index=%s\ngpu_name=%s\ngpu_uuid=%s\nlock=%s\n' "$GPU_INDEX" "$GPU_NAME" "$GPU_UUID" "$LOCK" > "$RUN_ROOT/gpu_reservation.txt"
release_lock() { local rc=$?; if [[ -n "${LOCK:-}" && -f "$LOCK/run_id.txt" && "$(cat "$LOCK/run_id.txt")" = "$RUN_ID" ]]; then date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/reservation_released_utc.txt"; rm -f "$LOCK/run_id.txt" "$LOCK/acquired_utc.txt" "$LOCK/resource_reservation.json"; rmdir "$LOCK"; fi; [[ -e "$RUN_ROOT/exit_code.txt" ]] || printf '%s\n' "$rc" > "$RUN_ROOT/exit_code.txt"; return 0; }
trap release_lock EXIT
UTIL2=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' %')
USED2=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' MiB')
APPS2=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>&1 || true)
printf 'index=%s\nuuid=%s\nutil=%s\nmemory_mib=%s\ncompute_apps=%s\n' "$GPU_INDEX" "$GPU_UUID" "$UTIL2" "$USED2" "$APPS2" > "$RUN_ROOT/post_reservation_check.txt"
if [[ "$UTIL2" != 0 || ! "$USED2" =~ ^[0-9]+$ || "$USED2" -gt 64 ]] || grep -Fq "$GPU_UUID" <<< "$APPS2"; then echo 'GPU became busy after reservation; refusing training' >> "$RUN_ROOT/post_reservation_check.txt"; printf '5\n' > "$RUN_ROOT/exit_code.txt"; exit 5; fi
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader >> "$RUN_ROOT/post_reservation_check.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/training_started_utc.txt"
set +e
CUDA_VISIBLE_DEVICES="$GPU_INDEX" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$EXEC:$PROJECT" \
  "$PY" -u "$TRAINER" --repo-root "$EXEC" --model-root "$MODEL_ROOT" --data-root "$FLOW_BANK" \
    --config "$CONFIG" --output-root "$RUN_ROOT/training_output" --device cuda:0 \
    > "$RUN_ROOT/trainer.stdout.log" 2> "$RUN_ROOT/trainer.stderr.log"
RC=$?
set -e
printf '%s\n' "$RC" > "$RUN_ROOT/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/stage8_finished_utc.txt"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu_after.csv" 2>&1 || true
if [[ "$RC" -eq 0 ]]; then sha256sum "$RUN_ROOT/training_output/summary.json" > "$RUN_ROOT/summary.sha256"; find "$RUN_ROOT/training_output" -maxdepth 2 -type f -name 'checkpoint_final.pt' -exec sha256sum {} \; > "$RUN_ROOT/checkpoints.sha256"; fi
exit "$RC"
