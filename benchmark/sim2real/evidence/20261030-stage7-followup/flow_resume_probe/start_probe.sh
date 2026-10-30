#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/home/zhaoyuhan/Eventbased_WAM
EXEC=$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
SRC=$EXEC/benchmark/sim2real/scripts
RUN_ROOT=$PROJECT/runs/sim2real/20261030-stage7-flow-resume-200to201-g1-run01
SOURCE_RUN=$PROJECT/runs/sim2real/20261029-stage5-hightexture-fixed-gtfit-g3-v2
RESERVATION_ROOT=$PROJECT/runs/sim2real/.gpu_reservations
GPU_INDEX=2
GPU_UUID=GPU-a02affe0-7598-579e-ca39-f551f35ad6e9
LOCK=$RESERVATION_ROOT/g1_$GPU_UUID
RUN_ID=20261030-stage7-flow-resume-200to201-g1-run01
SCRIPT_SHA=2184f94d120ca9447fc3c2e4b7280c4aed0acda12176e1bbc285ddfdd4ab3478
DRIVER_SHA=18ade169e24b89e1ea884117b009d0a214e3475aebd9b810202d44eccad8002d
CONFIG_SHA=ce7c7e6c7670f2514e75db821eb7ef93939ff6794bcb5af8bc90e024b4597665
DATA_SHA=02f8eab65559a9b8442fe04b46ae202fe23a15116ae62fe7b3385e7e6c3cf50d
EG_SHA=4da12339e52e29be1425287e3e4af5056cc3c7a125f7cc12f3ba326ad69f2a9b
V2E_SHA=07bae17f1c1d5c635e863b73349a13ea04c635f1a079c31f466b254142bf143c
STARTED=0
LOCK_ACQUIRED=0

cleanup_preflight() {
  rc=$?
  trap - EXIT
  if [ "$rc" -ne 0 ] && [ "$STARTED" -eq 0 ] && [ "$LOCK_ACQUIRED" -eq 1 ] \
      && [ -f "$LOCK/reservation.txt" ] \
      && grep -Fqx "run_id=$RUN_ID" "$LOCK/reservation.txt"; then
    cp "$LOCK/reservation.txt" "$PROJECT/runs/sim2real/${RUN_ID}.failed_preflight_reservation.txt" || true
    rm -f "$LOCK/reservation.txt"
    rmdir "$LOCK" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup_preflight EXIT

echo "preflight_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname -f) run_id=$RUN_ID"
echo "execution_base_commit=$(git -C "$EXEC" rev-parse HEAD)"
if [ -f "$SRC/driver.sh" ]; then
  if [ "$(sha256sum "$SRC/driver.sh" | awk '{print $1}')" = 92942585fbef0fa6aa9a46c0a1a2a6f1ce1752214deb87ef681181e0dc95b90b ]; then
    rm "$SRC/driver.sh"
  else
    echo "refusing unexpected scripts/driver.sh"
    exit 5
  fi
fi
echo "probe_script_sha256=$(sha256sum "$SRC/resume_stage5_flow_fit_one_update.py" | awk '{print $1}')"
echo "probe_driver_sha256=$(sha256sum "$SRC/stage7_flow_resume_probe_driver.sh" | awk '{print $1}')"
echo "config_sha256=$(sha256sum "$EXEC/benchmark/sim2real/configs/stage5_genesis_high_texture_fixed_sample_gt_fit_v1.json" | awk '{print $1}')"
echo "data_manifest_sha256=$(sha256sum "$SOURCE_RUN/paired_adapter/training_bank_manifest.json" | awk '{print $1}')"
test "$(sha256sum "$SRC/resume_stage5_flow_fit_one_update.py" | awk '{print $1}')" = "$SCRIPT_SHA"
test "$(sha256sum "$SRC/stage7_flow_resume_probe_driver.sh" | awk '{print $1}')" = "$DRIVER_SHA"
test "$(sha256sum "$EXEC/benchmark/sim2real/configs/stage5_genesis_high_texture_fixed_sample_gt_fit_v1.json" | awk '{print $1}')" = "$CONFIG_SHA"
test "$(sha256sum "$SOURCE_RUN/paired_adapter/training_bank_manifest.json" | awk '{print $1}')" = "$DATA_SHA"
test "$(sha256sum "$SOURCE_RUN/fixed_sample_fit/eventgenesis_last.pt" | awk '{print $1}')" = "$EG_SHA"
test "$(sha256sum "$SOURCE_RUN/fixed_sample_fit/v2e_last.pt" | awk '{print $1}')" = "$V2E_SHA"
test ! -e "$RUN_ROOT"
if tmux has-session -t "$RUN_ID" 2>/dev/null; then
  echo "tmux session already exists; refusing duplicate start"
  exit 8
fi

for file in "$RESERVATION_ROOT"/*/reservation.txt; do
  [ -f "$file" ] || continue
  if grep -Fqx "gpu_uuid=$GPU_UUID" "$file"; then
    echo "GPU UUID already reserved by $file"
    cat "$file"
    exit 6
  fi
done
mkdir "$LOCK"
LOCK_ACQUIRED=1
printf 'schema=project_gpu_reservation_v1\nrun_id=%s\nstatus=reserved\nhost=g1.pkuml.org\nuser=zhaoyuhan\ngpu_index=%s\ngpu_uuid=%s\nlocal_repository_commit=4c81857\nexecution_base_commit=d0e36796f04826a14a0931878405abed943a49a3\nprobe_script_sha256=%s\ndriver_sha256=%s\nconfig_sha256=%s\ndata_manifest_sha256=%s\neventgenesis_source_checkpoint_sha256=%s\nv2e_source_checkpoint_sha256=%s\npurpose=independent_one_step_resume_probe_200_to_201\ncreated_utc=%s\nreal_data_used=false\n' \
  "$RUN_ID" "$GPU_INDEX" "$GPU_UUID" "$SCRIPT_SHA" "$DRIVER_SHA" "$CONFIG_SHA" "$DATA_SHA" "$EG_SHA" "$V2E_SHA" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK/reservation.txt"

ROW=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader -i "$GPU_INDEX")
echo "gpu_recheck=$ROW"
UUID=$(echo "$ROW" | cut -d, -f1 | tr -d ' ')
MEM=$(echo "$ROW" | cut -d, -f2 | tr -dc '0-9')
UTIL=$(echo "$ROW" | cut -d, -f3 | tr -dc '0-9')
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep -F "$GPU_UUID" || true)
if [ "$UUID" != "$GPU_UUID" ] || [ "$MEM" -gt 50 ] || [ "$UTIL" -gt 5 ] || [ -n "$APPS" ]; then
  echo "GPU resource recheck failed: uuid=$UUID mem_mib=$MEM util_pct=$UTIL apps=$APPS"
  exit 7
fi
echo "scheduler=$(command -v squeue || command -v qstat || echo none-visible)"

mkdir -p "$RUN_ROOT"
chmod +x "$SRC/stage7_flow_resume_probe_driver.sh"
tmux new-session -d -s "$RUN_ID" "bash $SRC/stage7_flow_resume_probe_driver.sh"
STARTED=1
tmux has-session -t "$RUN_ID"
sleep 2
echo "tmux_session=$RUN_ID"
echo "driver_pid=$(cat "$RUN_ROOT/driver.pid" 2>/dev/null || echo starting)"
echo "run_state=$(cat "$RUN_ROOT/run_state.txt" 2>/dev/null || echo starting)"
if tmux has-session -t "$RUN_ID"; then tmux capture-pane -t "$RUN_ID" -p | tail -20; else echo "tmux_session=completed_or_exited"; fi
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader -i "$GPU_INDEX"
