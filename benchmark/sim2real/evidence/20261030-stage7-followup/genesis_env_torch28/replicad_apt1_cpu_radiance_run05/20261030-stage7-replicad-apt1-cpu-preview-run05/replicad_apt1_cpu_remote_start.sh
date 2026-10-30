#!/usr/bin/env bash
set -euo pipefail

RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-replicad-apt1-cpu-preview-run05
ASSET_ROOT=/home/zhaoyuhan/Eventbased_WAM/external/sim2real_assets/replicacad_interactive_v1_5/extracted_v1_5
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python
SESSION=sim2real_replicad_apt1_cpu_20261030_05
CACHE=/tmp/sim2real-stage7-replicad-apt1-cpu-preview-run05
GENESIS_ROOT=/home/zhaoyuhan/Eventbased_WAM/runtime/sim2real/genesis_hdr_v2

cd /home/zhaoyuhan/Eventbased_WAM
test "$(pwd)" = /home/zhaoyuhan/Eventbased_WAM
test -x "$PY"
test -d "$ASSET_ROOT"
test -f "$GENESIS_ROOT/genesis/vis/camera.py"
test ! -e "$RUN/started_utc.txt"
LOAD1=$(awk '{print $1}' /proc/loadavg)
NCPU=$(getconf _NPROCESSORS_ONLN)
MEM=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if ! awk -v l="$LOAD1" -v c="$NCPU" 'BEGIN {exit !(l < c * .8)}'; then
  echo "CPU gate refused: load=$LOAD1 cpus=$NCPU" >&2
  exit 5
fi
if [[ "$MEM" -lt 16777216 ]]; then
  echo "memory gate refused: available_kb=$MEM" >&2
  exit 5
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "refusing duplicate tmux session $SESSION" >&2
  exit 6
fi
mkdir -p "$RUN" "$CACHE/quadrants" "$CACHE/genesis"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/preflight_utc.txt"
{
  printf 'host=%s\nproject=/home/zhaoyuhan/Eventbased_WAM\n' "$(hostname)"
  printf 'execution_checkout_commit='
  git -C /home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030 rev-parse HEAD
  printf 'local_source_repo_commit='
  cat "$RUN/local_repo_commit.txt"
  printf 'python=%s\nasset_root=%s\nload1=%s\ncpu_count=%s\nmem_available_kb=%s\n' \
    "$PY" "$ASSET_ROOT" "$LOAD1" "$NCPU" "$MEM"
  printf 'device_mode=CPU; CUDA_VISIBLE_DEVICES empty; Mesa software rasterizer requested\n'
  printf 'genesis_overlay=%s\n' "$GENESIS_ROOT"
  printf 'quadrants_cache=%s/quadrants\ngenesis_cache=%s/genesis\n' "$CACHE" "$CACHE"
  sha256sum "$ASSET_ROOT/../ReplicaCAD_dataset_v1.5.zip" 2>/dev/null || \
    printf 'archive_sha256=d571c2c1315601c7faed35452e54a15c56646393a4c330235f20fb6fc8d8edfd (previously verified)\n'
  sha256sum "$ASSET_ROOT/configs/scenes/apt_1.scene_instance.json" \
    "$GENESIS_ROOT/genesis/vis/camera.py" \
    "$RUN/replicad_apt1_cpu_import_preview.py" "$PY"
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  df -h "$RUN" /tmp
} > "$RUN/environment_and_inputs.txt" 2>&1
tmux new-session -d -s "$SESSION" "bash '$RUN/replicad_apt1_cpu_driver.sh'"
tmux has-session -t "$SESSION"
PANE_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
printf '%s\n' "$SESSION" > "$RUN/tmux_session.txt"
printf '%s\n' "$PANE_PID" > "$RUN/tmux_pane_pid.txt"
ps -o pid,ppid,etime,args -p "$PANE_PID" > "$RUN/detach_verification.txt"
cat "$RUN/environment_and_inputs.txt"
printf 'tmux_session=%s pane_pid=%s\n' "$SESSION" "$PANE_PID"
