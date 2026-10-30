#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-recon-loader-parity-run03
REPO=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
DRIVER="$REPO/benchmark/sim2real/evidence/20261030-stage7-followup/reconstruction_loader_parity/driver.sh"
SESSION=20261030-stage7-recon-loader-parity-g1-run03
PYTHON=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python
DATA_ROOT="$PROJECT/data/sim2real/stage4/e2vid_corrected_10ms_20261019_v1"

test -f "$DRIVER"
test -x "$PYTHON"
test -d "$PROJECT/external/sources/downstream/event_cnn_minimal"
test -d "$PROJECT/external/sources/downstream/EVREAL"
test -f "$DATA_ROOT/eventgenesis/train/s0_00.h5"
test -f "$DATA_ROOT/v2e/train/s0_00.h5"
test ! -e "$RUN_ROOT"

load1=$(awk '{print $1}' /proc/loadavg)
ncpu=$(nproc)
max_load=$(awk -v n="$ncpu" 'BEGIN{x=n*0.8;if(x<32)x=32;print x}')
mem_available_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
awk -v l1="$load1" -v limit="$max_load" 'BEGIN{exit !(l1<limit)}' || {
  echo "CPU preflight rejected: load=$load1 limit=$max_load" >&2; exit 20;
}
if [ "$mem_available_kb" -lt 16777216 ]; then
  echo "CPU preflight rejected: MemAvailable_kB=$mem_available_kb < 16777216" >&2; exit 21
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists; refusing duplicate start: $SESSION" >&2; exit 22
fi

printf 'preflight_utc=%s host=%s load1=%s ncpu=%s max_load=%s mem_available_kb=%s session=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -f)" "$load1" "$ncpu" "$max_load" "$mem_available_kb" "$SESSION"
tmux new-session -d -s "$SESSION" "bash '$DRIVER'"
sleep 1
tmux has-session -t "$SESSION"
printf 'detached_session=%s\n' "$SESSION"
tmux capture-pane -pt "$SESSION" -S -8 || true
