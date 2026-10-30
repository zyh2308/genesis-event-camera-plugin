#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/zhaoyuhan/Eventbased_WAM
RUN_ROOT=$PROJECT/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1
RETRY_ROOT=$RUN_ROOT/v2e_retry01
SESSION=sim2real_stage7_multiaxis_v2e_retry01_20261031
cd "$PROJECT"
test "$(pwd)" = "$PROJECT"
test -f "$RETRY_ROOT/run_replicad_multiaxis_v2e_gpu_retry01.sh"
test -f "$RETRY_ROOT/run_v2e_same_radiance_camera_motion_retry01.py"
test -f "$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030/benchmark/sim2real/scripts/generate_stage5_flow_coverage_bank.py"
test ! -e "$RETRY_ROOT/started_utc.txt"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "refusing duplicate tmux session $SESSION" >&2
  exit 6
fi
mkdir -p "$RETRY_ROOT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RETRY_ROOT/launcher_gpu_check_utc.txt"
{
  nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
  uptime
  free -h
  df -h "$RUN_ROOT" /tmp
  sha256sum "$RETRY_ROOT/run_replicad_multiaxis_v2e_gpu_retry01.sh"
  sha256sum "$RETRY_ROOT/run_v2e_same_radiance_camera_motion_retry01.py"
  sha256sum "$PROJECT/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030/benchmark/sim2real/scripts/generate_stage5_flow_coverage_bank.py"
} > "$RETRY_ROOT/launcher_gpu_check.txt" 2>&1
tmux new-session -d -s "$SESSION" "bash '$RETRY_ROOT/run_replicad_multiaxis_v2e_gpu_retry01.sh' > '$RETRY_ROOT/driver.log' 2>&1"
tmux has-session -t "$SESSION"
PANE_PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
printf '%s\n' "$SESSION" > "$RETRY_ROOT/tmux_session.txt"
printf '%s\n' "$PANE_PID" > "$RETRY_ROOT/tmux_pane_pid.txt"
ps -o pid,ppid,etime,args -p "$PANE_PID" > "$RETRY_ROOT/detach_verification.txt"
cat "$RETRY_ROOT/detach_verification.txt"
