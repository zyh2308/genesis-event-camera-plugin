#!/usr/bin/env bash
set -euo pipefail

repo=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-cadence-sweep-20261019d
python=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
reference=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016
ours_checkpoint=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-nativehdr-20261019a/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/ours_fit/spatial_event_state.pt
v2e_checkpoint=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-nativehdr-20261019a/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/v2e_fit/spatial_event_state.pt
config=benchmark/robot_closed_loop/configs/r1_5_endpoint_terminal_cadence_isolation_20261019e.json
scene_config=benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json
control_config=benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json
output=benchmark/robot_closed_loop/results/r1_5_endpoint_terminal_cadence_isolation_20261019e
session=r1_5_endpoint_terminal_cadence_isolation_20261019e_1
log=benchmark/robot_closed_loop/results/runtime_logs/r1_5_endpoint_terminal_cadence_isolation_20261019e.log
pid_file=benchmark/robot_closed_loop/results/runtime_logs/r1_5_endpoint_terminal_cadence_isolation_20261019e.pid
exit_file=benchmark/robot_closed_loop/results/runtime_logs/r1_5_endpoint_terminal_cadence_isolation_20261019e.exit

cd "$repo"
test -x "$python"
test -f "$config"
test -f "$scene_config"
test -f "$control_config"
test -f "$reference/SHA256SUMS"
test -f "$ours_checkpoint"
test -f "$v2e_checkpoint"
test ! -e "$output"
test ! -e "$log"
test ! -e "$pid_file"
test ! -e "$exit_file"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "refusing to reuse existing tmux session: $session" >&2
  exit 2
fi

# Replay immutable logged states only. Genesis physics, event emulation, and
# model inference use CPU; idle GPU 0 is reserved for the headless renderer.
tmux new-session -d -s "$session" \
  "cd '$repo' && printf '%s\\n' \$\$ > '$pid_file' && env CUDA_VISIBLE_DEVICES=0 EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. '$python' benchmark/robot_closed_loop/diagnostics/diagnose_r1_ours_localization_cadence.py --config '$config' --scene-config '$scene_config' --control-config '$control_config' --reference-root '$reference' --ours-checkpoint '$ours_checkpoint' --v2e-checkpoint '$v2e_checkpoint' --output '$output' > '$log' 2>&1; rc=\$?; printf '%s\\n' \$rc > '$exit_file'"
sleep 2
tmux list-panes -t "$session" -F 'session=#{session_name} pane_pid=#{pane_pid} command=#{pane_current_command}'
printf 'launcher_pid_file='; cat "$pid_file"
printf 'output_path=%s\nlog_path=%s\n' "$output" "$log"
