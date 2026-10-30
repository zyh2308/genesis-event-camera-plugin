#!/usr/bin/env bash
set -euo pipefail

REPO=/home/科研/Eventbased_WAM/repositories/gec-sim2real
SSH_EXE=/mnt/c/Windows/System32/OpenSSH/ssh.exe
HOST=zhaoyuhan@g1.pkuml.org
PORT=22
REMOTE_RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-apt1-v2e-camera-motion-cpu-run08
HERE=benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28
cd "$REPO"
[[ -x "$SSH_EXE" ]]
bash -n "$HERE/run_v2e_same_radiance_camera_motion_driver.sh"
bash -n "$HERE/run_v2e_same_radiance_camera_motion_remote_start.sh"
python -m py_compile "$HERE/run_v2e_same_radiance_camera_motion.py"
PAYLOAD_DIR=$(mktemp -d)
trap 'rm -rf "$PAYLOAD_DIR"' EXIT
cp "$HERE/run_v2e_same_radiance_camera_motion.py" "$HERE/run_v2e_same_radiance_camera_motion_driver.sh" "$HERE/run_v2e_same_radiance_camera_motion_remote_start.sh" "$PAYLOAD_DIR/"
git rev-parse HEAD > "$PAYLOAD_DIR/local_repo_commit.txt"
tar -czf - -C "$PAYLOAD_DIR" . | \
  "$SSH_EXE" -o ConnectTimeout=10 -p "$PORT" "$HOST" \
  "set -euo pipefail; RUN='/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-apt1-v2e-camera-motion-cpu-run08'; test ! -e \"\$RUN\"; mkdir -p \"\$RUN\"; tar -xzf - -C \"\$RUN\"; bash \"\$RUN/run_v2e_same_radiance_camera_motion_remote_start.sh\""
