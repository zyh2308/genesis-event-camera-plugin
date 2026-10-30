#!/usr/bin/env bash
set -euo pipefail
REPO=/home/科研/Eventbased_WAM/repositories/gec-sim2real
SSH_EXE=/mnt/c/Windows/System32/OpenSSH/ssh.exe
HOST=zhaoyuhan@g1.pkuml.org
PORT=22
REMOTE_RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-replicad-v3-sc2-v2e-gpu-run21
HERE=benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28/replicad_layout_runs/v3_sc2_staging_00/v2e_run21
cd "$REPO"
[[ -x "$SSH_EXE" ]]
for script in "$HERE/driver.sh" "$HERE/remote_start.sh" "$HERE/start.sh"; do bash -n "$script"; done
python -m py_compile benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28/run_v2e_same_radiance_camera_motion.py
PAYLOAD_DIR=$(mktemp -d); trap 'rm -rf "$PAYLOAD_DIR"' EXIT
cp benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28/run_v2e_same_radiance_camera_motion.py "$HERE/registration.json" "$HERE/driver.sh" "$HERE/remote_start.sh" "$PAYLOAD_DIR/"
git -C "$REPO" rev-parse HEAD > "$PAYLOAD_DIR/local_repo_commit.txt"
tar -czf - -C "$PAYLOAD_DIR" . | "$SSH_EXE" -o ConnectTimeout=10 -p "$PORT" "$HOST" "set -euo pipefail; RUN='$REMOTE_RUN'; test ! -e \"\$RUN\"; mkdir -p \"\$RUN\"; tar -xzf - -C \"\$RUN\"; bash \"\$RUN/remote_start.sh\""
