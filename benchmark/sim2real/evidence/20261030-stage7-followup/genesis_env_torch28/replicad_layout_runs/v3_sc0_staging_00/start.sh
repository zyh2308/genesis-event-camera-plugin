#!/usr/bin/env bash
set -euo pipefail
REPO=/home/科研/Eventbased_WAM/repositories/gec-sim2real
SSH_EXE=/mnt/c/Windows/System32/OpenSSH/ssh.exe
HOST=zhaoyuhan@g1.pkuml.org
PORT=22
REMOTE_RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-replicad-v3_sc0_staging_00-fast-run14
HERE=benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28
cd "$REPO"
[[ -x "$SSH_EXE" ]]
bash -n "$HERE/replicad_layout_runs/v3_sc0_staging_00/driver.sh"
bash -n "$HERE/replicad_layout_runs/v3_sc0_staging_00/remote_start.sh"
python -c 'import py_compile,sys; py_compile.compile(sys.argv[1], cfile="/tmp/sim2real_v3_sc0_staging_00.pyc", doraise=True)' "$HERE/replicad_layout_camera_motion_pilot.py"
PAYLOAD_DIR=$(mktemp -d); trap 'rm -rf "$PAYLOAD_DIR"' EXIT
cp "$HERE/replicad_layout_camera_motion_pilot.py" "$HERE/replicad_layout_camera_motion_pilot_registration_v1.json" "$HERE/replicad_layout_runs/v3_sc0_staging_00/driver.sh" "$HERE/replicad_layout_runs/v3_sc0_staging_00/remote_start.sh" "$PAYLOAD_DIR/"
git -C "$REPO" rev-parse HEAD > "$PAYLOAD_DIR/local_repo_commit.txt"
tar -czf - -C "$PAYLOAD_DIR" . | "$SSH_EXE" -o ConnectTimeout=10 -p "$PORT" "$HOST" "set -euo pipefail; RUN='/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-replicad-v3_sc0_staging_00-fast-run14'; test ! -e \"\$RUN\"; mkdir -p \"\$RUN\"; tar -xzf - -C \"\$RUN\"; bash \"\$RUN/remote_start.sh\""
