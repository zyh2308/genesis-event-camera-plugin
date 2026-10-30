#!/usr/bin/env bash
set -euo pipefail

REPO=/home/科研/Eventbased_WAM/repositories/gec-sim2real
SSH_EXE=/mnt/c/Windows/System32/OpenSSH/ssh.exe
HOST=zhaoyuhan@g1.pkuml.org
REMOTE_CHECKOUT=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
REL=benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28

cd "$REPO"
[[ -x "$SSH_EXE" ]]
[[ -f "$REL/driver_run02.sh" && -f "$REL/launch_run02.sh" ]]
PAYLOAD_DIR=$(mktemp -d)
trap 'rm -rf "$PAYLOAD_DIR"' EXIT
mkdir -p "$PAYLOAD_DIR/$REL"
cp "$REL/driver_run02.sh" "$REL/launch_run02.sh" "$PAYLOAD_DIR/$REL/"
tar -czf - -C "$PAYLOAD_DIR" benchmark | base64 -w0 | \
  "$SSH_EXE" -p 22 "$HOST" \
  "test -d '$REMOTE_CHECKOUT' && base64 -d | tar -xz -C '$REMOTE_CHECKOUT' && mkdir -p /tmp/sim2real-stage7-torch28-20261030-run02 && bash '$REMOTE_CHECKOUT/$REL/launch_run02.sh'"
