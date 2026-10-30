#!/usr/bin/env bash
set -euo pipefail

REPO=/home/科研/Eventbased_WAM/repositories/gec-sim2real
SSH_EXE=/mnt/c/Windows/System32/OpenSSH/ssh.exe
HOST=zhaoyuhan@g3.pkuml.org
REMOTE_CHECKOUT=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030
REL=benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28
SESSION=sim2real_g3_torch28_wheel_download_20261030_01

cd "$REPO"
[[ -x "$SSH_EXE" ]]
[[ -f "$REL/driver_g3_download.sh" ]]
PAYLOAD_DIR=$(mktemp -d)
trap 'rm -rf "$PAYLOAD_DIR"' EXIT
mkdir -p "$PAYLOAD_DIR/$REL"
cp "$REL/driver_g3_download.sh" "$PAYLOAD_DIR/$REL/"
tar -czf - -C "$PAYLOAD_DIR" benchmark | base64 -w0 | \
  "$SSH_EXE" -p 22 "$HOST" \
  "test -d '$REMOTE_CHECKOUT' && mkdir -p '$REMOTE_CHECKOUT/$REL' && base64 -d | tar -xz -C '$REMOTE_CHECKOUT' && tmux new-session -d -s '$SESSION' \"bash '$REMOTE_CHECKOUT/$REL/driver_g3_download.sh'\" && tmux list-sessions | grep -F '$SESSION'"
