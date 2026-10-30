#!/usr/bin/env bash
set -euo pipefail

REPO=/home/科研/Eventbased_WAM/repositories/gec-sim2real
SSH_EXE=/mnt/c/Windows/System32/OpenSSH/ssh.exe
HOST=zhaoyuhan@g1.pkuml.org
HERE=benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28
REMOTE_RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261030-stage7-genesis-radiance-cpu-probe-run02

cd "$REPO"
[[ -x "$SSH_EXE" ]]
bash -n "$HERE/radiance_probe_remote.sh"
python - <<'PY'
import py_compile
py_compile.compile(
    'benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28/probe_genesis_radiance_cpu.py',
    cfile='/tmp/sim2real-genesis-radiance-probe.pyc', doraise=True)
PY
PAYLOAD=$(mktemp -d)
trap 'rm -rf "$PAYLOAD"' EXIT
cp "$HERE/probe_genesis_radiance_cpu.py" "$HERE/radiance_probe_remote.sh" "$PAYLOAD/"
tar -czf - -C "$PAYLOAD" . | \
  "$SSH_EXE" -o ConnectTimeout=10 -p 22 "$HOST" \
  "set -euo pipefail; RUN='$REMOTE_RUN'; test ! -e \"\$RUN\"; mkdir -p \"\$RUN\"; tar -xzf - -C \"\$RUN\"; bash \"\$RUN/radiance_probe_remote.sh\""
