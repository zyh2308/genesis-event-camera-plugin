#!/usr/bin/env python3
"""Low-frequency WSL watcher for g1/g3; launches one reserved CUDA probe."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SSH = Path("/mnt/c/Windows/System32/OpenSSH/ssh.exe")
REMOTE_AGENT = ROOT / "benchmark/robot_closed_loop/diagnostics/launch_r1_5_candidate_cuda_parity_remote.sh"
RUN_ID = "r1_5_core_cuda_parity_20261030a"
BRANCH = "exp/r1_5-core-candidate-20261030"
HOSTS = ("g1.pkuml.org", "g3.pkuml.org")
PROBE = r'''set -u
HOME_ROOT=/home/zhaoyuhan/Eventbased_WAM
LOCK_PARENT="$HOME_ROOT/runs/robot_scenes/.gpu_reservations"
load=$(awk '{print $1}' /proc/loadavg)
cores=$(nproc)
mem=$(awk '/MemAvailable:/ {print $2*1024}' /proc/meminfo)
disk=$(df -Pk "$HOME_ROOT" | awk 'NR==2 {print $4*1024}')
fstype=$(findmnt -T "$HOME_ROOT" -no FSTYPE 2>/dev/null || echo unknown)
jobs=0
if command -v squeue >/dev/null 2>&1; then
  jobs=$(squeue -h -u zhaoyuhan -o '%T' 2>/dev/null | awk '$1=="PENDING" || $1=="RUNNING" {n++} END {print n+0}')
fi
printf 'META\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(hostname -s)" "$load" "$cores" "$mem" "$disk" "$fstype" "$jobs"
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null || true)
pmon=$(nvidia-smi pmon -c 1 -s um 2>/dev/null || true)
while IFS=, read -r index uuid name util memused total; do
  index=$(printf '%s' "$index" | xargs); uuid=$(printf '%s' "$uuid" | xargs)
  name=$(printf '%s' "$name" | xargs); util=$(printf '%s' "$util" | xargs)
  memused=$(printf '%s' "$memused" | xargs); total=$(printf '%s' "$total" | xargs)
  [[ -n "$uuid" ]] || continue
  app=0; procmon=0; reserved=0
  printf '%s\n' "$apps" | grep -Fq "$uuid" && app=1
  printf '%s\n' "$pmon" | awk -v i="$index" '$1==i && $2 ~ /^[0-9]+$/ {found=1} END {exit !found}' && procmon=1
  [[ -e "$LOCK_PARENT/$uuid" ]] && reserved=1
  printf 'GPU\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$index" "$uuid" "$name" "$util" "$memused" "$total" "$app" "$procmon" "$reserved"
done < <(nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
'''


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def log(stream, message: str) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    stream.write(line + "\n")
    stream.flush()


def probe(host: str, timeout_s: float) -> dict:
    command = [
        str(SSH), "-T", "-p", "22", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes", "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=2", f"zhaoyuhan@{host}", "bash -s",
    ]
    result = subprocess.run(command, input=PROBE, text=True, capture_output=True, timeout=timeout_s)
    if result.returncode:
        raise RuntimeError(f"ssh exit={result.returncode}: {(result.stderr or result.stdout).strip()[-600:]}")
    meta = None
    gpus = []
    for line in result.stdout.splitlines():
        parts = line.strip().split("\t")
        if parts[0] == "META" and len(parts) == 8:
            meta = {
                "hostname": parts[1], "load1": float(parts[2]), "cpu_count": int(parts[3]),
                "mem_available_bytes": int(parts[4]), "home_free_bytes": int(parts[5]),
                "home_fstype": parts[6], "scheduler_jobs": int(parts[7]),
            }
        elif parts[0] == "GPU" and len(parts) == 10:
            gpus.append({
                "index": int(parts[1]), "uuid": parts[2], "name": parts[3],
                "util_pct": int(parts[4]), "memory_used_mib": int(parts[5]),
                "memory_total_mib": int(parts[6]), "compute_process": bool(int(parts[7])),
                "pmon_process": bool(int(parts[8])), "project_reserved": bool(int(parts[9])),
            })
    if not meta or not gpus:
        raise RuntimeError(f"probe output was incomplete: {result.stdout[-800:]}")
    meta["host_gate"] = (
        meta["home_fstype"] == "nfs"
        and meta["load1"] <= 1.25 * meta["cpu_count"]
        and meta["mem_available_bytes"] >= 8 * 1024**3
        and meta["home_free_bytes"] >= 2 * 1024**3
        and meta["scheduler_jobs"] == 0
    )
    for gpu in gpus:
        gpu["free_candidate"] = bool(
            meta["host_gate"] and gpu["util_pct"] <= 5 and gpu["memory_used_mib"] <= 512
            and not gpu["compute_process"] and not gpu["pmon_process"] and not gpu["project_reserved"]
        )
    return {"host": host, "meta": meta, "gpus": gpus, "stderr": result.stderr.strip()}


def launch(host: str, gpu: dict, source_commit: str, timeout_s: float) -> str:
    remote_script = REMOTE_AGENT.read_text(encoding="utf-8")
    remote_command = "bash -s -- {} {} {} {} {} {}".format(
        "g1" if host.startswith("g1.") else "g3", RUN_ID, BRANCH, source_commit,
        gpu["index"], gpu["uuid"],
    )
    command = [
        str(SSH), "-T", "-p", "22", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes", "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=2", f"zhaoyuhan@{host}", remote_command,
    ]
    result = subprocess.run(command, input=remote_script, text=True, capture_output=True, timeout=timeout_s)
    combined = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode:
        raise RuntimeError(f"remote launcher exit={result.returncode}: {combined[-1600:]}")
    if "LAUNCHED" not in combined:
        raise RuntimeError(f"remote launcher did not confirm detached launch: {combined[-1600:]}")
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--log-dir", type=Path, default=Path.home() / "Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/r1_5_core_cuda_parity_20261030a")
    parser.add_argument("--ssh-timeout-seconds", type=float, default=25.0)
    args = parser.parse_args()
    if not SSH.exists():
        raise FileNotFoundError(f"required Windows OpenSSH client missing: {SSH}")
    if not REMOTE_AGENT.is_file():
        raise FileNotFoundError(f"remote launch script missing: {REMOTE_AGENT}")
    if args.poll_seconds < 60:
        raise ValueError("poll interval must be at least 60 seconds")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / "watcher.log"
    source_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    timeout_at = time.monotonic() + args.max_hours * 3600
    streaks: dict[tuple[str, str], int] = {}
    prior_free: set[tuple[str, str]] = set()
    failed_hosts: set[str] = set()
    host_failures: dict[str, int] = {host: 0 for host in HOSTS}
    next_probe_at: dict[str, float] = {host: 0.0 for host in HOSTS}

    with log_path.open("a", encoding="utf-8", buffering=1) as stream:
        log(stream, f"START run_id={RUN_ID} branch={BRANCH} commit={source_commit} poll_s={args.poll_seconds} hosts={','.join(HOSTS)}")
        log(stream, "criteria: two probes >=60s apart; util<=5%; memory<=512MiB; no compute/pmon PID; no UUID reservation; host load/memory/disk/scheduler gates; shared atomic reservation and final recheck before tmux launch")
        while time.monotonic() < timeout_at:
            free_now: set[tuple[str, str]] = set()
            probe_end = time.monotonic()
            for host in HOSTS:
                if host in failed_hosts or time.monotonic() < next_probe_at[host]:
                    continue
                try:
                    result = probe(host, args.ssh_timeout_seconds)
                    host_failures[host] = 0
                    log(stream, f"PROBE {json.dumps(result, separators=(',', ':'))}")
                    for gpu in result["gpus"]:
                        key = (host, gpu["uuid"])
                        if gpu["free_candidate"]:
                            streaks[key] = streaks.get(key, 0) + 1 if key in prior_free else 1
                            free_now.add(key)
                        else:
                            streaks[key] = 0
                except subprocess.TimeoutExpired as exc:
                    error = f"SSH timeout after {exc.timeout}s"
                    host_failures[host] += 1
                    log(stream, f"SSH_ERROR host={host} {error} failure={host_failures[host]}; cooldown={args.poll_seconds}s")
                    next_probe_at[host] = time.monotonic() + max(60, args.poll_seconds)
                    if host_failures[host] >= 2:
                        failed_hosts.add(host)
                        log(stream, f"HOST_DISABLED host={host} reason=second consecutive timeout")
                except Exception as exc:
                    message = str(exc)
                    host_failures[host] += 1
                    hard = any(token in message.lower() for token in ("host key verification failed", "permission denied", "no supported authentication", "remote host identification has changed"))
                    log(stream, f"SSH_ERROR host={host} failure={host_failures[host]} hard={hard} detail={message!r}")
                    if hard or host_failures[host] >= 2:
                        failed_hosts.add(host)
                        log(stream, f"HOST_DISABLED host={host} reason=authentication/host-key error or second failure")
                    else:
                        next_probe_at[host] = time.monotonic() + max(60, args.poll_seconds)
            stable = [key for key in free_now if key in prior_free and streaks.get(key, 0) >= 2]
            if stable:
                # Honor g1 preference, then a stable UUID order for reproducibility.
                rank = {"g1.pkuml.org": 0, "g3.pkuml.org": 1}
                # The next probe is the source of truth for the selected exact UUID.
                stable.sort(key=lambda item: (rank.get(item[0], 9), item[1]))
                host, uuid = stable[0]
                snapshot = None
                try:
                    snapshot = probe(host, args.ssh_timeout_seconds)
                    log(stream, f"PRELAUNCH_RECHECK {json.dumps(snapshot, separators=(',', ':'))}")
                    gpu = next((item for item in snapshot["gpus"] if item["uuid"] == uuid), None)
                    if not gpu or not gpu["free_candidate"]:
                        log(stream, f"PRELAUNCH_RECHECK_FAILED host={host} uuid={uuid}; returning to monitoring")
                    else:
                        detail = launch(host, gpu, source_commit, max(60.0, args.ssh_timeout_seconds * 3))
                        log(stream, f"DEPLOYED {detail}")
                        return 0
                except Exception as exc:
                    log(stream, f"DEPLOY_ERROR host={host} uuid={uuid} detail={str(exc)!r}; will not retry deployment until fresh stable probes")
                    streaks[(host, uuid)] = 0
            prior_free = free_now
            remaining = timeout_at - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(args.poll_seconds, remaining))
        log(stream, f"STOP reason=timeout_or_hosts_disabled failed_hosts={sorted(failed_hosts)}")
    return 2 if len(failed_hosts) == len(HOSTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
