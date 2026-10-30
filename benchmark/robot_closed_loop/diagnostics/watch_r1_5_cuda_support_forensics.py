#!/usr/bin/env python3
"""Low-frequency WSL watcher for the R1.5 support forensics CUDA run."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SSH = Path("/mnt/c/Windows/System32/OpenSSH/ssh.exe")
SCP = Path("/mnt/c/Windows/System32/OpenSSH/scp.exe")
REMOTE_AGENT = ROOT / "benchmark/robot_closed_loop/diagnostics/launch_r1_5_cuda_support_forensics_remote.sh"
RUN_ID = "r1_5_cuda_support_forensics_20261031b"
BRANCH = "exp/r1_5-core-candidate-20261030"
HOSTS = ("g1.pkuml.org", "g3.pkuml.org")
REMOTE_RUN_ROOT = f"/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/{RUN_ID}"
LOCAL_PROJECT = ROOT.parent.parent
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


def ssh_command(host: str, remote_command: str) -> list[str]:
    return [
        str(SSH), "-T", "-p", "22", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes", "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=2", f"zhaoyuhan@{host}", remote_command,
    ]


def stage_source(host: str, source_commit: str, timeout_s: float) -> str:
    files = [
        "genesis_event_plugin",
        "benchmark/robot_closed_loop/diagnostics/validate_r1_5_core_candidate.py",
        "benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_core_candidate_cuda_parity.py",
        "benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_cuda_support_forensics.py",
        "benchmark/robot_closed_loop/diagnostics/launch_r1_5_cuda_support_forensics_worker.sh",
        "benchmark/robot_closed_loop/configs/r1_5_cuda_support_forensics_20261031b.json",
        "benchmark/robot_closed_loop/results/r1_5_core_cuda_parity_20261030c/cuda_parity_arrays.npz",
    ]
    bundle = io.BytesIO()
    with tarfile.open(fileobj=bundle, mode="w:gz") as archive:
        for relative in files:
            archive.add(ROOT / relative, arcname=relative, filter=lambda item: None if "__pycache__" in item.name or item.name.endswith(".pyc") else item)
    payload = bundle.getvalue()
    bundle_hash = hashlib.sha256(payload).hexdigest()
    provenance = json.dumps({
        "source_commit": source_commit,
        "branch": BRANCH,
        "bundle_sha256": bundle_hash,
        "staged_utc": utc_now(),
    }, separators=(",", ":"))
    source_root = f"{REMOTE_RUN_ROOT}/source"
    command = (
        f"mkdir -m 700 -p '{REMOTE_RUN_ROOT}' && test ! -e '{source_root}' && "
        f"mkdir -m 700 '{source_root}' && tar -xz -C '{source_root}' && "
        f"printf '%s\\n' '{provenance}' > '{source_root}/source_provenance.json' && "
        f"chmod 700 '{source_root}/benchmark/robot_closed_loop/diagnostics/launch_r1_5_cuda_support_forensics_worker.sh' && "
        f"test -f '{source_root}/benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_cuda_support_forensics.py' && "
        f"test -f '{source_root}/benchmark/robot_closed_loop/results/r1_5_core_cuda_parity_20261030c/cuda_parity_arrays.npz'"
    )
    result = subprocess.run(ssh_command(host, command), input=payload, capture_output=True, timeout=timeout_s)
    if result.returncode:
        raise RuntimeError(f"source staging ssh exit={result.returncode}: {(result.stderr or result.stdout).decode(errors='replace')[-800:]}")
    return bundle_hash


def launch(host: str, gpu: dict, source_commit: str, timeout_s: float) -> str:
    remote_script = REMOTE_AGENT.read_text(encoding="utf-8")
    remote_command = "bash -s -- {} {} {} {} {}".format(
        "g1" if host.startswith("g1.") else "g3", RUN_ID, source_commit, gpu["index"], gpu["uuid"],
    )
    result = subprocess.run(ssh_command(host, remote_command), input=remote_script, text=True, capture_output=True, timeout=timeout_s)
    combined = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode:
        raise RuntimeError(f"remote launcher exit={result.returncode}: {combined[-1600:]}")
    if "LAUNCHED" not in combined:
        raise RuntimeError(f"remote launcher did not confirm detached launch: {combined[-1600:]}")
    return combined


def read_remote_status(host: str, timeout_s: float) -> dict | None:
    path = f"{REMOTE_RUN_ROOT}/logs/status.json"
    command = f"if test -f '{path}'; then cat '{path}'; else printf 'RUNNING\\n'; fi"
    result = subprocess.run(ssh_command(host, command), text=True, capture_output=True, timeout=timeout_s)
    if result.returncode:
        raise RuntimeError(f"status ssh exit={result.returncode}: {(result.stderr or result.stdout).strip()[-600:]}")
    content = result.stdout.strip()
    if content == "RUNNING":
        return None
    return json.loads(content)


def archive_results(host: str, source_commit: str, status: dict, log_stream, timeout_s: float) -> None:
    windows_temp = Path("/mnt/c/Users/zhaoyuhan/AppData/Local/Temp") / RUN_ID
    if not windows_temp.parent.exists():
        raise FileNotFoundError(f"Windows temp path unavailable: {windows_temp.parent}")
    if windows_temp.exists():
        raise FileExistsError(f"refusing to overwrite download staging path: {windows_temp}")
    windows_temp.mkdir(parents=True)
    sources = [f"zhaoyuhan@{host}:{REMOTE_RUN_ROOT}/logs"]
    if status.get("output_exists"):
        sources.insert(0, f"zhaoyuhan@{host}:{REMOTE_RUN_ROOT}/output")
    command = [
        str(SCP), "-r", "-P", "22", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2", *sources,
        "C:/Users/zhaoyuhan/AppData/Local/Temp/" + RUN_ID + "/",
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout_s)
    if result.returncode:
        raise RuntimeError(f"artifact scp exit={result.returncode}: {(result.stderr or result.stdout).strip()[-1000:]}")
    local_output = ROOT / "benchmark/robot_closed_loop/results" / RUN_ID
    local_logs = ROOT / "benchmark/robot_closed_loop/results/runtime_logs" / RUN_ID
    if local_output.exists() or local_logs.exists():
        raise FileExistsError("refusing to overwrite existing local result or log directory")
    downloaded_output = windows_temp / "output"
    downloaded_logs = windows_temp / "logs"
    if downloaded_output.is_dir():
        shutil.copytree(downloaded_output, local_output)
    if not downloaded_logs.is_dir():
        raise FileNotFoundError("remote logs directory was not downloaded")
    shutil.copytree(downloaded_logs, local_logs)
    if local_output.is_dir() and (local_output / "SHA256SUMS").is_file():
        verification = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=local_output, text=True, capture_output=True)
        if verification.returncode:
            raise RuntimeError(f"downloaded output checksum failure: {verification.stdout} {verification.stderr}")
    log(log_stream, f"ARCHIVED remote={REMOTE_RUN_ROOT} local_output={local_output} local_logs={local_logs}")
    current = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if current != source_commit:
        log(log_stream, f"ARCHIVE_NOT_COMMITTED branch_head_moved expected={source_commit} actual={current}; files retained locally")
        return
    summary_path = local_output / "summary.json"
    rel_paths = [
        f"benchmark/robot_closed_loop/results/runtime_logs/{RUN_ID}",
        "docs/coordination/robot_status.md",
    ]
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        acceptance = summary.get("acceptance_result", {})
        status_line = (
            f"\n#### R1.5 support forensics run `{RUN_ID}` ({summary.get('gpu', {}).get('name', 'GPU')}; "
            f"{summary.get('acceptance_result', {}).get('all_gates_pass', 'unknown')})\n\n"
            f"- Source commit: `{source_commit}`; device `{summary.get('gpu', {}).get('name', 'unknown')}`. "
            f"Acceptance: `{json.dumps(acceptance, separators=(',', ':'))}`.\n"
            f"- Detailed summary/arrays: [`{RUN_ID}`](../../benchmark/robot_closed_loop/results/{RUN_ID}/README.md); "
            f"runtime log copy: `benchmark/robot_closed_loop/results/runtime_logs/{RUN_ID}/`.\n"
        )
        status_doc = ROOT / "docs/coordination/robot_status.md"
        with status_doc.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(status_line)
        rel_paths.insert(0, f"benchmark/robot_closed_loop/results/{RUN_ID}")
    else:
        status_doc = ROOT / "docs/coordination/robot_status.md"
        with status_doc.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"\n#### R1.5 CUDA parity run `{RUN_ID}`\n\n- Remote exit code `{status.get('exit_code')}`; no result summary was produced. Runtime logs are in `benchmark/robot_closed_loop/results/runtime_logs/{RUN_ID}/`.\n")
    (local_logs / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(local_logs).as_posix()}\n"
            for path in sorted(local_logs.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"
        ), encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(ROOT), "add", "-f", *rel_paths], check=True)
    subprocess.run(["git", "-C", str(ROOT), "commit", "-m", f"Record {RUN_ID} CUDA parity diagnostic"], check=True)
    pushed = subprocess.run(["git", "-C", str(ROOT), "push", "origin", BRANCH], text=True, capture_output=True, timeout=120)
    if pushed.returncode:
        log(log_stream, f"ARCHIVE_PUSH_FAILED {pushed.stderr.strip()[-800:]}; local commit retained")
    else:
        log(log_stream, f"ARCHIVE_PUSHED {pushed.stdout.strip()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--log-dir", type=Path, default=ROOT.parent.parent / f"runs/robot_scenes/genesis_event_camera_plugin/{RUN_ID}")
    parser.add_argument("--ssh-timeout-seconds", type=float, default=25.0)
    args = parser.parse_args()
    if not SSH.exists() or not SCP.exists():
        raise FileNotFoundError(f"required Windows OpenSSH client missing: {SSH}")
    if not REMOTE_AGENT.is_file():
        raise FileNotFoundError(f"remote launch script missing: {REMOTE_AGENT}")
    if args.poll_seconds < 60:
        raise ValueError("poll interval must be at least 60 seconds")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / "watcher.log"
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("candidate checkout must be clean before staging source")
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
        try:
            bundle_hash = stage_source("g1.pkuml.org", source_commit, args.ssh_timeout_seconds)
            log(stream, f"STAGED source_root={REMOTE_RUN_ROOT}/source bundle_sha256={bundle_hash} bytes=see local staging log")
        except Exception as exc:
            log(stream, f"SOURCE_STAGE_FAILED {str(exc)!r}; no GPU job was launched")
            return 3
        while time.monotonic() < timeout_at:
            free_now: set[tuple[str, str]] = set()
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
                        run_deadline = min(timeout_at, time.monotonic() + 6 * 3600)
                        while time.monotonic() < run_deadline:
                            time.sleep(args.poll_seconds)
                            try:
                                status = read_remote_status(host, args.ssh_timeout_seconds)
                            except Exception as status_exc:
                                log(stream, f"STATUS_POLL_ERROR {str(status_exc)!r}; stopping local polling and preserving remote job/output")
                                return 4
                            if status is not None:
                                log(stream, f"REMOTE_JOB_FINISHED {json.dumps(status, separators=(',', ':'))}")
                                try:
                                    archive_results(host, source_commit, status, stream, max(90.0, args.ssh_timeout_seconds * 4))
                                except Exception as archive_exc:
                                    log(stream, f"ARTIFACT_ARCHIVE_FAILED {str(archive_exc)!r}; remote files preserved at {REMOTE_RUN_ROOT}")
                                    return 5
                                return int(status.get("exit_code", 1))
                        log(stream, f"REMOTE_JOB_STILL_RUNNING status_path={REMOTE_RUN_ROOT}/logs/status.json; remote tmux job continues")
                        return 6
                except Exception as exc:
                    log(stream, f"DEPLOY_ERROR host={host} uuid={uuid} detail={str(exc)!r}; no automatic redeploy after an uncertain launch")
                    return 7
            prior_free = free_now
            remaining = timeout_at - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(args.poll_seconds, remaining))
        log(stream, f"STOP reason=timeout_or_hosts_disabled failed_hosts={sorted(failed_hosts)}")
    return 2 if len(failed_hosts) == len(HOSTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
