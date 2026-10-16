#!/usr/bin/env python3
"""Run pinned V2E on a disjoint list of immutable Stage-4 rollout frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--bank-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    args = p.parse_args()
    project, bank, out = args.project_root.resolve(), args.bank_root.resolve(), args.output_root.resolve()
    source = project / "external/sources/v2e-cac6384445/v2e-pinned"
    model = project / "external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt"
    entry = source / "v2e.py"
    out.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    started = datetime.now(timezone.utc).isoformat()
    records = []
    for rollout in args.episodes:
        inp = bank / "v2e_input" / rollout
        raw = out / f"{rollout}.h5"
        folder = out / rollout
        if not inp.is_dir() or len(list(inp.glob("*.png"))) != 201:
            raise RuntimeError(f"{rollout}: expected exactly 201 immutable input PNGs at {inp}")
        if raw.exists() or folder.exists():
            raise FileExistsError(f"refusing to overwrite prior V2E artifact for {rollout}")
        cmd = [
            sys.executable, str(entry), "--input", str(inp),
            "--input_frame_rate", "100", "--auto_timestamp_resolution", "False",
            "--timestamp_resolution", "0.000025", "--slomo_model", str(model),
            "--output_folder", str(folder), "--unique_output_folder", "False",
            "--skip_video_output", "--no_preview", "--dvs_h5", str(raw),
            "--dvs_aedat2", "None", "--dvs_aedat4", "None", "--dvs_text", "None",
            "--pos_thres", "0.2", "--neg_thres", "0.2", "--sigma_thres", "0.03",
            "--cutoff_hz", "300", "--leak_rate_hz", "0.01", "--leak_jitter_fraction", "0.1",
            "--noise_rate_cov_decades", "0.1", "--shot_noise_rate_hz", "0.001",
            "--refractory_period", "0.0005", "--dvs_emulator_seed", "42",
        ]
        log = out / f"{rollout}.stdout.log"
        err = out / f"{rollout}.stderr.log"
        t0 = time.monotonic()
        run_started = datetime.now(timezone.utc).isoformat()
        atomic_json(out / "shard_progress.json", {
            "gpu_physical_id": args.gpu, "started_utc": started,
            "completed_rollouts": len(records), "requested_rollouts": len(args.episodes),
            "active_rollout": rollout, "active_started_utc": run_started,
            "episodes": records, "status": "running", "updated_utc": run_started,
        })
        with log.open("w") as stdout, err.open("w") as stderr:
            result = subprocess.run(cmd, cwd=source, env=os.environ.copy(), stdout=stdout, stderr=stderr)
        row = {
            "rollout": rollout, "gpu_physical_id": args.gpu, "started_utc": run_started,
            "elapsed_s": round(time.monotonic() - t0, 3), "exit_code": result.returncode,
            "command": cmd, "input_frames": 201, "timestamp_resolution_s": 2.5e-5,
            "raw_h5": str(raw), "raw_h5_sha256": hashlib.sha256(raw.read_bytes()).hexdigest() if raw.exists() else None,
            "stdout_log": str(log), "stderr_log": str(err),
        }
        records.append(row)
        atomic_json(out / f"{rollout}.status.json", row)
        atomic_json(out / "shard_progress.json", {
            "gpu_physical_id": args.gpu, "started_utc": started, "completed_rollouts": len(records),
            "requested_rollouts": len(args.episodes), "episodes": records,
            "status": "failed" if result.returncode else "running",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        if result.returncode:
            raise SystemExit(result.returncode)
    atomic_json(out / "shard_progress.json", {
        "gpu_physical_id": args.gpu, "started_utc": started, "completed_rollouts": len(records),
        "requested_rollouts": len(args.episodes), "episodes": records, "status": "complete",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
