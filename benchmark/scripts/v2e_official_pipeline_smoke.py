#!/usr/bin/env python3
"""Run the official V2E image-folder path with SuperSloMo enabled.

This script intentionally refuses the direct EventEmulator shortcut.  It is a
readiness smoke only: it checks intermediate frames, output timestamps, and
the official numerical warning log, never a real-vs-sim metric.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _find_h5_events(path: Path):
    import h5py
    with h5py.File(path, "r") as handle:
        candidates = []
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and obj.shape[1] >= 4:
                candidates.append((name, obj.shape, obj.dtype))
        handle.visititems(visit)
        if not candidates:
            raise RuntimeError(f"no 2D event dataset found in {path}")
        name, shape, dtype = candidates[0]
        sample = handle[name][:]
    return name, shape, dtype, sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--python", dest="python_exe", default=sys.executable)
    parser.add_argument("--slomo-model", type=Path, default=None)
    args = parser.parse_args()

    source = args.source or Path(os.environ.get("V2E_SOURCE", "/home/科研/Eventbased_WAM/code/v2e_source"))
    v2e = source / "v2e.py"
    model = args.slomo_model or Path(os.environ.get("V2E_SLOMO_MODEL", ""))
    if not v2e.is_file():
        return _stop("V2E source/v2e.py not found", source=str(source))
    if not model.is_file():
        return _stop(
            "official SuperSloMo checkpoint is missing; direct EventEmulator is forbidden",
            expected_model=str(model),
        )
    frames = sorted(args.frames.glob("*.png"))
    if len(frames) < 3:
        return _stop("same-Genesis replay did not produce at least 3 intermediate frames", frame_count=len(frames))

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    event_file = output / "events.h5"
    log_file = output / "official_v2e.log"
    cmd = [
        args.python_exe, str(v2e),
        "--input", str(args.frames),
        "--input_frame_rate", "100",
        "--auto_timestamp_resolution", "False",
        "--timestamp_resolution", "0.0001",
        "--cutoff_hz", "300",
        "--slomo_model", str(model),
        "--output_folder", str(output),
        "--dvs_h5", event_file.name,
        "--dvs_text", "None",
        "--skip_video_output",
        "--no_preview",
        "--overwrite",
    ]
    completed = subprocess.run(cmd, cwd=source, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_file.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        return _stop("official V2E pipeline exited non-zero", returncode=completed.returncode, log=str(log_file))
    if not event_file.is_file():
        return _stop("official V2E pipeline produced no HDF5 event output", log=str(log_file))

    dataset, shape, dtype, sample = _find_h5_events(event_file)
    if sample.shape[0] == 0:
        return _stop("official V2E pipeline produced an empty event stream", log=str(log_file))
    # V2E output is conventionally [t,x,y,p].  We inspect its time span only;
    # no distributional or quality metric is computed in this phase.
    timestamps = np.asarray(sample[:, 0], dtype=np.float64)
    duration = float(timestamps.max() - timestamps.min())
    forbidden = re.compile(r"(undersampl|cutoff.*warning|warning.*300|disable_slomo|rescal.*timestamp)", re.I)
    forbidden_lines = [line for line in completed.stdout.splitlines() if forbidden.search(line)]
    if forbidden_lines:
        return _stop("official V2E log contains a forbidden cadence/timestamp warning", lines=forbidden_lines[:10])
    result = {
        "status": "PASS",
        "formal_metrics_executed": False,
        "official_pipeline": True,
        "super_slomo_enabled": True,
        "source_frame_count": len(frames),
        "effective_timestamp_resolution_s": 0.0001,
        "dataset": dataset,
        "event_shape": list(shape),
        "dtype": str(dtype),
        "output_duration_s": duration,
        "forbidden_warning_lines": [],
        "config": str(ROOT / "benchmark/configs/v2e.yaml"),
        "seed": 42,
    }
    (output / "v2e_pipeline_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def _stop(message: str, **fields) -> int:
    result = {"status": "STOP", "formal_metrics_executed": False, "error": message, **fields}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
