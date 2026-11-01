#!/usr/bin/env python3
"""Run pinned V2E on exact saved scene-linear frames from a motion pilot."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_registered_conversion(source: Path):
    tree = ast.parse(source.read_text())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "radiance_to_v2e_luma_u8"
    )
    module = ast.Module(body=[node], type_ignores=[])
    scope = {"np": np}
    exec(compile(module, str(source), "exec"), scope)
    return scope[node.name]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiance-frames", type=Path, required=True)
    parser.add_argument("--conversion-source", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default="",
                        help="CUDA device list inherited by V2E; default empty keeps the run CPU-only")
    args = parser.parse_args()
    radiance_path = args.radiance_frames.resolve()
    conversion_source = args.conversion_source.resolve()
    project = args.project_root.resolve()
    out = args.output_root.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    frames = np.load(radiance_path, allow_pickle=False)
    if frames.shape != (4, 240, 320, 3) or frames.dtype != np.float32 or not np.isfinite(frames).all():
        raise RuntimeError(f"expected finite float32 [4,240,320,3] radiance, got {frames.shape} {frames.dtype}")
    conversion = extract_registered_conversion(conversion_source)
    out.mkdir(parents=True)
    input_dir = out / "v2e_input"
    input_dir.mkdir()
    frame_rows = []
    for index, radiance in enumerate(frames):
        luma = conversion(radiance)
        if luma.shape != (240, 320) or luma.dtype != np.uint8:
            raise RuntimeError(f"registered luma adapter returned {luma.shape} {luma.dtype}")
        image_path = input_dir / f"{index:08d}.png"
        Image.fromarray(luma, mode="L").save(image_path)
        frame_rows.append({
            "index": index, "time_s": index / 100.0,
            "path": str(image_path), "size_bytes": image_path.stat().st_size,
            "sha256": sha256(image_path), "luma_min": int(luma.min()),
            "luma_max": int(luma.max()), "luma_mean": float(luma.mean()),
        })
    source = project / "external/sources/v2e-cac6384445/v2e-pinned"
    entry = source / "v2e.py"
    model = project / "external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt"
    h5_path = out / "v2e_events.h5"
    output_folder = out / "v2e_output"
    cmd = [
        sys.executable, str(entry), "--input", str(input_dir),
        "--input_frame_rate", "100", "--auto_timestamp_resolution", "False",
        "--timestamp_resolution", "0.000025", "--slomo_model", str(model),
        "--output_folder", str(output_folder), "--unique_output_folder", "False",
        "--skip_video_output", "--no_preview", "--dvs_h5", str(h5_path),
        "--dvs_aedat2", "None", "--dvs_aedat4", "None", "--dvs_text", "None",
        "--pos_thres", "0.2", "--neg_thres", "0.2", "--sigma_thres", "0.03",
        "--cutoff_hz", "300", "--leak_rate_hz", "0.01", "--leak_jitter_fraction", "0.1",
        "--noise_rate_cov_decades", "0.1", "--shot_noise_rate_hz", "0.001",
        "--refractory_period", "0.0005", "--dvs_emulator_seed", "42",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["PYTHONUNBUFFERED"] = "1"
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    (out / "command.json").write_text(json.dumps({"argv": cmd, "cwd": str(source), "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices, "threads": 2}, indent=2) + "\n")
    with (out / "v2e.stdout.log").open("w") as stdout, (out / "v2e.stderr.log").open("w") as stderr:
        proc = subprocess.run(cmd, cwd=source, env=env, stdout=stdout, stderr=stderr, check=False)
    ended = datetime.now(timezone.utc).isoformat()
    result = {
        "schema": "stage7_v2e_same_radiance_camera_motion_pilot_v1",
        "status": "failed_v2e" if proc.returncode else "v2e_exited_zero_pending_h5_validation",
        "started_utc": started, "finished_utc": ended,
        "elapsed_s": round(time.monotonic() - t0, 3), "exit_code": int(proc.returncode),
        "radiance_frames": {"path": str(radiance_path), "sha256": sha256(radiance_path), "shape": list(frames.shape), "dtype": str(frames.dtype)},
        "conversion": {"function": "radiance_to_v2e_luma_u8", "source": str(conversion_source), "source_sha256": sha256(conversion_source), "frames": frame_rows},
        "v2e_source": {"path": str(source), "pinned_commit": "cac638444557a224ba6ee5c29260916f152c00bd", "revision_basis": "the project-pinned extracted bundle is named v2e-cac6384445 and has no .git directory; full revision is recorded in the versioned protocol/source lock", "entry_sha256": sha256(entry), "slomo_sha256": sha256(model)},
        "protocol": {"input_frame_rate_hz": 100, "timestamp_resolution_s": 2.5e-5, "generator_sensor_parameters": "copied from existing Stage-5 V2E runner", "cuda_visible_devices": args.cuda_visible_devices, "device_policy": "V2E inherits CUDA_VISIBLE_DEVICES from this exact run argument", "real_data_used": False},
        "command": cmd,
    }
    validation_rc = int(proc.returncode)
    if proc.returncode == 0:
        if not h5_path.is_file():
            result["status"] = "failed_missing_h5"
            validation_rc = 1
        else:
            with h5py.File(h5_path, "r") as h5:
                if "events" not in h5:
                    result["status"] = "failed_missing_events_dataset"
                    validation_rc = 1
                else:
                    events = np.asarray(h5["events"], dtype=np.uint32)
            if validation_rc == 0 and (events.ndim != 2 or events.shape[1] != 4):
                result["status"] = "failed_event_shape"
                result["observed_event_shape"] = list(events.shape)
                validation_rc = 1
            if validation_rc == 0:
                if len(events):
                    ts = events[:, 0].astype(np.int64)
                    if np.any(np.diff(ts) < 0) or ts.min() < 0 or ts.max() > 30_000:
                        result["status"] = "failed_event_timestamps"
                        validation_rc = 1
                    if events[:, 1].max() >= 320 or events[:, 2].max() >= 240:
                        result["status"] = "failed_event_coordinates"
                        validation_rc = 1
                if validation_rc == 0:
                    result.update({
                        "status": "passed_pinned_v2e_same_frames",
                        "events": {"path": str(h5_path), "sha256": sha256(h5_path), "size_bytes": h5_path.stat().st_size, "shape": list(events.shape), "dtype": str(events.dtype), "timestamp_unit": "microseconds", "timestamp_range_us": ([int(events[:, 0].min()), int(events[:, 0].max())] if len(events) else None), "counts_by_10ms_interval": [int(((events[:, 0] > i * 10000) & (events[:, 0] <= (i + 1) * 10000)).sum()) for i in range(3)] if len(events) else [0, 0, 0]},
                    })
    result["validation_exit_code"] = validation_rc
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "exit_code": validation_rc, "elapsed_s": result["elapsed_s"], "events": result.get("events")}, indent=2), flush=True)
    (out / "exit_code.txt").write_text(f"{validation_rc}\n")
    return validation_rc


if __name__ == "__main__":
    raise SystemExit(main())
