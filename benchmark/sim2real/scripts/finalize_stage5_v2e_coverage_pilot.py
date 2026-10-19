#!/usr/bin/env python3
"""Validate and hash pinned V2E outputs for the isolated eight-rollout pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from stage4_v2e_artifacts import locate_shard_artifacts


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank-root", type=Path, required=True)
    p.add_argument("--v2e-root", type=Path, required=True)
    args = p.parse_args()
    bank, root = args.bank_root.resolve(), args.v2e_root.resolve()
    split = json.loads((bank / "registered_split.json").read_text())
    if len(split) != 8 or sum(row["split"] == "train" for row in split) != 4:
        raise RuntimeError("registered pilot split must contain 8 rollout rows with 4 train")
    rows = []
    for item in split:
        rollout = item["id"]
        path, status_path = locate_shard_artifacts(root, rollout)
        status = json.loads(status_path.read_text())
        if (status.get("rollout") != rollout or status.get("exit_code") != 0
                or Path(status.get("raw_h5", "")).resolve() != path.resolve()
                or status.get("raw_h5_sha256") != sha256(path)):
            raise RuntimeError(f"V2E execution/hash record failed for {rollout}")
        with h5py.File(path, "r") as h5:
            if "events" not in h5:
                raise RuntimeError(f"{path} lacks events dataset")
            events = np.asarray(h5["events"], dtype=np.uint32)
        if events.ndim != 2 or events.shape[1] != 4:
            raise RuntimeError(f"{rollout}: V2E raw events shape {events.shape}")
        if len(events):
            ts = events[:, 0].astype(np.int64)
            if np.any(np.diff(ts) < 0) or int(ts.min()) < 0 or int(ts.max()) > 2_000_000:
                raise RuntimeError(f"{rollout}: raw V2E timestamp order/range invalid")
            if int(events[:, 1].max()) >= 346 or int(events[:, 2].max()) >= 260:
                raise RuntimeError(f"{rollout}: V2E coordinates outside registered raster")
        input_files = sorted((bank / "v2e_input" / rollout).glob("*.png"))
        if len(input_files) != 201:
            raise RuntimeError(f"{rollout}: expected 201 V2E input endpoints, got {len(input_files)}")
        input_frames = [{"path": str(image), "size_bytes": image.stat().st_size, "sha256": sha256(image)}
                        for image in input_files]
        rows.append({
            "rollout_id": rollout,
            "split": item["split"],
            "stratum": item["stratum"],
            "coverage_factors": {key: item[key] for key in (
                "texture_pattern", "texture_contrast", "speed_class", "speed_scale",
                "amplitude_class", "amplitude_scale", "size_class", "size_range_m", "layout_class")},
            "raw_events_path": str(path),
            "raw_events_sha256": sha256(path),
            "raw_events_size_bytes": path.stat().st_size,
            "event_count": int(len(events)),
            "event_dtype": str(events.dtype),
            "event_shape": list(events.shape),
            "timestamp_unit": "microsecond integer in pinned V2E HDF5",
            "timestamp_resolution_s": 2.5e-5,
            "input_frames": len(input_files),
            "input_frame_manifest": input_frames,
            "status_record": str(status_path),
            "command": status["command"],
            "started_utc": status.get("started_utc"),
            "elapsed_s": status.get("elapsed_s"),
            "exit_code": status["exit_code"],
        })
    if len(rows) != 8 or sum(row["split"] == "train" for row in rows) != 4:
        raise RuntimeError("V2E paired pilot bank incomplete or split mismatch")
    manifest = {
        "schema": "stage5_v2e_coverage_pilot_raw_events_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bank_manifest_sha256": sha256(bank / "bank_manifest.json"),
        "config_sha256": sha256(bank / "stage5_coverage_pilot_config.json"),
        "config": {"timestamp_resolution_s": 2.5e-5, "input_rate_hz": 100,
                   "slomo": "SuperSloMo39.ckpt", "real_data_used": False,
                   "sensor_thresholds_changed_for_coverage": False},
        "episodes": rows,
    }
    out = root / "v2e_complete_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete", "episodes": len(rows), "manifest": str(out),
                      "sha256": sha256(out)}, indent=2))


if __name__ == "__main__":
    main()
