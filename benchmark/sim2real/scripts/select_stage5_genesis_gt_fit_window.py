#!/usr/bin/env python3
"""Select a synthetic flow-fit window from physical labels and texture only.

The script deliberately never opens either generator's event dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    bank = args.bank_root.resolve()
    bank_path = bank / "shared_trajectory.h5"
    manifest_path = bank / "bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != config["shared_bank"]["schema"]:
        raise RuntimeError("shared-bank schema mismatch")
    if manifest.get("bank_sha256") != config["shared_bank"]["sha256"]:
        raise RuntimeError("bank SHA differs from the preregistered value")
    bank_sha = sha256(bank_path)
    if bank_sha != manifest["bank_sha256"]:
        raise RuntimeError("shared bank HDF5 hash mismatch")
    rollout = config["training_rollout"]["rollout_id"]
    split = json.loads((bank / "registered_split.json").read_text())
    row = next((item for item in split if item.get("id") == rollout), None)
    if row is None or row.get("split") != "train" or row.get("stratum") != config["training_rollout"]["stratum"]:
        raise RuntimeError("training rollout does not match the registered split and stratum")

    rows = []
    with h5py.File(bank_path, "r") as shared:
        group = shared[f"episodes/{rollout}"]
        times = np.asarray(group["frame_timestamps_s"], dtype=np.float64)
        flow = np.asarray(group["flow_forward_30ms_px"], dtype=np.float32)
        valid = np.asarray(group["flow_valid_30ms"], dtype=np.uint8).astype(bool)
        signal = np.asarray(group["linearized_display_rgb_proxy"], dtype=np.float32)
        if (times.shape != (201,) or flow.shape != (66, 2, 260, 346) or
                valid.shape != (66, 260, 346) or signal.shape != (201, 260, 346, 3)):
            raise RuntimeError("unexpected registered sample/label/image shapes")
        if not np.isfinite(flow).all() or not np.isfinite(signal).all():
            raise RuntimeError("non-finite labels or shared rendered images")
        gray = signal @ np.asarray([.2126, .7152, .0722], dtype=np.float32)
        for index in range(66):
            endpoint = index * 3
            gx = cv2.Sobel(gray[endpoint], cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(gray[endpoint], cv2.CV_32F, 0, 1, ksize=3) / 8.0
            texture = np.hypot(gx, gy) > (1.0 / 255.0)
            magnitude = np.linalg.norm(flow[index], axis=0)
            eligible = valid[index] & (magnitude >= 0.5) & texture
            rows.append({
                "window_index": index,
                "sample_id": f"{rollout}/dt1_{endpoint * 10:03d}",
                "endpoint_indices": [endpoint, endpoint + 3],
                "window_seconds": [float(times[endpoint]), float(times[endpoint + 3])],
                "shared_valid_pixels": int(valid[index].sum()),
                "gt_moving_pixels_ge_0_5": int((valid[index] & (magnitude >= 0.5)).sum()),
                "gt_moving_texture_score": int(eligible.sum()),
                "gt_magnitude_p50_p90_p99_px_per_30ms": np.quantile(
                    magnitude[valid[index]], [0.5, 0.9, 0.99]).astype(float).tolist(),
            })
    selected = min(rows, key=lambda item: (-item["gt_moving_texture_score"], item["window_index"]))
    result = {
        "schema": config["schema"],
        "status": "selected_by_preregistered_physical_signal_rule",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256(args.config.resolve()),
        "bank_path": str(bank_path),
        "bank_sha256": bank_sha,
        "selection_rule": config["selection_rule"],
        "evidence_contract": {
            "generator_event_dataset_opened": False,
            "generator_event_count_used": False,
            "model_loaded": False,
            "model_output_used": False,
            "real_data_used": False,
        },
        "training_rollout": rollout,
        "candidate_window_count": len(rows),
        "selected_training_sample": selected,
        "all_candidate_scores": rows,
        "independent_validation_rollout": config["independent_validation"]["rollout_id"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "selected_training_sample": selected,
                      "top_score": selected["gt_moving_texture_score"],
                      "candidate_window_count": len(rows), "bank_sha256": bank_sha,
                      "config_sha256": result["config_sha256"],
                      "generator_events_opened": False}, indent=2))


if __name__ == "__main__":
    main()
