#!/usr/bin/env python3
"""Serially download the preregistered MVSEC HDF5 inputs from the official Drive folder.

This prepares real files and hashes only. It never loads a model or computes a score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gdown
import h5py
import numpy as np

OFFICIAL_PAGE = "https://daniilidis-group.github.io/mvsec/download/#hdf5-files"
SOURCE_FOLDER = "https://drive.google.com/open?id=1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV"

FILES = {
    "indoor_flying1_data.hdf5": ("indoor_flying", "18lISHaEtIHxTETuuNHBRHPTtsVhq29l_", "data"),
    "indoor_flying1_gt.hdf5": ("indoor_flying", "1U9NChDGHdBGVSZrQJeQLSG0_i_z3NquN", "gt"),
    "indoor_flying2_data.hdf5": ("indoor_flying", "1BD7cVNUdDgvqTYPdzxy9Tcyl1FrjhrjF", "data"),
    "indoor_flying2_gt.hdf5": ("indoor_flying", "1zKJS-2Ayn2FitjRw40x_qVYLQDmh-C4u", "gt"),
    "indoor_flying3_data.hdf5": ("indoor_flying", "1Q7Mm_oZrSI_cTRktY8LcUaLSrGUmAERJ", "data"),
    "indoor_flying3_gt.hdf5": ("indoor_flying", "1DnT5SOOuKP2xuM3clMd_L54qnRG-JOI6", "gt"),
    "outdoor_day1_data.hdf5": ("outdoor_day", "1JLIrw2L24zIQBmqaWvef7G2t9tsMY3H0", "data"),
    "outdoor_day1_gt.hdf5": ("outdoor_day", "1wzUmTBxQ5wtSpB0KBogliB2IGTrCtJ7e", "gt"),
    "indoor_flying_calib.zip": ("indoor_flying", "1dZNctjh7i_Myq-wACiUuBPli9FR7d96U", "calibration"),
    "outdoor_day_calib.zip": ("outdoor_day", "1Y0sPP0ebX_cEKUCVVhJLej9TZgCxS3ME", "calibration"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def validate_hdf5(path: Path, kind: str) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        base = "davis/left"
        required = (
            ["events", "image_raw", "image_raw_ts", "image_raw_event_inds"]
            if kind == "data"
            else ["flow_dist", "flow_dist_ts"]
        )
        missing = [name for name in required if f"{base}/{name}" not in handle]
        if missing:
            raise ValueError(f"{path} lacks {kind} datasets: {missing}")
        info = {name: {"shape": list(handle[f"{base}/{name}"].shape),
                       "dtype": str(handle[f"{base}/{name}"].dtype)} for name in required}
        if kind == "data":
            if len(info["events"]["shape"]) != 2 or info["events"]["shape"][1] != 4:
                raise ValueError(f"unexpected raw event shape {info['events']['shape']}")
            if info["image_raw"]["shape"][1:] != [260, 346]:
                raise ValueError(f"unexpected APS shape {info['image_raw']['shape']}")
            aps_ts = np.asarray(handle[f"{base}/image_raw_ts"], dtype=np.float64)
            if not np.isfinite(aps_ts).all() or np.any(np.diff(aps_ts) <= 0):
                raise ValueError("APS timestamps are non-finite or not strictly increasing")
        else:
            flow_shape = info["flow_dist"]["shape"]
            if len(flow_shape) != 4 or flow_shape[1:] != [2, 260, 346]:
                raise ValueError(f"unexpected flow GT shape {flow_shape}")
            flow_ts = np.asarray(handle[f"{base}/flow_dist_ts"], dtype=np.float64)
            if len(flow_ts) not in (flow_shape[0], flow_shape[0] + 1):
                raise ValueError("flow timestamp count does not match maps/endpoints")
            if not np.isfinite(flow_ts).all() or np.any(np.diff(flow_ts) <= 0):
                raise ValueError("flow GT timestamps are non-finite or not strictly increasing")
    return info


def validate_archive(path: Path) -> list[str]:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"calibration download is not a valid ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"calibration ZIP has a corrupt member: {path}")
        return [name for name in archive.namelist() if not name.endswith("/")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="new Stage-5 MVSEC root; existing Stage-3 files are never overwritten")
    parser.add_argument("--sequences", nargs="+",
                        choices=["indoor_flying1", "indoor_flying2", "indoor_flying3", "outdoor_day1"],
                        default=["indoor_flying1", "indoor_flying2", "indoor_flying3", "outdoor_day1"])
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "public_download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("source_folder") != SOURCE_FOLDER:
            raise ValueError("existing manifest source folder differs; refusing to reuse it")
    else:
        manifest = {
            "schema": "mvsec_public_hdf5_download_manifest_v1",
            "status": "running", "created_utc": utc_now(),
            "official_page": OFFICIAL_PAGE, "source_folder": SOURCE_FOLDER,
            "real_scoring_enabled": False, "real_scores_computed": False,
            "requested_sequences": args.sequences, "files": {},
        }

    expected_names = []
    for sequence in args.sequences:
        for name in FILES:
            if name.startswith(sequence + "_"):
                expected_names.append(name)
    if not args.skip_calibration:
        expected_names.extend(["indoor_flying_calib.zip", "outdoor_day_calib.zip"])

    failures = []
    for name in expected_names:
        folder, file_id, kind = FILES[name]
        path = root / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = manifest["files"].get(name)
        if path.is_file() and existing and existing.get("sha256") == sha256(path):
            print(json.dumps({"event": "skip_verified", "name": name,
                              "size_bytes": path.stat().st_size, "sha256": existing["sha256"]}))
            continue
        if path.exists() and not existing:
            raise FileExistsError(f"unregistered existing file; refusing to overwrite: {path}")
        entry = {
            "name": name, "path": str(path), "drive_file_id": file_id,
            "source_folder": folder, "kind": kind, "status": "downloading",
            "started_utc": utc_now(),
        }
        manifest["files"][name] = entry
        manifest["status"] = "running"
        atomic_json(manifest_path, manifest)
        print(json.dumps({"event": "download_start", "name": name,
                          "drive_file_id": file_id, "output": str(path)}), flush=True)
        try:
            downloaded = gdown.download(
                id=file_id, output=str(path), quiet=False,
                resume=True, use_cookies=False, verify=True,
            )
            if downloaded is None or not path.is_file() or path.stat().st_size == 0:
                raise IOError("gdown did not produce a non-empty output file")
            validation = validate_hdf5(path, kind) if name.endswith((".h5", ".hdf5")) else validate_archive(path)
            entry.update({
                "status": "complete", "finished_utc": utc_now(),
                "size_bytes": path.stat().st_size, "sha256": sha256(path),
                "hdf5_datasets": validation,
            })
            print(json.dumps({"event": "download_complete", "name": name,
                              "size_bytes": entry["size_bytes"], "sha256": entry["sha256"],
                              "hdf5_datasets": validation}), flush=True)
        except Exception as exc:
            entry.update({"status": "failed", "finished_utc": utc_now(),
                          "error_type": type(exc).__name__, "error": str(exc)})
            failures.append(name)
            atomic_json(manifest_path, manifest)
            raise
        atomic_json(manifest_path, manifest)

    required = expected_names
    manifest["status"] = "complete" if all(
        manifest["files"].get(name, {}).get("status") == "complete" for name in required
    ) else "partial"
    manifest["finished_utc"] = utc_now()
    manifest["failed_files"] = failures
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path),
                      "required_hdf5_files": len(required), "real_scoring_enabled": False}, indent=2))


if __name__ == "__main__":
    main()
