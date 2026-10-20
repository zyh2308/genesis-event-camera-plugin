#!/usr/bin/env python3
"""Inventory registered MVSEC dt1 real inputs without computing flow scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py

SEQUENCES = ("indoor_flying1", "indoor_flying2", "indoor_flying3", "outdoor_day1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def h5_index(path: Path) -> dict:
    datasets = {}
    with h5py.File(path, "r") as handle:
        attrs = {key: str(value) for key, value in handle.attrs.items()}
        def visit(name, item):
            if isinstance(item, h5py.Dataset):
                datasets[name] = {"shape": list(item.shape), "dtype": str(item.dtype)}
        handle.visititems(visit)
    return {"root_attributes": attrs, "datasets": datasets}


def find_one(root: Path, filename: str) -> Path | None:
    found = sorted(root.rglob(filename))
    if len(found) > 1:
        raise RuntimeError(f"ambiguous MVSEC file {filename}: {found}")
    return found[0] if found else None


def find_registered_fragment_pairs(root: Path, sequence: str) -> list[dict]:
    pairs = []
    for data_path in sorted(root.rglob(f"{sequence}_registered_*_data.hdf5")):
        gt_path = data_path.with_name(data_path.name[:-len("_data.hdf5")] + "_gt.hdf5")
        pair = {
            "fragment_id": data_path.name[:-len("_data.hdf5")],
            "data": describe(data_path),
            "ground_truth": describe(gt_path if gt_path.is_file() else None),
            "status": "available_unscored" if gt_path.is_file() else "data_without_matching_gt",
        }
        pairs.append(pair)
    return pairs


def describe(path: Path | None) -> dict:
    if path is None:
        return {"exists": False}
    row = {"exists": True, "path": str(path.resolve()), "size_bytes": path.stat().st_size,
           "sha256": sha256(path)}
    if path.suffix.lower() in (".h5", ".hdf5"):
        row["hdf5_index"] = h5_index(path)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    rows = []
    for sequence in SEQUENCES:
        data = find_one(root, f"{sequence}_data.hdf5")
        gt = find_one(root, f"{sequence}_gt.hdf5")
        fragments = find_registered_fragment_pairs(root, sequence)
        full_pair_available = bool(data and gt)
        fragments_available = sum(row["status"] == "available_unscored" for row in fragments)
        rows.append({"sequence": sequence, "data": describe(data), "ground_truth": describe(gt),
                     "registered_fragment_pairs": fragments,
                     "status": "available_unscored" if full_pair_available else (
                         "registered_fragment_only" if fragments_available else "incomplete")})
    calibration = sorted(root.rglob("*calib*.zip"))
    calib_rows = [{"path": str(p.resolve()), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in calibration]
    available = sum(row["status"] == "available_unscored" for row in rows)
    fragment_count = sum(
        1 for row in rows for fragment in row["registered_fragment_pairs"]
        if fragment["status"] == "available_unscored"
    )
    result = {
        "schema": "mvsec_dt1_file_manifest_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://daniilidis-group.github.io/mvsec/",
        "candidate_protocol": "benchmark/sim2real/protocol/mvsec_dt1_protocol_candidate_20261019.json",
        "registered_sequence_count": len(SEQUENCES), "available_sequence_count": available,
        "available_registered_fragment_pair_count": fragment_count,
        "calibration_files": calib_rows,
        "real_scores_computed": False, "real_scoring_enabled": False,
        "status": "complete_unscored_inventory" if available == len(SEQUENCES) else (
            "registered_fragments_only" if available == 0 and fragment_count else "partial_inventory"),
        "sequences": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "available": available, "expected": len(SEQUENCES),
                      "missing": [row["sequence"] for row in rows if row["status"] != "available_unscored"],
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
