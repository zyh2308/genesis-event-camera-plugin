#!/usr/bin/env python3
"""Hash and validate available HQF bags and EVREAL MemMap sequences."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark/sim2real/scripts"))
from infer_hqf_e2vid_candidate import EXPECTED_DATA_FILES, validate_sequence_files

SEQUENCES = (
    "bike_bay_hdr", "boxes", "desk", "desk_fast", "desk_hand_only", "desk_slow",
    "engineering_posters", "high_texture_plants", "poster_pillar_1", "poster_pillar_2",
    "reflective_materials", "slow_and_fast_desk", "slow_hand", "still_life",
)
MEMMAP_FILES = (*EXPECTED_DATA_FILES, "conversion_manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_row(name: str, bag_root: Path, memmap_root: Path) -> dict:
    bag = bag_root / f"{name}.bag"
    sequence_dir = memmap_root / name
    row = {
        "sequence": name,
        "raw_bag": {"path": str(bag), "exists": bag.is_file()},
        "evreal_memmap": {"path": str(sequence_dir), "exists": sequence_dir.is_dir()},
        "status": "missing",
    }
    if bag.is_file():
        row["raw_bag"].update({"size_bytes": bag.stat().st_size, "sha256": sha256(bag)})
    if sequence_dir.is_dir():
        file_rows, missing = {}, []
        for filename in MEMMAP_FILES:
            path = sequence_dir / filename
            if not path.is_file():
                missing.append(filename)
            else:
                file_rows[filename] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        row["evreal_memmap"]["files"] = file_rows
        row["evreal_memmap"]["missing_files"] = missing
        if not missing:
            info = validate_sequence_files(sequence_dir)
            arrays = {
                "events_ts": np.load(sequence_dir / "events_ts.npy", mmap_mode="r").squeeze(),
                "images_ts": np.load(sequence_dir / "images_ts.npy", mmap_mode="r").squeeze(),
                "image_event_indices": np.load(sequence_dir / "image_event_indices.npy", mmap_mode="r").squeeze(),
            }
            indices = arrays["image_event_indices"].astype(np.int64)
            if np.any(indices >= len(arrays["events_ts"])):
                raise RuntimeError(f"{name}: image_event_indices points outside event timestamps")
            if np.any(arrays["events_ts"][indices] > arrays["images_ts"] + 1e-12):
                raise RuntimeError(f"{name}: an image endpoint maps to a future event")
            conversion = json.loads((sequence_dir / "conversion_manifest.json").read_text())
            if bag.is_file() and conversion.get("source_bag", {}).get("bag_sha256") != row["raw_bag"].get("sha256"):
                raise RuntimeError(f"{name}: MemMap conversion was built from a different raw bag hash")
            row["evreal_memmap"].update({
                "schema_checks": info,
                "event_image_endpoint_association": "image_event_indices selects latest event <= each image timestamp; next event, if present, is strictly later",
                "reference_images_are_evaluation_targets_only": True,
            })
            row["status"] = "ready_unscored"
        else:
            row["status"] = "memmap_incomplete"
    elif bag.is_file():
        row["status"] = "raw_bag_only"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-root", type=Path, required=True)
    parser.add_argument("--memmap-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows = [sequence_row(name, args.bag_root.resolve(), args.memmap_root.resolve()) for name in SEQUENCES]
    ready = sum(row["status"] == "ready_unscored" for row in rows)
    result = {
        "schema": "hqf_14_file_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_source": "https://timostoff.github.io/20ecnn",
        "official_download_folder": "https://drive.google.com/drive/folders/18Xdr6pxJX0ZXTrXW9tK0hC3ZpmKDIt6_",
        "locked_evreal_commit": "398f03551b7f5d150ae10aa89e4e731557fce240",
        "expected_sequence_count": 14,
        "ready_unscored_sequence_count": ready,
        "real_scoring_enabled": False,
        "real_gt_metrics_computed": False,
        "sequences": rows,
        "status": "ready_unscored" if ready == len(SEQUENCES) else "partial_inventory",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "expected": len(SEQUENCES), "ready": ready,
                      "missing": [row["sequence"] for row in rows if row["status"] != "ready_unscored"],
                      "output": str(args.output.resolve())}, indent=2))
    if args.require_complete and result["status"] != "ready_unscored":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
