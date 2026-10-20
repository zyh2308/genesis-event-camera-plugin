#!/usr/bin/env python3
"""Export the registered eight-rollout Stage-5 pilot for STE diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py

from prepare_stage4_steflow_bank import convert_rollout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--v2e-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    bank, v2e, output = args.bank_root.resolve(), args.v2e_root.resolve(), args.output_root.resolve()
    bank_path = bank / "shared_trajectory.h5"
    bank_manifest_path = bank / "bank_manifest.json"
    split_path = bank / "registered_split.json"
    v2e_manifest_path = v2e / "v2e_complete_manifest.json"
    if not all(path.is_file() for path in (bank_path, bank_manifest_path, split_path, v2e_manifest_path)):
        raise FileNotFoundError("complete shared bank, split and V2E manifest are required")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic data root: {output}")

    bank_manifest = json.loads(bank_manifest_path.read_text())
    split = json.loads(split_path.read_text())
    v2e_manifest = json.loads(v2e_manifest_path.read_text())
    if (bank_manifest.get("schema") != "stage5_flow_scene_coverage_shared_bank_v3"
            or bank_manifest.get("scope") != "complete_balanced_8_rollout_pilot"
            or bank_manifest.get("complete_balanced_pilot") is not True
            or len(split) != 8 or len(bank_manifest.get("episodes", [])) != 8):
        raise RuntimeError("input is not the registered complete v3 eight-rollout bank")
    if (v2e_manifest.get("schema") != "stage5_v2e_coverage_pilot_raw_events_v3"
            or v2e_manifest.get("complete_balanced_pilot") is not True
            or len(v2e_manifest.get("episodes", [])) != 8):
        raise RuntimeError("input is not the registered complete v3 paired V2E artifact")
    if sha256(bank_path) != bank_manifest.get("bank_sha256"):
        raise RuntimeError("physical bank HDF5 hash differs from its manifest")
    if v2e_manifest.get("bank_manifest_sha256") != sha256(bank_manifest_path):
        raise RuntimeError("V2E events were not generated from this exact shared bank manifest")
    expected_ids = {row["id"] for row in split}
    if ({row["id"] for row in bank_manifest["episodes"]} != expected_ids
            or {row["rollout_id"] for row in v2e_manifest["episodes"]} != expected_ids):
        raise RuntimeError("physical bank, split and paired V2E episode IDs differ")
    if sum(row["split"] == "train" for row in split) != 4 or sum(row["split"] == "validation" for row in split) != 4:
        raise RuntimeError("the registered physical-rollout split must be 4 train / 4 validation")

    v2e_rows = {row["rollout_id"]: row for row in v2e_manifest["episodes"]}
    output.mkdir(parents=True)
    entries = []
    with h5py.File(bank_path, "r") as shared:
        if set(shared["episodes"].keys()) != expected_ids:
            raise RuntimeError("HDF5 episode groups differ from the frozen split")
        for generator in ("eventgenesis", "v2e"):
            for item in sorted(split, key=lambda row: row["id"]):
                rollout = item["id"]
                destination = output / generator / item["split"] / f"{rollout}.npz"
                entries.append(convert_rollout(generator, rollout, shared, v2e, destination, v2e_rows))

    by_pair = {(row["generator"], row["rollout_id"]): row for row in entries}
    for rollout in sorted(expected_ids):
        eg = by_pair[("eventgenesis", rollout)]
        v2 = by_pair[("v2e", rollout)]
        if eg["label_sha256"] != v2["label_sha256"]:
            raise RuntimeError(f"shared flow GT/validity changed between generators for {rollout}")
    for generator in ("eventgenesis", "v2e"):
        train_windows = sum(row["samples"] for row in entries if row["generator"] == generator and row["split"] == "train")
        validation_windows = sum(row["samples"] for row in entries if row["generator"] == generator and row["split"] == "validation")
        if (train_windows, validation_windows) != (264, 264):
            raise RuntimeError(f"{generator}: expected 264 train / 264 validation windows, got {train_windows}/{validation_windows}")

    manifest = {
        "schema": "stage5_steflow_diagnostic_bank_v1",
        "status": "paired_synthetic_diagnostic_data_ready_no_training_result",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "real_data_used": False,
        "real_scoring_enabled": False,
        "checkpoint_selection_allowed": False,
        "flow_gt_used_for_self_supervised_optimizer": False,
        "shared_physical_bank": {"path": str(bank_path), "size_bytes": bank_path.stat().st_size,
                                 "sha256": sha256(bank_path), "manifest_sha256": sha256(bank_manifest_path)},
        "registered_split": {"path": str(split_path), "sha256": sha256(split_path),
                             "split_unit": "independent full physical rollout", "train_rollouts": 4,
                             "validation_rollouts": 4},
        "v2e_manifest": {"path": str(v2e_manifest_path), "sha256": sha256(v2e_manifest_path)},
        "event_representation": "pinned Stage-4 five temporal bins per polarity; same gaussian_count_representation encoder",
        "window_rule": "66 non-overlapping 30 ms windows per two-second rollout, (t_i,t_i+3]",
        "flow_unit": "forward source-pixel displacement per 30 ms endpoint interval",
        "paired_labels_identical": True,
        "train_windows_per_generator": 264,
        "validation_windows_per_generator": 264,
        "event_count_filtering": "none",
        "entries": entries,
    }
    path = output / "training_bank_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_sha256"] = sha256(path)
    (output / "training_bank_manifest.sha256").write_text(manifest["manifest_sha256"] + "  training_bank_manifest.json\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "entries"}, indent=2))


if __name__ == "__main__":
    main()
