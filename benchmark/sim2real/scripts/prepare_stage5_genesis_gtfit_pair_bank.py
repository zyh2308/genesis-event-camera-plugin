#!/usr/bin/env python3
"""Prepare only the preregistered train window and independent validation rollout.

This narrow adapter is for the Stage-5 fixed-sample STE network-fit diagnostic.
It does not replace the eight-rollout pilot exporter and never filters samples by
generator event count or model output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py

from prepare_stage4_steflow_bank import convert_rollout
from stage4_v2e_artifacts import locate_shard_artifacts


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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    bank, v2e = args.bank_root.resolve(), args.v2e_root.resolve()
    config_path, output = args.config.resolve(), args.output_root.resolve()
    bank_h5 = bank / "shared_trajectory.h5"
    bank_manifest_path = bank / "bank_manifest.json"
    split_path = bank / "registered_split.json"
    if not all(path.is_file() for path in (bank_h5, bank_manifest_path, split_path)):
        raise FileNotFoundError("shared bank HDF5, manifest and registered split are required")
    config = json.loads(config_path.read_text())
    if config.get("schema") != "stage5_genesis_high_texture_fixed_sample_gt_fit_v1":
        raise RuntimeError("unexpected fixed-sample GT-fit config schema")
    bank_manifest = json.loads(bank_manifest_path.read_text())
    if (bank_manifest.get("schema") != "stage5_flow_scene_coverage_shared_bank_v3"
            or bank_manifest.get("scope") != "complete_balanced_8_rollout_pilot"
            or bank_manifest.get("complete_balanced_pilot") is not True):
        raise RuntimeError("input must be the complete registered shared v3 eight-rollout bank")
    expected_bank = config["shared_bank"]
    if sha256(bank_h5) != expected_bank["sha256"] or sha256(bank_h5) != bank_manifest.get("bank_sha256"):
        raise RuntimeError("shared physical bank hash differs from the frozen config/manifest")
    split = json.loads(split_path.read_text())
    split_rows = {row["id"]: row for row in split}
    train_id = config["sample"]["rollout_id"]
    val_id = config["independent_validation"]["rollout_id"]
    if train_id == val_id or train_id not in split_rows or val_id not in split_rows:
        raise RuntimeError("fixed train and independent validation rollouts must be distinct registered IDs")
    if split_rows[train_id].get("split") != "train" or split_rows[val_id].get("split") != "validation":
        raise RuntimeError("fixed train/validation rollout roles disagree with the preregistered split")
    bank_rows = {row["id"]: row for row in bank_manifest.get("episodes", [])}
    if set(split_rows) != set(bank_rows):
        raise RuntimeError("shared bank episode IDs differ from the registered split")
    for rollout in (train_id, val_id):
        if bank_rows[rollout].get("frames") != 201:
            raise RuntimeError(f"{rollout}: expected 201 shared trajectory frames")
        input_files = sorted((bank / "v2e_input" / rollout).glob("*.png"))
        if len(input_files) != 201:
            raise RuntimeError(f"{rollout}: expected 201 immutable V2E input frames, got {len(input_files)}")

    v2e_rows: dict[str, dict] = {}
    for rollout in (train_id, val_id):
        raw_path, status_path = locate_shard_artifacts(v2e, rollout)
        status = json.loads(status_path.read_text())
        raw_hash = sha256(raw_path)
        if (status.get("rollout") != rollout or status.get("exit_code") != 0
                or Path(status.get("raw_h5", "")).resolve() != raw_path.resolve()
                or status.get("raw_h5_sha256") != raw_hash):
            raise RuntimeError(f"{rollout}: V2E output lacks a matching successful hash-bound run record")
        v2e_rows[rollout] = {
            "rollout_id": rollout, "raw_events_path": str(raw_path),
            "raw_events_sha256": raw_hash, "status_record": str(status_path),
            "exit_code": status["exit_code"], "command": status.get("command"),
            "started_utc": status.get("started_utc"), "elapsed_s": status.get("elapsed_s"),
        }

    if output.exists():
        raise FileExistsError(f"refusing to overwrite pair adapter directory: {output}")
    output.mkdir(parents=True)
    entries = []
    with h5py.File(bank_h5, "r") as shared:
        ids = set(shared["episodes"].keys())
        if ids != set(split_rows):
            raise RuntimeError("HDF5 episode IDs differ from registered split")
        for rollout in (train_id, val_id):
            group = shared[f"episodes/{rollout}"]
            if str(group.attrs["split"]) != split_rows[rollout]["split"]:
                raise RuntimeError(f"{rollout}: HDF5 split disagrees with registered split")
            for generator in ("eventgenesis", "v2e"):
                target = output / generator / split_rows[rollout]["split"] / f"{rollout}.npz"
                entries.append(convert_rollout(generator, rollout, shared, v2e, target, v2e_rows))

    by_key = {(row["generator"], row["rollout_id"]): row for row in entries}
    for rollout in (train_id, val_id):
        eg, v2 = by_key[("eventgenesis", rollout)], by_key[("v2e", rollout)]
        if eg["label_sha256"] != v2["label_sha256"]:
            raise RuntimeError(f"{rollout}: paired physical labels differ between generators")
        if eg["samples"] != 66 or v2["samples"] != 66:
            raise RuntimeError(f"{rollout}: expected all 66 registered 30 ms windows per arm")
    train_artifact = Path(by_key[("eventgenesis", train_id)]["path"])
    import numpy as np
    with np.load(train_artifact, allow_pickle=False) as archive:
        expected_sample = config["sample"]["sample_id"]
        if str(archive["sample_ids"][int(config["sample"]["window_index"])]) != expected_sample:
            raise RuntimeError("adapted fixed training window ID differs from the preregistered sample")
    for generator in ("eventgenesis", "v2e"):
        val_rows = [row for row in entries if row["generator"] == generator and row["rollout_id"] == val_id]
        if len(val_rows) != 1 or val_rows[0]["split"] != "validation":
            raise RuntimeError(f"{generator}: independent validation rollout is not isolated")

    manifest = {
        "schema": "stage5_genesis_gtfit_fixed_pair_bank_v1",
        "status": "paired_train_sample_and_independent_validation_ready",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "only the preregistered fixed train rollout and independent validation rollout",
        "config_path": str(config_path), "config_sha256": sha256(config_path),
        "real_data_used": False, "real_scoring_enabled": False,
        "checkpoint_selection_allowed": False,
        "flow_gt_used_for_optimizer": "yes, for the separately registered supervised fit diagnostic only",
        "event_only_model_input": True,
        "shared_physical_bank": {"path": str(bank_h5), "size_bytes": bank_h5.stat().st_size,
                                 "sha256": sha256(bank_h5),
                                 "manifest_path": str(bank_manifest_path),
                                 "manifest_sha256": sha256(bank_manifest_path),
                                 "split_sha256": sha256(split_path)},
        "v2e_artifacts": v2e_rows,
        "train_rollout": train_id, "fixed_sample_id": config["sample"]["sample_id"],
        "independent_validation_rollout": val_id,
        "window_rule": "66 non-overlapping 30 ms windows using the shared (t_i,t_i+3] timestamp rule",
        "flow_unit": "pixels per 30ms endpoint interval",
        "paired_labels_identical": True, "event_count_filtering": "none",
        "entries": entries,
    }
    manifest_path = output / "training_bank_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_hash = sha256(manifest_path)
    (output / "training_bank_manifest.sha256").write_text(manifest_hash + "  training_bank_manifest.json\n")
    print(json.dumps({"status": manifest["status"], "train_rollout": train_id,
                      "validation_rollout": val_id, "entries": len(entries),
                      "manifest": str(manifest_path), "sha256": manifest_hash}, indent=2))


if __name__ == "__main__":
    main()
