#!/usr/bin/env python3
"""Verify paired Stage-5 receipts and write an immutable training gate manifest."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmark.sim2real.compat.e2vid_temporal_consistency import (
    PINNED_UPSTREAM_SHA256,
    validate_upstream,
)
from benchmark.sim2real.e2vid_stage5_gate import (
    FROZEN_CORE,
    canonical_sha256,
    collect_data_files,
    sha256_file,
    verify_model_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--joint-augmentation-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    model_root, data_root = args.model_root.resolve(), args.data_root.resolve()
    config_path, preflight_root = args.config.resolve(), args.preflight_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite registered run manifest: {output}")
    if not config_path.is_file() or not (data_root / "adapter_manifest.json").is_file():
        raise FileNotFoundError("registered config or E2VID data manifest is missing")
    trainer_path = repo / "benchmark/sim2real/scripts/train_e2vid_stage5.py"
    compat_path = repo / "benchmark/sim2real/compat/e2vid_temporal_consistency.py"
    geometry_path = repo / "benchmark/sim2real/scripts/audit_e2vid_tc_geometry.py"
    gate_module_path = repo / "benchmark/sim2real/e2vid_stage5_gate.py"
    source_bundle_path = repo / "benchmark/sim2real/manifests/downstream_source_bundle.json"
    for path in (trainer_path, compat_path, geometry_path, gate_module_path, source_bundle_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    model_bundle = verify_model_bundle(model_root, source_bundle_path)
    tc_upstream = validate_upstream(model_root)
    data_manifest_path = data_root / "adapter_manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text())
    if data_manifest.get("frozen_simulator_core") != FROZEN_CORE:
        raise RuntimeError("shared data bank does not match the frozen simulator core")
    if data_manifest.get("real_data_used") is not False or data_manifest.get("real_scoring_enabled") is not False:
        raise RuntimeError("real-data isolation flag is not closed in the data manifest")
    data_files = collect_data_files(data_root, data_manifest, verify=True)

    fingerprints = {
        "trainer_script_sha256": sha256_file(trainer_path),
        "compat_source_sha256": sha256_file(compat_path),
        "geometry_script_sha256": sha256_file(geometry_path),
        "gate_module_sha256": sha256_file(gate_module_path),
        "upstream_loss_sha256": PINNED_UPSTREAM_SHA256,
        "config_sha256": sha256_file(config_path),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "source_bundle_manifest_sha256": sha256_file(source_bundle_path),
        "model_source_tree_sha256": model_bundle["included_tree_sha256"],
    }
    if tc_upstream["sha256"] != fingerprints["upstream_loss_sha256"]:
        raise RuntimeError("registered upstream loss hash unexpectedly changed")

    receipts = {}
    for arm in ("eventgenesis", "v2e"):
        path = preflight_root / arm / "preflight_receipt.json"
        if not path.is_file():
            raise FileNotFoundError(f"required preflight receipt is missing: {path}")
        receipt = json.loads(path.read_text())
        if receipt.get("status") != "pass" or receipt.get("generator") != arm or receipt.get("seed") != 0:
            raise RuntimeError(f"invalid preflight receipt for {arm}: status/generator/seed")
        if receipt.get("fingerprints") != fingerprints:
            raise RuntimeError(f"preflight fingerprint is stale for {arm}")
        if receipt.get("geometry_contract", {}).get("status") != "pass":
            raise RuntimeError(f"geometry contract did not pass for {arm}")
        if receipt.get("real_model_tc_probe", {}).get("status") != "pass":
            raise RuntimeError(f"real model TC backprop did not pass for {arm}")
        if receipt.get("recurrent_state", {}).get("status") != "pass":
            raise RuntimeError(f"recurrent-state fixture did not pass for {arm}")
        if receipt.get("loader", {}).get("status") != "pass":
            raise RuntimeError(f"loader contract did not pass for {arm}")
        actual_arm_data_hash = canonical_sha256(data_files[arm])
        if receipt.get("data_files_sha256") != actual_arm_data_hash:
            raise RuntimeError(f"preflight data files are stale or incomplete for {arm}")
        receipts[arm] = {
            "path": str(path), "sha256": sha256_file(path),
            "initial_model_sha256": receipt["initial_model_sha256"],
            "data_files_sha256": receipt["data_files_sha256"],
        }
    initial_hashes = {row["initial_model_sha256"] for row in receipts.values()}
    if len(initial_hashes) != 1:
        raise RuntimeError(f"paired initial model states differ: {sorted(initial_hashes)}")

    joint_path = args.joint_augmentation_receipt.resolve()
    joint = json.loads(joint_path.read_text())
    if joint.get("status") != "pass" or joint.get("trainer_script_sha256") != fingerprints["trainer_script_sha256"]:
        raise RuntimeError("joint augmentation receipt missing, failed, or stale")
    if not all(all(checks.values()) for row in joint.get("results", []) for checks in [row["checks"]]):
        raise RuntimeError("joint augmentation receipt contains a failed transform check")

    manifest = {
        "schema": "e2vid_stage5_paired_training_gate_v1",
        "group_id": "e2vid-tc-aligncorners-true-seed0-20261019",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preflight_pass_training_authorized",
        "frozen_simulator_core": FROZEN_CORE,
        "real_data_used": False,
        "real_scoring_enabled": False,
        "generator_comparison_variable": "event generator only",
        "paired_initial_model_sha256": next(iter(initial_hashes)),
        "seed": 0,
        "optimizer_budget_per_arm": 12000,
        "training_config_path": str(config_path),
        "model_root": str(model_root),
        "model_source_bundle": model_bundle,
        "data_root": str(data_root),
        "data_manifest_path": str(data_manifest_path),
        "source_bank": {
            "path": data_manifest["source_bank_path"],
            "size_bytes": data_manifest["source_bank_size_bytes"],
            "sha256": data_manifest["source_bank_sha256"],
        },
        "data_files": data_files,
        "fingerprints": fingerprints,
        "preflight_receipts": receipts,
        "joint_augmentation_receipt": {
            "path": str(joint_path), "sha256": sha256_file(joint_path),
            "seed_count": joint["seed_count"],
        },
        "tc_geometry_checks_required": [
            "zero_flow_identity", "zero_flow_tc_loss", "known_integer_translation_and_sign",
            "subpixel_displacement", "valid_boundary_last_in_bounds_pixel",
            "invalid_boundary_zero_padding", "independent_sampling_oracle",
        ],
        "training_recipe": "existing corrected-v2 reconstruction recipe; fresh shared random initialization; 40-frame clips; synthetic LPIPS-VGG validation selection",
        "output_root": "/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-e2vid-tc-aligntrue-seed0",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": manifest["status"], "manifest_path": str(output),
                      "paired_initial_model_sha256": manifest["paired_initial_model_sha256"],
                      "manifest_sha256": sha256_file(output),
                      "data_files_per_generator": {key: len(value) for key, value in data_files.items()},
                      "fingerprints": fingerprints}, indent=2))


if __name__ == "__main__":
    main()
