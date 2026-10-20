#!/usr/bin/env python3
"""Create the immutable run-specific config after the paired pilot bank is complete."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    model = args.model_root.resolve()
    bank = args.bank_root.resolve()
    adapter = args.adapter_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite registered config: {output}")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    if dirty:
        raise RuntimeError("config must be generated from the same clean immutable source checkout as training")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    bank_manifest_path = bank / "bank_manifest.json"
    bank_path = bank / "shared_trajectory.h5"
    split_path = bank / "registered_split.json"
    adapter_manifest_path = adapter / "training_bank_manifest.json"
    for path in (bank_manifest_path, bank_path, split_path, adapter_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    bank_manifest = json.loads(bank_manifest_path.read_text())
    adapter_manifest = json.loads(adapter_manifest_path.read_text())
    if (bank_manifest.get("schema") != "stage5_flow_scene_coverage_shared_bank_v3"
            or bank_manifest.get("scope") != "complete_balanced_8_rollout_pilot"
            or bank_manifest.get("complete_balanced_pilot") is not True
            or len(bank_manifest.get("episodes", [])) != 8):
        raise RuntimeError("registered physical bank is not the complete v3 eight-rollout pilot")
    if sha256(bank_path) != bank_manifest.get("bank_sha256"):
        raise RuntimeError("physical bank file does not match its manifest hash")
    if (adapter_manifest.get("schema") != "stage5_steflow_diagnostic_bank_v1"
            or adapter_manifest.get("status") != "paired_synthetic_diagnostic_data_ready_no_training_result"
            or adapter_manifest.get("real_data_used") is not False
            or adapter_manifest.get("flow_gt_used_for_self_supervised_optimizer") is not False
            or adapter_manifest.get("train_windows_per_generator") != 264
            or adapter_manifest.get("validation_windows_per_generator") != 264
            or adapter_manifest.get("paired_labels_identical") is not True):
        raise RuntimeError("paired adapter manifest fails the registered synthetic-only contract")

    code_paths = {
        "benchmark/sim2real/scripts/register_stage5_flow_selfsup_diagnostic_config.py": HERE,
        "benchmark/sim2real/scripts/train_stage5_flow_selfsup_diagnostic.py": repo / "benchmark/sim2real/scripts/train_stage5_flow_selfsup_diagnostic.py",
        "benchmark/sim2real/scripts/prepare_stage5_flow_diagnostic_bank.py": repo / "benchmark/sim2real/scripts/prepare_stage5_flow_diagnostic_bank.py",
        "benchmark/sim2real/scripts/prepare_stage4_steflow_bank.py": repo / "benchmark/sim2real/scripts/prepare_stage4_steflow_bank.py",
        "benchmark/sim2real/scripts/prepare_steflow_stage0.py": repo / "benchmark/sim2real/scripts/prepare_steflow_stage0.py",
        "benchmark/sim2real/scripts/run_steflow_stage2_pilot.py": repo / "benchmark/sim2real/scripts/run_steflow_stage2_pilot.py",
        "benchmark/sim2real/scripts/train_steflow_stage4.py": repo / "benchmark/sim2real/scripts/train_steflow_stage4.py",
        "benchmark/sim2real/compat/steflow_align_corners.py": repo / "benchmark/sim2real/compat/steflow_align_corners.py",
        "@model/models/steflow.py": model / "models/steflow.py",
        "@model/models/corr.py": model / "models/corr.py",
        "@model/multiscaleloss.py": model / "multiscaleloss.py",
    }
    missing = [name for name, path in code_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"registered source files missing: {missing}")
    code_hashes = {name: sha256(path) for name, path in code_paths.items()}
    model_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=model, text=True).strip()
    model_dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=model, text=True).splitlines()

    config = {
        "schema": "stage5_flow_selfsup_diagnostic_v1",
        "purpose": "short paired synthetic-only check of the pinned STE-FlowNet self-supervised objective on a registered 8-rollout shared bank",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_simulator_core": "e79e656ac00f7847b98db9e12bf43b093f3afff1",
        "real_data_used": False,
        "real_scoring_enabled": False,
        "checkpoint_selection_allowed": False,
        "validation_selection_performed": False,
        "code": {
            "source_commit": commit,
            "file_sha256": code_hashes,
            "model_root": str(model),
            "model_source_commit": model_commit,
            "model_source_dirty_paths": model_dirty,
        },
        "data": {
            "shared_physical_bank_path": str(bank_path),
            "expected_bank_sha256": sha256(bank_path),
            "bank_manifest_sha256": sha256(bank_manifest_path),
            "split_sha256": sha256(split_path),
            "paired_adapter_root": str(adapter),
            "expected_adapter_manifest_sha256": sha256(adapter_manifest_path),
            "train_rollouts_per_generator": 4,
            "validation_rollouts_per_generator": 4,
            "train_windows_per_generator": 264,
            "validation_windows_per_generator": 264,
            "split_unit": "independent physical rollout",
            "flow_gt_used_in_optimizer": False,
            "event_count_filtering": "none",
        },
        "training": {
            "seed": 0,
            "expected_paired_initial_state_sha256": "acc6bbd11488a21868c5f147c7f397521480166c24914cc7988a6428d6ed71cf",
            "optimizer_updates_per_arm": 100,
            "batch_size": 8,
            "optimizer": "Adam",
            "learning_rate": 0.0004,
            "betas": [0.9, 0.999],
            "weight_decay_weights": 0.0004,
            "weight_decay_biases": 0.0,
            "scheduler": {"milestones_epoch": [5, 10, 20], "gamma": 0.7},
            "augmentation": "pinned Stage-4 joint crop, resize, flip and rotation; paired RNG seed sequence per sample and update",
            "sampling": "deterministic epoch permutation over all 264 training windows, no event-count filtering",
            "objective": "pinned STE-FlowNet multi-scale photometric objective plus 10x second-order smoothness; unchanged official Stage-4 path",
            "checkpoint_selection": False,
        },
        "validation": {
            "primary_diagnostic_mask": "shared flow-valid AND union of nonzero pixels in paired NORMAL EventGenesis/V2E input event tensors",
            "diagnostic_masks": ["full shared valid flow region", "common normal-input event-active union intersect shared valid flow"],
            "metrics": ["strict EPE in px per 30ms", "zero-flow EPE", "prediction/GT magnitude", "direction cosine"],
            "fixed_input_perturbations": ["all-zero event tensor", "reverse five-bin temporal axis within each polarity"],
            "perturbation_mask": "frozen from normal paired inputs; never recomputed after corruption",
            "sample_coverage": "all 264 windows from four held-out physical rollouts per arm",
        },
        "stop_rule": "run exactly 100 optimizer updates per arm if contracts and finite losses/gradients hold; this is a diagnostic only and does not launch a formal long run",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "registered", "output": str(output),
                      "config_sha256": sha256(output), "source_commit": commit,
                      "bank_sha256": config["data"]["expected_bank_sha256"],
                      "adapter_manifest_sha256": config["data"]["expected_adapter_manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
