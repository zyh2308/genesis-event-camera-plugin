#!/usr/bin/env python3
"""Run the fixed 64-window synthetic motion diagnostic on async best checkpoints.

The prior epoch-23 diagnostic remains untouched. This adapter uses the existing
fixed-sample diagnostic implementation and changes only per-arm checkpoint
boundary validation so each independently selected synthetic-best checkpoint
is checked against its own epoch/update/scheduler metadata.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "benchmark/sim2real/configs/stage4_final_best_motion_diagnostic_20261019.json"
BASE_SCRIPT = REPO_ROOT / "benchmark/sim2real/scripts/stage4_motion_learning_diagnostic.py"
FROZEN_PLAN = REPO_ROOT / "benchmark/sim2real/evidence/S4/20261018-motion-diagnostic/DIAGNOSTIC_PLAN.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_literal_assignments(source_path: Path) -> dict[str, Any]:
    tree = ast.parse(source_path.read_text())
    found: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return found


def load_and_validate_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != 1 or not config.get("asynchronous_checkpoint_pairing"):
        raise ValueError("expected schema v1 with explicit asynchronous checkpoint pairing")
    fixed = config["fixed_diagnostic_source"]
    if sha256(BASE_SCRIPT) != fixed["script_sha256"]:
        raise RuntimeError("fixed motion diagnostic implementation changed; update/review its registered hash first")
    if sha256(FROZEN_PLAN) != fixed["plan_sha256"]:
        raise RuntimeError("fixed epoch-23 diagnostic plan changed; refusing to silently change sample/mask protocol")
    assignments = parse_literal_assignments(BASE_SCRIPT)
    checks = {
        "SAMPLE_INDICES": tuple(fixed["fixed_sample_adapter_indices_per_rollout"]),
        "EXPECTED_RESOLUTION": (260, 346),
        "FLOW_UNIT": fixed["flow_unit"],
        "REGION_NAMES": tuple(fixed["regions"]),
        "GT_MOTION_THRESHOLDS_PX_PER_30MS": tuple(fixed["gt_motion_thresholds_px_per_30ms"]),
        "NEAR_ZERO_THRESHOLDS_PX_PER_30MS": tuple(fixed["near_zero_prediction_thresholds_px_per_30ms"]),
    }
    for name, expected in checks.items():
        if assignments.get(name) != expected:
            raise RuntimeError(f"fixed diagnostic contract mismatch for {name}: {assignments.get(name)!r} != {expected!r}")
    derived_ids = [f"dt1_{index * 3:03d}" for index in fixed["fixed_sample_adapter_indices_per_rollout"]]
    if derived_ids != fixed["expected_sample_ids_per_rollout"]:
        raise ValueError("fixed adapter rows no longer map to the registered physical sample IDs")
    if fixed["expected_rollout_count"] * len(fixed["fixed_sample_adapter_indices_per_rollout"]) != fixed["expected_paired_window_count"]:
        raise ValueError("registered rollout/sample census is inconsistent")
    if tuple(assignments.get("STRATA", {}).values()) != tuple(fixed["strata"]):
        raise RuntimeError("fixed diagnostic scenario strata changed")
    if fixed["interval_seconds"] != 0.03 or fixed["direction_is_reported_only_when_prediction_magnitude_at_least_px"] != 0.1:
        raise ValueError("diagnostic unit or direction threshold differs from the fixed protocol")
    previous_output = REPO_ROOT / "benchmark/sim2real/evidence/S4/20261018-motion-diagnostic/motion_diagnostic.json"
    output = Path(config["run"]["output"])
    if not output.is_absolute():
        output = REPO_ROOT / output
    if output.resolve() == previous_output.resolve():
        raise ValueError("new final-best run must not overwrite the epoch-23 evidence")
    arms = config["checkpoints"]
    if set(arms) != {"eventgenesis", "v2e"}:
        raise ValueError("exactly EventGenesis and V2E checkpoints are required")
    initial_states = set()
    for arm, spec in arms.items():
        epoch = int(spec["synthetic_best_epoch"])
        if int(spec["seed"]) != 0:
            raise ValueError(f"{arm}: only registered seed 0 is permitted")
        # The training checkpoint serializes zero-based `epoch` and one-based
        # `best_epoch`/completed-update count. Preserve this repository format.
        if int(spec["completed_epoch_in_checkpoint"]) != epoch - 1 or int(spec["next_epoch"]) != epoch:
            raise ValueError(f"{arm}: checkpoint epoch metadata is inconsistent")
        if int(spec["optimizer_update"]) != epoch * 198:
            raise ValueError(f"{arm}: optimizer update does not match 198 updates per completed epoch")
        if int(spec["scheduler_last_epoch"]) != int(spec["next_epoch"]):
            raise ValueError(f"{arm}: expected scheduler boundary must match its own next_epoch")
        score = float(spec["best_synthetic_validation_epe_px_per_30ms"])
        if not math.isfinite(score) or score <= 0:
            raise ValueError(f"{arm}: invalid registered synthetic-best score")
        if spec["generator"] != arm:
            raise ValueError(f"{arm}: generator identity mismatch")
        if spec.get("initial_state_sha256"):
            initial_states.add(spec["initial_state_sha256"])
    if len({int(v["synthetic_best_epoch"]) for v in arms.values()}) != 2:
        raise ValueError("registered final-best pair must retain independently selected, asynchronous epochs")
    if len({int(v["optimizer_update"]) for v in arms.values()}) != 2:
        raise ValueError("registered final-best pair must retain independently selected optimizer updates")
    if len(initial_states) > 1:
        raise ValueError("paired synthetic training initialization hashes differ")
    return config


def validate_checkpoint_metadata(arm_cfg: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Pure-Python per-arm metadata check, intentionally independent of Torch."""
    expected = {
        "generator": arm_cfg["generator"],
        "seed": int(arm_cfg["seed"]),
        "epoch": int(arm_cfg["completed_epoch_in_checkpoint"]),
        "optimizer_update": int(arm_cfg["optimizer_update"]),
        "next_epoch": int(arm_cfg["next_epoch"]),
        "best_epoch": int(arm_cfg["synthetic_best_epoch"]),
    }
    mismatch = {key: (metadata.get(key), value) for key, value in expected.items()
                if metadata.get(key) != value}
    try:
        actual_best = float(metadata.get("best_synthetic_validation_epe"))
        expected_best = float(arm_cfg["best_synthetic_validation_epe_px_per_30ms"])
        if abs(actual_best - expected_best) > 1e-7:
            mismatch["best_synthetic_validation_epe"] = (actual_best, expected_best)
        actual_scheduler = int(metadata["scheduler"]["last_epoch"])
        if actual_scheduler != int(arm_cfg["scheduler_last_epoch"]):
            mismatch["scheduler.last_epoch"] = (actual_scheduler, int(arm_cfg["scheduler_last_epoch"]))
    except (KeyError, TypeError, ValueError) as exc:
        mismatch["checkpoint_boundary_metadata"] = (str(exc), "required epoch/best/scheduler fields")
        actual_scheduler = -1
        expected_best = float(arm_cfg["best_synthetic_validation_epe_px_per_30ms"])
    expected_init = arm_cfg.get("initial_state_sha256")
    if expected_init and metadata.get("initial_state_sha256") != expected_init:
        mismatch["initial_state_sha256"] = (metadata.get("initial_state_sha256"), expected_init)
    if mismatch:
        raise ValueError(f"{arm_cfg['generator']} checkpoint differs from its own synthetic-best registration: {mismatch}")


def self_test(config_path: Path) -> dict[str, Any]:
    config = load_and_validate_config(config_path)
    for arm, arm_cfg in config["checkpoints"].items():
        synthetic_meta = {
            "generator": arm,
            "seed": 0,
            "epoch": arm_cfg["completed_epoch_in_checkpoint"],
            "optimizer_update": arm_cfg["optimizer_update"],
            "next_epoch": arm_cfg["next_epoch"],
            "best_epoch": arm_cfg["synthetic_best_epoch"],
            "best_synthetic_validation_epe": arm_cfg["best_synthetic_validation_epe_px_per_30ms"],
            "scheduler": {"last_epoch": arm_cfg["scheduler_last_epoch"]},
            "initial_state_sha256": arm_cfg["initial_state_sha256"],
        }
        validate_checkpoint_metadata(arm_cfg, synthetic_meta)
    bad_meta = dict(synthetic_meta)
    bad_meta["scheduler"] = {"last_epoch": synthetic_meta["scheduler"]["last_epoch"] + 1}
    try:
        validate_checkpoint_metadata(config["checkpoints"]["v2e"], bad_meta)
    except ValueError:
        pass
    else:
        raise AssertionError("per-arm scheduler boundary mismatch must fail closed")
    bad_update = dict(synthetic_meta)
    bad_update["optimizer_update"] = int(bad_update["optimizer_update"]) + 198
    try:
        validate_checkpoint_metadata(config["checkpoints"]["v2e"], bad_update)
    except ValueError:
        pass
    else:
        raise AssertionError("per-arm optimizer-update mismatch must fail closed")
    return {
        "status": "pass",
        "scope": "self-contained static registration check; no torch, CUDA, data, checkpoint, or model loaded",
        "fixed_script_sha256": config["fixed_diagnostic_source"]["script_sha256"],
        "fixed_plan_sha256": config["fixed_diagnostic_source"]["plan_sha256"],
        "sample_indices_per_rollout": config["fixed_diagnostic_source"]["fixed_sample_adapter_indices_per_rollout"],
        "expected_paired_windows": config["fixed_diagnostic_source"]["expected_paired_window_count"],
        "units": config["fixed_diagnostic_source"]["flow_unit"],
        "event_active_rule": config["fixed_diagnostic_source"]["shared_event_active_union_rule"],
        "checkpoint_pair": {
            arm: {key: config["checkpoints"][arm][key] for key in (
                "synthetic_best_epoch", "completed_epoch_in_checkpoint", "optimizer_update", "next_epoch",
                "best_synthetic_validation_epe_px_per_30ms")}
            for arm in ("eventgenesis", "v2e")
        },
        "different_epochs_and_updates_accepted": True,
        "negative_controls": ["incorrect per-arm scheduler boundary rejected", "incorrect per-arm optimizer update rejected"],
        "epoch23_output_preserved": True,
    }


def run(config_path: Path) -> None:
    config = load_and_validate_config(config_path)
    run_cfg = config["run"]
    output = Path(run_cfg["output"])
    if not output.is_absolute():
        output = REPO_ROOT / output
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final-best diagnostic output: {output}")
    unverified_output = output.with_name(output.name + ".unverified")
    if unverified_output.exists():
        raise FileExistsError(f"refusing to overwrite prior unverified attempt: {unverified_output}")
    checkpoints: dict[str, dict[str, Any]] = config["checkpoints"]
    for arm, spec in checkpoints.items():
        checkpoint = Path(spec["path"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{arm} synthetic-best checkpoint not found: {checkpoint}")
        actual_sha = sha256(checkpoint)
        expected_sha = spec.get("sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise RuntimeError(f"{arm} checkpoint SHA-256 must be refreshed from the g1 artifact index before inference")
        if actual_sha != expected_sha:
            raise RuntimeError(f"{arm} checkpoint SHA-256 differs from refreshed g1 artifact index")

    source_path = BASE_SCRIPT.resolve()
    module_name = "stage4_motion_learning_diagnostic_fixed_base"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import fixed diagnostic implementation at {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_loader = module.load_checkpoint
    restored_metadata: dict[str, dict[str, Any]] = {}

    def load_one_registered_best(path: Path, generator: str, model_root: Path,
                                 data_manifest_sha: str, config_sha: str, device: Any):
        arm_cfg = checkpoints[generator]
        # synthetic_best checkpoints are deliberately inference-only
        # (resume_phase=inference, resume_allowed=False). The old diagnostic
        # loader was written for resumable last.pt and rejects this valid
        # lifecycle contract. Validate the inference boundary explicitly,
        # then restore the same pinned model using its ordinary eval path.
        saved = module.torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            "generator": generator,
            "seed": int(arm_cfg["seed"]),
            "epoch": int(arm_cfg["completed_epoch_in_checkpoint"]),
            "optimizer_update": int(arm_cfg["optimizer_update"]),
            "next_epoch": int(arm_cfg["next_epoch"]),
            "next_position_samples": 0,
            "flow_unit": module.FLOW_UNIT,
            "data_manifest_sha256": data_manifest_sha,
            "config_sha256": config_sha,
            "resume_phase": "inference",
            "resume_allowed": False,
        }
        mismatch = {key: (saved.get(key), value) for key, value in expected.items()
                    if saved.get(key) != value}
        if saved.get("best_epoch") != int(arm_cfg["synthetic_best_epoch"]):
            mismatch["best_epoch"] = (saved.get("best_epoch"), int(arm_cfg["synthetic_best_epoch"]))
        if mismatch:
            raise ValueError(f"{generator} inference checkpoint contract mismatch: {mismatch}")
        if not isinstance(saved.get("scheduler"), dict):
            raise ValueError(f"{generator} inference checkpoint has no scheduler boundary")
        scheduler_epoch = int(saved["scheduler"].get("last_epoch", -1))
        if scheduler_epoch != int(arm_cfg["scheduler_last_epoch"]):
            raise ValueError(f"{generator} inference checkpoint scheduler boundary mismatch: {scheduler_epoch}")
        sys.path.insert(0, str(model_root))
        compat_dir = Path(__file__).resolve().parent.parent / "compat"
        sys.path.insert(0, str(compat_dir))
        from run_steflow_stage2_pilot import model_output_to_flow_hw, pure_torch_spatial_correlation
        from steflow_align_corners import patch_ste

        corr = importlib.import_module("models.corr")
        corr.spatial_correlation_sample = pure_torch_spatial_correlation
        models = importlib.import_module("models")
        multiscaleloss = importlib.import_module("multiscaleloss")
        source_hashes = patch_ste(models, multiscaleloss, model_root)
        model = models.steflow().to(device)
        model.load_state_dict(saved["state_dict"], strict=True)
        actual_state = module.state_sha256(model)
        if actual_state != saved.get("current_state_sha256"):
            raise ValueError(f"{generator} inference checkpoint model-state hash mismatch")
        model.eval()
        metadata = {key: saved.get(key) for key in (
            "generator", "seed", "epoch", "optimizer_update", "next_epoch",
            "next_position_samples", "scheduler", "current_state_sha256",
            "initial_state_sha256", "best_epoch", "best_synthetic_validation_epe",
        )}
        metadata["scheduler"] = {"last_epoch": scheduler_epoch}
        del saved
        decoder = model_output_to_flow_hw
        validate_checkpoint_metadata(arm_cfg, metadata)
        actual_best = float(metadata.get("best_synthetic_validation_epe"))
        expected_best = float(arm_cfg["best_synthetic_validation_epe_px_per_30ms"])
        actual_scheduler = int(metadata["scheduler"]["last_epoch"])
        restored_metadata[generator] = {
            **metadata,
            "scheduler": {"last_epoch": actual_scheduler},
            "checkpoint_path": str(Path(path).resolve()),
            "checkpoint_sha256": sha256(Path(path)),
            "checkpoint_size_bytes": Path(path).stat().st_size,
            "expected_synthetic_best_epe": expected_best,
        }
        # The frozen base diagnostic historically requires identical scheduler
        # dicts. Both real per-arm scheduler boundaries have already been
        # validated above; normalize only the transient metadata passed to its
        # equal-init/pairing assertion. The emitted JSON is restored below.
        normalized = {**metadata, "scheduler": {"last_epoch": -1}}
        return model, decoder, source_hashes, normalized

    module.load_checkpoint = load_one_registered_best
    args = [
        "--model-root", run_cfg["model_root"],
        "--data-root", run_cfg["data_root"],
        "--shared-bank", run_cfg["shared_bank"],
        "--config", run_cfg["training_config"],
        "--eventgenesis-checkpoint", checkpoints["eventgenesis"]["path"],
        "--v2e-checkpoint", checkpoints["v2e"]["path"],
        "--output", str(unverified_output),
        "--device", run_cfg["device"],
    ]
    previous_argv = sys.argv
    try:
        sys.argv = [str(source_path), *args]
        module.main()
    finally:
        sys.argv = previous_argv
        module.load_checkpoint = original_loader
    result = json.loads(unverified_output.read_text())
    if set(restored_metadata) != {"eventgenesis", "v2e"}:
        raise RuntimeError("fixed diagnostic did not restore both registered best checkpoints")
    actual_init = {restored_metadata[a].get("initial_state_sha256") for a in restored_metadata}
    if len(actual_init) != 1:
        raise RuntimeError("asynchronously selected checkpoints do not retain the paired initialization")
    result["protocol"]["checkpoint_selection"] = (
        "independent synthetic-validation-best checkpoints; EventGenesis epoch 33/update 6534 and "
        "V2E epoch 20/update 3960. Epochs and scheduler boundaries are validated per arm; equal update/epoch is not required."
    )
    result["protocol"]["checkpoint_selection_metric"] = "full synthetic validation EPE only"
    result["protocol"]["same_optimizer_update_required"] = False
    result["protocol"]["same_checkpoint_epoch_required"] = False
    result["protocol"]["paired_initialization_sha256"] = next(iter(actual_init))
    result["async_checkpoint_validation"] = {
        "status": "pass",
        "same_initialization_required_and_verified": True,
        "per_arm_epoch_update_best_score_scheduler_verified": True,
        "fixed_samples_regions_masks_and_units_inherited_from_epoch23_plan": True,
        "source_diagnostic_script_sha256": config["fixed_diagnostic_source"]["script_sha256"],
        "source_plan_sha256": config["fixed_diagnostic_source"]["plan_sha256"],
        "checkpoints": restored_metadata,
        "epoch23_evidence_preserved_at": "benchmark/sim2real/evidence/S4/20261018-motion-diagnostic/",
    }
    result["paths"]["checkpoints"] = restored_metadata
    result["paths"]["final_best_diagnostic_config"] = str(Path(config_path).resolve())
    result["paths"]["final_best_diagnostic_config_sha256"] = sha256(Path(config_path).resolve())
    result["real_data_used"] = False
    result["training_or_checkpoint_selection_changed"] = False
    temp = output.with_suffix(output.suffix + ".async.partial")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temp, output)
    unverified_output.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--self-test", action="store_true",
                        help="static source/config checks only; does not load torch, data, or checkpoints")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.config), indent=2))
    else:
        run(args.config)


if __name__ == "__main__":
    main()
