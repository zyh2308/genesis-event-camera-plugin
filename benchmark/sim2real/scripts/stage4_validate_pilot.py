#!/usr/bin/env python3
"""Fail-closed validator for the paired, one-epoch official-entrypoint pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows, summaries = {}, {}
    for generator in ("eventgenesis", "v2e"):
        root = args.run_root.resolve() / generator
        status = json.loads((root / "live_status.json").read_text())
        metadata = json.loads((root / "run_metadata.json").read_text())
        lines = [json.loads(line) for line in (root / "training_metrics.jsonl").read_text().splitlines() if line.strip()]
        updates = [row for row in lines if "optimizer_update" in row]
        validations = [row for row in lines if row.get("event") == "synthetic_validation"]
        initial = [row for row in lines if row.get("event") == "initial_synthetic_validation"]
        checkpoint_path = root / "last.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if status.get("status") != "paused_at_update_budget" or status.get("completed_updates") != 198:
            raise RuntimeError(f"{generator}: did not stop cleanly at update 198: {status}")
        if len(updates) != 198 or [row["optimizer_update"] for row in updates] != list(range(1, 199)):
            raise RuntimeError(f"{generator}: update log is incomplete or non-contiguous")
        if len(initial) != 1 or len(validations) != 1:
            raise RuntimeError(f"{generator}: expected one initial and one completed synthetic validation")
        if checkpoint["optimizer_update"] != 198 or checkpoint["next_epoch"] != 1 or checkpoint["next_position_samples"] != 0:
            raise RuntimeError(f"{generator}: checkpoint does not record the post-validation epoch boundary")
        if checkpoint.get("resume_allowed") is not True or checkpoint.get("resume_phase") != "train":
            raise RuntimeError(f"{generator}: epoch-boundary last checkpoint is not resumable")
        if metadata.get("real_data_used") is not False or metadata.get("flow_gt_used_in_optimizer") is not False:
            raise RuntimeError(f"{generator}: synthetic-only training guard failed")
        if any(not math.isfinite(float(row[key])) or float(row[key]) <= 0
               for row in updates for key in ("loss", "gradient_l2_norm")):
            raise RuntimeError(f"{generator}: invalid loss or gradient in update log")
        if any(not all(math.isfinite(float(row[key])) for key in ("photometric_loss", "smoothness_loss"))
               for row in updates):
            raise RuntimeError(f"{generator}: invalid loss component")
        validation = validations[0]
        if len(validation.get("model_epe_by_rollout", {})) != 8 or len(validation.get("zero_flow_epe_by_rollout", {})) != 8:
            raise RuntimeError(f"{generator}: synthetic validation did not cover all eight rollouts")
        visual_dir = root / "fixed_validation_visuals/epoch_01"
        visuals = sorted(visual_dir.glob("*.png"))
        if len(visuals) != 4 or any(path.stat().st_size <= 0 for path in visuals):
            raise RuntimeError(f"{generator}: fixed validation visual evidence missing")
        rows[generator] = updates
        summaries[generator] = {
            "updates": len(updates), "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha(checkpoint_path),
            "initial_state_sha256": checkpoint["initial_state_sha256"],
            "model_state_sha256": checkpoint["current_state_sha256"],
            "scheduler_last_epoch": checkpoint["scheduler"]["last_epoch"],
            "next_epoch": checkpoint["next_epoch"], "next_position_samples": checkpoint["next_position_samples"],
            "validation_model_epe_px_per_30ms": validation["model_mean_epe_px_per_30ms"],
            "validation_zero_flow_epe_px_per_30ms": validation["zero_flow_mean_epe_px_per_30ms"],
            "strata": validation["strata"], "fixed_validation_visuals": [str(path) for path in visuals],
            "first_update": updates[0], "last_update": updates[-1],
        }
    if summaries["eventgenesis"]["initial_state_sha256"] != summaries["v2e"]["initial_state_sha256"]:
        raise RuntimeError("paired pilot arms do not share identical seed-0 initialization")
    for a, b in zip(rows["eventgenesis"], rows["v2e"]):
        if a["optimizer_update"] != b["optimizer_update"] or a["rollout_sample_ids"] != b["rollout_sample_ids"] or a["augmentation_seed_sequences"] != b["augmentation_seed_sequences"]:
            raise RuntimeError(f"paired sample/augmentation schedule mismatch at update {a.get('optimizer_update')}")
    result = {"schema": "stage4_paired_formal_pilot_v1", "status": "PASS", "optimizer_updates_each": 198,
              "epochs_each": 1, "paired_initialization": summaries["eventgenesis"]["initial_state_sha256"],
              "sample_order_and_augmentation_match": True, "arms": summaries,
              "all_training_synthetic_only": True}
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
