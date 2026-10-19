#!/usr/bin/env python3
"""Evaluate fixed flow-GT pilot checkpoints on full and common active support."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark/sim2real/scripts"))
from stage5_flow_supervised_pilot import (  # noqa: E402
    FLOW_UNIT,
    epe_pair,
    load_sample,
    model_output_to_flow_hw,
    model_source,
    pad_sample,
    sha256,
    state_sha256,
)


def _mean_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eventgenesis-checkpoint", type=Path, required=True)
    parser.add_argument("--v2e-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("active-region flow evaluation requires CUDA")
    torch.cuda.set_device(device)
    manifest_path = args.data_root / "training_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha, config_sha = sha256(manifest_path), sha256(args.config)
    if manifest.get("flow_unit") != "pixels per 30ms endpoint interval":
        raise RuntimeError("unexpected flow label unit")
    by_arm = {(row["generator"], row["rollout_id"]): row for row in manifest["entries"]}
    train_rollouts = [f"s{i}_00" for i in range(4)]
    validation_rollouts = sorted(
        rid for arm, rid in by_arm
        if arm == "eventgenesis" and by_arm[(arm, rid)]["split"] == "validation"
    )
    if len(validation_rollouts) != 8:
        raise RuntimeError(f"expected 8 held-out rollouts, found {validation_rollouts}")
    indices = np.floor((np.arange(2, dtype=np.float64) + 0.5) * 64 / 2).astype(int).tolist()

    sample_rows = []
    for split, rollouts, window_indices in (
        ("train", train_rollouts, indices),
        ("validation", validation_rollouts, [0]),
    ):
        for rollout_id in rollouts:
            selected = window_indices if split == "train" else [0]
            for sample_index in selected:
                paired = {}
                for arm in ("eventgenesis", "v2e"):
                    row = by_arm[(arm, rollout_id)]
                    if row["split"] != split:
                        raise RuntimeError(f"split mismatch for {arm}/{rollout_id}")
                    path = Path(row["path"])
                    if not path.is_file() or sha256(path) != row["sha256"]:
                        raise RuntimeError(f"registered adapter changed: {path}")
                    paired[arm] = load_sample(path, sample_index)
                eg, v2 = paired["eventgenesis"], paired["v2e"]
                for arm, sample in paired.items():
                    if sample["events"].shape != (2, 260, 346, 5):
                        raise RuntimeError(f"{arm}: unexpected event tensor shape {sample['events'].shape}")
                    if sample["flow_gt"].shape != (2, 260, 346):
                        raise RuntimeError(f"{arm}: unexpected GT shape {sample['flow_gt'].shape}")
                    if sample["flow_valid"].shape != (260, 346):
                        raise RuntimeError(f"{arm}: unexpected valid mask shape {sample['flow_valid'].shape}")
                    if not np.isfinite(sample["events"]).all() or not np.isfinite(sample["flow_gt"]).all():
                        raise RuntimeError(f"{arm}: non-finite event tensor or flow GT")
                if not (
                    np.array_equal(eg["flow_gt"], v2["flow_gt"])
                    and np.array_equal(eg["flow_valid"], v2["flow_valid"])
                    and np.array_equal(eg["sample_ids"], v2["sample_ids"])
                    and np.array_equal(eg["label_sha256"], v2["label_sha256"])
                ):
                    raise RuntimeError(f"shared GT mismatch at {rollout_id}/{sample_index}")
                active = {
                    arm: np.any(sample["events"] != 0, axis=(0, 3))
                    for arm, sample in paired.items()
                }
                common_active = active["eventgenesis"] | active["v2e"]
                valid = eg["flow_valid"].astype(bool)
                sample_rows.append({
                    "split": split,
                    "rollout_id": rollout_id,
                    "sample_index": sample_index,
                    "sample_id": str(eg["sample_ids"]),
                    "label_sha256": str(eg["label_sha256"]),
                    "common_valid_pixels": int(valid.sum()),
                    "common_event_active_pixels": int((valid & common_active).sum()),
                    "active_coverage_of_valid": float((valid & common_active).sum() / max(valid.sum(), 1)),
                    "arms": paired,
                    "valid": valid,
                    "active": valid & common_active,
                })

    results = {
        "schema": "stage5_flow_gt_subset_active_eval_v1",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "data_manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "flow_unit": FLOW_UNIT,
        "training_sample_indices_per_rollout": indices,
        "validation_sample_index": 0,
        "mask_rule": "common valid GT intersected with union of pixels active in the paired normal EventGenesis and V2E five-bin inputs; same mask is used for both trained models",
        "real_data_used": False,
        "arms": {},
    }
    for arm, checkpoint_path in (
        ("eventgenesis", args.eventgenesis_checkpoint.resolve()),
        ("v2e", args.v2e_checkpoint.resolve()),
    ):
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = {
            "generator": arm,
            "seed": 0,
            "flow_unit": FLOW_UNIT,
            "data_manifest_sha256": manifest_sha,
            "config_sha256": config_sha,
            "real_data_used": False,
            "validation_used_for_selection": False,
        }
        mismatch = {key: (saved.get(key), value) for key, value in expected.items() if saved.get(key) != value}
        if mismatch:
            raise RuntimeError(f"{arm} checkpoint metadata mismatch: {mismatch}")
        model, source_patch = model_source(args.model_root.resolve(), device)
        model.load_state_dict(saved["state_dict"], strict=True)
        model.eval()
        if state_sha256(model) != saved.get("state_sha256"):
            raise RuntimeError(f"{arm} checkpoint state hash mismatch")
        split_metrics = {"train": [], "validation": []}
        with torch.inference_mode():
            for row in sample_rows:
                sample = row["arms"][arm]
                events, _, _ = pad_sample(sample, device)
                pred = model_output_to_flow_hw(model(events, 352, use_dt4=False, flow_seq=False))[:, 46:306, 3:349]
                if pred.shape != (2, 260, 346) or not np.isfinite(pred).all():
                    raise RuntimeError(f"{arm} returned invalid dense flow on {row['sample_id']}")
                gt = sample["flow_gt"].astype(np.float32, copy=False)
                split_metrics[row["split"]].append({
                    "sample_id": row["sample_id"],
                    "rollout_id": row["rollout_id"],
                    "valid_pixels": row["common_valid_pixels"],
                    "common_event_active_pixels": row["common_event_active_pixels"],
                    "active_coverage_of_valid": row["active_coverage_of_valid"],
                    "full_valid_epe_px_per_30ms": epe_pair(pred, gt, row["valid"]),
                    "full_valid_zero_flow_epe_px_per_30ms": epe_pair(np.zeros_like(gt), gt, row["valid"]),
                    "common_active_epe_px_per_30ms": epe_pair(pred, gt, row["active"]),
                    "common_active_zero_flow_epe_px_per_30ms": epe_pair(np.zeros_like(gt), gt, row["active"]),
                    "prediction_mean_magnitude_full_valid": float(np.hypot(pred[0], pred[1])[row["valid"]].mean()) if row["valid"].any() else None,
                    "prediction_mean_magnitude_active": float(np.hypot(pred[0], pred[1])[row["active"]].mean()) if row["active"].any() else None,
                })
        results["arms"][arm] = {
            "checkpoint": {
                "path": str(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "sha256": sha256(checkpoint_path),
                "state_sha256": saved["state_sha256"],
            },
            "source_patch": source_patch,
            "train_full_valid_macro_epe": float(np.mean([x["full_valid_epe_px_per_30ms"] for x in split_metrics["train"]])),
            "train_active_macro_epe": _mean_present([x["common_active_epe_px_per_30ms"] for x in split_metrics["train"]]),
            "validation_full_valid_macro_epe": float(np.mean([x["full_valid_epe_px_per_30ms"] for x in split_metrics["validation"]])),
            "validation_active_macro_epe": _mean_present([x["common_active_epe_px_per_30ms"] for x in split_metrics["validation"]]),
            "train": split_metrics["train"],
            "validation": split_metrics["validation"],
        }
        del saved, model
        torch.cuda.empty_cache()

    results["status"] = "complete"
    results["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": results["status"],
        "output": str(args.output),
        "summary": {
            arm: {
                key: value[key]
                for key in ("train_full_valid_macro_epe", "train_active_macro_epe", "validation_full_valid_macro_epe", "validation_active_macro_epe")
            }
            for arm, value in results["arms"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
