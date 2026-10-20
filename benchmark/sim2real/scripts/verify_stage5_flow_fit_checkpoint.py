#!/usr/bin/env python3
"""Independently restore and inspect one fixed-sample STE-FlowNet checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark/sim2real/scripts"))
from stage5_flow_supervised_pilot import (  # noqa: E402
    FLOW_UNIT,
    load_sample,
    model_output_to_flow_hw,
    model_source,
    pad_sample,
    sha256,
    state_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    sample_cfg = config["sample"]
    manifest_path = args.data_root.resolve() / "training_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected = {
        "sample_id": sample_cfg["sample_id"],
        "data_manifest_sha256": sha256(manifest_path),
        "flow_unit": FLOW_UNIT,
        "real_data_used": False,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"checkpoint {key} mismatch: {checkpoint.get(key)!r} != {value!r}")
    if checkpoint.get("updates") != int(config["diagnostic_budget"]["updates_per_arm"]):
        raise RuntimeError("checkpoint does not contain the registered number of optimizer updates")

    row = next(row for row in manifest["entries"]
               if row["generator"] == checkpoint["generator"]
               and row["rollout_id"] == sample_cfg["rollout_id"])
    sample_path = Path(row["path"])
    if not sample_path.is_file() or sha256(sample_path) != row["sha256"]:
        raise RuntimeError(f"registered fixed-window event/label artifact hash mismatch: {sample_path}")
    sample = load_sample(sample_path, int(sample_cfg["window_index"]))
    if str(sample["sample_ids"]) != sample_cfg["sample_id"]:
        raise RuntimeError("fixed-sample identity differs from the preregistered sample")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("checkpoint restore probe requires CUDA")
    torch.cuda.set_device(device)
    model, patch = model_source(args.model_root.resolve(), device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    restored_state_hash = state_sha256(model)
    if restored_state_hash != checkpoint.get("state_sha256"):
        raise RuntimeError("restored model state hash differs from the state recorded before serialization")
    optimizer = torch.optim.Adam([
        {"params": model.bias_parameters(), "weight_decay": 0.0},
        {"params": model.weight_parameters(), "weight_decay": 0.0004},
    ], lr=0.0004, betas=(0.9, 0.999))
    optimizer.load_state_dict(checkpoint["optimizer"])
    step_values = sorted({int(state["step"].item()) for state in optimizer.state.values() if "step" in state})
    if step_values != [int(checkpoint["updates"])]:
        raise RuntimeError(f"restored optimizer step mismatch: {step_values}")

    events, _, _ = pad_sample(sample, device)
    model.eval()
    with torch.inference_mode():
        first = model_output_to_flow_hw(model(events, 352, use_dt4=False, flow_seq=False))
        second = model_output_to_flow_hw(model(events, 352, use_dt4=False, flow_seq=False))
    if first.shape != second.shape or not np.isfinite(first).all():
        raise RuntimeError(f"restored prediction shape/finite check failed: {first.shape}, {second.shape}")
    max_abs = float(np.max(np.abs(first - second)))
    if max_abs > 1e-6:
        raise RuntimeError(f"ordinary eval prediction is not repeatable within 1e-6 after checkpoint load: {max_abs}")

    result = {
        "schema": "stage5_flow_fit_checkpoint_restore_v1",
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": checkpoint["generator"],
        "sample_id": checkpoint["sample_id"],
        "updates": int(checkpoint["updates"]),
        "checkpoint": {"path": str(args.checkpoint.resolve()), "size_bytes": args.checkpoint.stat().st_size,
                       "sha256": sha256(args.checkpoint)},
        "data_manifest_sha256": sha256(manifest_path),
        "source_patch": patch,
        "restored_state_sha256": restored_state_hash,
        "restored_optimizer_step_values": step_values,
        "prediction_shape": list(first.shape),
        "prediction_finite": True,
        "repeat_eval_prediction_max_abs": max_abs,
        "repeat_eval_prediction_max_abs_tolerance": 1e-6,
        "real_data_used": False,
        "real_scoring_enabled": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
