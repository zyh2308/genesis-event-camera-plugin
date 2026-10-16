#!/usr/bin/env python3
"""Measure same-process and independently restored STE-FlowNet output equality."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from run_steflow_stage2_pilot import (
    FLOW_UNIT,
    compute_epe,
    event_input,
    model_output_to_flow_hw,
    read_episode,
    sha256,
    state_sha256,
)
from verify_pinned_source import verify_pinned_source


COMMIT = "eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6"
SOURCE_MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "downstream_source_bundle.json"


def difference(left: np.ndarray, right: np.ndarray) -> dict:
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        "shape_equal": left.shape == right.shape,
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()),
        "allclose_at_1e-6": bool(np.allclose(left, right, rtol=1e-6, atol=1e-6)),
        "allclose_at_1e-4": bool(np.allclose(left, right, rtol=1e-4, atol=1e-4)),
        "finite": bool(np.isfinite(left).all() and np.isfinite(right).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", choices=(1, 4), type=int, required=True)
    parser.add_argument("--generator", choices=("eventgenesis", "v2e"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    source_verification = verify_pinned_source(args.model_root, "STE-FlowNet", COMMIT, SOURCE_MANIFEST)
    sys.path.insert(0, str(args.model_root.resolve()))
    corr_module = importlib.import_module("models.corr")
    from run_steflow_stage2_pilot import pure_torch_spatial_correlation
    corr_module.spatial_correlation_sample = pure_torch_spatial_correlation
    import models

    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = models.steflow().to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    state_hash = state_sha256(model)
    if checkpoint.get("final_state_sha256") and state_hash != checkpoint["final_state_sha256"]:
        raise RuntimeError("checkpoint's stored state hash does not match the restored parameters")

    episode_path = sorted((args.data_root / args.generator / "validation").glob("*.npz"))[0]
    episode = read_episode(episode_path)
    index = 0
    x = event_input(episode, index, args.dt, device)
    input_before = x.detach().cpu().numpy().copy()
    preds = []
    with torch.inference_mode():
        for _ in range(3):
            preds.append(model_output_to_flow_hw(model(x, 256, use_dt4=(args.dt == 4), flow_seq=False)))
    input_after = x.detach().cpu().numpy()
    restored = models.steflow().to(device)
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    restored.eval()
    restored_hash = state_sha256(restored)
    with torch.inference_mode():
        reloaded = model_output_to_flow_hw(restored(x, 256, use_dt4=(args.dt == 4), flow_seq=False))
    gt = episode[f"flow_gt_dt{args.dt}"][index]
    mask = episode[f"flow_valid_dt{args.dt}"][index]
    epes = [compute_epe(pred, gt, mask, prediction_unit=FLOW_UNIT, gt_unit=FLOW_UNIT)[0]
            for pred in preds + [reloaded]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "completed_diagnostic",
        "scope": "one fixed synthetic validation input; same-model repeats and a separate state_dict restore",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_state_sha256": state_hash,
        "restored_state_sha256": restored_hash,
        "state_hash_equal": state_hash == restored_hash,
        "generator": args.generator,
        "dt": args.dt,
        "rollout_id": episode_path.stem,
        "interval_index": index,
        "input_shape": list(x.shape),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "input_unchanged": bool(np.array_equal(input_before, input_after)),
        "same_model_repeat_1_vs_2": difference(preds[0], preds[1]),
        "same_model_repeat_2_vs_3": difference(preds[1], preds[2]),
        "same_model_vs_independent_restore": difference(preds[0], reloaded),
        "epe_px_per_interval_repeats_then_restore": epes,
        "source_verification": source_verification,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
