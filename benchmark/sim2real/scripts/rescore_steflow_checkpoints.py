#!/usr/bin/env python3
"""Recompute Stage 2 synthetic EPE from the four saved STE-FlowNet checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from verify_pinned_source import verify_pinned_source

PINNED_STEFLOW_COMMIT = "eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6"
SOURCE_BUNDLE_MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "downstream_source_bundle.json"

from run_steflow_stage2_pilot import (
    FLOW_UNIT,
    compute_epe,
    event_input,
    model_output_to_flow_hw,
    read_episode,
    sha256,
    state_sha256,
)


def checkpoint_state(payload: dict, generator: str, dt: int) -> dict:
    if payload.get("generator") != generator or int(payload.get("dt", -1)) != dt:
        raise ValueError(
            f"checkpoint identity mismatch: expected {generator}/dt{dt}, "
            f"got {payload.get('generator')}/dt{payload.get('dt')}"
        )
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint is missing a nonempty state_dict")
    return state


def evaluate_one(checkpoint_path: Path, data_root: Path, model_root: Path,
                 generator: str, dt: int, device: torch.device) -> dict:
    sys.path.insert(0, str(model_root.resolve()))
    corr_module = importlib.import_module("models.corr")
    from run_steflow_stage2_pilot import pure_torch_spatial_correlation
    corr_module.spatial_correlation_sample = pure_torch_spatial_correlation
    import models

    checkpoint_path = checkpoint_path.resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint_state(payload, generator, dt)
    model = models.steflow().to(device)
    model.load_state_dict(state, strict=True)
    loaded_hash = state_sha256(model)
    expected_hash = payload.get("final_state_sha256")
    if expected_hash and loaded_hash != expected_hash:
        raise RuntimeError(f"checkpoint model-state hash mismatch for {checkpoint_path}")
    model.eval()

    validation_dir = data_root / generator / "validation"
    episodes = sorted(validation_dir.glob("*.npz"))
    if not episodes:
        raise FileNotFoundError(f"no synthetic validation rollout files in {validation_dir}")
    flow_key = f"flow_gt_dt{dt}"
    valid_key = f"flow_valid_dt{dt}"
    event_key = f"events_dt{dt}"
    detail = []
    restored_probe = None
    peak_allocated = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for episode_path in episodes:
            episode = read_episode(episode_path)
            if flow_key not in episode or valid_key not in episode or event_key not in episode:
                raise KeyError(f"{episode_path} missing {event_key}/{flow_key}/{valid_key}")
            events = episode[event_key]
            labels = episode[flow_key]
            masks = episode[valid_key]
            if events.ndim != 5 or events.shape[1:] != (2, 256, 256, 5 * dt):
                raise ValueError(f"{episode_path}: malformed {event_key} shape {events.shape}")
            if labels.ndim != 4 or labels.shape[1] != 2:
                raise ValueError(f"{episode_path}: malformed {flow_key} shape {labels.shape}")
            if len(events) != len(labels) or masks.shape != (len(labels), *labels.shape[-2:]):
                raise ValueError(f"{episode_path}: event/GT/mask interval count mismatch")
            if len(labels) == 0:
                raise ValueError(f"{episode_path}: empty synthetic validation rollout")
            for index in range(len(labels)):
                tensor = event_input(episode, index, dt, device)
                output = model(tensor, 256, use_dt4=(dt == 4), flow_seq=False)
                prediction = model_output_to_flow_hw(output)
                epe, valid_pixels = compute_epe(
                    prediction, labels[index], masks[index],
                    prediction_unit=FLOW_UNIT, gt_unit=FLOW_UNIT,
                )
                if restored_probe is None:
                    restored_probe = {
                        "episode": episode_path.name,
                        "interval_index": index,
                        "input": tensor.detach().cpu(),
                        "prediction": prediction.copy(),
                        "epe": epe,
                    }
                detail.append({
                    "rollout_id": episode_path.stem,
                    "interval_index": index,
                    "prediction_shape": list(prediction.shape),
                    "gt_shape": list(labels[index].shape),
                    "valid_mask_shape": list(masks[index].shape),
                    "valid_pixels": valid_pixels,
                    "epe_px_per_interval": epe,
                    "epe_unit": FLOW_UNIT,
                })
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(device))

    # Independently instantiate and restore the same checkpoint. Verify both
    # full-vector output and strict EPE on the exact first rollout/index.
    restored = models.steflow().to(device)
    restored.load_state_dict(state, strict=True)
    restored.eval()
    restored_state_hash = state_sha256(restored)
    if restored_state_hash != loaded_hash:
        raise RuntimeError("independent checkpoint restore state hash differs from original load")
    with torch.inference_mode():
        restored_output = restored(
            restored_probe["input"].to(device), 256,
            use_dt4=(dt == 4), flow_seq=False,
        )
        restored_flow = model_output_to_flow_hw(restored_output)
    if restored_flow.shape != restored_probe["prediction"].shape:
        raise RuntimeError("restored prediction tensor shape differs from original")
    restore_delta = np.abs(restored_flow.astype(np.float64) - restored_probe["prediction"].astype(np.float64))
    if not np.array_equal(restored_flow, restored_probe["prediction"]):
        raise RuntimeError(
            "restored checkpoint full 2-D prediction differs under deterministic inference; "
            f"max_abs={float(restore_delta.max())}, mean_abs={float(restore_delta.mean())}"
        )
    probe_episode = read_episode(validation_dir / restored_probe["episode"])
    probe_gt = probe_episode[flow_key][restored_probe["interval_index"]]
    probe_mask = probe_episode[valid_key][restored_probe["interval_index"]]
    restored_epe, restored_valid_pixels = compute_epe(
        restored_flow, probe_gt, probe_mask,
        prediction_unit=FLOW_UNIT, gt_unit=FLOW_UNIT,
    )
    if restored_valid_pixels != detail[0]["valid_pixels"] or not np.isclose(
        restored_epe, restored_probe["epe"], rtol=1e-6, atol=1e-6
    ):
        raise RuntimeError("restored checkpoint EPE/mask result differs from scored probe")
    return {
        "generator": generator,
        "dt": dt,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_saved_state_sha256": expected_hash,
        "restored_state_sha256": restored_state_hash,
        "state_hash_equal": loaded_hash == expected_hash,
        "validation_rollouts": [path.stem for path in episodes],
        "validation_interval_count": len(detail),
        "epe_unit": FLOW_UNIT,
        "corrected_synthetic_validation_epe_mean": float(np.mean([row["epe_px_per_interval"] for row in detail])),
        "corrected_synthetic_validation_epe_median": float(np.median([row["epe_px_per_interval"] for row in detail])),
        "valid_pixel_count_total": int(sum(row["valid_pixels"] for row in detail)),
        "checkpoint_restore_probe": {
            "rollout_id": Path(restored_probe["episode"]).stem,
            "interval_index": restored_probe["interval_index"],
            "full_two_channel_prediction_equal": True,
            "prediction_max_abs_difference": float(restore_delta.max(initial=0.0)),
            "restored_epe_px_per_interval": restored_epe,
            "valid_pixels": restored_valid_pixels,
        },
        "elapsed_seconds": elapsed,
        "peak_allocated_gpu_bytes": peak_allocated,
        "intervals": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pinned STE-FlowNet inference")
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    source_verification = verify_pinned_source(
        args.model_root, "STE-FlowNet", PINNED_STEFLOW_COMMIT, SOURCE_BUNDLE_MANIFEST
    )
    results = []
    for dt in (1, 4):
        for generator in ("eventgenesis", "v2e"):
            checkpoint = args.checkpoint_dir / f"{generator}_dt{dt}_final.pt"
            results.append(evaluate_one(
                checkpoint, args.data_root, args.model_root,
                generator, dt, device,
            ))
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema_version": 1,
        "status": "pass",
        "scope": "corrected re-score of existing Stage 2 synthetic validation only; no retraining",
        "old_stage2_epe": "INVALID: runner extracted only horizontal flow u and NumPy broadcast it across GT channels",
        "prediction_target_mask_contract": "same [2,H,W], [2,H,W], [H,W]; px_per_interval; finite values; nonempty binary mask",
        "data_root": str(args.data_root.resolve()),
        "model_root": str(args.model_root.resolve()),
        "steflow_source_commit": PINNED_STEFLOW_COMMIT,
        "steflow_source_verification": source_verification,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deterministic_inference": {
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "results": results,
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
