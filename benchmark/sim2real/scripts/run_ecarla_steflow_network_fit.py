#!/usr/bin/env python3
"""Bounded STE-FlowNet fit on a preregistered public synthetic-flow sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark/sim2real/scripts"))
from stage5_flow_supervised_pilot import (  # noqa: E402
    model_output_to_flow_hw,
    model_source,
    multiscale_supervised_epe,
    sha256,
    state_sha256,
)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_rows(path: Path, indices: list[int], crop: dict, split: str) -> list[dict]:
    rows = []
    top, left = int(crop["top"]), int(crop["left"])
    height, width = int(crop["height"]), int(crop["width"])
    with np.load(path, allow_pickle=False) as z:
        for i in indices:
            if split == "train":
                j = i - 100
            elif split == "validation":
                j = i - 600
            else:
                raise ValueError(f"unsupported fixed split: {split}")
            events = np.asarray(z[f"{split}_events"][j])
            flow = np.asarray(z[f"{split}_flow"][j])
            valid = np.asarray(z[f"{split}_valid"][j]).astype(bool)
            sample_id = f"flow_{i:04d}"
            rows.append({
                "index": i, "sample_id": sample_id,
                "events": events[:, top:top + height, left:left + width, :].astype(np.float32),
                "flow": flow[:, top:top + height, left:left + width].astype(np.float32),
                "valid": valid[top:top + height, left:left + width],
            })
    return rows


def tensors(row: dict, device: torch.device, variant: str = "normal"):
    events = row["events"]
    if variant == "zero_events":
        events = np.zeros_like(events)
    elif variant == "reverse_temporal_bins":
        events = events[..., ::-1].copy()
    elif variant != "normal":
        raise ValueError(variant)
    x = torch.from_numpy(events).unsqueeze(0).to(device)
    y = torch.from_numpy(row["flow"]).unsqueeze(0).to(device)
    mask = torch.from_numpy(row["valid"].astype(np.float32)).view(1, 1, *row["valid"].shape).to(device)
    return x, y, mask


def numpy_epe(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    error = np.sqrt(np.square(pred.astype(np.float64) - gt.astype(np.float64)).sum(axis=0))
    return float(error[mask].mean())


def metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, active: np.ndarray) -> dict:
    magnitude = np.hypot(gt[0], gt[1])
    moving = valid & (magnitude >= 0.5)
    pred_mag = np.hypot(pred[0], pred[1])
    cosine_mask = active & moving
    if np.any(cosine_mask):
        cosine = (pred[0] * gt[0] + pred[1] * gt[1]) / (pred_mag * magnitude + 1e-6)
        mean_cosine = float(cosine[cosine_mask].mean())
    else:
        mean_cosine = None
    active_mask = valid & active
    return {
        "valid_pixels": int(valid.sum()), "fixed_normal_event_active_pixels": int(active_mask.sum()),
        "fixed_normal_event_active_coverage_of_valid": float(active_mask.sum() / max(valid.sum(), 1)),
        "full_valid_epe_px_per_40ms": numpy_epe(pred, gt, valid),
        "fixed_active_epe_px_per_40ms": numpy_epe(pred, gt, active_mask),
        "full_valid_zero_flow_epe_px_per_40ms": numpy_epe(np.zeros_like(gt), gt, valid),
        "fixed_active_zero_flow_epe_px_per_40ms": numpy_epe(np.zeros_like(gt), gt, active_mask),
        "prediction_mean_magnitude_full_valid": float(pred_mag[valid].mean()) if valid.any() else None,
        "prediction_mean_magnitude_fixed_active": float(pred_mag[active_mask].mean()) if active_mask.any() else None,
        "mean_prediction_gt_direction_cosine_on_active_gt_motion": mean_cosine,
        "mean_gt_magnitude_on_active_gt_motion": float(magnitude[cosine_mask].mean()) if cosine_mask.any() else None,
    }


def evaluate(model, rows: list[dict], device: torch.device, out_path: Path | None = None) -> list[dict]:
    model.eval()
    results = []
    with torch.inference_mode():
        for row in rows:
            active = np.any(row["events"] != 0, axis=(0, 3)) & row["valid"]
            variants = {}
            for variant in ("normal", "zero_events", "reverse_temporal_bins"):
                x, _, _ = tensors(row, device, variant)
                flow = model_output_to_flow_hw(model(x, int(row["events"].shape[1]), use_dt4=False, flow_seq=False))
                if flow.shape != row["flow"].shape:
                    raise RuntimeError(f"prediction shape mismatch {flow.shape} != {row['flow'].shape}")
                # The fixed active mask always comes from the normal input, even for perturbations.
                variants[variant] = {
                    **metrics(flow, row["flow"], row["valid"], active),
                    "prediction_sha256": hashlib.sha256(np.ascontiguousarray(flow).view(np.uint8)).hexdigest(),
                }
            results.append({"sample_id": row["sample_id"], "flow_index": row["index"], "variants": variants})
    if out_path is not None:
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="corrected eCARLA audit-v2 NPZ")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    out.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this diagnostic requires CUDA")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    crop = config["crop"]
    train_rows = load_rows(args.data, [int(i) for i in config["train_flow_indices"]], crop, "train")
    validation_rows = load_rows(args.data, [int(i) for i in config["validation_flow_indices"]], crop, "validation")
    if not train_rows or not validation_rows:
        raise RuntimeError("fixed train and validation windows must both be non-empty")
    for row in train_rows + validation_rows:
        if row["events"].shape != (2, int(crop["height"]), int(crop["width"]), 5):
            raise RuntimeError(f"unexpected event shape for {row['sample_id']}: {row['events'].shape}")
        if row["flow"].shape != (2, int(crop["height"]), int(crop["width"])):
            raise RuntimeError(f"unexpected flow shape for {row['sample_id']}: {row['flow'].shape}")
        if not np.isfinite(row["events"]).all() or not np.isfinite(row["flow"]).all() or not row["valid"].any():
            raise RuntimeError(f"invalid diagnostic sample {row['sample_id']}")

    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    model, source_patch = model_source(args.model_root.resolve(), device)
    initial_sha = state_sha256(model)
    optimizer = torch.optim.Adam([
        {"params": model.bias_parameters(), "weight_decay": 0.0},
        {"params": model.weight_parameters(), "weight_decay": float(config["budget"]["weight_decay_weights"])},
    ], lr=float(config["budget"]["learning_rate"]), betas=tuple(config["budget"]["betas"]))
    result = {
        "schema": "ecarla_steflow_network_fit_v1", "status": "running", "started_utc": timestamp(),
        "config": config, "config_sha256": sha256(args.config.resolve()),
        "data_path": str(args.data.resolve()), "data_size_bytes": args.data.stat().st_size,
        "data_sha256": sha256(args.data.resolve()), "model_root": str(args.model_root.resolve()),
        "pinned_model_commit": "eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6",
        "initial_model_sha256": initial_sha,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device), "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "source_patch": source_patch,
        "train_windows": [r["sample_id"] for r in train_rows],
        "validation_windows": [r["sample_id"] for r in validation_rows],
        "real_data_used": False, "formal_checkpoint_selection": False,
    }
    (out / "status.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    train_logs = out / "updates.jsonl"
    order = list(range(len(train_rows)))
    initial_evals = evaluate(model, validation_rows, device, out / "validation_initial.json")
    with train_logs.open("w") as log:
        for update in range(int(config["budget"]["updates"])):
            row = train_rows[order[update % len(order)]]
            events, gt, valid = tensors(row, device)
            model.train()
            outputs = model(events, int(crop["height"]), use_dt4=False, flow_seq=False)
            loss, scale_losses = multiscale_supervised_epe(outputs, gt, valid)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at update {update + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise RuntimeError(f"invalid gradients at update {update + 1}")
            grad_norm = float(torch.sqrt(sum(g.detach().float().square().sum() for g in grads)).cpu())
            if grad_norm <= 0 or not np.isfinite(grad_norm):
                raise RuntimeError(f"zero or non-finite gradient norm at update {update + 1}")
            optimizer.step()
            torch.cuda.synchronize(device)
            record = {
                "update": update + 1, "sample_id": row["sample_id"], "loss": float(loss.detach().cpu()),
                "scale_losses": scale_losses, "gradient_l2": grad_norm,
                "parameter_sha256": state_sha256(model), "gpu_allocated_bytes": torch.cuda.memory_allocated(device),
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "time_utc": timestamp(),
            }
            log.write(json.dumps(record, sort_keys=True) + "\n"); log.flush()
    model.eval()
    checkpoint = out / "checkpoint.pt"
    state = {
        "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
        "updates": int(config["budget"]["updates"]), "seed": 0,
        "initial_model_sha256": initial_sha, "model_sha256": state_sha256(model),
        "data_sha256": result["data_sha256"], "config_sha256": result["config_sha256"],
        "real_data_used": False, "validation_used_for_selection": False,
    }
    torch.save(state, checkpoint)
    final_eval = evaluate(model, validation_rows, device, out / "validation_final.json")
    restored, _ = model_source(args.model_root.resolve(), device)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    restored.eval()
    restored_eval = evaluate(restored, validation_rows, device, out / "validation_restored.json")
    restore_equal = [
        [entry["variants"][k]["prediction_sha256"] for k in ("normal", "zero_events", "reverse_temporal_bins")]
        for entry in final_eval
    ] == [
        [entry["variants"][k]["prediction_sha256"] for k in ("normal", "zero_events", "reverse_temporal_bins")]
        for entry in restored_eval
    ]
    if not restore_equal or state_sha256(restored) != state["model_sha256"]:
        raise RuntimeError("checkpoint restore did not reproduce model state and ordinary eval predictions")
    result.update({
        "status": "complete", "finished_utc": timestamp(), "elapsed_seconds": time.monotonic() - started,
        "checkpoint": {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size,
                       "sha256": sha256(checkpoint), "model_sha256": state["model_sha256"],
                       "restore_state_equal": True, "ordinary_eval_outputs_equal": True},
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "initial_validation": initial_evals, "final_validation": final_eval,
        "summary": {
            variant: {
                "mean_fixed_active_epe": float(np.mean([
                    row["variants"][variant]["fixed_active_epe_px_per_40ms"] for row in final_eval
                    if row["variants"][variant]["fixed_active_epe_px_per_40ms"] is not None])),
                "mean_full_epe": float(np.mean([
                    row["variants"][variant]["full_valid_epe_px_per_40ms"] for row in final_eval
                    if row["variants"][variant]["full_valid_epe_px_per_40ms"] is not None])),
            } for variant in ("normal", "zero_events", "reverse_temporal_bins")
        },
        "validation_zero_flow_mean_epe": float(np.mean([
            row["variants"]["normal"]["fixed_active_zero_flow_epe_px_per_40ms"] for row in final_eval
            if row["variants"]["normal"]["fixed_active_zero_flow_epe_px_per_40ms"] is not None])),
        "real_scoring_enabled": False,
    })
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out / "status.json").write_text(json.dumps({"status": "complete", "summary": str(summary_path)}, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "elapsed_seconds", "gpu_peak_allocated_bytes", "summary", "validation_zero_flow_mean_epe", "checkpoint")}, indent=2))


if __name__ == "__main__":
    main()
