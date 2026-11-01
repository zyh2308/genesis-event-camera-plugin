#!/usr/bin/env python3
"""Compare the actual STE-FlowNet photometric objective on fixed GT/zero/reverse flow."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def flow_pyramid(full_flow: torch.Tensor, shapes: list[tuple[int, int]]) -> tuple[torch.Tensor, ...]:
    if full_flow.ndim != 4 or full_flow.shape[0:2] != (1, 2):
        raise ValueError(f"full flow must be [1,2,H,W], got {tuple(full_flow.shape)}")
    full_h, full_w = full_flow.shape[-2:]
    result = []
    for height, width in shapes:
        scaled = F.interpolate(full_flow, size=(height, width), mode="bilinear", align_corners=False)
        scaled = scaled.clone()
        scaled[:, 0] *= width / full_w
        scaled[:, 1] *= height / full_h
        result.append(scaled)
    return tuple(result)


def inbounds(flow: torch.Tensor) -> torch.Tensor:
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must be [B,2,H,W], got {tuple(flow.shape)}")
    _, _, height, width = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    x = xx[None] + flow[:, 0]
    y = yy[None] + flow[:, 1]
    return (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)


def texture_mask(previous: np.ndarray, following: np.ndarray, threshold: float) -> np.ndarray:
    magnitudes = []
    for image in (previous, following):
        dy, dx = np.gradient(image.astype(np.float32, copy=False))
        magnitudes.append(np.hypot(dx, dy))
    return (magnitudes[0] > threshold) & (magnitudes[1] > threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    run_root = args.run_root.resolve()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output_root}")
    sys.path[:0] = [str(repo / "benchmark/sim2real/scripts"), str(repo / "benchmark/sim2real/compat")]
    from run_steflow_stage2_pilot import objective_dt1, pure_torch_spatial_correlation, verify_correlation
    from steflow_align_corners import patch_ste
    from train_stage7_replicad_selfsup_diagnostic import load_bank, padded_tensors
    from verify_pinned_source import verify_pinned_source

    config = json.loads(config_path.read_text())
    if config.get("registered_before_execution") is not True or config.get("real_scoring_enabled") is not False:
        raise RuntimeError("diagnostic requires a preregistered config with real scoring disabled")
    script_path = Path(__file__).resolve()
    if sha256(script_path) != config["code"]["script_sha256"]:
        raise RuntimeError("executed diagnostic script differs from preregistered SHA256")
    for relative, expected in config["code"]["pinned_helpers"].items():
        path = repo / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"pinned helper hash mismatch: {relative}")

    manifest_path = run_root / "flow_bank_retry01/training_bank_manifest.json"
    manifest_hash = sha256(manifest_path)
    if manifest_hash != config["data"]["manifest_sha256"]:
        raise RuntimeError("shared bank manifest hash changed")
    bank = load_bank(manifest_path)
    expected_rollouts = config["data"]["validation_rollouts"]
    observed_rollouts = [row["rollout_id"] for row in bank["validation"]]
    if observed_rollouts != expected_rollouts:
        raise RuntimeError(f"validation rollouts changed: {observed_rollouts}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this registered run requires the assigned CUDA device")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False

    model_root = args.model_root.resolve()
    source = verify_pinned_source(
        model_root, "STE-FlowNet", config["model"]["source_commit"],
        repo / "benchmark/sim2real/manifests/downstream_source_bundle.json",
    )
    # The external source bundle is intentionally not installed as a package.
    # Put its verified root first so imports cannot resolve to a stale checkout
    # elsewhere on PYTHONPATH.
    sys.path.insert(0, str(model_root))
    import models
    import multiscaleloss
    if Path(models.__file__).resolve().parent != (model_root / "models").resolve():
        raise RuntimeError(f"STE-FlowNet models imported from unexpected path: {models.__file__}")
    if Path(multiscaleloss.__file__).resolve() != (model_root / "multiscaleloss.py").resolve():
        raise RuntimeError(
            f"STE-FlowNet multiscaleloss imported from unexpected path: {multiscaleloss.__file__}"
        )
    corr_module = importlib.import_module("models.corr")
    corr_module.spatial_correlation_sample = pure_torch_spatial_correlation
    correlation_check = verify_correlation(device)
    patch_record = patch_ste(models, multiscaleloss, model_root)

    # A single actual model forward identifies the four exact scales expected by
    # the pinned loss. Its predictions/weights are not used in the oracle result.
    first = bank["validation"][0]
    event0 = first["arms"]["eventgenesis"]["events"]
    event_tensor, prev, next_ = padded_tensors(event0, first["gray"]["prev"], first["gray"]["next"], device)
    torch.manual_seed(0)
    model = models.steflow().to(device)
    model.train()
    with torch.no_grad():
        model_outputs = model(event_tensor, 352, use_dt4=False, flow_seq=False)
    if not isinstance(model_outputs, (tuple, list)) or len(model_outputs) != 4:
        raise RuntimeError("training-mode model must return the four scales consumed by the official objective")
    shapes = [(int(x.shape[-2]), int(x.shape[-1])) for x in model_outputs]
    if any(tuple(x.shape[:2]) != (1, 2) for x in model_outputs):
        raise RuntimeError("model scale flow contract is not [1,2,H,W]")
    del model_outputs, model, event_tensor, prev, next_
    torch.cuda.empty_cache()

    results = []
    texture_threshold = float(config["diagnostic_regions"]["luma_gradient_threshold"])
    motion_threshold = float(config["diagnostic_regions"]["gt_motion_threshold_px"])
    for row in bank["validation"]:
        eg = row["arms"]["eventgenesis"]
        v2 = row["arms"]["v2e"]
        if not (np.array_equal(eg["flow_gt"], v2["flow_gt"])
                and np.array_equal(eg["flow_valid"], v2["flow_valid"])
                and np.array_equal(eg["sample_ids"], v2["sample_ids"])):
            raise RuntimeError(f"paired physical labels differ for {row['rollout_id']}")
        if row["gray"]["radiance_sha256"] != config["data"]["radiance_sha256_by_rollout"][row["rollout_id"]]:
            raise RuntimeError(f"radiance source changed for {row['rollout_id']}")

        gt_np = np.asarray(eg["flow_gt"], dtype=np.float32)
        valid = np.asarray(eg["flow_valid"], dtype=bool)
        gray_prev_np = np.asarray(row["gray"]["prev"], dtype=np.float32)
        gray_next_np = np.asarray(row["gray"]["next"], dtype=np.float32)
        gt_mag = np.hypot(gt_np[0], gt_np[1])
        texture = texture_mask(gray_prev_np, gray_next_np, texture_threshold)
        motion_texture_valid = valid & (gt_mag >= motion_threshold) & texture
        pair_active = (np.any(eg["events"] != 0, axis=(0, 3))
                       | np.any(v2["events"] != 0, axis=(0, 3))) & valid

        full_flows_native = {
            "gt": torch.from_numpy(gt_np).unsqueeze(0).to(device),
            "zero": torch.zeros((1, 2, *gt_np.shape[1:]), dtype=torch.float32, device=device),
            "reverse_gt": torch.from_numpy(-gt_np).unsqueeze(0).to(device),
        }
        candidates_padded = {
            name: F.pad(flow, (16, 16, 56, 56)) for name, flow in full_flows_native.items()
        }
        common_inbounds = np.ones(gt_np.shape[1:], dtype=bool)
        inbounds_by_variant = {}
        for name, flow in candidates_padded.items():
            mask = inbounds(flow)[0, 56:296, 16:336].detach().cpu().numpy()
            inbounds_by_variant[name] = mask
            common_inbounds &= mask
        region = motion_texture_valid & common_inbounds

        event_inputs = {}
        for arm in ("eventgenesis", "v2e"):
            event, gray0, gray1 = padded_tensors(
                row["arms"][arm]["events"], gray_prev_np, gray_next_np, device
            )
            event_inputs[arm] = (event, gray0, gray1)

        variant_metrics = {}
        arm_photo = {"eventgenesis": {}, "v2e": {}}
        for variant, full_flow in full_flows_native.items():
            pyramid = flow_pyramid(candidates_padded[variant], shapes)
            for arm in ("eventgenesis", "v2e"):
                event, gray0, gray1 = event_inputs[arm]
                total, photo, smooth = objective_dt1(
                    pyramid, gray0, gray1, event, multiscaleloss
                )
                one_scale_photo = []
                if gray0.shape != (1, 352, 352) or gray1.shape != (1, 352, 352):
                    raise RuntimeError(
                        f"pinned photometric adapter expects [B,H,W] padded grayscale, got "
                        f"{tuple(gray0.shape)}/{tuple(gray1.shape)}"
                    )
                gray0_cpu = gray0.detach().cpu().contiguous()
                gray1_cpu = gray1.detach().cpu().contiguous()
                event_image = torch.sum(event, dim=4)
                for flow_scale in pyramid:
                    one_scale_photo.append(float(multiscaleloss.compute_photometric_loss(
                        gray0_cpu, gray1_cpu, event_image, (flow_scale,), weights=[1.0]
                    ).detach().cpu()))
                predicted_photo_mean = float(photo.detach().cpu())
                if not np.isclose(predicted_photo_mean, np.mean(one_scale_photo), rtol=1e-5, atol=1e-4):
                    raise RuntimeError("official scalar photometric loss does not match its per-scale decomposition")
                arm_photo[arm][variant] = predicted_photo_mean
                if arm == "eventgenesis":
                    # This auxiliary regional MAE is computed at native
                    # resolution; the official objective above evaluates its
                    # own image resize separately at each pyramid scale.
                    native_padded_flow = candidates_padded[variant]
                    warped = multiscaleloss.warp(gray1[:, None], native_padded_flow)
                    error = (warped[0, 0, 56:296, 16:336] - gray0[0, 56:296, 16:336]).abs()
                    direct_mae = float(error[region].mean().detach().cpu()) if region.any() else None
                    inbounds_full = float(inbounds_by_variant[variant].mean())
                    variant_metrics[variant] = {
                        "official_photometric_loss_mean_over_4_scales": predicted_photo_mean,
                        "official_photometric_loss_per_scale": one_scale_photo,
                        "official_smoothness_loss": float(smooth.detach().cpu()),
                        "official_total_loss_photo_plus_10x_smoothness": float(total.detach().cpu()),
                        "native_warp_inbounds_fraction": inbounds_full,
                        "shared_visible_texture_motion_mae": direct_mae,
                        "common_inbounds_pixels": int(common_inbounds.sum()),
                    }
                del total, photo, smooth, gray0_cpu, gray1_cpu, event_image
            del pyramid

        if not np.allclose(
            [arm_photo["eventgenesis"][v] for v in arm_photo["eventgenesis"]],
            [arm_photo["v2e"][v] for v in arm_photo["eventgenesis"]],
            rtol=0.0, atol=1e-6,
        ):
            raise RuntimeError("same flow/radiance produced different photometric losses by event generator")
        results.append({
            "rollout_id": row["rollout_id"],
            "sample_id": str(eg["sample_ids"]),
            "flow_unit": "pixels per 30ms endpoint interval",
            "label_sha256": row["shared_label_sha256"],
            "radiance_sha256": row["gray"]["radiance_sha256"],
            "flow_valid_pixels": int(valid.sum()),
            "common_event_active_valid_pixels": int(pair_active.sum()),
            "texture_pixels_both_endpoints": int(texture.sum()),
            "gt_motion_texture_valid_pixels": int(motion_texture_valid.sum()),
            "gt_motion_texture_common_inbounds_pixels": int(region.sum()),
            "gt_motion_texture_common_inbounds_fraction_of_valid": float(region.sum() / max(int(valid.sum()), 1)),
            "luma_range": [float(min(gray_prev_np.min(), gray_next_np.min())),
                           float(max(gray_prev_np.max(), gray_next_np.max()))],
            "luma_gradient_threshold": texture_threshold,
            "gt_motion_threshold_px": motion_threshold,
            "gt_flow_abs_max_px_per_30ms": float(np.max(np.abs(gt_np))),
            "candidate_inbounds_fraction_native": {k: float(v.mean()) for k, v in inbounds_by_variant.items()},
            "actual_objective": variant_metrics,
            "photometric_loss_by_generator_repeated_call": arm_photo,
            "event_effect_on_direct_oracle_loss": "none: the pinned photometric operator receives event images but does not use them; event affects this objective through the model-predicted flow only",
        })

    output_root.mkdir(parents=True)
    result = {
        "schema": "stage7_gt_zero_reverse_actual_photometric_oracle_v1",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_head": config["source"]["source_head"],
        "script_sha256": sha256(script_path),
        "config_sha256": sha256(config_path),
        "manifest_sha256": manifest_hash,
        "model_source_verification": source,
        "correlation_operator_verification": correlation_check,
        "warp_patch": patch_record,
        "gpu": {"name": torch.cuda.get_device_name(device), "index_visible_to_process": 0},
        "real_data_used": False,
        "real_scoring_enabled": False,
        "training_performed": False,
        "model_scale_shapes_hw": [list(x) for x in shapes],
        "objective_definition": config["objective"],
        "validation_rows": results,
        "interpretation_limits": config["limits"],
    }
    (output_root / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "rows": len(results), "gpu": result["gpu"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
