#!/usr/bin/env python3
"""Evaluate a trained E2VID checkpoint on a paired synthetic artifact only.

This fail-closed metric plumbing entry requires an exported Stage-5 synthetic
fixture and full-resolution float outputs from ``infer_hqf_e2vid_candidate``.
It cannot score HQF or any other real dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def load_lpips_alex(model_root: Path, device: torch.device):
    model_root = model_root.resolve()
    sys.path.insert(0, str(model_root))
    perceptual_root = model_root / "PerceptualSimilarity"
    sys.path.insert(0, str(perceptual_root))
    try:
        import IPython  # noqa: F401
    except ImportError:
        # The pinned metric imports IPython only for an optional interactive
        # debug branch. Keep the inference/evaluation process non-interactive.
        import types
        stub = types.ModuleType("IPython")
        stub.embed = lambda *unused_args, **unused_kwargs: None
        sys.modules["IPython"] = stub
    # PerceptualSimilarity v0.1 imports the pre-0.18 scikit-image name even
    # though the current environment exposes the same implementation as
    # skimage.metrics.structural_similarity. Provide that source-compatible
    # alias before importing the pinned package; LPIPS weights/forward path
    # remain the pinned v0.1 implementation.
    import skimage.measure
    import skimage.metrics
    if not hasattr(skimage.measure, "compare_ssim"):
        skimage.measure.compare_ssim = skimage.metrics.structural_similarity
    # Stage-5's model.loss.perceptual_loss delegates to this exact pinned
    # PerceptualSimilarity class. Import it directly so EVREAL's own `utils`
    # package (loaded for MemMapDataset) cannot shadow event_cnn_minimal.utils.
    perceptual_models = importlib.import_module("PerceptualSimilarity.models")
    metric = perceptual_models.PerceptualLoss(net="alex", use_gpu=(device.type == "cuda"))
    metric = metric.to(device)
    metric.eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    source_path = Path(inspect.getsourcefile(perceptual_models.PerceptualLoss)).resolve()
    weights_path = perceptual_root / "models" / "weights" / "v0.1" / "alex.pth"
    if not weights_path.is_file():
        raise FileNotFoundError(f"pinned PerceptualSimilarity v0.1 AlexNet weights are missing: {weights_path}")
    backbone_weights_path = Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth"
    if not backbone_weights_path.is_file():
        raise FileNotFoundError(f"torchvision's pinned AlexNet ImageNet weights are missing: {backbone_weights_path}")
    return metric, source_path, weights_path.resolve(), backbone_weights_path.resolve()


def validate_pair(prediction: np.ndarray, target_u8: np.ndarray, expected_hw: tuple[int, int]) -> np.ndarray:
    pred = np.asarray(prediction)
    target = np.asarray(target_u8)
    if pred.dtype != np.float32 or pred.shape != expected_hw:
        raise ValueError(f"prediction must be float32 {expected_hw}, got {pred.dtype} {pred.shape}")
    if target.dtype != np.uint8 or target.shape != expected_hw:
        raise ValueError(f"target must be uint8 {expected_hw}, got {target.dtype} {target.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("prediction contains NaN or Inf")
    if not (np.all(pred >= 0.0) and np.all(pred <= 1.0)):
        raise ValueError("prediction lies outside the registered [0,1] evaluation domain")
    target01 = target.astype(np.float32) / np.float32(255.0)
    if not np.isfinite(target01).all() or np.any(target01 < 0.0) or np.any(target01 > 1.0):
        raise ValueError("target conversion produced invalid [0,1] values")
    return target01


def classical_metrics(prediction: np.ndarray, target01: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target01 = np.asarray(target01, dtype=np.float32)
    if prediction.shape != target01.shape or prediction.ndim != 2:
        raise ValueError(f"MSE/PSNR/SSIM require matching [H,W] arrays, got {prediction.shape}/{target01.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target01).all():
        raise ValueError("MSE/PSNR/SSIM input contains NaN or Inf")
    if np.any(prediction < 0) or np.any(prediction > 1) or np.any(target01 < 0) or np.any(target01 > 1):
        raise ValueError("MSE/PSNR/SSIM input lies outside [0,1]")
    delta = prediction - target01
    mse = float(np.mean(delta * delta, dtype=np.float64))
    psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError("the pinned synthetic/real evaluation environment must provide scikit-image") from exc
    ssim = float(structural_similarity(target01, prediction, data_range=1.0))
    return {"mse": mse, "psnr_db": psnr, "ssim": ssim}


def lpips_input(image01: np.ndarray, device: torch.device) -> torch.Tensor:
    array = np.asarray(image01, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
        raise ValueError("LPIPS input must be a finite grayscale [H,W] float image in [0,1]")
    tensor = torch.from_numpy(array[None, None]).to(device=device)
    return tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, required=True,
                        help="directory containing synthetic_fixture_manifest.json and one MemMap sequence")
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    fixture_root = args.fixture_root.resolve()
    sequence_root = args.sequence_root.resolve()
    inference_root = args.inference_root.resolve()
    output = args.output.resolve()
    fixture_manifest_path = fixture_root / "synthetic_fixture_manifest.json"
    inference_manifest_path = inference_root / "inference_manifest.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite metric result: {output}")
    if not fixture_manifest_path.is_file() or not inference_manifest_path.is_file():
        raise FileNotFoundError("fixture or inference manifest is missing")
    fixture = json.loads(fixture_manifest_path.read_text())
    inference = json.loads(inference_manifest_path.read_text())
    if (fixture.get("schema") != "stage5_synthetic_e2vid_memmap_fixture_v1"
            or fixture.get("status") != "synthetic_inference_and_metric_plumbing_only"
            or fixture.get("real_data_used") is not False
            or fixture.get("real_scoring_enabled") is not False
            or fixture.get("checkpoint_selection_allowed") is not False):
        raise RuntimeError("input fixture is not a registered synthetic-only plumbing artifact")
    if (inference.get("status") != "synthetic_fixture_event_only_inference_completed_no_GT_scoring"
            or inference.get("self_test", {}).get("is_synthetic_fixture") is not True
            or inference.get("model", {}).get("real_GT_scoring_performed") is not False
            or inference.get("model", {}).get("reference_frame_pixels_used_as_input") is not False):
        raise RuntimeError("inference manifest is not a synthetic event-only candidate run")
    if sequence_root.parent != fixture_root:
        raise RuntimeError("synthetic sequence must be a direct child of its registered fixture root")
    if Path(inference["sequence"]["path"]).resolve() != sequence_root:
        raise RuntimeError("inference sequence path differs from the supplied synthetic artifact")
    if fixture.get("initial_anchor_excluded_from_model_targets") is not True:
        raise RuntimeError("synthetic fixture must register the sequence-start anchor exclusion")
    if fixture["target_frame_count"] != int(fixture["image_endpoint_count_including_anchor"]) - 1:
        raise RuntimeError("synthetic target count must exclude exactly one initial boundary image")
    if inference["sequence"]["num_samples_processed"] != fixture["target_frame_count"]:
        raise RuntimeError("inference sample count differs from the inferable synthetic target count")
    if inference["registered_inference"].get("save_every_n_samples") != 1:
        raise RuntimeError("full synthetic metric run must save every endpoint")
    if inference["registered_inference"].get("save_float_arrays") is not True:
        raise RuntimeError("full synthetic metric run must save float32 outputs")

    for name, record in fixture.get("files", {}).items():
        path = Path(record.get("path", "")).resolve()
        if path.parent != sequence_root or path.name != name or not path.is_file():
            raise RuntimeError(f"synthetic fixture file path mismatch or missing file: {name}")
        if sha256(path) != record.get("sha256"):
            raise RuntimeError(f"synthetic fixture file hash mismatch: {name}")
    checkpoint_path = Path(inference["model"]["checkpoint_path"]).resolve()
    if not checkpoint_path.is_file() or sha256(checkpoint_path) != inference["model"].get("checkpoint_sha256"):
        raise RuntimeError("candidate checkpoint path/hash differs from the event-only inference manifest")
    bank_path = Path(fixture["physical_bank_path"]).resolve()
    if (not bank_path.is_file() or bank_path.stat().st_size != fixture.get("physical_bank_size_bytes")
            or sha256(bank_path) != fixture.get("physical_bank_sha256")):
        raise RuntimeError("shared physical bank path/size/hash differs from the fixture manifest")

    target_path = sequence_root / "images.npy"
    target_ts_path = sequence_root / "images_ts.npy"
    target_indices_path = sequence_root / "image_event_indices.npy"
    targets = np.load(target_path, mmap_mode="r")
    target_times = np.load(target_ts_path, mmap_mode="r").reshape(-1)
    target_indices = np.load(target_indices_path, mmap_mode="r").reshape(-1)
    event_times = np.load(sequence_root / "events_ts.npy", mmap_mode="r").reshape(-1)
    n, height, width, channels = targets.shape
    if (channels != 1 or n != fixture["image_endpoint_count_including_anchor"]
            or fixture["target_frame_count"] != n - 1 or len(target_times) != n or len(target_indices) != n):
        raise RuntimeError("synthetic target arrays have inconsistent lengths/shapes")
    if not np.isfinite(target_times).all() or np.any(np.diff(target_times) <= 0):
        raise RuntimeError("synthetic target times must be finite and strictly increasing")
    if not np.isfinite(event_times).all() or np.any(np.diff(event_times) < 0):
        raise RuntimeError("synthetic event times must be finite and monotonically non-decreasing")
    independently_counted_offsets = np.searchsorted(event_times, target_times, side="right").astype(np.int64)
    if not np.array_equal(target_indices.astype(np.int64), independently_counted_offsets):
        raise RuntimeError("fixture event offsets differ from independent timestamp recount")
    if independently_counted_offsets[0] != 0:
        raise RuntimeError("the t0 recurrent boundary must not contain events from before/at its timestamp")

    inference_rows = inference["registered_inference"]["saved_images"]
    expected_sample_indices = list(range(1, n))
    if [row["sample_index"] for row in inference_rows] != expected_sample_indices:
        raise RuntimeError("saved inference outputs do not cover every inferable interval endpoint exactly once")
    if [row["sample_index"] for row in inference["sequence"]["samples"]] != expected_sample_indices:
        raise RuntimeError("inference sample manifest is incomplete or reordered")
    inference_samples = inference["sequence"]["samples"]
    for row in inference_samples:
        index = int(row["sample_index"])
        expected_start = int(independently_counted_offsets[index - 1])
        expected_end = int(independently_counted_offsets[index])
        if (row["target_timestamp_s"] != float(target_times[index])
                or row["event_start_index_inclusive"] != expected_start
                or row["event_end_index_exclusive"] != expected_end
                or row["event_count"] != expected_end - expected_start):
            raise RuntimeError(f"EVREAL event window/target endpoint mismatch at synthetic sample {index}")
        interval_times = event_times[expected_start:expected_end]
        if interval_times.size and (np.any(interval_times <= target_times[index - 1])
                                    or np.any(interval_times > target_times[index])):
            raise RuntimeError(f"event timestamp fell outside (previous,target] at sample {index}")
    if inference["sequence"].get("initial_anchor_endpoint_excluded_from_model_and_metrics") is not True:
        raise RuntimeError("event-only inference did not declare the zero-event sequence anchor exclusion")

    device = torch.device(args.device)
    metric, metric_source_path, metric_weights_path, backbone_weights_path = load_lpips_alex(args.model_root, device)
    metric_source_sha = sha256(metric_source_path)
    rows = []
    with torch.inference_mode():
        for saved in inference_rows:
            index = int(saved["sample_index"])
            prediction_path = Path(saved.get("float_array_path", "")).resolve()
            if not prediction_path.is_file() or sha256(prediction_path) != saved.get("float_array_sha256"):
                raise RuntimeError(f"prediction artifact missing/hash mismatch at sample {index}")
            prediction = np.load(prediction_path, allow_pickle=False)
            target = validate_pair(prediction, np.asarray(targets[index, :, :, 0]), (height, width))
            classical = classical_metrics(prediction, target)
            mse, psnr, ssim = classical["mse"], classical["psnr_db"], classical["ssim"]
            pred_rgb = lpips_input(prediction, device)
            target_rgb = lpips_input(target, device)
            # `lpips_input` already maps grayscale [0,1] to the registered
            # three-channel [-1,1] domain. Avoid the wrapper's default second
            # normalization and invoke its pinned metric module explicitly.
            lpips_value = metric.forward(pred_rgb, target_rgb, normalize=False)
            lpips = float(lpips_value.detach().float().mean().cpu())
            values = {"mse": mse, "psnr_db": psnr, "ssim": ssim, "lpips_alex": lpips}
            if not all(math.isfinite(value) for key, value in values.items() if key != "psnr_db"):
                raise RuntimeError(f"non-finite synthetic reconstruction metric at sample {index}: {values}")
            if not (math.isfinite(psnr) or math.isinf(psnr)):
                raise RuntimeError(f"invalid PSNR at sample {index}: {psnr}")
            rows.append({
                "sample_index": index,
                "target_timestamp_s": float(target_times[index]),
                "event_end_offset_exclusive": int(target_indices[index]),
                "prediction_sha256": saved["float_array_sha256"],
                "target_uint8_sha256": sha256_array(np.asarray(targets[index])),
                "prediction_range": [float(prediction.min()), float(prediction.max())],
                "target_range": [float(target.min()), float(target.max())],
                "metrics": {**values, "psnr_db": psnr if math.isfinite(psnr) else None,
                            "psnr_exact_match": bool(math.isinf(psnr))},
            })

    summary = {
        name: float(np.mean([row["metrics"][name] for row in rows]))
        for name in ("mse", "ssim", "lpips_alex")
    }
    finite_psnr = [row["metrics"]["psnr_db"] for row in rows if row["metrics"]["psnr_db"] is not None]
    summary["psnr_db"] = float(np.mean(finite_psnr)) if finite_psnr else None
    summary["psnr_exact_match_frames"] = sum(row["metrics"]["psnr_exact_match"] for row in rows)
    result = {
        "schema": "stage5_synthetic_reconstruction_metric_result_v1",
        "status": "synthetic_plumbing_pass_not_model_quality_or_generator_result",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "real_data_used": False,
        "real_scoring_enabled": False,
        "checkpoint_selection_allowed": False,
        "scope": "one paired synthetic rollout; every inferable event endpoint after the initial boundary; metric/shape/range/hash plumbing only",
        "fixture_manifest": {"path": str(fixture_manifest_path), "sha256": sha256(fixture_manifest_path)},
        "inference_manifest": {"path": str(inference_manifest_path), "sha256": sha256(inference_manifest_path)},
        "checkpoint": inference["model"]["checkpoint_path"],
        "checkpoint_sha256": inference["model"]["checkpoint_sha256"],
        "evaluation_script": {"path": str(Path(__file__).resolve()),
                              "sha256": sha256(Path(__file__).resolve())},
        "lpips_source": {"path": str(metric_source_path), "sha256": metric_source_sha,
                         "weights_path": str(metric_weights_path), "weights_sha256": sha256(metric_weights_path),
                         "backbone_weights_path": str(backbone_weights_path),
                         "backbone_weights_sha256": sha256(backbone_weights_path),
                         "model": "pinned PerceptualSimilarity v0.1 AlexNet with torchvision ImageNet backbone",
                         "input": "grayscale repeated to RGB; [0,1] mapped linearly to [-1,1]"},
        "other_metric_definitions": {
            "mse": "mean squared full-frame error on float32 grayscale [0,1]",
            "psnr_db": "10 log10(1/MSE), data_range=1; +inf for an exact match",
            "ssim": "skimage structural_similarity, grayscale float32, data_range=1",
        },
        "aggregation": "arithmetic mean over all registered inferable synthetic endpoints after the boundary image; no frame- or output-dependent normalization",
        "image_endpoint_count_including_anchor": n,
        "target_frame_count": len(rows),
        "initial_anchor_excluded_from_model_targets": True,
        "metric_mean": summary,
        "per_frame": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "frames": len(rows),
                      "metric_mean": summary, "output": str(output), "sha256": sha256(output)},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
