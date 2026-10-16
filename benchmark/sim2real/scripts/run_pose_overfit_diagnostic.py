#!/usr/bin/env python3
"""Tiny one-sample pose diagnostic; not a benchmark or model-selection run."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch

from verify_pinned_source import verify_pinned_source


ROOT = Path(__file__).resolve().parents[3]
PINNED_POSE_COMMIT = "2c282619101f9f31666dd18b42393375bea1600f"
FIXTURE_SCRIPT = Path(__file__).with_name("audit_pose_event_windows.py")
SOURCE_BUNDLE_MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "downstream_source_bundle.json"


def import_fixture_module():
    spec = importlib.util.spec_from_file_location("pose_event_window_audit", FIXTURE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot import pose fixture helper {FIXTURE_SCRIPT}")
    spec.loader.exec_module(module)
    return module


def decode_error(runner, output, cell, gt, scale):
    prediction, confidence, actual_cell = runner.decode_normalized_keypoints(output, cell)
    error = float(np.linalg.norm((prediction[1:] - gt[1:]) * scale, axis=1).mean())
    return {
        "cell_yx": list(actual_cell),
        "confidence": confidence,
        "keypoint_mean_epe_px": error,
        "prediction_xy_min_max": [float(prediction.min()), float(prediction.max())],
        "prediction_outside_unit_square_count": int(
            np.count_nonzero((prediction < 0.0) | (prediction >= 1.0))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pose-source", type=Path,
        default=ROOT.parent.parent / "external_sources" / "SingleShotPose",
    )
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2718)
    args = parser.parse_args()
    if args.updates < 1:
        raise ValueError("--updates must be positive")
    source = args.pose_source.resolve()
    source_verification = verify_pinned_source(
        source, "SingleShotPose", PINNED_POSE_COMMIT, SOURCE_BUNDLE_MANIFEST
    )
    if not torch.cuda.is_available():
        raise RuntimeError("the tiny overfit diagnostic requires CUDA")

    # Python 3.14 removed imghdr; pinned utils imports it only for unrelated
    # image-directory enumeration, so provide a no-op compatibility module.
    if "imghdr" not in sys.modules:
        sys.modules["imghdr"] = types.SimpleNamespace(what=lambda *args, **kwargs: None)
    sys.path.insert(0, str(source))
    from darknet import Darknet
    from region_loss import RegionLoss, build_targets

    fixture = import_fixture_module()
    runner = fixture.POSE
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    with tempfile.TemporaryDirectory(prefix="pose-overfit-diagnostic-") as temp:
        temp_root = Path(temp)
        shared = temp_root / "shared.h5"
        v2e_root = temp_root / "v2e"
        fixture.write_fixture(shared, v2e_root)
        sample = runner.collect_samples(
            "eventgenesis", shared, v2e_root, [0.1, 0.1, 0.1],
            "train", windows_per_rollout=2, count_scale=8,
        )[0]
        gt = sample["uv_normalized"]
        if gt.shape != (runner.NUM_KEYPOINTS, 2) or not np.isfinite(gt).all():
            raise RuntimeError("fixture generated malformed pose targets")
        if np.any(gt < 0.0) or np.any(gt >= 1.0):
            raise RuntimeError("fixture target coordinates are outside [0,1)")

        model = Darknet(str(source / "cfg" / "yolo-pose.cfg")).to(device)
        loss_layer = RegionLoss(
            num_keypoints=runner.NUM_KEYPOINTS, num_classes=runner.NUM_CLASSES,
            anchors=[], num_anchors=1, pretrain_num_epochs=15,
        )
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005,
        )
        inputs = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)
        target = torch.from_numpy(sample["label"]).unsqueeze(0)
        losses = []
        model.train()
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.updates):
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            loss, components = runner.patched_official_region_loss(
                output, target, loss_layer, 1, build_targets,
            )
            if components["nGT"] != 1 or not torch.isfinite(loss):
                raise RuntimeError("tiny diagnostic encountered invalid target/loss")
            loss.backward()
            optimizer.step()
            loss_layer.seen += 1
            losses.append(float(loss.detach().cpu()))
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        output_grid_shape = (int(output.shape[-2]), int(output.shape[-1]))
        centroid_cell = (
            min(output_grid_shape[0] - 1, int(np.floor(float(gt[0, 1]) * output_grid_shape[0]))),
            min(output_grid_shape[1] - 1, int(np.floor(float(gt[0, 0]) * output_grid_shape[1]))),
        )
        scale = np.asarray([runner.NETWORK_SIZE, runner.NETWORK_SIZE], dtype=np.float32)

        model.eval()
        with torch.no_grad():
            eval_output = model(inputs)
        eval_argmax = decode_error(runner, eval_output, None, gt, scale)
        eval_gt_cell = decode_error(runner, eval_output, centroid_cell, gt, scale)

        # This pass uses current-batch BN statistics and updates running stats;
        # the following evaluation pass records the resulting state effect.
        model.train()
        with torch.no_grad():
            train_batch_output = model(inputs)
        train_batch_gt_cell = decode_error(runner, train_batch_output, centroid_cell, gt, scale)
        model.eval()
        with torch.no_grad():
            post_bn_eval_output = model(inputs)
        post_bn_eval_gt_cell = decode_error(runner, post_bn_eval_output, centroid_cell, gt, scale)

        print(json.dumps({
            "status": "completed",
            "scope": "one corrected synthetic event window; diagnostic only; no checkpoint selection",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "source_commit": PINNED_POSE_COMMIT,
            "source_verification": source_verification,
            "seed": args.seed,
            "updates": args.updates,
            "sample_id": sample["sample_id"],
            "rollout_id": sample["rollout_id"],
            "event_start_s": sample["event_start_s"],
            "event_end_s": sample["event_end_s"],
            "label_time_s": sample["label_time_s"],
            "event_count": sample["event_count"],
            "target_xy_min_max": [float(gt.min()), float(gt.max())],
            "target_xy_finite": bool(np.isfinite(gt).all()),
            "output_shape_b_ch_h_w": list(output.shape),
            "output_grid_shape_h_w": list(output_grid_shape),
            "gt_centroid_cell_yx": list(centroid_cell),
            "confidence_loss_included": False,
            "batchnorm2d_count": sum(1 for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm2d)),
            "loss_first_last_min": [losses[0], losses[-1], min(losses)],
            "train_wall_seconds": elapsed,
            "peak_allocated_gpu_mib": torch.cuda.max_memory_allocated(device) / 1024.0 ** 2,
            "eval_mode_confidence_argmax_decode": eval_argmax,
            "eval_mode_gt_centroid_cell_decode": eval_gt_cell,
            "train_mode_gt_centroid_cell_decode": train_batch_gt_cell,
            "eval_after_train_mode_bn_update_gt_centroid_cell_decode": post_bn_eval_gt_cell,
            "interpretation_limit": "single tiny fixture; state-mode gap is diagnostic evidence, not a causal estimate for Stage 2 aggregate errors",
        }, indent=2))


if __name__ == "__main__":
    main()
