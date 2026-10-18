#!/usr/bin/env python3
"""Tiny synthetic-only probe of SingleShotPose eval BN and confidence behavior.

This is a mechanism check, not a benchmark or checkpoint-selection run. It uses
the public-shape timestamp fixture, trains on its synthetic event inputs only,
re-estimates BN buffers from the same train split, and performs all reported
predictions with ``model.eval()`` plus the normal confidence argmax decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import types
from pathlib import Path

import numpy as np
import torch
import cv2

from verify_pinned_source import verify_pinned_source


ROOT = Path(__file__).resolve().parents[3]
PINNED_POSE_COMMIT = "2c282619101f9f31666dd18b42393375bea1600f"
SOURCE_BUNDLE_MANIFEST = ROOT / "manifests" / "downstream_source_bundle.json"
POSE_RUNNER = Path(__file__).with_name("run_pose_stage2_pilot.py")
WINDOW_FIXTURE = Path(__file__).with_name("audit_pose_event_windows.py")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(parameter.detach().cpu().numpy()).view(np.uint8))
    return digest.hexdigest()


def eval_report(runner, model, samples, device) -> dict:
    model.eval()
    rows = []
    scale = np.asarray([runner.NETWORK_SIZE, runner.NETWORK_SIZE], dtype=np.float32)
    with torch.no_grad():
        for sample in samples:
            prediction, confidence, cell, grid_shape = runner.predict_normalized_keypoints(
                model, sample["image"], device
            )
            gt = sample["uv_normalized"]
            epe = float(np.linalg.norm((prediction[1:] - gt[1:]) * scale, axis=1).mean())
            pnp_metrics = {"pnp_succeeded": False}
            try:
                pnp_ok, rotation_vector, translation = cv2.solvePnP(
                    sample["keypoints_obj"][1:].astype(np.float64),
                    (prediction[1:] * scale).astype(np.float64),
                    sample["camera_matrix_416"].astype(np.float64),
                    np.zeros((4, 1), dtype=np.float64),
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            except cv2.error as exc:
                pnp_ok = False
                pnp_metrics["pnp_error"] = str(exc)
            pnp_metrics["pnp_succeeded"] = bool(pnp_ok)
            if pnp_ok:
                rotation, _ = cv2.Rodrigues(rotation_vector)
                gt_transform = sample["camera_from_object"]
                rotation_delta = rotation @ gt_transform[:3, :3].T
                cosine = np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
                pnp_metrics.update({
                    "rotation_error_deg": float(np.degrees(np.arccos(cosine))),
                    "translation_error_m": float(np.linalg.norm(
                        translation.reshape(3) - gt_transform[:3, 3]
                    )),
                    "translation_unit": "synthetic fixture world unit (metre)",
                    "input_contract": "ordinary eval-mode confidence-argmax corners, fixed synthetic CAD points, fixed synthetic K",
                })
            gt_center_cell = [
                min(grid_shape[0] - 1, int(float(gt[0, 1]) * grid_shape[0])),
                min(grid_shape[1] - 1, int(float(gt[0, 0]) * grid_shape[1])),
            ]
            rows.append({
                "sample_id": sample["sample_id"],
                "event_start_s": sample["event_start_s"],
                "event_end_s": sample["event_end_s"],
                "label_time_s": sample["label_time_s"],
                "event_count": sample["event_count"],
                "inference_mode": "model.eval()",
                "decoder": "normal global confidence argmax",
                "confidence_argmax_cell_yx": list(cell),
                "gt_center_cell_yx_for_metric_only": gt_center_cell,
                "selected_cell_matches_gt_center_cell": list(cell) == gt_center_cell,
                "selected_confidence": confidence,
                "keypoint_mean_epe_px": epe,
                "prediction_xy_min_max": [float(prediction.min()), float(prediction.max())],
                "prediction_outside_unit_square_count": int(
                    np.count_nonzero((prediction < 0.0) | (prediction >= 1.0))
                ),
                **pnp_metrics,
            })
    return {
        "model_training_flag_after_inference": bool(model.training),
        "mean_keypoint_epe_px": float(np.mean([row["keypoint_mean_epe_px"] for row in rows])),
        "details": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-source", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2718)
    args = parser.parse_args()

    # The pinned legacy helper imports imghdr for unrelated directory listing;
    # Python 3.14 removed it. Match the existing compatibility shim.
    if "imghdr" not in sys.modules:
        sys.modules["imghdr"] = types.SimpleNamespace(what=lambda *a, **k: None)
    source = args.pose_source.resolve()
    source_verification = verify_pinned_source(
        source, "SingleShotPose", PINNED_POSE_COMMIT, SOURCE_BUNDLE_MANIFEST
    )
    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires CUDA to match the pinned pose diagnostic path")

    sys.path.insert(0, str(source))
    from darknet import Darknet
    from region_loss import RegionLoss, build_targets

    runner = load_module("pose_stage2_runner", POSE_RUNNER)
    fixture = load_module("pose_window_fixture", WINDOW_FIXTURE)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")

    import tempfile
    with tempfile.TemporaryDirectory(prefix="pose-eval-contract-") as temp:
        temp_root = Path(temp)
        shared_h5 = temp_root / "shared.h5"
        v2e_root = temp_root / "v2e"
        fixture.write_fixture(shared_h5, v2e_root)
        train_samples = runner.collect_samples(
            "eventgenesis", shared_h5, v2e_root, [0.1, 0.1, 0.1],
            "train", windows_per_rollout=2, count_scale=8,
        )

        model = Darknet(str(source / "cfg" / "yolo-pose.cfg")).to(device)
        loss_layer = RegionLoss(
            num_keypoints=runner.NUM_KEYPOINTS,
            num_classes=runner.NUM_CLASSES,
            anchors=[], num_anchors=1, pretrain_num_epochs=15,
        )
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005,
        )
        sample0 = train_samples[0]
        input0 = torch.from_numpy(sample0["image"]).unsqueeze(0).to(device)
        target0 = torch.from_numpy(sample0["label"]).unsqueeze(0)
        warmup_losses = []
        model.train()
        for epoch in range(1, 17):
            optimizer.zero_grad(set_to_none=True)
            output = model(input0)
            loss, components = runner.patched_official_region_loss(
                output, target0, loss_layer, epoch, build_targets,
            )
            if components["nGT"] != 1 or not torch.isfinite(loss):
                raise RuntimeError("invalid synthetic warmup loss")
            loss.backward()
            optimizer.step()
            loss_layer.seen += 1
            warmup_losses.append(float(loss.detach().cpu()))

        parameters_before_calibration = parameter_sha256(model)
        uncalibrated_eval = eval_report(runner, model, train_samples, device)
        calibration_first = runner.recalibrate_batchnorm_from_synthetic_train(
            model, train_samples, device,
        )
        parameters_after_calibration = parameter_sha256(model)
        calibrated_eval_before_confidence_phase = eval_report(runner, model, train_samples, device)
        if parameters_before_calibration != parameters_after_calibration:
            raise RuntimeError("BN calibration changed trainable parameters")

        # Keep the pinned 15-epoch confidence burn-in and add a small synthetic
        # confidence-supervised phase to reveal whether ordinary eval selection
        # becomes learnable. These repeated fixture windows are not a budget
        # recommendation or generalization evaluation.
        confidence_losses = []
        model.train()
        for step in range(64):
            sample = train_samples[step % len(train_samples)]
            inputs = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)
            target = torch.from_numpy(sample["label"]).unsqueeze(0)
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            loss, components = runner.patched_official_region_loss(
                output, target, loss_layer, 16, build_targets,
            )
            if components["nGT"] != 1 or not torch.isfinite(loss):
                raise RuntimeError("invalid synthetic confidence-phase loss")
            loss.backward()
            optimizer.step()
            loss_layer.seen += 1
            confidence_losses.append(float(loss.detach().cpu()))

        eval_before_final_calibration = eval_report(runner, model, train_samples, device)
        parameters_before_final_calibration = parameter_sha256(model)
        calibration_final = runner.recalibrate_batchnorm_from_synthetic_train(
            model, train_samples, device,
        )
        parameters_after_final_calibration = parameter_sha256(model)
        final_eval = eval_report(runner, model, train_samples, device)
        if parameters_before_final_calibration != parameters_after_final_calibration:
            raise RuntimeError("final BN calibration changed trainable parameters")

        checkpoint_path = temp_root / "calibrated_fixture.pt"
        final_state_hash = runner.state_sha256(model.state_dict())
        torch.save({"model": model.state_dict()}, checkpoint_path)
        restored = Darknet(str(source / "cfg" / "yolo-pose.cfg")).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        restored.load_state_dict(checkpoint["model"])
        restored.eval()
        restored_state_hash = runner.state_sha256(restored.state_dict())
        restored_eval = eval_report(runner, restored, train_samples, device)
        restored_eval_matches = restored_eval == final_eval
        if restored_state_hash != final_state_hash or not restored_eval_matches:
            raise RuntimeError("reloaded calibrated checkpoint changed ordinary eval inference")

    print(json.dumps({
        "status": "completed",
        "scope": "two-window synthetic training fixture only; no independent rollout, real data, or checkpoint selection",
        "source_commit": PINNED_POSE_COMMIT,
        "source_verification": source_verification,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "sample_ids": [sample["sample_id"] for sample in train_samples],
        "same_rollout": len({sample["rollout_id"] for sample in train_samples}) == 1,
        "model_inference_inputs": ["full-frame synthetic event ON/OFF/total count channels"],
        "forbidden_inputs_used": [],
        "training_updates": {
            "warmup_with_official_confidence_exclusion": len(warmup_losses),
            "confidence_supervised": len(confidence_losses),
            "warmup_loss_first_last_min": [warmup_losses[0], warmup_losses[-1], min(warmup_losses)],
            "confidence_phase_loss_first_last_min": [confidence_losses[0], confidence_losses[-1], min(confidence_losses)],
        },
        "bn_calibration_passes": [calibration_first, calibration_final],
        "bn_calibration_changed_parameters": False,
        "checkpoint_restore": {
            "state_sha256": final_state_hash,
            "restored_state_sha256": restored_state_hash,
            "state_hash_equal": restored_state_hash == final_state_hash,
            "ordinary_eval_outputs_equal": restored_eval_matches,
            "checkpoint_stored_only_in_temporary_directory": True,
        },
        "eval_reports": {
            "after_16_updates_before_bn_calibration": uncalibrated_eval,
            "after_train_only_bn_calibration_before_confidence_phase": calibrated_eval_before_confidence_phase,
            "after_64_confidence_updates_before_final_bn_calibration": eval_before_final_calibration,
            "after_64_confidence_updates_and_final_train_only_bn_calibration": final_eval,
        },
        "interpretation": "The run isolates two fixable training-state causes: the 16-update warmup leaves confidence largely unsupervised, while recalibrating BN buffers from synthetic train event tensors restores ordinary eval behavior on the tiny training fixture. This is only an overfit/mechanism check; held-out-rollout eval must pass before pose training is considered ready.",
    }, indent=2))


if __name__ == "__main__":
    main()
