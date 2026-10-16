#!/usr/bin/env python3
"""Event-only SingleShotPose pilot on paired shared Genesis rollouts.

This adapter uses the pinned SingleShotPose Darknet and its original
build_targets objective. The event input is a fixed three-channel count image;
no RGB, mask, crop, segmentation, depth, or pose enters the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
POSE_SOURCE = ROOT.parent.parent / "external" / "sources" / "downstream" / "SingleShotPose"
NUM_KEYPOINTS = 9
NUM_CLASSES = 1
NETWORK_SIZE = 416
MAX_GT = 50
LABEL_SIZE = 2 * NUM_KEYPOINTS + 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).view(np.uint8))
    return digest.hexdigest()


def box_keypoints(size_xyz: list[float]) -> np.ndarray:
    hx, hy, hz = (float(value) / 2.0 for value in size_xyz)
    # SingleShotPose label_file_creation.md order: object centroid then the
    # eight [min/max x, min/max y, min/max z] cuboid corners.
    corners = np.asarray([
        [-hx, -hy, -hz], [-hx, -hy, hz],
        [-hx, hy, -hz], [-hx, hy, hz],
        [hx, -hy, -hz], [hx, -hy, hz],
        [hx, hy, -hz], [hx, hy, hz],
    ], dtype=np.float64)
    return np.concatenate((np.zeros((1, 3), dtype=np.float64), corners), axis=0)


def project_keypoints(keypoints_obj: np.ndarray, camera_to_world: np.ndarray,
                      object_to_world: np.ndarray, intrinsics: np.ndarray,
                      width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_from_object = np.linalg.inv(camera_to_world) @ object_to_world
    points_h = np.concatenate((keypoints_obj, np.ones((len(keypoints_obj), 1))), axis=1)
    points_cam = (camera_from_object @ points_h.T).T[:, :3]
    if np.any(points_cam[:, 2] <= 0) or not np.isfinite(points_cam).all():
        raise ValueError("pose pilot selected a keypoint behind or invalid in the camera")
    homogeneous = (intrinsics @ points_cam.T).T
    uv = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.isfinite(uv).all():
        raise ValueError("projected pose keypoints contain non-finite values")
    return uv, points_cam, camera_from_object


def read_event_table(generator: str, episode_id: str, shared: h5py.File,
                     v2e_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if generator == "eventgenesis":
        table = np.asarray(shared[f"episodes/{episode_id}/eventgenesis_events_txyp"], dtype=np.float64).reshape(-1, 4)
        timestamp_s = table[:, 0]
        x, y = table[:, 1].astype(np.int64), table[:, 2].astype(np.int64)
        polarity = (table[:, 3] > 0).astype(np.uint8)
        source_schema = "t_seconds,x,y,p_signed"
    elif generator == "v2e":
        path = v2e_root / f"{episode_id}.h5"
        with h5py.File(path, "r") as handle:
            table = np.asarray(handle["events"], dtype=np.uint32).reshape(-1, 4)
        timestamp_s = table[:, 0].astype(np.float64) * 1e-6
        x, y = table[:, 1].astype(np.int64), table[:, 2].astype(np.int64)
        polarity = table[:, 3].astype(np.uint8)
        source_schema = "t_microseconds,x,y,p_01"
    else:
        raise ValueError(f"unsupported generator {generator}")
    if len(table) and (
        not np.isfinite(timestamp_s).all()
        or not np.all(np.diff(timestamp_s) >= 0)
        or not np.isin(polarity, (0, 1)).all()
    ):
        raise ValueError(f"invalid event table for {generator}/{episode_id}")
    return timestamp_s, x, y, polarity, source_schema


def encode_event_window(timestamp_s: np.ndarray, x: np.ndarray, y: np.ndarray,
                        polarity: np.ndarray, start_s: float, end_s: float,
                        width: int, height: int, count_scale: float) -> np.ndarray:
    selected = (timestamp_s > start_s) & (timestamp_s <= end_s)
    xs, ys, ps = x[selected], y[selected], polarity[selected]
    if len(xs) and (
        xs.min() < 0 or xs.max() >= width or ys.min() < 0 or ys.max() >= height
    ):
        raise ValueError("event coordinate is outside the shared sensor")
    on = np.zeros((height, width), dtype=np.float32)
    off = np.zeros_like(on)
    if len(xs):
        np.add.at(on, (ys[ps == 1], xs[ps == 1]), 1.0)
        np.add.at(off, (ys[ps == 0], xs[ps == 0]), 1.0)
    # Fixed event-count scaling, independent of generator and sample maxima.
    scale = np.float32(np.log1p(count_scale))
    on = np.minimum(np.log1p(on) / scale, 1.0)
    off = np.minimum(np.log1p(off) / scale, 1.0)
    counts = np.zeros_like(on)
    if len(xs):
        np.add.at(counts, (ys, xs), 1.0)
    total = np.minimum(np.log1p(counts) / scale, 1.0)
    frame = np.stack((on, off, total), axis=-1)
    resized = np.stack([
        cv2.resize(frame[..., channel], (NETWORK_SIZE, NETWORK_SIZE), interpolation=cv2.INTER_LINEAR)
        for channel in range(3)
    ], axis=0)
    return np.ascontiguousarray(resized, dtype=np.float32)


def build_label(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    normalized = uv / np.asarray([width, height], dtype=np.float64)
    if np.any(normalized[0] <= 0) or np.any(normalized[0] >= 1):
        raise ValueError("object centroid is outside the model image")
    corners = normalized[1:]
    extent = corners.max(axis=0) - corners.min(axis=0)
    if np.any(extent <= 0):
        raise ValueError("projected cuboid has a degenerate image extent")
    row = np.zeros((MAX_GT, LABEL_SIZE), dtype=np.float32)
    row[0, 0] = 0.0
    row[0, 1:19] = normalized.reshape(-1).astype(np.float32)
    row[0, 19:21] = extent.astype(np.float32)
    return row.reshape(-1)


def select_intervals(interval_count: int, sample_count: int) -> list[int]:
    if interval_count < 1:
        return []
    return sorted(set(np.linspace(
        min(2, interval_count - 1), interval_count - 1,
        num=min(sample_count, interval_count), dtype=int,
    ).tolist()))


def collect_samples(generator: str, shared_path: Path, v2e_root: Path,
                    target_size: list[float], split: str,
                    windows_per_rollout: int, count_scale: float) -> list[dict]:
    rows: list[dict] = []
    keypoints_obj = box_keypoints(target_size)
    with h5py.File(shared_path, "r") as shared:
        for episode_id in sorted(shared["episodes"].keys()):
            episode = shared[f"episodes/{episode_id}"]
            if str(episode.attrs["split"]) != split:
                continue
            times = np.asarray(episode["frame_timestamps_s"], dtype=np.float64)
            K = np.asarray(episode["camera_intrinsics_K_px"], dtype=np.float64)
            camera_poses = np.asarray(episode["camera_to_world_cv"], dtype=np.float64)
            object_poses = np.asarray(episode["target_pose"], dtype=np.float64)
            height, width = np.asarray(episode["v2e_display_luma_u8"]).shape[1:]
            timestamp_s, xs, ys, polarity, event_schema = read_event_table(
                generator, episode_id, shared, v2e_root
            )
            for interval_index in select_intervals(len(times) - 1, windows_per_rollout):
                label_time_index = interval_index + 1
                uv, xyz_cam, camera_from_object = project_keypoints(
                    keypoints_obj,
                    camera_poses[label_time_index],
                    object_poses[label_time_index],
                    K,
                    width,
                    height,
                )
                if np.any(uv < 0) or np.any(uv[:, 0] >= width) or np.any(uv[:, 1] >= height):
                    continue
                label = build_label(uv, width, height)
                image = encode_event_window(
                    timestamp_s, xs, ys, polarity,
                    float(times[interval_index]), float(times[label_time_index]),
                    width, height, count_scale,
                )
                resize_K = np.diag([
                    NETWORK_SIZE / width, NETWORK_SIZE / height, 1.0
                ]) @ K
                rows.append({
                    "generator": generator,
                    "episode_id": episode_id,
                    "interval_index": interval_index,
                    "event_count": int(((timestamp_s > times[interval_index]) & (timestamp_s <= times[label_time_index])).sum()),
                    "event_schema": event_schema,
                    "image": image,
                    "label": label,
                    "uv": uv.astype(np.float32),
                    "uv_normalized": (uv / np.asarray([width, height])).astype(np.float32),
                    "keypoints_obj": keypoints_obj.astype(np.float32),
                    "xyz_cam": xyz_cam.astype(np.float32),
                    "camera_matrix_416": resize_K.astype(np.float64),
                    "camera_from_object": camera_from_object.astype(np.float64),
                })
    if not rows:
        raise RuntimeError(f"no valid {split} samples for {generator}")
    return rows


def patched_official_region_loss(output: torch.Tensor, target: torch.Tensor,
                                loss_layer, epoch: int,
                                official_build_targets) -> tuple[torch.Tensor, dict]:
    """PyTorch 2.x device/API compatibility for the pinned RegionLoss objective."""
    batch, _, height, width = output.shape
    anchors = loss_layer.num_anchors
    classes = loss_layer.num_classes
    keypoints = loss_layer.num_keypoints
    view = output.view(batch, anchors, 2 * keypoints + 1 + classes, height, width)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=output.device, dtype=output.dtype),
        torch.arange(width, device=output.device, dtype=output.dtype),
        indexing="ij",
    )
    pred_pairs = []
    xs, ys = [], []
    for point in range(keypoints):
        raw_x, raw_y = view[:, :, 2 * point], view[:, :, 2 * point + 1]
        pred_x = torch.sigmoid(raw_x) if point == 0 else raw_x
        pred_y = torch.sigmoid(raw_y) if point == 0 else raw_y
        xs.append(pred_x)
        ys.append(pred_y)
        pred_pairs.extend((
            ((pred_x.detach() + grid_x) / width).reshape(-1),
            ((pred_y.detach() + grid_y) / height).reshape(-1),
        ))
    pred_corners = torch.stack(pred_pairs, dim=1).cpu()
    noobj_conf = torch.sigmoid(view[:, :, 2 * keypoints])
    nGT, nCorrect, coord_mask, conf_mask, cls_mask, txs, tys, tconf, tcls = official_build_targets(
        pred_corners, target.detach().cpu(), keypoints, anchors, classes, height, width,
        loss_layer.noobject_scale, loss_layer.object_scale, loss_layer.thresh, loss_layer.seen,
    )
    coord_mask = coord_mask.to(output.device)
    conf_mask = conf_mask.to(output.device).sqrt()
    target_conf = tconf.to(output.device)
    loss_x = torch.stack([
        F.mse_loss(xs[point] * coord_mask, txs[point].to(output.device) * coord_mask, reduction="sum") / 2.0
        for point in range(keypoints)
    ]).sum()
    loss_y = torch.stack([
        F.mse_loss(ys[point] * coord_mask, tys[point].to(output.device) * coord_mask, reduction="sum") / 2.0
        for point in range(keypoints)
    ]).sum()
    loss_conf = F.mse_loss(noobj_conf * conf_mask, target_conf * conf_mask, reduction="sum") / 2.0
    total = loss_x + loss_y + (loss_conf if epoch > loss_layer.pretrain_num_epochs else 0.0)
    return total, {
        "loss_x": float(loss_x.detach().cpu()),
        "loss_y": float(loss_y.detach().cpu()),
        "loss_conf": float(loss_conf.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
        "nGT": int(nGT),
        "nCorrect": int(nCorrect),
    }


def predict_normalized_keypoints(model, image: np.ndarray, device: torch.device) -> tuple[np.ndarray, float]:
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    batch, _, height, width = output.shape
    view = output.view(batch, 1, 2 * NUM_KEYPOINTS + 2, height, width)
    confidence = torch.sigmoid(view[0, 0, 2 * NUM_KEYPOINTS])
    index = int(confidence.reshape(-1).argmax().item())
    grid_y, grid_x = divmod(index, width)
    result = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    for point in range(NUM_KEYPOINTS):
        rx = view[0, 0, 2 * point, grid_y, grid_x]
        ry = view[0, 0, 2 * point + 1, grid_y, grid_x]
        px = torch.sigmoid(rx) if point == 0 else rx
        py = torch.sigmoid(ry) if point == 0 else ry
        result[point, 0] = float((px + grid_x) / width)
        result[point, 1] = float((py + grid_y) / height)
    return result, float(confidence[grid_y, grid_x])


def train_generator(generator: str, train_samples: list[dict], val_samples: list[dict],
                    args, device: torch.device, official_build_targets) -> dict:
    from darknet import Darknet
    from region_loss import RegionLoss

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = Darknet(str(POSE_SOURCE / "cfg" / "yolo-pose.cfg")).to(device)
    init_hash = state_sha256(model.state_dict())
    loss_layer = RegionLoss(
        num_keypoints=NUM_KEYPOINTS, num_classes=NUM_CLASSES,
        anchors=[], num_anchors=1, pretrain_num_epochs=15,
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    records = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    model.train()
    for epoch in range(1, args.epochs + 1):
        for sample_index, sample in enumerate(train_samples):
            inputs = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)
            target = torch.from_numpy(sample["label"]).unsqueeze(0)
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            loss, losses = patched_official_region_loss(
                output, target, loss_layer, epoch, official_build_targets
            )
            if losses["nGT"] != 1:
                raise RuntimeError(f"{generator}: official target builder found {losses['nGT']} objects, expected one")
            if not torch.isfinite(loss):
                raise RuntimeError(f"{generator}: non-finite pose loss")
            loss.backward()
            grad_norm = torch.sqrt(torch.stack([
                parameter.grad.detach().pow(2).sum()
                for parameter in model.parameters() if parameter.grad is not None
            ]).sum())
            if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
                raise RuntimeError(f"{generator}: missing/non-finite pose gradient")
            optimizer.step()
            loss_layer.seen += 1
            records.append({
                "epoch": epoch,
                "sample_index": sample_index,
                "episode_id": sample["episode_id"],
                "interval_index": sample["interval_index"],
                "event_count": sample["event_count"],
                "gradient_norm": float(grad_norm.detach().cpu()),
                **losses,
            })
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    final_hash = state_sha256(model.state_dict())
    if final_hash == init_hash:
        raise RuntimeError(f"{generator}: model parameters did not change")

    model.eval()
    validation = []
    for sample in val_samples:
        predicted, confidence = predict_normalized_keypoints(model, sample["image"], device)
        gt = sample["uv_normalized"]
        scale = np.asarray([NETWORK_SIZE, NETWORK_SIZE], dtype=np.float32)
        keypoint_epe = float(np.linalg.norm((predicted[1:] - gt[1:]) * scale, axis=1).mean())
        pnp_metrics = {"pnp_succeeded": False}
        try:
            pnp_ok, rotation_vector, translation = cv2.solvePnP(
                sample["keypoints_obj"][1:].astype(np.float64),
                (predicted[1:] * scale).astype(np.float64),
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
                "translation_error_m": float(np.linalg.norm(translation.reshape(3) - gt_transform[:3, 3])),
            })
        validation.append({
            "episode_id": sample["episode_id"],
            "interval_index": sample["interval_index"],
            "event_count": sample["event_count"],
            "keypoint_mean_epe_px": keypoint_epe,
            "confidence": confidence,
            **pnp_metrics,
        })

    checkpoint = args.output_root / f"{generator}.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "generator": generator,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_rollouts": sorted({row["episode_id"] for row in train_samples}),
        "validation_rollouts": sorted({row["episode_id"] for row in val_samples}),
        "input_contract": "three fixed-scale full-frame channels: log-count ON, log-count OFF, log-count total",
        "upstream_source_commit": "2c282619101f9f31666dd18b42393375bea1600f",
    }, checkpoint)
    restored = Darknet(str(POSE_SOURCE / "cfg" / "yolo-pose.cfg")).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    restored.load_state_dict(saved["model"])
    restored.eval()
    restored_hash = state_sha256(restored.state_dict())
    if restored_hash != final_hash:
        raise RuntimeError(f"{generator}: pose checkpoint restore hash mismatch")
    restored_first, _ = predict_normalized_keypoints(restored, val_samples[0]["image"], device)
    if not np.allclose(restored_first, predict_normalized_keypoints(model, val_samples[0]["image"], device)[0], atol=1e-6):
        raise RuntimeError(f"{generator}: restored pose validation output differs")

    return {
        "generator": generator,
        "initial_state_sha256": init_hash,
        "final_state_sha256": final_hash,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_path": str(checkpoint),
        "train_sample_count": len(train_samples),
        "validation_sample_count": len(validation),
        "train_rollouts": sorted({row["episode_id"] for row in train_samples}),
        "validation_rollouts": sorted({row["episode_id"] for row in val_samples}),
        "train_wall_seconds": train_seconds,
        "peak_allocated_gpu_mib": peak_memory_mib,
        "train_loss_mean": float(np.mean([row["loss_total"] for row in records])),
        "train_gradient_norm_min_max": [
            float(min(row["gradient_norm"] for row in records)),
            float(max(row["gradient_norm"] for row in records)),
        ],
        "validation_keypoint_mean_epe_px": float(np.mean([row["keypoint_mean_epe_px"] for row in validation])),
        "validation_detail": validation,
        "checkpoint_restore_hash_equal": restored_hash == final_hash,
        "training_detail": records,
        "objective": "pinned SingleShotPose RegionLoss coordinate pretraining phase; default first-15-epoch confidence exclusion retained",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-h5", required=True, type=Path)
    parser.add_argument("--v2e-root", required=True, type=Path)
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--pilot-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    if not POSE_SOURCE.is_dir():
        raise FileNotFoundError(f"pinned SingleShotPose source missing: {POSE_SOURCE}")
    args.output_root.mkdir(parents=True, exist_ok=False)

    sys.path.insert(0, str(POSE_SOURCE))
    from region_loss import build_targets

    if not torch.cuda.is_available():
        raise RuntimeError("pose pilot requires the assigned CUDA GPU")
    device = torch.device("cuda:0")
    data_config = json.loads(args.data_config.read_text())
    pilot_config = json.loads(args.pilot_config.read_text())
    training_config = pilot_config["training"]
    input_config = pilot_config["input"]
    args.seed = training_config["seed"] if args.seed is None else args.seed
    args.epochs = training_config["epochs"] if args.epochs is None else args.epochs
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if input_config["resize_hw"] != [NETWORK_SIZE, NETWORK_SIZE]:
        raise ValueError(f"pilot resize_hw must match pinned model size {[NETWORK_SIZE, NETWORK_SIZE]}")
    args.learning_rate = training_config["learning_rate"]
    args.momentum = training_config["momentum"]
    args.weight_decay = training_config["weight_decay"]
    size = data_config["simulation"]["target_box"]["size_m"]
    generators = {}
    for generator in ("eventgenesis", "v2e"):
        train = collect_samples(
            generator, args.shared_h5, args.v2e_root, size, "train",
            int(training_config["windows_per_physical_rollout"]),
            float(input_config["fixed_count_scale_per_channel"]),
        )
        validation = collect_samples(
            generator, args.shared_h5, args.v2e_root, size, "validation",
            int(training_config["windows_per_physical_rollout"]),
            float(input_config["fixed_count_scale_per_channel"]),
        )
        generators[generator] = {"train": train, "validation": validation}
    ev_rows = generators["eventgenesis"]
    v2e_rows = generators["v2e"]
    if [(r["episode_id"], r["interval_index"]) for r in ev_rows["train"]] != [
        (r["episode_id"], r["interval_index"]) for r in v2e_rows["train"]
    ] or [(r["episode_id"], r["interval_index"]) for r in ev_rows["validation"]] != [
        (r["episode_id"], r["interval_index"]) for r in v2e_rows["validation"]
    ]:
        raise RuntimeError("paired pose samples do not use identical rollout windows")
    for sample_ev, sample_v2e in zip(ev_rows["train"] + ev_rows["validation"], v2e_rows["train"] + v2e_rows["validation"]):
        if not np.array_equal(sample_ev["label"], sample_v2e["label"]):
            raise RuntimeError("pose labels differ between paired generator inputs")

    results = []
    for generator in ("eventgenesis", "v2e"):
        results.append(train_generator(
            generator,
            generators[generator]["train"],
            generators[generator]["validation"],
            args,
            device,
            build_targets,
        ))
    if results[0]["initial_state_sha256"] != results[1]["initial_state_sha256"]:
        raise RuntimeError("paired pose training did not start from identical initialization")
    report = {
        "schema_version": 1,
        "scope": "short synthetic-only event-only SingleShotPose pilot; not E-POSE real evaluation",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": "2c282619101f9f31666dd18b42393375bea1600f",
        "shared_h5": str(args.shared_h5.resolve()),
        "shared_h5_sha256": sha256_file(args.shared_h5),
        "v2e_root": str(args.v2e_root.resolve()),
        "data_config": str(args.data_config.resolve()),
        "data_config_sha256": sha256_file(args.data_config),
        "pilot_config": str(args.pilot_config.resolve()),
        "pilot_config_sha256": sha256_file(args.pilot_config),
        "seed": args.seed,
        "epochs": args.epochs,
        "model_inference_inputs": ["raw-event-derived full-frame three-channel count tensor only"],
        "training_supervision": ["shared Genesis object pose and camera K projected cuboid keypoints"],
        "forbidden_inputs_not_used": ["RGB", "GT mask", "GT crop", "GT box", "segmented events", "depth", "segmentation", "pose at inference"],
        "input_encoding": input_config,
        "physical_object_scope": "generic Genesis box with configured dimensions; no E-POSE class/CAD match in this pilot",
        "paired_labels_equal": True,
        "results": results,
    }
    (args.output_root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
