#!/usr/bin/env python3
"""Build immutable E2VID Stage-4 inputs with exact adjacent 10 ms geometry labels.

This deliberately reprojects the shared physical state at endpoint i into the
stored state at endpoint i+1.  It never derives 10 ms flow by scaling the
registered 30 ms STE-FlowNet labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from event_window_audit import audit_capture_partition
from stage4_v2e_artifacts import locate_shard_artifacts
from genesis_event_plugin.physics_warp import project_endpoint_to_alpha


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def load_generator_events(generator: str, episode: h5py.Group, v2e_root: Path):
    if generator == "eventgenesis":
        raw = np.asarray(episode["eventgenesis_events_txyp"], dtype=np.float64).reshape(-1, 4)
        times, xs, ys = raw[:, 0], raw[:, 1], raw[:, 2]
        polarity = (raw[:, 3] > 0).astype(np.uint8)
        metadata = {"source": "shared physical bank", "time_unit": "seconds", "polarity": "signed -1/+1"}
    elif generator == "v2e":
        path, status_path = locate_shard_artifacts(v2e_root, str(episode.name.rsplit("/", 1)[-1]))
        status = json.loads(status_path.read_text())
        event_file_hash = sha256(path)
        if int(status.get("exit_code", -1)) != 0 or status.get("raw_h5_sha256") != event_file_hash:
            raise RuntimeError(f"V2E shard status/hash mismatch: {path}")
        with h5py.File(path, "r") as src:
            raw = np.asarray(src["events"], dtype=np.uint32).reshape(-1, 4)
        times = raw[:, 0].astype(np.float64) * 1e-6
        xs, ys, polarity = raw[:, 1], raw[:, 2], raw[:, 3].astype(np.uint8)
        metadata = {"source": str(path), "source_sha256": event_file_hash, "time_unit": "microseconds converted to seconds", "polarity": "OFF=0, ON=1"}
    else:
        raise ValueError(generator)
    events = np.column_stack((times, xs, ys, polarity)).reshape(-1, 4)
    if not np.isfinite(events).all() or (len(events) > 1 and np.any(np.diff(events[:, 0]) < 0)):
        raise RuntimeError(f"invalid or unsorted {generator} events for {episode.name}")
    if len(events) and (not np.isin(events[:, 3], (0, 1)).all()):
        raise RuntimeError(f"bad polarity in {generator}/{episode.name}")
    return events, metadata


def project_pair(episode: h5py.Group, i: int):
    """Direct source→next-endpoint reprojection and target-depth visibility."""
    depth = episode["depth_m"]
    seg = episode["segmentation_link_idx"]
    cam = episode["camera_to_world_cv"]
    obj = episode["object_to_world"]
    obj_ids = np.asarray(episode["object_segmentation_link_ids"], dtype=np.int32)
    K = np.asarray(episode["camera_intrinsics_K_px"], dtype=np.float64)
    source_depth = np.asarray(depth[i], dtype=np.float32)
    source_seg = np.asarray(seg[i], dtype=np.int32)
    object0 = {int(sid): np.asarray(obj[i, j], dtype=np.float64) for j, sid in enumerate(obj_ids)}
    object1 = {int(sid): np.asarray(obj[i + 1, j], dtype=np.float64) for j, sid in enumerate(obj_ids)}
    uv, projected_z, geometry_valid = project_endpoint_to_alpha(
        source_depth, source_seg, K,
        np.asarray(cam[i], dtype=np.float64), np.asarray(cam[i + 1], dtype=np.float64), 1.0,
        source_index=0, object_world0=object0, object_world1=object1,
    )
    H, W = source_depth.shape
    tx, ty = uv[0], uv[1]
    inside = geometry_valid & (tx >= 0.0) & (tx <= W - 1) & (ty >= 0.0) & (ty <= H - 1)
    sampled_target_depth = cv2.remap(
        np.asarray(depth[i + 1], dtype=np.float32),
        np.where(inside, tx, -1.0).astype(np.float32),
        np.where(inside, ty, -1.0).astype(np.float32),
        cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    tolerance = np.maximum(0.02, 0.02 * projected_z)
    visible = inside & np.isfinite(sampled_target_depth) & (sampled_target_depth > 1e-6)
    visible &= np.abs(sampled_target_depth - projected_z) <= tolerance
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    flow = np.stack((tx - xx, ty - yy), axis=0).astype(np.float32)
    flow[:, ~visible] = 0.0
    return flow, visible.astype(np.uint8)


def build_labels(episode: h5py.Group, verify_registered_30ms: bool = True):
    times = np.asarray(episode["frame_timestamps_s"], dtype=np.float64)
    if times.shape != (201,) or not np.allclose(np.diff(times), 0.01, atol=1e-7, rtol=0):
        raise RuntimeError(f"{episode.name}: endpoints must be 201 uniformly spaced 10 ms timestamps")
    flows, valids = [], []
    for i in range(200):
        flow, valid = project_pair(episode, i)
        if not np.isfinite(flow).all():
            raise RuntimeError(f"non-finite 10 ms flow in {episode.name} interval {i}")
        flows.append(flow)
        valids.append(valid)
    flow10, valid10 = np.stack(flows), np.stack(valids)
    if flow10.shape != (200, 2, 260, 346) or valid10.shape != (200, 260, 346):
        raise RuntimeError(f"{episode.name}: wrong direct 10 ms label shapes {flow10.shape}/{valid10.shape}")

    exact_30_checks = 0
    if verify_registered_30ms:
        # Recompute the original interval labels independently and require bitwise
        # parity on the registered endpoint pairs; this verifies we preserved the
        # corrected bank geometry/mask while adding the adjacent 10 ms labels.
        for i in range(0, 198, 3):
            depth = episode["depth_m"]
            seg = episode["segmentation_link_idx"]
            cam = episode["camera_to_world_cv"]
            obj = episode["object_to_world"]
            ids = np.asarray(episode["object_segmentation_link_ids"], dtype=np.int32)
            object0 = {int(sid): np.asarray(obj[i, j], dtype=np.float64) for j, sid in enumerate(ids)}
            object1 = {int(sid): np.asarray(obj[i + 3, j], dtype=np.float64) for j, sid in enumerate(ids)}
            uv, pz, geom = project_endpoint_to_alpha(
                np.asarray(depth[i], dtype=np.float32), np.asarray(seg[i], dtype=np.int32),
                np.asarray(episode["camera_intrinsics_K_px"], dtype=np.float64),
                np.asarray(cam[i], dtype=np.float64), np.asarray(cam[i + 3], dtype=np.float64), 1.0,
                source_index=0, object_world0=object0, object_world1=object1,
            )
            H, W = np.asarray(depth[i]).shape
            inside = geom & (uv[0] >= 0) & (uv[0] <= W - 1) & (uv[1] >= 0) & (uv[1] <= H - 1)
            zt = cv2.remap(
                np.asarray(depth[i + 3], dtype=np.float32), np.where(inside, uv[0], -1).astype(np.float32),
                np.where(inside, uv[1], -1).astype(np.float32), cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
            )
            mask = inside & np.isfinite(zt) & (zt > 1e-6) & (np.abs(zt - pz) <= np.maximum(0.02, 0.02 * pz))
            yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
            direct = np.stack((uv[0] - xx, uv[1] - yy), axis=0).astype(np.float32)
            direct[:, ~mask] = 0.0
            old_flow = np.asarray(episode["flow_forward_30ms_px"][i // 3], dtype=np.float32)
            old_valid = np.asarray(episode["flow_valid_30ms"][i // 3], dtype=np.uint8)
            if not np.array_equal(direct, old_flow) or not np.array_equal(mask.astype(np.uint8), old_valid):
                raise RuntimeError(f"direct 30 ms parity failed in {episode.name}, interval {i}")
            exact_30_checks += 1
    return flow10, valid10, exact_30_checks


def export_h5(source: h5py.File, labels: dict, generator: str, episode_id: str,
              v2e_root: Path, out_path: Path):
    episode = source[f"episodes/{episode_id}"]
    times = np.asarray(episode["frame_timestamps_s"], dtype=np.float64)
    luma = np.asarray(episode["v2e_display_luma_u8"], dtype=np.uint8)
    if luma.shape[0] != 201:
        raise RuntimeError(f"{episode_id}: expected 201 shared target endpoints")
    events, event_source = load_generator_events(generator, episode, v2e_root)
    start, end = float(times[0]), float(times[-1])
    events = events[(events[:, 0] > start) & (events[:, 0] <= end)]
    frame_times = np.concatenate((times[1:], times[-1:]))
    if frame_times.shape != (201,):
        raise RuntimeError("adapter requires 201 image index entries for 200 intervals")
    ends = np.searchsorted(events[:, 0], frame_times, side="right")
    starts = np.concatenate(([0], ends[:-1]))
    counts = ends - starts
    if len(events) and ends[-2] != len(events):
        raise RuntimeError(f"{generator}/{episode_id}: event endpoint partition omitted events")
    if generator == "eventgenesis":
        capture_counts = np.asarray(episode["eventgenesis_interval_event_counts_10ms"], dtype=np.int64)[1:]
        audit = audit_capture_partition(events[:, 0], times, counts[:-1], capture_counts)
    else:
        audit = None
    flow, valid = labels[episode_id]
    label_hash = hash_array(flow) + hash_array(valid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as out:
        H, W = luma.shape[1:]
        out.attrs["sensor_resolution"] = np.asarray([H, W], dtype=np.int32)
        out.attrs["num_events"] = np.int64(len(events))
        out.attrs["num_imgs"] = np.int64(201)
        out.attrs["source"] = "unknown"
        out.attrs["sim2real_generator"] = generator
        out.attrs["sim2real_episode_id"] = episode_id
        out.attrs["sim2real_split"] = str(episode.attrs["split"])
        out.attrs["event_window"] = "(t_i,t_i_plus_1]"
        out.attrs["event_timestamp_unit"] = "seconds"
        out.attrs["event_label_alignment"] = "each event window right endpoint equals reconstruction target timestamp"
        out.attrs["target_image_mapping"] = "shared Rec.709 linear luminance, exposure 1, Reinhard, sRGB transfer, uint8"
        out.attrs["flow_label_contract"] = "direct source endpoint i to target endpoint i+1 reprojection; px per 10 ms interval"
        out.attrs["flow_validity_contract"] = "source geometry, in-bounds next-endpoint projection, next-depth agreement within max(2 cm,2%); common across generators"
        out.attrs["flow_label_hash"] = label_hash
        out.attrs["event_source_metadata"] = json.dumps(event_source, sort_keys=True)
        if audit is not None:
            out.attrs["event_capture_partition_audit"] = json.dumps(audit, sort_keys=True)
        event_group = out.create_group("events")
        event_group.create_dataset("xs", data=events[:, 1].astype(np.uint16), compression="gzip", compression_opts=1)
        event_group.create_dataset("ys", data=events[:, 2].astype(np.uint16), compression="gzip", compression_opts=1)
        event_group.create_dataset("ts", data=events[:, 0].astype(np.float64), compression="gzip", compression_opts=1)
        event_group.create_dataset("ps", data=events[:, 3].astype(np.uint8), compression="gzip", compression_opts=1)
        event_group.create_dataset("interval_counts", data=counts[:-1].astype(np.int64))
        images = out.create_group("images")
        right_images = np.concatenate((luma[1:], luma[-1:]), axis=0)
        for i, (frame, timestamp, event_index) in enumerate(zip(right_images, frame_times, ends)):
            image = images.create_dataset(f"image{i:09d}", data=frame, compression="gzip", compression_opts=1)
            image.attrs["timestamp"] = np.float64(timestamp)
            image.attrs["event_idx"] = np.int64(event_index)
        flow_group, valid_group = out.create_group("flow"), out.create_group("flow_valid")
        for i in range(200):
            safe = flow[i].copy()
            safe[:, ~valid[i].astype(bool)] = 0.0
            flow_group.create_dataset(f"flow{i:09d}", data=safe, compression="gzip", compression_opts=1)
            valid_group.create_dataset(f"flow_valid{i:09d}", data=valid[i], compression="gzip", compression_opts=1)
    return {
        "generator": generator, "rollout_id": episode_id, "split": str(episode.attrs["split"]),
        "path": str(out_path), "size_bytes": out_path.stat().st_size, "sha256": sha256(out_path),
        "event_count": int(len(events)), "interval_event_count_min": int(counts[:-1].min()),
        "interval_event_count_max": int(counts[:-1].max()), "flow_label_sha256": label_hash,
        "valid_fraction_mean": float(valid.mean()), "target_timestamps_s": [float(times[1]), float(times[-1])],
        "event_window_convention": "(t_i,t_i+1] with all endpoint ties assigned right",
        "capture_partition_audit": audit,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory-h5", type=Path, required=True)
    p.add_argument("--v2e-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--limit-rollouts", type=int, default=0, help="nonzero for a one-time adapter preflight only")
    args = p.parse_args()
    source_path, v2e_root, root = args.trajectory_h5.resolve(), args.v2e_root.resolve(), args.output_root.resolve()
    if not source_path.is_file() or not v2e_root.is_dir() or root.exists():
        raise FileNotFoundError(f"input missing or output already exists (fail-closed): {source_path}, {v2e_root}, {root}")
    root.mkdir(parents=True)
    labels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    label_rows = []
    with h5py.File(source_path, "r") as source:
        episode_ids = sorted(source["episodes"].keys())
        if args.limit_rollouts:
            episode_ids = episode_ids[:args.limit_rollouts]
        for episode_id in episode_ids:
            episode = source[f"episodes/{episode_id}"]
            flow, valid, parity_count = build_labels(episode)
            labels[episode_id] = (flow, valid)
            selected_magnitude = np.linalg.norm(flow, axis=1)[valid.astype(bool)]
            if selected_magnitude.size == 0:
                raise RuntimeError(f"{episode_id} has no geometry-valid 10 ms flow pixels")
            label_rows.append({
                "rollout_id": episode_id, "split": str(episode.attrs["split"]),
                "stratum": str(episode.attrs["stratum"]), "intervals": 200,
                "flow_shape": list(flow.shape), "valid_shape": list(valid.shape),
                "flow_sha256": hash_array(flow), "valid_sha256": hash_array(valid),
                "combined_sha256": hash_array(flow) + hash_array(valid),
                "valid_fraction": float(valid.mean()), "direct_30ms_parity_pairs": parity_count,
                "flow_magnitude_px_p50": float(np.percentile(selected_magnitude, 50)),
            })
        bank_hash = sha256(source_path)
        for arm in ("eventgenesis", "v2e"):
            for episode_id in episode_ids:
                row = export_h5(
                    source, labels, arm, episode_id, v2e_root,
                    root / arm / str(source[f"episodes/{episode_id}"].attrs["split"]) / f"{episode_id}.h5",
                )
                label_rows.append(row)
    for episode_id in episode_ids:
        pair = [row for row in label_rows if row.get("rollout_id") == episode_id and "generator" in row]
        if len(pair) != 2 or pair[0]["flow_label_sha256"] != pair[1]["flow_label_sha256"]:
            raise RuntimeError(f"paired 10 ms label hash mismatch for {episode_id}")
    for arm in ("eventgenesis", "v2e"):
        for split in ("train", "validation"):
            paths = sorted((root / arm / split).glob("*.h5"))
            (root / f"{arm}_{split}_files.txt").write_text("".join(f"{path}\n" for path in paths))
    script_hash = sha256(Path(__file__).resolve())
    manifest = {
        "schema": "stage4_e2vid_shared_10ms_labels_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_bank_path": str(source_path), "source_bank_size_bytes": source_path.stat().st_size,
        "source_bank_sha256": bank_hash, "v2e_root": str(v2e_root),
        "builder_path": str(Path(__file__).resolve()), "builder_sha256": script_hash,
        "frozen_simulator_core": "e79e656ac00f7847b98db9e12bf43b093f3afff1",
        "real_data_used": False, "real_scoring_enabled": False,
        "label_derivation": "per-endpoint depth+segmentation+K+camera/object poses; project endpoint i directly to i+1; nearest target depth; validity intersection",
        "explicitly_not_used": "flow_forward_30ms_px divided or scaled by 3",
        "rollouts": len(episode_ids), "rollout_rows": label_rows,
    }
    (root / "adapter_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "adapter_manifest.sha256").write_text(sha256(root / "adapter_manifest.json") + "  adapter_manifest.json\n")
    print(json.dumps({"status": "complete", "output_root": str(root), "rollouts": len(episode_ids),
                      "source_bank_sha256": bank_hash, "adapter_manifest_sha256": sha256(root / "adapter_manifest.json"),
                      "paired_label_hashes_match": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
