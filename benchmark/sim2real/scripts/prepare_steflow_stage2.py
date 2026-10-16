#!/usr/bin/env python3
"""Build native STE-FlowNet inputs and shared dt=1/dt=4 labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from prepare_steflow_stage0 import gaussian_count_representation, pad_center, read_events


SIZE = 256
SLICES = 5


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def convert_episode(source: h5py.File, generator: str, episode_id: str, v2e_root: Path, output: Path) -> dict:
    episode = source[f"episodes/{episode_id}"]
    times = np.asarray(episode["frame_timestamps_s"], dtype=np.float64)
    gray = np.asarray(episode["v2e_display_luma_u8"], dtype=np.float32) / 255.0
    flow_dt1 = np.asarray(episode["flow_forward_px_per_interval"], dtype=np.float32)
    valid_dt1 = np.asarray(episode["flow_valid"], dtype=np.uint8)
    flow_dt4 = np.asarray(episode["flow_forward_dt4_px_per_interval"], dtype=np.float32)
    valid_dt4 = np.asarray(episode["flow_valid_dt4"], dtype=np.uint8)
    height, width = gray.shape[1:]
    if len(flow_dt1) != len(times) - 1 or len(flow_dt4) != max(0, len(times) - 4):
        raise ValueError(f"flow timestamp alignment failed in {episode_id}")

    raw = read_events(generator, episode, v2e_root)
    raw = raw[(raw[:, 0] > times[0]) & (raw[:, 0] <= times[-1])]
    if len(raw) and not np.all(np.diff(raw[:, 0]) >= 0):
        raise ValueError(f"unsorted {generator} events in {episode_id}")
    start = 0
    reps = []
    interval_counts = []
    for left, right in zip(times[:-1], times[1:]):
        end = int(np.searchsorted(raw[:, 0], right, side="right"))
        interval = raw[start:end]
        if len(interval) and (np.any(interval[:, 0] <= left) or np.any(interval[:, 0] > right)):
            raise ValueError(f"event window boundary mismatch in {generator}/{episode_id}")
        local = interval.copy()
        if len(local):
            local[:, 0] -= left
        rep = gaussian_count_representation(local, height, width)
        padded = np.zeros((2, SIZE, SIZE, SLICES), dtype=np.uint8)
        yoff, xoff = (SIZE - height) // 2, (SIZE - width) // 2
        padded[:, yoff:yoff + height, xoff:xoff + width, :] = rep
        reps.append(padded)
        interval_counts.append(int(len(interval)))
        start = end
    if start != len(raw):
        raise ValueError(f"unassigned events: {len(raw) - start} in {generator}/{episode_id}")

    event_dt1 = np.stack(reps, axis=0)
    if len(reps) >= 4:
        event_dt4 = np.stack([
            np.concatenate(reps[index:index + 4], axis=-1)
            for index in range(len(reps) - 3)
        ], axis=0)
    else:
        event_dt4 = np.zeros((0, 2, SIZE, SIZE, 4 * SLICES), dtype=np.uint8)

    yoff, xoff = (SIZE - height) // 2, (SIZE - width) // 2
    gray_padded = pad_center(gray, yoff, xoff, SIZE, SIZE)
    flow1_padded = pad_center(flow_dt1, yoff, xoff, SIZE, SIZE)
    valid1_padded = pad_center(valid_dt1, yoff, xoff, SIZE, SIZE)
    flow4_padded = pad_center(flow_dt4, yoff, xoff, SIZE, SIZE)
    valid4_padded = pad_center(valid_dt4, yoff, xoff, SIZE, SIZE)
    # EPE is only evaluated on shared geometric validity. Invalid flow values
    # remain zeroed in the stored copy and are excluded by the adjacent mask.
    flow1_padded = flow1_padded.copy()
    flow4_padded = flow4_padded.copy()
    flow1_padded *= valid1_padded[:, None, :, :].astype(np.float32)
    flow4_padded *= valid4_padded[:, None, :, :].astype(np.float32)

    label_hash_dt1 = sha256_array(flow1_padded) + sha256_array(valid1_padded)
    label_hash_dt4 = sha256_array(flow4_padded) + sha256_array(valid4_padded)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        events_dt1=event_dt1,
        events_dt4=event_dt4,
        gray_frames=gray_padded,
        flow_gt_dt1=flow1_padded.astype(np.float32),
        flow_valid_dt1=valid1_padded.astype(np.uint8),
        flow_gt_dt4=flow4_padded.astype(np.float32),
        flow_valid_dt4=valid4_padded.astype(np.uint8),
        interval_event_counts=np.asarray(interval_counts, dtype=np.int32),
        endpoint_timestamps_s=times,
        camera_intrinsics_K_px=np.asarray(episode["camera_intrinsics_K_px"], dtype=np.float64),
        source_resolution_hw=np.asarray([height, width], dtype=np.int32),
        pad_yx=np.asarray([yoff, xoff], dtype=np.int32),
    )
    return {
        "generator": generator,
        "episode_id": episode_id,
        "split": str(episode.attrs["split"]),
        "path": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "raw_event_count": int(len(raw)),
        "event_counts_dt1": interval_counts,
        "event_representation_dt1_shape": list(event_dt1.shape),
        "event_representation_dt4_shape": list(event_dt4.shape),
        "uint8_max_dt1": int(event_dt1.max(initial=0)),
        "uint8_max_dt4": int(event_dt4.max(initial=0)),
        "flow_label_sha256_dt1": label_hash_dt1,
        "flow_label_sha256_dt4": label_hash_dt4,
        "valid_pixels_dt1": valid1_padded.reshape(len(valid1_padded), -1).sum(axis=1).astype(int).tolist(),
        "valid_pixels_dt4": valid4_padded.reshape(len(valid4_padded), -1).sum(axis=1).astype(int).tolist(),
        "camera_intrinsics_K_px": np.asarray(episode["camera_intrinsics_K_px"]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-h5", type=Path, required=True)
    parser.add_argument("--v2e-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = []
    with h5py.File(args.trajectory_h5, "r") as source:
        for generator in ("eventgenesis", "v2e"):
            for episode_id in sorted(source["episodes"].keys()):
                split = str(source[f"episodes/{episode_id}"].attrs["split"])
                output = args.output_root / generator / split / f"{episode_id}.npz"
                manifest.append(convert_episode(source, generator, episode_id, args.v2e_root, output))
    by_pair = {(item["generator"], item["episode_id"]): item for item in manifest}
    for episode_id in sorted({item["episode_id"] for item in manifest}):
        for task in ("dt1", "dt4"):
            key = f"flow_label_sha256_{task}"
            if by_pair[("eventgenesis", episode_id)][key] != by_pair[("v2e", episode_id)][key]:
                raise RuntimeError(f"paired {task} flow labels differ for {episode_id}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
