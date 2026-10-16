#!/usr/bin/env python3
"""Export paired Stage-2 trajectories for the pinned E2VID+ HDF5 loader.

The flow target is a shared per-endpoint displacement. It is never divided by
an observed event span. A benchmark-local dataset adapter restores this target
after the pinned loader's legacy ``flow *= event_dt`` behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def array_sha256(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def load_events(generator: str, episode: h5py.Group, v2e_root: Path) -> tuple[np.ndarray, dict]:
    if generator == "eventgenesis":
        raw = np.asarray(episode["eventgenesis_events_txyp"], dtype=np.float64).reshape(-1, 4)
        if raw.ndim != 2 or raw.shape[1] != 4:
            raise ValueError(f"unexpected EventGenesis event shape {raw.shape}")
        times, xs, ys = raw[:, 0], raw[:, 1], raw[:, 2]
        polarity = (raw[:, 3] > 0).astype(np.uint8)
        metadata = {"schema": "t_seconds,x,y,p_signed", "timestamp_unit": "s", "polarity": "signed -1/+1"}
    elif generator == "v2e":
        path = v2e_root / f"{episode.name.rsplit('/', 1)[-1]}.h5"
        with h5py.File(path, "r") as event_file:
            raw = np.asarray(event_file["events"], dtype=np.uint32).reshape(-1, 4)
        if raw.ndim != 2 or raw.shape[1] != 4:
            raise ValueError(f"unexpected V2E event shape {raw.shape} in {path}")
        times = raw[:, 0].astype(np.float64) * 1e-6
        xs, ys = raw[:, 1], raw[:, 2]
        polarity = raw[:, 3].astype(np.uint8)
        metadata = {"schema": "t_microseconds,x,y,p_01", "timestamp_unit": "us", "polarity": "OFF=0, ON=1"}
    else:
        raise ValueError(f"unsupported generator {generator}")
    events = np.column_stack((times, xs, ys, polarity)).reshape(-1, 4)
    return events, metadata


def export_episode(
    source: h5py.File,
    generator: str,
    episode_id: str,
    v2e_root: Path,
    output_path: Path,
) -> dict:
    episode = source[f"episodes/{episode_id}"]
    endpoint_times = np.asarray(episode["frame_timestamps_s"], dtype=np.float64)
    luma = np.asarray(episode["v2e_display_luma_u8"], dtype=np.uint8)
    flow = np.asarray(episode["flow_forward_px_per_interval"], dtype=np.float32)
    valid = np.asarray(episode["flow_valid"], dtype=np.uint8)
    if len(endpoint_times) != len(luma) or len(flow) != len(endpoint_times) - 1:
        raise ValueError(f"frame/flow lengths differ in {episode_id}")
    events, event_metadata = load_events(generator, episode, v2e_root)
    if events.shape != (len(events), 4) or not np.isfinite(events[:, 0]).all():
        raise ValueError(f"invalid event table for {generator}/{episode_id}")
    if len(events) and not np.all(np.diff(events[:, 0]) >= 0):
        raise ValueError(f"events are not time sorted for {generator}/{episode_id}")
    height, width = luma.shape[1:]
    if len(events) and (
        events[:, 1].min() < 0 or events[:, 1].max() >= width
        or events[:, 2].min() < 0 or events[:, 2].max() >= height
    ):
        raise ValueError(f"event coordinate outside shared image for {generator}/{episode_id}")
    if len(events) and not np.isin(events[:, 3], (0, 1)).all():
        raise ValueError(f"invalid polarity in {generator}/{episode_id}")

    # HDF5 loader timestamps attach each event window to its right endpoint.
    # searchsorted(side=right) assigns all equal-time events consistently.
    t0, tlast = endpoint_times[0], endpoint_times[-1]
    events = events[(events[:, 0] > t0) & (events[:, 0] <= tlast)]
    frame_times = np.concatenate((endpoint_times[1:], endpoint_times[-1:]))
    event_ends = np.searchsorted(events[:, 0], frame_times, side="right")
    if np.any(np.diff(event_ends) < 0) or (len(event_ends) > 1 and event_ends[-2] != len(events)):
        raise ValueError(f"right-endpoint partition failed for {generator}/{episode_id}")
    interval_starts = np.concatenate(([0], event_ends[:-2]))
    interval_ends = event_ends[:-1]
    interval_counts = interval_ends - interval_starts

    if generator == "eventgenesis":
        expected = np.asarray(episode["eventgenesis_interval_event_counts"], dtype=np.int64)
        if not np.array_equal(interval_counts, expected):
            raise ValueError(f"EventGenesis interval counts differ: {interval_counts.tolist()} != {expected.tolist()}")

    label_hash = array_sha256(flow) + array_sha256(valid)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as out:
        out.attrs["sensor_resolution"] = np.asarray([height, width], dtype=np.int32)
        out.attrs["num_events"] = np.int64(len(events))
        out.attrs["num_imgs"] = np.int64(len(frame_times))
        out.attrs["source"] = "unknown"
        out.attrs["sim2real_generator"] = generator
        out.attrs["sim2real_episode_id"] = episode_id
        out.attrs["event_window"] = "(t_i,t_i_plus_1]"
        out.attrs["event_timestamp_unit"] = "seconds"
        out.attrs["polarity_encoding"] = "OFF=0, ON=1; pinned loader maps to -1/+1"
        out.attrs["target_image_mapping"] = "shared fixed radiance-to-display luma uint8"
        out.attrs["flow_label_contract"] = "shared forward source-to-target endpoint displacement, pixels per endpoint interval"
        out.attrs["flow_label_hash"] = label_hash
        out.attrs["flow_validity_contract"] = str(episode.attrs["flow_valid_contract"])
        out.attrs["event_source_metadata"] = json.dumps(event_metadata, sort_keys=True)

        event_group = out.create_group("events")
        event_group.create_dataset("xs", data=events[:, 1].astype(np.uint16), compression="gzip", compression_opts=1)
        event_group.create_dataset("ys", data=events[:, 2].astype(np.uint16), compression="gzip", compression_opts=1)
        event_group.create_dataset("ts", data=events[:, 0].astype(np.float64), compression="gzip", compression_opts=1)
        event_group.create_dataset("ps", data=events[:, 3].astype(np.uint8), compression="gzip", compression_opts=1)
        event_group.create_dataset("interval_counts", data=interval_counts.astype(np.int64))

        images = out.create_group("images")
        right_images = np.concatenate((luma[1:], luma[-1:]), axis=0)
        for i, (frame, timestamp, event_index) in enumerate(zip(right_images, frame_times, event_ends)):
            image = images.create_dataset(f"image{i:09d}", data=frame, compression="gzip", compression_opts=1)
            image.attrs["timestamp"] = np.float64(timestamp)
            image.attrs["event_idx"] = np.int64(event_index)

        flow_group = out.create_group("flow")
        valid_group = out.create_group("flow_valid")
        for i, (flow_i, valid_i) in enumerate(zip(flow, valid)):
            safe_flow = flow_i.copy()
            safe_flow[:, ~valid_i.astype(bool)] = 0.0
            flow_group.create_dataset(f"flow{i:09d}", data=safe_flow, compression="gzip", compression_opts=1)
            valid_group.create_dataset(f"flow_valid{i:09d}", data=valid_i.astype(np.uint8), compression="gzip", compression_opts=1)

    return {
        "generator": generator,
        "episode_id": episode_id,
        "split": str(episode.attrs["split"]),
        "path": str(output_path),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "event_count": int(len(events)),
        "interval_event_counts": interval_counts.astype(int).tolist(),
        "image_count": int(len(frame_times)),
        "flow_count": int(len(flow)),
        "flow_label_sha256": label_hash,
        "valid_pixels_per_interval": valid.reshape(len(valid), -1).sum(axis=1).astype(int).tolist(),
        "resolution_hw": [height, width],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-h5", type=Path, required=True)
    parser.add_argument("--v2e-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)

    manifests = []
    with h5py.File(args.trajectory_h5, "r") as source:
        episode_ids = sorted(source["episodes"].keys())
        for generator in ("eventgenesis", "v2e"):
            for episode_id in episode_ids:
                episode = source[f"episodes/{episode_id}"]
                split = str(episode.attrs["split"])
                if split not in ("train", "validation"):
                    raise ValueError(f"unexpected split {split!r} for {episode_id}")
                output = args.output_root / generator / split / f"{episode_id}.h5"
                manifests.append(export_episode(source, generator, episode_id, args.v2e_root, output))

    by_pair = {(row["generator"], row["episode_id"]): row for row in manifests}
    for episode_id in sorted({row["episode_id"] for row in manifests}):
        if by_pair[("eventgenesis", episode_id)]["flow_label_sha256"] != by_pair[("v2e", episode_id)]["flow_label_sha256"]:
            raise RuntimeError(f"paired flow labels differ for {episode_id}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "conversion_manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
