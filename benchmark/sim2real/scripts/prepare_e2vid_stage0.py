#!/usr/bin/env python3
"""Convert the tiny shared trajectory to the pinned E2VID+ HDF5 loader schema.

This is a data-format adapter only. It does not modify the upstream source or
use any real data. The explicit right-endpoint interval search fixes the
upstream binary search's ambiguous behavior on duplicate event timestamps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def load_events(generator: str, episode: h5py.Group, v2e_root: Path | None) -> tuple[np.ndarray, dict]:
    if generator == "eventgenesis":
        raw = np.asarray(episode["eventgenesis_events_txyp"], dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 4:
            raise ValueError(f"unexpected EventGenesis event shape {raw.shape}")
        times = raw[:, 0]
        xs, ys = raw[:, 1], raw[:, 2]
        ps = (raw[:, 3] > 0).astype(np.uint8)
        metadata = {"source_schema": "t_seconds,x,y,p_signed", "timestamp_unit": "s"}
    elif generator == "v2e":
        if v2e_root is None:
            raise ValueError("--v2e-root is required for generator=v2e")
        episode_id = episode.name.rsplit("/", 1)[-1]
        path = v2e_root / f"{episode_id}.h5"
        with h5py.File(path, "r") as event_file:
            raw = np.asarray(event_file["events"], dtype=np.uint32)
        if raw.ndim != 2 or raw.shape[1] != 4:
            raise ValueError(f"unexpected V2E event shape {raw.shape} in {path}")
        times = raw[:, 0].astype(np.float64) * 1e-6
        xs, ys = raw[:, 1], raw[:, 2]
        ps = raw[:, 3].astype(np.uint8)
        metadata = {"source_schema": "t_microseconds,x,y,p_01", "timestamp_unit": "us"}
    else:
        raise ValueError(f"unsupported generator: {generator}")
    return np.stack((times, xs, ys, ps), axis=1), metadata


def write_one(
    source_h5: h5py.File,
    generator: str,
    episode_id: str,
    v2e_root: Path | None,
    output_path: Path,
) -> dict:
    episode = source_h5[f"episodes/{episode_id}"]
    endpoint_times = np.asarray(episode["frame_timestamps_s"], dtype=np.float64)
    luma = np.asarray(episode["v2e_display_luma_u8"], dtype=np.uint8)
    flow_disp = np.asarray(episode["flow_forward_px_per_interval"], dtype=np.float32)
    flow_valid = np.asarray(episode["flow_valid"], dtype=bool)
    height, width = luma.shape[1:]
    if len(endpoint_times) != len(luma) or len(flow_disp) != len(endpoint_times) - 1:
        raise ValueError(f"inconsistent frame/flow lengths for {episode_id}")

    raw, event_metadata = load_events(generator, episode, v2e_root)
    t0, t_last = endpoint_times[0], endpoint_times[-1]
    keep = (raw[:, 0] > t0) & (raw[:, 0] <= t_last)
    raw = raw[keep]
    if len(raw) and not np.all(np.diff(raw[:, 0]) >= 0):
        raise ValueError(f"events are not time sorted: {generator}/{episode_id}")
    xs, ys = raw[:, 1].astype(np.uint16), raw[:, 2].astype(np.uint16)
    if len(raw) and (xs.max() >= width or ys.max() >= height):
        raise ValueError(f"event coordinates out of bounds: {generator}/{episode_id}")
    ps = raw[:, 3].astype(np.uint8)
    if len(raw) and not np.isin(ps, (0, 1)).all():
        raise ValueError(f"invalid polarity values: {generator}/{episode_id}")
    times = raw[:, 0].astype(np.float64)

    frame_times = np.concatenate((endpoint_times[1:], endpoint_times[-1:]))
    end_indices = np.searchsorted(times, frame_times, side="right")
    if np.any(np.diff(end_indices) < 0) or end_indices[-2] != len(times):
        raise ValueError(f"endpoint event partition does not cover all events: {generator}/{episode_id}")
    interval_starts = np.concatenate(([0], end_indices[:-2]))
    interval_ends = end_indices[:-1]
    interval_counts = interval_ends - interval_starts

    if generator == "eventgenesis":
        expected_counts = np.asarray(episode["eventgenesis_interval_event_counts"], dtype=np.int64)
        # The plugin's first capture interval is the shared (t_i,t_i+1] window.
        if not np.array_equal(interval_counts, expected_counts):
            raise ValueError(
                f"EventGenesis HDF5 timestamp partition differs from capture chunks: "
                f"{interval_counts.tolist()} != {expected_counts.tolist()}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as out:
        out.attrs["sensor_resolution"] = np.asarray([height, width], dtype=np.int32)
        out.attrs["num_events"] = np.int64(len(raw))
        out.attrs["num_imgs"] = np.int64(len(frame_times))
        out.attrs["source"] = "unknown"
        out.attrs["sim2real_generator"] = generator
        out.attrs["sim2real_episode_id"] = episode_id
        out.attrs["event_window"] = "(t_i,t_i_plus_1]"
        out.attrs["event_timestamp_unit"] = "seconds"
        out.attrs["polarity_encoding"] = "OFF=0, ON=1; upstream loader maps 2p-1"
        out.attrs["target_image_mapping"] = "shared fixed radiance-to-display luma uint8"
        out.attrs["flow_label_adapter"] = "per-interval displacement divided by observed event dt; upstream multiplies by dt"
        out.attrs["event_source_metadata"] = json.dumps(event_metadata, sort_keys=True)

        event_group = out.create_group("events")
        event_group.create_dataset("xs", data=xs, compression="gzip", compression_opts=1)
        event_group.create_dataset("ys", data=ys, compression="gzip", compression_opts=1)
        event_group.create_dataset("ts", data=times, compression="gzip", compression_opts=1)
        event_group.create_dataset("ps", data=ps, compression="gzip", compression_opts=1)
        event_group.create_dataset("interval_counts", data=interval_counts)

        images = out.create_group("images")
        right_endpoint_images = np.concatenate((luma[1:], luma[-1:]), axis=0)
        for index, (frame, timestamp, event_index) in enumerate(
            zip(right_endpoint_images, frame_times, end_indices)
        ):
            image = images.create_dataset(f"image{index:09d}", data=frame, compression="gzip", compression_opts=1)
            image.attrs["timestamp"] = np.float64(timestamp)
            image.attrs["event_idx"] = np.int64(event_index)

        flows = out.create_group("flow")
        for index in range(len(flow_disp)):
            start, end = int(interval_starts[index]), int(interval_ends[index])
            interval_times = times[start:end]
            if len(interval_times) < 3:
                flow_rate = np.zeros_like(flow_disp[index], dtype=np.float32)
            else:
                observed_dt = float(interval_times[-1] - interval_times[0])
                if observed_dt <= 0:
                    flow_rate = np.zeros_like(flow_disp[index], dtype=np.float32)
                else:
                    flow_rate = flow_disp[index].copy() / observed_dt
            flow_rate[:, ~flow_valid[index]] = 0.0
            flows.create_dataset(
                f"flow{index:09d}", data=flow_rate, compression="gzip", compression_opts=1
            )

    return {
        "generator": generator,
        "episode_id": episode_id,
        "output": str(output_path),
        "event_count": int(len(raw)),
        "interval_event_counts": interval_counts.tolist(),
        "image_count": int(len(frame_times)),
        "flow_count": int(len(flow_disp)),
        "resolution_hw": [int(height), int(width)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-h5", type=Path, required=True)
    parser.add_argument("--v2e-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = []
    with h5py.File(args.trajectory_h5, "r") as source:
        for generator in ("eventgenesis", "v2e"):
            for episode_id in ("train_00", "validation_00"):
                split = "train" if episode_id.startswith("train") else "validation"
                output = args.output_root / generator / f"{split}.h5"
                manifest.append(write_one(source, generator, episode_id, args.v2e_root, output))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
