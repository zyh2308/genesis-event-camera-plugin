#!/usr/bin/env python3
"""Build official STE-FlowNet polarity/Gaussian inputs from Stage-0 raw events."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


SIZE = 256
SLICES = 5


def gaussian_count_representation(events: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return (polarity=ON/OFF, H, W, time-slice) uint8 maps.

    This follows the upstream equal-event-count Gaussian-weighted encoding,
    while retaining remainder events and explicitly guarding empty or tied-time
    slices. Any count beyond uint8 range fails closed instead of wrapping.
    """
    result = np.zeros((2, height, width, SLICES), dtype=np.uint8)
    for slice_index, event_slice in enumerate(np.array_split(events, SLICES)):
        if len(event_slice) == 0:
            continue
        ts = event_slice[:, 0].astype(np.float64)
        span = float(ts.max() - ts.min())
        normalized_ts = (ts - ts.min()) / span if span > 0 else np.zeros_like(ts)
        x = event_slice[:, 1].astype(np.int64)
        y = event_slice[:, 2].astype(np.int64)
        p = event_slice[:, 3].astype(np.int8)

        for polarity_index, polarity in enumerate((1, -1)):
            selected = p == polarity
            if not np.any(selected):
                continue
            polarity_ts = normalized_ts[selected]
            if len(polarity_ts) > 1:
                std = float(np.std(polarity_ts, ddof=1)) + 1e-4
            else:
                std = 1.0
            mean = float(polarity_ts.mean())
            gaussian = (1.0 / np.sqrt(2.0 * np.pi) / std) * np.exp(
                -((polarity_ts - mean) ** 2) / (2.0 * std**2)
            )
            normalizer = float(np.abs(p[selected]).sum()) / float(gaussian.sum())
            per_event_count = np.ceil(gaussian * normalizer).astype(np.uint32)
            plane = np.zeros((height, width), dtype=np.uint32)
            np.add.at(plane, (y[selected], x[selected]), per_event_count)
            maximum = int(plane.max())
            if maximum > 255:
                raise OverflowError(
                    f"STE representation exceeds uint8 at slice {slice_index}, "
                    f"polarity {polarity}: max={maximum}; no wrap/saturation policy selected"
                )
            result[polarity_index, :, :, slice_index] = plane.astype(np.uint8)
    return result


def read_events(generator: str, episode: h5py.Group, v2e_root: Path) -> np.ndarray:
    if generator == "eventgenesis":
        raw = np.asarray(episode["eventgenesis_events_txyp"], dtype=np.float64)
        return np.stack((raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]), axis=1)
    episode_id = episode.name.rsplit("/", 1)[-1]
    with h5py.File(v2e_root / f"{episode_id}.h5", "r") as source:
        raw = np.asarray(source["events"], dtype=np.uint32)
    return np.stack((raw[:, 0].astype(np.float64) * 1e-6,
                     raw[:, 1].astype(np.float64),
                     raw[:, 2].astype(np.float64),
                     np.where(raw[:, 3] > 0, 1.0, -1.0)), axis=1)


def pad_center(array: np.ndarray, y_offset: int, x_offset: int, target_h: int, target_w: int) -> np.ndarray:
    shape = (*array.shape[:-2], target_h, target_w)
    padded = np.zeros(shape, dtype=array.dtype)
    height, width = array.shape[-2:]
    padded[..., y_offset:y_offset + height, x_offset:x_offset + width] = array
    return padded


def convert_episode(source: h5py.File, generator: str, episode_id: str, v2e_root: Path,
                    output: Path) -> dict:
    episode = source[f"episodes/{episode_id}"]
    times = np.asarray(episode["frame_timestamps_s"], dtype=np.float64)
    raw = read_events(generator, episode, v2e_root)
    keep = (raw[:, 0] > times[0]) & (raw[:, 0] <= times[-1])
    raw = raw[keep]
    if len(raw) and not np.all(np.diff(raw[:, 0]) >= 0):
        raise ValueError(f"unsorted {generator} events in {episode_id}")
    height, width = episode["v2e_display_luma_u8"].shape[1:]
    if len(raw) and (raw[:, 1].min() < 0 or raw[:, 1].max() >= width or
                     raw[:, 2].min() < 0 or raw[:, 2].max() >= height):
        raise ValueError(f"out-of-range event coordinate in {generator}/{episode_id}")

    x_offset = (SIZE - width) // 2
    y_offset = (SIZE - height) // 2
    interval_start = 0
    reps, counts = [], []
    for left, right in zip(times[:-1], times[1:]):
        interval_end = int(np.searchsorted(raw[:, 0], right, side="right"))
        interval_events = raw[interval_start:interval_end]
        # Check and preserve the shared (t_i,t_i+1] window definition.
        if np.any(interval_events[:, 0] <= left) or np.any(interval_events[:, 0] > right):
            raise ValueError(f"event interval boundary mismatch at [{left},{right}]")
        local = interval_events.copy()
        local[:, 0] -= left  # retain float64 relative time before any tensor conversion
        representation = gaussian_count_representation(local, height, width)
        padded = np.zeros((2, SIZE, SIZE, SLICES), dtype=representation.dtype)
        padded[:, y_offset:y_offset + height, x_offset:x_offset + width, :] = representation
        reps.append(padded)
        counts.append(int(len(interval_events)))
        interval_start = interval_end
    if interval_start != len(raw):
        raise ValueError(f"some events were not assigned to a frame window: {len(raw)-interval_start}")
    if generator == "eventgenesis":
        expected = np.asarray(episode["eventgenesis_interval_event_counts"], dtype=np.int64)
        if not np.array_equal(np.asarray(counts, dtype=np.int64), expected):
            raise ValueError(f"EventGenesis window counts differ: {counts} != {expected.tolist()}")

    # Use the exact same fixed-mapped gray endpoints for both generators.
    gray = np.asarray(episode["v2e_display_luma_u8"], dtype=np.float32) / 255.0
    gray = pad_center(gray, y_offset, x_offset, SIZE, SIZE)
    flow = np.asarray(episode["flow_forward_px_per_interval"], dtype=np.float32)
    valid = np.asarray(episode["flow_valid"], dtype=np.uint8)
    flow = pad_center(flow, y_offset, x_offset, SIZE, SIZE)
    valid = pad_center(valid, y_offset, x_offset, SIZE, SIZE)

    input_events = np.stack(reps, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        events=input_events,
        gray_prev=gray[:-1],
        gray_next=gray[1:],
        flow_gt=flow,
        flow_valid=valid,
        interval_event_counts=np.asarray(counts, dtype=np.int32),
        source_resolution_hw=np.asarray([height, width], dtype=np.int32),
        pad_yx=np.asarray([y_offset, x_offset], dtype=np.int32),
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {
        "generator": generator,
        "episode_id": episode_id,
        "path": str(output),
        "sha256": digest,
        "event_count": int(len(raw)),
        "interval_event_counts": counts,
        "representation_shape": list(input_events.shape),
        "representation_dtype": str(input_events.dtype),
        "max_count": int(input_events.max(initial=0)),
        "gray_shape": list(gray.shape),
        "flow_shape": list(flow.shape),
        "flow_valid_pixels": int(valid.sum()),
        "center_padding_yx": [y_offset, x_offset],
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
                manifest.append(convert_episode(
                    source, generator, episode_id, args.v2e_root,
                    args.output_root / generator / f"{split}.npz",
                ))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
