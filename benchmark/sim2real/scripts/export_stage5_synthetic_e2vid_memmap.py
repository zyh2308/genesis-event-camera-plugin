#!/usr/bin/env python3
"""Export one paired Stage-5 synthetic rollout to the pinned EVREAL MemMap schema.

This is an inference/metric plumbing fixture. It does not change the candidate
checkpoint, training split, or model selection. The reconstructed target comes
from the same shared synthetic endpoint that both generators see; only raw
events differ by generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from build_e2vid_stage4_10ms_data import load_generator_events


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-h5", type=Path, required=True)
    parser.add_argument("--rollout-id", required=True)
    parser.add_argument("--generator", choices=("eventgenesis", "v2e"), required=True)
    parser.add_argument("--v2e-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    bank_path = args.bank_h5.resolve()
    v2e_root = args.v2e_root.resolve()
    output_root = args.output_root.resolve()
    if not bank_path.is_file() or not v2e_root.is_dir():
        raise FileNotFoundError(f"bank/V2E root missing: {bank_path}, {v2e_root}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite synthetic fixture: {output_root}")

    with h5py.File(bank_path, "r") as bank:
        episode_path = f"episodes/{args.rollout_id}"
        if episode_path not in bank:
            raise KeyError(f"rollout absent from registered physical bank: {args.rollout_id}")
        episode = bank[episode_path]
        times = np.asarray(episode["frame_timestamps_s"], dtype=np.float64)
        target_luma = np.asarray(episode["v2e_display_luma_u8"], dtype=np.uint8)
        if times.ndim != 1 or target_luma.ndim != 3 or len(times) != len(target_luma):
            raise RuntimeError("shared synthetic timestamps and luma endpoint arrays are inconsistent")
        if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise RuntimeError("shared synthetic endpoints must be finite and strictly increasing")
        events, event_source = load_generator_events(args.generator, episode, v2e_root)
        split = str(episode.attrs["split"])
        stratum = str(episode.attrs["stratum"])
        height, width = map(int, target_luma.shape[1:])

    # EVREAL's between_frames loader uses image 0 as the sequence boundary:
    # sample 0 pairs that endpoint with an empty event window, and samples
    # 1..N-1 use events in (t[i-1], t[i]] for the matching target image.
    # Retain the t0 image so all later right endpoints stay aligned.
    target_times = times
    target_images = target_luma[:, :, :, None]
    events = events[(events[:, 0] > times[0]) & (events[:, 0] <= times[-1])]
    if len(events) and (events[:, 1].min() < 0 or events[:, 1].max() >= width
                        or events[:, 2].min() < 0 or events[:, 2].max() >= height):
        raise RuntimeError("raw synthetic events lie outside the sensor raster")
    event_indices = np.searchsorted(events[:, 0], target_times, side="right").astype(np.int64)
    if len(event_indices) != len(target_times) or event_indices[-1] != len(events):
        raise RuntimeError("synthetic endpoint offsets do not partition the complete event stream")
    if len(events):
        has_previous = event_indices > 0
        has_next = event_indices < len(events)
        if np.any(events[event_indices[has_previous] - 1, 0] > target_times[has_previous]):
            raise RuntimeError("an event included at an endpoint occurs after that target timestamp")
        if np.any(events[event_indices[has_next], 0] <= target_times[has_next]):
            raise RuntimeError("an event at or before a target timestamp was omitted by end-exclusive offsets")

    output_root.mkdir(parents=True)
    sequence_name = f"{args.rollout_id}_{args.generator}"
    sequence_root = output_root / sequence_name
    sequence_root.mkdir()
    arrays = {
        "events_ts.npy": events[:, 0].astype(np.float64),
        "events_xy.npy": events[:, 1:3].astype(np.int32),
        "events_p.npy": events[:, 3].astype(np.uint8),
        "images.npy": target_images.astype(np.uint8),
        "images_ts.npy": target_times.astype(np.float64)[:, None],
        "image_event_indices.npy": event_indices.astype(np.int64)[:, None],
    }
    file_rows = {}
    for name, array in arrays.items():
        path = sequence_root / name
        np.save(path, array, allow_pickle=False)
        file_rows[name] = {
            "path": str(path), "shape": list(array.shape), "dtype": str(array.dtype),
            "size_bytes": path.stat().st_size, "sha256": sha256(path),
            "array_sha256": sha256_array(array),
        }
    metadata = {
        "sensor_resolution": [height, width],
        "dataset_role": "synthetic_pipeline_fixture_only",
        "target_image_mapping": "shared Rec.709 linear luminance, exposure 1, Reinhard, sRGB transfer, uint8 round-to-nearest",
        "event_interval": "sample 0 is the t0 recurrent boundary with an empty event interval; samples i>0 use (t[i-1],t[i]]",
    }
    metadata_path = sequence_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    file_rows["metadata.json"] = {
        "path": str(metadata_path), "size_bytes": metadata_path.stat().st_size,
        "sha256": sha256(metadata_path),
    }
    manifest = {
        "schema": "stage5_synthetic_e2vid_memmap_fixture_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "synthetic_inference_and_metric_plumbing_only",
        "real_data_used": False,
        "real_scoring_enabled": False,
        "checkpoint_selection_allowed": False,
        "physical_bank_path": str(bank_path),
        "physical_bank_size_bytes": bank_path.stat().st_size,
        "physical_bank_sha256": sha256(bank_path),
        "rollout_id": args.rollout_id,
        "split": split,
        "stratum": stratum,
        "generator": args.generator,
        "event_source": event_source,
        "event_window": "sample 0 is t0 with zero events; sample i>0 is (t[i-1],t[i]]",
        "target_times": [float(target_times[0]), float(target_times[-1])],
        "image_endpoint_count_including_anchor": int(len(target_images)),
        "target_frame_count": int(len(target_images) - 1),
        "initial_anchor_excluded_from_model_targets": True,
        "event_count": int(len(events)),
        "sensor_resolution_hw": [height, width],
        "event_tensor_input_to_model": "EVREAL pinned MemMapDataset, 5 signed temporal bins, between_frames; reference image discarded before model call",
        "shared_target_image_sha256": sha256_array(target_images),
        "files": file_rows,
    }
    manifest_path = output_root / "synthetic_fixture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": manifest["status"], "fixture_root": str(output_root),
                      "sequence_root": str(sequence_root), "rollout_id": args.rollout_id,
                      "generator": args.generator, "event_count": len(events),
                      "image_endpoints_including_anchor": len(target_images),
                      "inferable_target_frames": len(target_images) - 1,
                      "manifest_sha256": sha256(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
