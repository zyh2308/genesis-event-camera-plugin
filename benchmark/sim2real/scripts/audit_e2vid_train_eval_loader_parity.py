#!/usr/bin/env python3
"""Compare the actual E2VID HDF5 and EVREAL memmap input loaders on paired data.

The HDF5 file is converted in a temporary directory to the EVREAL file schema;
both loaders consume the same raw event arrays, target images, and endpoint
times. This checks event windows/voxel tensors/target scaling, not model quality.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evreal_dataset(evreal_root: Path):
    # event_cnn_minimal and EVREAL both expose a top-level package named utils.
    # The benchmark process starts clean; remove the training package before
    # loading EVREAL so its voxel implementation is genuinely imported.
    for key in list(sys.modules):
        if key == "utils" or key.startswith("utils."):
            del sys.modules[key]
    sys.path[:] = [str(evreal_root)] + [p for p in sys.path if p != str(evreal_root)]
    source = evreal_root / "dataset.py"
    spec = importlib.util.spec_from_file_location("stage7_pinned_evreal_dataset", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load EVREAL dataset source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != source.resolve():
        raise RuntimeError("EVREAL dataset source path mismatch")
    return module.MemMapDataset


def load_training_dataset(model_root: Path):
    sys.path[:] = [str(model_root)] + [p for p in sys.path if p != str(model_root)]
    from data_loader.dataset import DynamicH5Dataset

    class RegisteredStage5Dataset(DynamicH5Dataset):
        # This is the exact Stage-5 correction: use right-inclusive endpoint
        # offsets. Independently check these against HDF5 event_idx metadata.
        def compute_frame_indices(self):
            event_ts = np.asarray(self.h5_file["events/ts"], dtype=np.float64)
            result, start = [], 0
            for timestamp in self.frame_ts:
                end = int(np.searchsorted(event_ts, timestamp, side="right"))
                result.append([start, end])
                start = end
            return result

    return RegisteredStage5Dataset


def run_one(h5_path: Path, arm: str, model_root: Path, evreal_root: Path) -> dict:
    import torch

    torch.set_num_threads(2)
    TrainingDataset = load_training_dataset(model_root)
    train = TrainingDataset(
        str(h5_path), sensor_resolution=None, num_bins=5,
        voxel_method={"method": "between_frames"}, max_length=None,
        combined_voxel_channels=True, filter_hot_events=False, transforms={},
    )
    if len(train) != 200 or len(train.frame_ts) != 201:
        raise RuntimeError(f"expected 200 intervals + 201 endpoint rows, got {len(train)}/{len(train.frame_ts)}")

    with h5py.File(h5_path, "r") as h5:
        event_ts = np.asarray(h5["events/ts"], dtype=np.float64)
        xs = np.asarray(h5["events/xs"], dtype=np.uint16)
        ys = np.asarray(h5["events/ys"], dtype=np.uint16)
        polarity = np.asarray(h5["events/ps"], dtype=np.uint8)
        frame_names = sorted(h5["images"].keys())
        frames = np.stack([np.asarray(h5[f"images/{name}"], dtype=np.uint8) for name in frame_names])
        frame_ts = np.asarray([h5[f"images/{name}"].attrs["timestamp"] for name in frame_names], dtype=np.float64)
        stored_ends = np.asarray([h5[f"images/{name}"].attrs["event_idx"] for name in frame_names], dtype=np.int64)
        height, width = map(int, h5.attrs["sensor_resolution"][:2])
        flow_unit = str(h5.attrs.get("flow_label_contract", ""))
        manifest_event_window = str(h5.attrs.get("event_window", ""))

    expected_ends = np.searchsorted(event_ts, frame_ts, side="right").astype(np.int64)
    if not np.array_equal(stored_ends, expected_ends):
        raise RuntimeError("stored HDF5 endpoint offsets differ from independent NumPy right-search")
    if not np.array_equal(np.asarray(train.frame_ts, dtype=np.float64), frame_ts):
        raise RuntimeError("training loader frame timestamps differ from raw HDF5 timestamps")

    # EVREAL between_frames needs an initial boundary image. It is not sent to
    # the model. HDF5 frame 0 is the first inferable target at t=10 ms; the
    # registered trajectory starts at t=0. Reconstruct only the event-only
    # adapter arrays in a private temporary directory.
    if not np.allclose(np.diff(frame_ts[:2]), 0.01, rtol=0, atol=1e-7):
        raise RuntimeError("registered synthetic target interval is not 10 ms")
    anchor_ts = float(frame_ts[0] - 0.01)
    eval_images = np.concatenate((np.zeros((1, height, width), dtype=np.uint8), frames[:200]), axis=0)[..., None]
    eval_times = np.concatenate(([anchor_ts], frame_ts[:200]))
    eval_offsets = np.concatenate(([0], expected_ends[:200])).astype(np.int64)
    with tempfile.TemporaryDirectory(prefix=f"sim2real-e2vid-{arm}-") as temp_name:
        sequence = Path(temp_name) / "sequence"
        sequence.mkdir()
        np.save(sequence / "events_ts.npy", event_ts.astype(np.float64), allow_pickle=False)
        np.save(sequence / "events_xy.npy", np.stack((xs, ys), axis=1).astype(np.int32), allow_pickle=False)
        np.save(sequence / "events_p.npy", polarity.astype(np.uint8), allow_pickle=False)
        np.save(sequence / "images.npy", eval_images, allow_pickle=False)
        np.save(sequence / "images_ts.npy", eval_times.astype(np.float64)[:, None], allow_pickle=False)
        np.save(sequence / "image_event_indices.npy", eval_offsets[:, None], allow_pickle=False)
        (sequence / "metadata.json").write_text(json.dumps({"sensor_resolution": [height, width]}) + "\n")

        # Exercise the actual evaluation loader after the actual training loader
        # has been imported. The helper isolates EVREAL's top-level `utils`.
        MemMapDataset = load_evreal_dataset(evreal_root)
        evaluation = MemMapDataset(
            str(sequence), sensor_resolution=(height, width), num_bins=5,
            voxel_method={"method": "between_frames"}, keep_ratio=1,
        )
        source_loader_length = len(evaluation)
        if source_loader_length != evaluation.num_frames - 1 or len(evaluation.frames_to_use) != evaluation.num_frames:
            raise RuntimeError("unexpected pinned EVREAL between_frames length/index contract")
        # Match the inference adapter: index 0 is an empty t0 boundary, and
        # indices 1..N-1 are the 200 actual model inputs.
        evaluation.length = evaluation.num_frames
        if len(evaluation) != 201:
            raise RuntimeError(f"expected 201 EVREAL indices including boundary, got {len(evaluation)}")

        mismatches = []
        exact_tensor_count = 0
        nonfinite = 0
        sparse_windows = 0
        zero_span_windows = 0
        max_abs = 0.0
        input_hash = hashlib.sha256()
        target_hash = hashlib.sha256()
        interval_counts = []
        started = time.monotonic()
        for index in range(200):
            train_item = train[index]
            eval_item = evaluation[index + 1]
            lo = 0 if index == 0 else int(expected_ends[index - 1])
            hi = int(expected_ends[index])
            independently_counted = hi - lo
            interval_counts.append(independently_counted)
            if int(eval_item["event_count"]) != independently_counted:
                raise RuntimeError(f"EVREAL event count mismatch at interval {index}")
            if independently_counted < 3:
                sparse_windows += 1
            if independently_counted > 0 and event_ts[lo] == event_ts[hi - 1]:
                zero_span_windows += 1
            train_voxel = train_item["events"].detach().cpu().numpy()
            eval_voxel = eval_item["events"].detach().cpu().numpy()
            if not np.isfinite(train_voxel).all() or not np.isfinite(eval_voxel).all():
                nonfinite += 1
            if train_voxel.shape != eval_voxel.shape:
                mismatches.append({"index": index, "reason": "shape", "train": list(train_voxel.shape), "eval": list(eval_voxel.shape)})
                continue
            delta = np.abs(train_voxel - eval_voxel)
            local_max = float(delta.max(initial=0.0))
            max_abs = max(max_abs, local_max)
            if np.array_equal(train_voxel, eval_voxel):
                exact_tensor_count += 1
            else:
                mismatches.append({"index": index, "event_count": independently_counted,
                                   "max_abs": local_max,
                                   "train_range": [float(train_voxel.min()), float(train_voxel.max())],
                                   "eval_range": [float(eval_voxel.min()), float(eval_voxel.max())]})
            train_frame = train_item["frame"].detach().cpu().numpy()
            eval_frame = eval_item["frame"].detach().cpu().numpy()
            if not np.array_equal(train_frame, eval_frame):
                raise RuntimeError(f"target image scaling/content differs at interval {index}")
            input_hash.update(np.ascontiguousarray(train_voxel).view(np.uint8))
            target_hash.update(np.ascontiguousarray(train_frame).view(np.uint8))
            train_time = float(np.asarray(train_item["timestamp"]).reshape(-1)[0])
            eval_time = float(eval_item["frame_timestamp"].item())
            if train_time != eval_time or train_time != float(frame_ts[index]):
                raise RuntimeError(f"target timestamp mismatch at interval {index}: {train_time}/{eval_time}")
        duration = time.monotonic() - started

    h5_attrs = {}
    with h5py.File(h5_path, "r") as h5:
        for key in ("sim2real_generator", "sim2real_episode_id", "sim2real_split"):
            h5_attrs[key] = str(h5.attrs.get(key, ""))
    return {
        "arm": arm,
        "hdf5_path": str(h5_path.resolve()),
        "hdf5_size_bytes": h5_path.stat().st_size,
        "hdf5_sha256": sha256(h5_path),
        "dataset_attributes": h5_attrs,
        "training_source_sha256": {
            "dataset_py": sha256(model_root / "data_loader/dataset.py"),
            "event_utils_py": sha256(model_root / "events_contrast_maximization/utils/event_utils.py"),
        },
        "evaluation_source_sha256": {
            "dataset_py": sha256(evreal_root / "dataset.py"),
            "event_utils_py": sha256(evreal_root / "utils/event_utils.py"),
        },
        "event_window_contract": {
            "stored_endpoint_offsets_equal_numpy_searchsorted_right": True,
            "manifest_event_window": manifest_event_window,
            "flow_label_contract": flow_unit,
            "training_and_eval_interval_mapping": "training index i is compared with EVREAL between_frames index i+1 after an unobserved t0 boundary",
        },
        "intervals": 200,
        "evreal_source_loader_length_before_extension": source_loader_length,
        "event_count_min_median_max": [int(min(interval_counts)), float(np.median(interval_counts)), int(max(interval_counts))],
        "intervals_with_lt3_events": sparse_windows,
        "nonempty_intervals_with_zero_event_time_span": zero_span_windows,
        "input_tensor_exact_matches": exact_tensor_count,
        "input_tensor_mismatches": mismatches,
        "nonfinite_input_windows": nonfinite,
        "max_abs_voxel_difference": max_abs,
        "all_target_images_and_timestamps_exact": True,
        "concatenated_training_input_sha256": input_hash.hexdigest(),
        "concatenated_target_sha256": target_hash.hexdigest(),
        "comparison_wall_seconds": duration,
        "scope": "synthetic loader/tensor parity only; no model quality claim and no real data",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--evreal-root", type=Path, required=True)
    parser.add_argument("--eventgenesis-h5", type=Path, required=True)
    parser.add_argument("--v2e-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output}")
    result = {
        "schema": "e2vid_train_eval_loader_parity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "torch": None,
        "numpy": np.__version__,
        "status": "pending",
        "arms": [],
        "real_data_used": False,
        "real_scoring_enabled": False,
    }
    import torch
    result["torch"] = torch.__version__
    result["arms"] = [
        run_one(args.eventgenesis_h5.resolve(), "eventgenesis", args.model_root.resolve(), args.evreal_root.resolve()),
        run_one(args.v2e_h5.resolve(), "v2e", args.model_root.resolve(), args.evreal_root.resolve()),
    ]
    passed = all(not a["input_tensor_mismatches"] and a["nonfinite_input_windows"] == 0
                 and a["input_tensor_exact_matches"] == a["intervals"] for a in result["arms"])
    result["status"] = "pass_tensor_parity" if passed else "mismatch_detected"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output),
                      "arms": [{"arm": a["arm"], "matches": a["input_tensor_exact_matches"],
                                "mismatches": len(a["input_tensor_mismatches"]),
                                "sparse": a["intervals_with_lt3_events"],
                                "zero_span": a["nonempty_intervals_with_zero_event_time_span"],
                                "max_abs": a["max_abs_voxel_difference"]} for a in result["arms"]]}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
