#!/usr/bin/env python3
"""Event-only, sequential HQF inference adapter for a Stage-2 E2VID+ checkpoint.

This script uses the pinned EVREAL MemMapDataset only to construct its native
between-frames voxel input. It never loads EVREAL's public E2VID weights and
never scores or selects a candidate using HQF reference frames.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from verify_pinned_source import verify_pinned_source

EVREAL_COMMIT = "398f03551b7f5d150ae10aa89e4e731557fce240"
MODEL_COMMIT = "78aa471a66593f55e349303563997297fbc816e8"
REGISTERED_SEQUENCES = (
    "bike_bay_hdr", "boxes", "desk", "desk_fast", "desk_hand_only", "desk_slow",
    "engineering_posters", "high_texture_plants", "poster_pillar_1", "poster_pillar_2",
    "reflective_materials", "slow_and_fast_desk", "slow_hand", "still_life",
)
REGISTERED_SEQUENCE = REGISTERED_SEQUENCES[0]
REGISTERED_SAVE_RULE = "save every 100th adapter output (0,100,...) and the final adapter output; deduplicate"
EXPECTED_DATA_FILES = (
    "events_ts.npy",
    "events_xy.npy",
    "events_p.npy",
    "images.npy",
    "images_ts.npy",
    "image_event_indices.npy",
    "metadata.json",
)
SOURCE_BUNDLE_MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "downstream_source_bundle.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evreal_dataset_class(evreal_root: Path):
    evreal_root_s = str(evreal_root)
    if evreal_root_s not in sys.path:
        sys.path.insert(0, evreal_root_s)
    dataset_py = evreal_root / "dataset.py"
    spec = importlib.util.spec_from_file_location("sim2real_pinned_evreal_dataset", dataset_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned EVREAL dataset source: {dataset_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != dataset_py.resolve():
        raise RuntimeError("loaded MemMapDataset source does not match requested EVREAL checkout")
    return module.MemMapDataset


def load_e2vid_class(model_root: Path):
    package_dir = model_root / "model"
    init_py = package_dir / "__init__.py"
    # event_cnn_minimal's legacy model module uses absolute imports such as
    # `model.submodules`; register its package under that expected name. EVREAL's
    # `utils` package is already loaded by the dataset import and remains pinned.
    existing = sys.modules.get("model")
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() != init_py.resolve():
        raise RuntimeError("another Python package named 'model' is already loaded; run this adapter in a clean process")
    if existing is None:
        spec = importlib.util.spec_from_file_location(
            "model", init_py, submodule_search_locations=[str(package_dir)]
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import E2VID model package: {init_py}")
        package = importlib.util.module_from_spec(spec)
        sys.modules["model"] = package
        spec.loader.exec_module(package)
    module = importlib.import_module("model.model")
    return module.E2VIDRecurrent, module.CropParameters


def validate_sequence_files(sequence_path: Path) -> dict[str, Any]:
    missing = [name for name in EXPECTED_DATA_FILES if not (sequence_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"HQF sequence is missing required EVREAL files: {missing}")
    hashes = {name: sha256_file(sequence_path / name) for name in EXPECTED_DATA_FILES}
    arrays = {
        "events_ts": np.load(sequence_path / "events_ts.npy", mmap_mode="r").squeeze(),
        "events_xy": np.load(sequence_path / "events_xy.npy", mmap_mode="r"),
        "events_p": np.load(sequence_path / "events_p.npy", mmap_mode="r").squeeze(),
        "images": np.load(sequence_path / "images.npy", mmap_mode="r"),
        "images_ts": np.load(sequence_path / "images_ts.npy", mmap_mode="r").squeeze(),
        "image_event_indices": np.load(sequence_path / "image_event_indices.npy", mmap_mode="r").squeeze(),
    }
    if arrays["events_xy"].ndim != 2 or arrays["events_xy"].shape[1] != 2:
        raise ValueError(f"events_xy must have shape [N,2], got {arrays['events_xy'].shape}")
    if arrays["images"].ndim != 4 or arrays["images"].shape[-1] != 1:
        raise ValueError(f"images must have shape [N,H,W,1], got {arrays['images'].shape}")
    n_events = int(len(arrays["events_ts"]))
    n_frames = int(len(arrays["images_ts"]))
    if n_events <= 0 or n_frames < 2:
        raise ValueError(f"need non-empty events and at least two frames; got events={n_events}, frames={n_frames}")
    for key in ("events_ts", "events_p"):
        if arrays[key].ndim != 1:
            raise ValueError(f"{key} must be one-dimensional after squeeze, got {arrays[key].shape}")
    if len(arrays["events_xy"]) != n_events or len(arrays["events_p"]) != n_events:
        raise ValueError("event timestamp, coordinate, and polarity counts differ")
    if len(arrays["image_event_indices"]) != n_frames:
        raise ValueError("image_event_indices count must equal image count")
    if not np.isfinite(arrays["events_ts"]).all() or not np.isfinite(arrays["images_ts"]).all():
        raise ValueError("dataset timestamps contain NaN or Inf")
    if np.any(np.diff(arrays["events_ts"]) < 0):
        raise ValueError("event timestamps are not monotonically non-decreasing")
    if np.any(np.diff(arrays["images_ts"]) <= 0):
        raise ValueError("image timestamps are not strictly increasing")
    if not np.isin(np.unique(arrays["events_p"]), (0, 1)).all():
        raise ValueError("EVREAL expects raw event polarity encoded as 0/1")
    if not np.issubdtype(arrays["image_event_indices"].dtype, np.integer):
        raise ValueError("image_event_indices must use an integer dtype")
    if np.any(arrays["image_event_indices"] < 0) or np.any(arrays["image_event_indices"] > n_events):
        raise ValueError("image_event_indices contain out-of-range event offsets")
    height, width = map(int, arrays["images"].shape[1:3])
    xy = arrays["events_xy"]
    if len(xy) and (xy[:, 0].min() < 0 or xy[:, 0].max() >= width or xy[:, 1].min() < 0 or xy[:, 1].max() >= height):
        raise ValueError("event coordinates lie outside the image sensor resolution")
    metadata = json.loads((sequence_path / "metadata.json").read_text())
    metadata_resolution = tuple(int(v) for v in metadata["sensor_resolution"])
    if metadata_resolution != (height, width):
        raise ValueError(f"metadata sensor_resolution {metadata_resolution} != images resolution {(height, width)}")
    return {
        "hashes": hashes,
        "shapes": {key: list(value.shape) for key, value in arrays.items()},
        "dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "num_events": n_events,
        "num_frames": n_frames,
        "sensor_resolution_hw": [height, width],
        "events_timestamp_first_s": float(arrays["events_ts"][0]),
        "events_timestamp_last_s": float(arrays["events_ts"][-1]),
        "images_timestamp_first_s": float(arrays["images_ts"][0]),
        "images_timestamp_last_s": float(arrays["images_ts"][-1]),
        "metadata": metadata,
    }


def checkpoint_to_model(
    checkpoint_path: Path,
    device: torch.device,
    model_cls,
    model_args_override: dict[str, Any] | None = None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint must contain the candidate model state under key 'model'")
    checkpoint_args = checkpoint.get("model_args")
    if checkpoint_args is None:
        if model_args_override is None:
            raise ValueError("checkpoint lacks model_args; pass the registered Stage-5 model config")
        checkpoint_args = model_args_override
    elif model_args_override is not None and checkpoint_args != model_args_override:
        raise ValueError("checkpoint model_args differ from the explicitly registered model config")
    model_args = copy.deepcopy(checkpoint_args)
    if not isinstance(model_args, dict) or model_args.get("num_bins") != 5:
        raise ValueError("E2VID model_args must be a dict with num_bins=5")
    # The pinned constructor mutates its argument dict; retain the registered
    # values for the manifest and give the model a copy.
    model = model_cls(copy.deepcopy(model_args)).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict checkpoint restore mismatch: {incompatible}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint, model_args


def save_gray_png(path: Path, image01: np.ndarray) -> None:
    import cv2

    if image01.ndim != 2 or not np.isfinite(image01).all():
        raise ValueError(f"PNG input must be a finite grayscale image, got {image01.shape}")
    gray_u8 = np.clip(np.rint(image01 * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), gray_u8, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise IOError(f"cv2.imwrite failed for {path}")


def run_inference(
    *,
    evreal_root: Path,
    model_root: Path,
    sequence_path: Path,
    sequence_name: str,
    checkpoint_path: Path,
    output_root: Path,
    device_name: str,
    model_args_override: dict[str, Any] | None = None,
    self_test: bool = False,
) -> dict[str, Any]:
    started_utc = utc_now()
    wall_start = time.monotonic()
    output_root.mkdir(parents=True, exist_ok=True)
    sequence_path = sequence_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    evreal_root = evreal_root.resolve()
    model_root = model_root.resolve()
    if not self_test and sequence_name not in REGISTERED_SEQUENCES:
        raise ValueError(f"real HQF sequence must be one of the 14 registered sequences, got {sequence_name!r}")
    if not self_test and sequence_path.name != sequence_name:
        raise ValueError(f"HQF sequence directory must be named {sequence_name!r}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    source_verification = {
        "EVREAL": verify_pinned_source(evreal_root, "EVREAL", EVREAL_COMMIT, SOURCE_BUNDLE_MANIFEST),
        "event_cnn_minimal": verify_pinned_source(
            model_root, "event_cnn_minimal", MODEL_COMMIT, SOURCE_BUNDLE_MANIFEST
        ),
    }
    dataset_class = load_evreal_dataset_class(evreal_root)
    model_cls, crop_parameters_cls = load_e2vid_class(model_root)
    data_info = validate_sequence_files(sequence_path)
    # EVREAL MemMapDataset's pinned __len__ reports N-1 although its frame and
    # event-index tables contain N endpoints and __getitem__(N-1) is valid.
    # For this registered inference-only full-sequence smoke, extend only this
    # instance to include the final endpoint; no shared EVREAL source is edited.
    dataset = dataset_class(
        str(sequence_path),
        sensor_resolution=data_info["sensor_resolution_hw"],
        num_bins=5,
        voxel_method={"method": "between_frames"},
        keep_ratio=1,
    )
    source_loader_length = len(dataset)
    if source_loader_length != dataset.num_frames - 1 or len(dataset.frames_to_use) != dataset.num_frames:
        raise RuntimeError("unexpected pinned EVREAL MemMapDataset between_frames length/index contract")
    dataset.length = dataset.num_frames
    if len(dataset) != dataset.num_frames:
        raise RuntimeError("local full-sequence length extension failed")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    model, checkpoint, model_args = checkpoint_to_model(
        checkpoint_path, device, model_cls, model_args_override=model_args_override
    )
    reset_calls = 0
    original_reset = model.reset_states

    def counted_reset():
        nonlocal reset_calls
        reset_calls += 1
        return original_reset()

    model.reset_states = counted_reset
    model.reset_states()
    if reset_calls != 1:
        raise RuntimeError(f"expected exactly one recurrent-state reset at sequence start, got {reset_calls}")

    height, width = data_info["sensor_resolution_hw"]
    cropper = crop_parameters_cls(width, height, model.num_encoders)
    padded_hw = [int(cropper.height_crop_size), int(cropper.width_crop_size)]
    if padded_hw[0] < height or padded_hw[1] < width:
        raise RuntimeError("E2VID cropper produced a padded size smaller than the sensor image")
    save_indices = sorted(set(range(0, len(dataset), 100)) | {len(dataset) - 1})
    save_index_set = set(save_indices)
    samples: list[dict[str, Any]] = []
    saved_images: list[dict[str, Any]] = []
    output_min = math.inf
    output_max = -math.inf
    previous_target_ts: float | None = None
    saved_index_to_path = {index: output_root / f"reconstruction_{index:06d}.png" for index in save_indices}

    with torch.inference_mode():
        for sample_index in range(len(dataset)):
            item = dataset[sample_index]
            expected_target_ts = float(dataset.frame_ts[sample_index])
            frame_ts = float(item["frame_timestamp"].item())
            voxel_ts = float(item["voxel_timestamp"].item())
            if not math.isfinite(frame_ts) or not math.isfinite(voxel_ts):
                raise ValueError(f"non-finite target timestamp at sample {sample_index}")
            if not math.isclose(frame_ts, expected_target_ts, rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(f"loader frame timestamp mismatch at sample {sample_index}: {frame_ts} != {expected_target_ts}")
            if not math.isclose(voxel_ts, frame_ts, rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(f"between_frames voxel timestamp differs from image endpoint at sample {sample_index}")
            if previous_target_ts is not None and frame_ts <= previous_target_ts:
                raise ValueError(f"non-increasing sample target timestamps at sample {sample_index}")
            previous_target_ts = frame_ts

            # Derive the source event interval from the exact index table used by
            # MemMapDataset.__getitem__; end index is exclusive.
            previous_endpoint = dataset.frames_to_use[sample_index - 1] if sample_index > 0 else 0
            current_endpoint = dataset.frames_to_use[sample_index]
            event_start = int(dataset.get_event_indices(previous_endpoint)[1])
            event_end = int(dataset.get_event_indices(current_endpoint)[1])
            event_count_value = item["event_count"]
            event_count = int(event_count_value.item() if torch.is_tensor(event_count_value) else event_count_value)
            if event_start < 0 or event_end < event_start or event_end > dataset.num_events:
                raise RuntimeError(f"invalid EVREAL event slice [{event_start},{event_end}) at {sample_index}")
            if event_count != event_end - event_start:
                raise RuntimeError(f"EVREAL event count mismatch at {sample_index}: {event_count} != {event_end-event_start}")
            raw_times = dataset.filehandle["t"][event_start:event_end]
            if len(raw_times) and float(raw_times[-1]) > frame_ts + 1e-10:
                raise RuntimeError(f"event window contains an event after its target timestamp at sample {sample_index}")

            event_tensor = item["events"]
            if tuple(event_tensor.shape) != (5, height, width):
                raise ValueError(f"expected EVREAL events [5,{height},{width}], got {tuple(event_tensor.shape)}")
            if not torch.isfinite(event_tensor).all():
                raise ValueError(f"non-finite 5-bin event tensor at sample {sample_index}")
            # Reference pixels are deliberately discarded; only the 5-bin event
            # tensor enters the candidate network.
            reference_shape = tuple(item["frame"].shape)
            del item["frame"]
            model_input = event_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
            if tuple(model_input.shape) != (1, 5, height, width):
                raise ValueError(f"model input must be [1,5,H,W], got {tuple(model_input.shape)}")
            padded_input = cropper.pad(model_input)
            prediction = model(padded_input)
            if not isinstance(prediction, dict) or "image" not in prediction:
                raise TypeError("E2VID model output must be a dict containing 'image'")
            padded_image = prediction["image"]
            expected_padded_shape = (1, 1, padded_hw[0], padded_hw[1])
            if tuple(padded_image.shape) != expected_padded_shape:
                raise ValueError(f"padded model output must be {expected_padded_shape}, got {tuple(padded_image.shape)}")
            image = cropper.crop(padded_image)
            expected_shape = (1, 1, height, width)
            if tuple(image.shape) != expected_shape:
                raise ValueError(f"cropped model output must be [1,1,H,W]={expected_shape}, got {tuple(image.shape)}")
            if not torch.isfinite(image).all():
                raise ValueError(f"non-finite reconstruction output at sample {sample_index}")
            frame_min = float(image.min().item())
            frame_max = float(image.max().item())
            output_min = min(output_min, frame_min)
            output_max = max(output_max, frame_max)

            event_min_ts = float(raw_times[0]) if len(raw_times) else None
            event_max_ts = float(raw_times[-1]) if len(raw_times) else None
            samples.append({
                "sample_index": sample_index,
                "event_start_index_inclusive": event_start,
                "event_end_index_exclusive": event_end,
                "event_count": event_count,
                "event_timestamp_first_s": event_min_ts,
                "event_timestamp_last_s": event_max_ts,
                "target_frame_index": sample_index,
                "target_timestamp_s": frame_ts,
                "voxel_timestamp_s": voxel_ts,
                "event_tensor_shape": list(event_tensor.shape),
                "model_input_shape": list(model_input.shape),
                "model_padded_input_shape": list(padded_input.shape),
                "model_padded_output_shape": list(padded_image.shape),
                "cropped_output_shape": list(image.shape),
                "reference_frame_shape_checked_only": list(reference_shape),
                "output_min": frame_min,
                "output_max": frame_max,
            })
            if sample_index in save_index_set:
                array = image[0, 0].detach().cpu().numpy()
                output_path = saved_index_to_path[sample_index]
                save_gray_png(output_path, array)
                saved_images.append({
                    "sample_index": sample_index,
                    "target_timestamp_s": frame_ts,
                    "path": str(output_path.resolve()),
                    "sha256": sha256_file(output_path),
                    "shape_hw": [height, width],
                    "dtype": "uint8",
                    "conversion": "clip raw candidate output to [0,1], round(255*x), uint8; no data-dependent normalization",
                })

    if reset_calls != 1:
        raise RuntimeError(f"recurrent reset count changed during sequence; expected 1, got {reset_calls}")
    if len(samples) != data_info["num_frames"]:
        raise RuntimeError(f"processed {len(samples)} samples but expected all {data_info['num_frames']} frame endpoints")
    if sorted(row["sample_index"] for row in saved_images) != save_indices:
        raise RuntimeError("saved output subset differs from preregistered deterministic index rule")

    finished_utc = utc_now()
    manifest = {
        "schema_version": 1,
        "status": "mechanics_self_test_passed_not_real_HQF" if self_test else "fixed_sample_event_only_inference_completed_no_GT_scoring",
        "created_at_utc": started_utc,
        "finished_at_utc": finished_utc,
        "elapsed_seconds": time.monotonic() - wall_start,
        "task_scope": "candidate event-only reconstruction inference; no real-data score, checkpoint choice, or parameter tuning",
        "sequence": {
            "name": sequence_name,
            "path": str(sequence_path),
            "num_events": data_info["num_events"],
            "num_frames_in_memmap": data_info["num_frames"],
            "num_samples_processed": len(samples),
            "source_evreal_memmapdataset_length": source_loader_length,
            "adapter_length": len(dataset),
            "sequence_start_timestamp_s": samples[0]["target_timestamp_s"],
            "sequence_end_timestamp_s": samples[-1]["target_timestamp_s"],
            "timestamps_strictly_increasing": True,
            "event_time_unit": "seconds (EVREAL bag_to_npy conversion contract)",
            "frame_time_unit": "seconds (EVREAL bag_to_npy conversion contract)",
            "sensor_resolution_hw": data_info["sensor_resolution_hw"],
            "raw_event_schema": "events_xy[N,2]=(x,y) pixels; events_ts[N] seconds; events_p[N] uint8 0=OFF/1=ON; EVREAL maps p to -1/+1 for its between_frames voxelizer",
            "event_voxel_schema": "EVREAL MemMapDataset, num_bins=5, method=between_frames, shape [5,H,W], no additional event normalization",
            "image_array_schema": "images[N,H,W,1] used only by EVREAL item packaging for shape bookkeeping; no reference pixel enters model or score",
            "source_default_length_note": "pinned MemMapDataset.__len__ returns num_frames-1; this local inference-only adapter extends this instance to num_frames so the final legal endpoint is also processed; no EVREAL source files changed",
            "data_shapes": data_info["shapes"],
            "data_dtypes": data_info["dtypes"],
            "timestamp_bounds": {
                "events_first_s": data_info["events_timestamp_first_s"],
                "events_last_s": data_info["events_timestamp_last_s"],
                "images_first_s": data_info["images_timestamp_first_s"],
                "images_last_s": data_info["images_timestamp_last_s"],
            },
            "samples": samples,
        },
        "registered_inference": {
            "registration_sequence": sequence_name,
            "registered_sequence_inventory": list(REGISTERED_SEQUENCES),
            "sequence_selection": "explicit sequence argument from the fixed 14-sequence inventory",
            "saved_subset_rule": REGISTERED_SAVE_RULE,
            "saved_sample_indices": save_indices,
            "saved_images": saved_images,
        },
        "model": {
            "architecture": "event_cnn_minimal.model.model.E2VIDRecurrent",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_generator_metadata": checkpoint.get("generator"),
            "checkpoint_model_args": model_args,
            "checkpoint_state_dict_key": "model",
            "public_EVREAL_weights_loaded": False,
            "model_source_path": str(model_root),
            "model_source_commit": MODEL_COMMIT,
            "source_verification": source_verification,
            "device": str(device),
            "deterministic_inference": {
                "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
            "recurrent_reset_calls": reset_calls,
            "reset_policy": "one reset at sequence start; no reset inside sequence",
            "input_contract": "event-only; [1,5,H,W]",
            "padded_resolution_hw": padded_hw,
            "output_contract_before_crop": [1, 1, *padded_hw],
            "output_contract_after_crop": [1, 1, *data_info["sensor_resolution_hw"]],
            "finite_output_global_min": output_min,
            "finite_output_global_max": output_max,
            "reference_frame_pixels_used_as_input": False,
            "real_GT_scoring_performed": False,
        },
        "provenance": {
            "evreal_source_path": str(evreal_root),
            "evreal_commit": EVREAL_COMMIT,
            "event_cnn_minimal_commit": MODEL_COMMIT,
            "source_bundle_manifest": str(SOURCE_BUNDLE_MANIFEST.resolve()),
            "frozen_simulator_core_commit": "e79e656ac00f7847b98db9e12bf43b093f3afff1",
            "data_files": {
                name: {
                    "path": str((sequence_path / name).resolve()),
                    "size_bytes": (sequence_path / name).stat().st_size,
                    "sha256": data_info["hashes"][name],
                }
                for name in EXPECTED_DATA_FILES
            },
        },
        "self_test": {
            "is_synthetic_fixture": self_test,
            "not_HQF_real_data": self_test,
            "not_a_model_quality_result": True,
        },
    }
    manifest_path = output_root / "inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "sequence": sequence_name,
        "samples_processed": len(samples),
        "saved_indices": save_indices,
        "reset_calls": reset_calls,
        "checkpoint_sha256": manifest["model"]["checkpoint_sha256"],
        "manifest": str(manifest_path.resolve()),
        "output_root": str(output_root.resolve()),
    }, indent=2))
    return manifest


def make_self_test_fixture(output_root: Path, evreal_root: Path, model_root: Path) -> tuple[Path, Path]:
    fixture_root = output_root / "selftest_fixture" / "bike_bay_hdr"
    fixture_root.mkdir(parents=True, exist_ok=True)
    h, w = 9, 11
    event_ts = np.asarray([0.001, 0.004, 0.008, 0.012, 0.016, 0.020, 0.024, 0.028, 0.032, 0.036], dtype=np.float64)
    event_xy = np.asarray([[i % w, (2 * i) % h] for i in range(len(event_ts))], dtype=np.int16)
    event_p = np.asarray([i % 2 for i in range(len(event_ts))], dtype=np.uint8)
    image_ts = np.asarray([[0.005], [0.015], [0.025], [0.035]], dtype=np.float64)
    event_indices = np.searchsorted(event_ts, image_ts.reshape(-1), side="right") - 1
    event_indices = np.clip(event_indices, 0, len(event_ts)).astype(np.int64).reshape(-1, 1)
    images = np.zeros((len(image_ts), h, w, 1), dtype=np.uint8)
    for name, arr in (
        ("events_ts.npy", event_ts),
        ("events_xy.npy", event_xy),
        ("events_p.npy", event_p),
        ("images.npy", images),
        ("images_ts.npy", image_ts),
        ("image_event_indices.npy", event_indices),
    ):
        np.save(fixture_root / name, arr, allow_pickle=False)
    (fixture_root / "metadata.json").write_text(json.dumps({"sensor_resolution": [h, w]}, indent=2) + "\n")

    # Construct a tiny random-weight model checkpoint in the same key layout
    # as Stage-2. It only verifies loading/inference mechanics, never quality.
    dataset_cls = load_evreal_dataset_class(evreal_root.resolve())
    model_cls, _ = load_e2vid_class(model_root.resolve())
    del dataset_cls
    torch.manual_seed(20261016)
    tiny_args = {
        "num_bins": 5,
        "skip_type": "sum",
        "recurrent_block_type": "convlstm",
        "num_encoders": 2,
        "base_num_channels": 2,
        "num_residual_blocks": 1,
        "use_upsample_conv": True,
        "final_activation": "sigmoid",
        "norm": "none",
        "kernel_size": 3,
    }
    tiny_model = model_cls(copy.deepcopy(tiny_args))
    checkpoint_path = output_root / "selftest_random_untrained_stage2_schema.pt"
    torch.save({"model": tiny_model.state_dict(), "model_args": tiny_args, "generator": "fixture_only_untrained"}, checkpoint_path)
    return fixture_root, checkpoint_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evreal-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--hqf-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sequence", choices=REGISTERED_SEQUENCES, default=REGISTERED_SEQUENCE)
    parser.add_argument("--model-config", type=Path,
                        help="Stage-5 JSON config; supplies reconstruction.model_args when the checkpoint omits them")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--self-test", action="store_true", help="create and run a tiny fabricated MemMapDataset; not real HQF")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        sequence_path, checkpoint_path = make_self_test_fixture(output_root, args.evreal_root, args.model_root)
        run_inference(
            evreal_root=args.evreal_root,
            model_root=args.model_root,
            sequence_path=sequence_path,
            sequence_name=REGISTERED_SEQUENCE,
            checkpoint_path=checkpoint_path,
            output_root=output_root / "selftest_outputs",
            device_name=args.device,
            self_test=True,
        )
        return 0
    if args.hqf_root is None or args.checkpoint is None:
        parser.error("real inference requires --hqf-root and --checkpoint")
    model_args_override = None
    if args.model_config is not None:
        config = json.loads(args.model_config.read_text())
        model_args_override = config["reconstruction"]["model_args"]
    sequence_path = args.hqf_root / args.sequence
    run_inference(
        evreal_root=args.evreal_root,
        model_root=args.model_root,
        sequence_path=sequence_path,
        sequence_name=args.sequence,
        checkpoint_path=args.checkpoint,
        output_root=output_root,
        device_name=args.device,
        model_args_override=model_args_override,
        self_test=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
