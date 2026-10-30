#!/usr/bin/env python3
"""Check E2VID checkpoint forward, recurrent reset, and exact pad/crop geometry.

This is a bounded synthetic-only interface audit. It uses two consecutive event
windows from a registered Stage-5 synthetic fixture, never uses reference pixels
as model input, and does not calculate reconstruction metrics.
"""
from __future__ import annotations

import argparse
import hashlib
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from infer_hqf_e2vid_candidate import (  # noqa: E402
    checkpoint_to_model,
    load_e2vid_class,
    load_evreal_dataset_class,
    sha256_file,
    validate_sequence_files,
)
from verify_pinned_source import verify_pinned_source  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def fixture_hashes(fixture_manifest_path: Path, generator: str) -> dict[str, Any]:
    manifest = json.loads(fixture_manifest_path.read_text())
    expected = {
        "schema": "stage5_synthetic_e2vid_memmap_fixture_v1",
        "generator": generator,
        "rollout_id": "s5q_02_train",
        "split": "train",
        "real_data_used": False,
        "real_scoring_enabled": False,
        "checkpoint_selection_allowed": False,
        "target_frame_count": 200,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"synthetic fixture contract mismatch for {key}: {manifest.get(key)!r}")
    checked_files = {}
    for name, item in manifest.get("files", {}).items():
        path = Path(item["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != item.get("sha256"):
            raise RuntimeError(f"fixture file hash mismatch: {path}")
        checked_files[name] = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest}
    if len(checked_files) != 7:
        raise RuntimeError(f"expected seven fixture data files, found {len(checked_files)}")
    return {
        "path": str(fixture_manifest_path.resolve()),
        "sha256": sha256_file(fixture_manifest_path),
        "physical_bank_path": manifest.get("physical_bank_path"),
        "physical_bank_sha256": manifest.get("physical_bank_sha256"),
        "files": checked_files,
    }


def tensor_state_summary(value: Any) -> dict[str, Any]:
    tensors: list[torch.Tensor] = []

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensors.append(item)
        elif isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    digest = hashlib.sha256()
    finite = True
    rows = []
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        finite = finite and bool(torch.isfinite(value).all())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
        rows.append(list(value.shape))
    return {"tensor_count": len(tensors), "shapes": rows, "all_finite": finite,
            "sha256": digest.hexdigest() if tensors else None}


def roundtrip_check(crop_parameters_cls, num_encoders: int, height: int, width: int) -> dict[str, Any]:
    cropper = crop_parameters_cls(width, height, num_encoders)
    padded_hw = [int(cropper.height_crop_size), int(cropper.width_crop_size)]
    if padded_hw[0] < height or padded_hw[1] < width:
        raise RuntimeError(f"cropper returned undersized raster {padded_hw} for {(height, width)}")
    # Each pixel encodes its coordinate and channel, so a crop offset or axis
    # swap cannot pass as a shape-only check.
    plane = torch.arange(height * width, dtype=torch.float32).reshape(height, width)
    coded = torch.stack([plane + channel * (height * width) for channel in range(5)])[None]
    restored = cropper.crop(cropper.pad(coded))
    if tuple(restored.shape) != tuple(coded.shape):
        raise RuntimeError(f"pad/crop changed shape: {tuple(coded.shape)} -> {tuple(restored.shape)}")
    max_abs = float((restored - coded).abs().max().item())
    if max_abs != 0.0:
        raise RuntimeError(f"coordinate-coded pad/crop is not exact: max_abs={max_abs}")
    return {"native_hw": [height, width], "padded_hw": padded_hw,
            "input_shape": list(coded.shape), "restored_shape": list(restored.shape),
            "coordinate_roundtrip_max_abs": max_abs, "exact": True}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    started = utc_now()
    wall_start = time.monotonic()

    fixture_path = args.fixture_manifest.resolve()
    fixture_info = fixture_hashes(fixture_path, args.generator)
    model_config_path = args.model_config.resolve()
    model_config = json.loads(model_config_path.read_text())
    model_args_override = model_config["reconstruction"]["model_args"]
    model_config_sha256 = sha256_file(model_config_path)
    sequence_path = fixture_path.parent / f"s5q_02_train_{args.generator}"
    data_info = validate_sequence_files(sequence_path)
    if data_info["num_frames"] != 201:
        raise RuntimeError(f"expected 201 shared image endpoints, got {data_info['num_frames']}")

    evreal_info = verify_pinned_source(args.evreal_root, "EVREAL", "398f03551b7f5d150ae10aa89e4e731557fce240",
                                       REPO / "benchmark/sim2real/manifests/downstream_source_bundle.json")
    model_info = verify_pinned_source(args.model_root, "event_cnn_minimal", "78aa471a66593f55e349303563997297fbc816e8",
                                      REPO / "benchmark/sim2real/manifests/downstream_source_bundle.json")
    dataset_cls = load_evreal_dataset_class(args.evreal_root)
    model_cls, crop_cls = load_e2vid_class(args.model_root)
    height, width = data_info["sensor_resolution_hw"]
    dataset = dataset_cls(str(sequence_path), sensor_resolution=(height, width), num_bins=5,
                          voxel_method={"method": "between_frames"}, keep_ratio=1)
    if len(dataset) != dataset.num_frames - 1:
        raise RuntimeError("pinned EVREAL interval indexing changed")

    checkpoint_path = args.checkpoint.resolve()
    model, checkpoint, model_args = checkpoint_to_model(
        checkpoint_path, device, model_cls, model_args_override=model_args_override
    )
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("candidate must run under ordinary eval mode with frozen parameters")
    if model_args.get("num_bins") != 5:
        raise RuntimeError("checkpoint is not the registered five-bin E2VID model")

    geometry = [roundtrip_check(crop_cls, model.num_encoders, height, width),
                roundtrip_check(crop_cls, model.num_encoders, 180, 240)]
    cropper = crop_cls(width, height, model.num_encoders)

    # Only EVREAL's event tensor enters the network. The packaged reference
    # frame is intentionally discarded without reading its pixel values.
    first_item = dataset[1]
    second_item = dataset[2]
    first_events = first_item["events"]
    second_events = second_item["events"]
    del first_item["frame"], second_item["frame"]
    if not torch.is_tensor(first_events):
        first_events = torch.as_tensor(first_events)
    if not torch.is_tensor(second_events):
        second_events = torch.as_tensor(second_events)
    first_events = first_events.to(dtype=torch.float32).unsqueeze(0)
    second_events = second_events.to(dtype=torch.float32).unsqueeze(0)
    expected_input_shape = (1, 5, height, width)
    if tuple(first_events.shape) != expected_input_shape or tuple(second_events.shape) != expected_input_shape:
        raise RuntimeError(f"event input shapes differ from {expected_input_shape}")
    if not torch.isfinite(first_events).all() or not torch.isfinite(second_events).all():
        raise RuntimeError("synthetic event inputs contain nonfinite values")
    if torch.equal(first_events, second_events):
        raise RuntimeError("selected consecutive fixture event windows are identical")

    def forward(event_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padded = cropper.pad(event_tensor)
        result = model(padded)
        if not isinstance(result, dict) or "image" not in result:
            raise RuntimeError("candidate forward did not return {'image': tensor}")
        padded_image = result["image"]
        if tuple(padded_image.shape) != (1, 1, *geometry[0]["padded_hw"]):
            raise RuntimeError(f"unexpected padded model output shape: {tuple(padded_image.shape)}")
        cropped = cropper.crop(padded_image)
        if tuple(cropped.shape) != (1, 1, height, width):
            raise RuntimeError(f"unexpected native model output shape: {tuple(cropped.shape)}")
        if not torch.isfinite(padded_image).all() or not torch.isfinite(cropped).all():
            raise RuntimeError("candidate output contains nonfinite values")
        return padded_image.detach().clone(), cropped.detach().clone()

    with torch.inference_mode():
        model.reset_states()
        reset_initial = tensor_state_summary(model.states)
        if reset_initial["tensor_count"] != 0:
            raise RuntimeError("reset_states did not clear recurrent tensors before the first interval")
        first_padded_a, first_cropped_a = forward(first_events)
        after_first = tensor_state_summary(model.states)
        if after_first["tensor_count"] == 0 or not after_first["all_finite"]:
            raise RuntimeError("candidate forward did not establish finite recurrent state")
        second_padded, second_cropped = forward(second_events)
        after_second = tensor_state_summary(model.states)
        if after_second["tensor_count"] == 0 or not after_second["all_finite"]:
            raise RuntimeError("second candidate forward left invalid recurrent state")
        model.reset_states()
        reset_after_second = tensor_state_summary(model.states)
        if reset_after_second["tensor_count"] != 0:
            raise RuntimeError("reset_states failed to clear recurrent state after sequence activity")
        first_padded_b, first_cropped_b = forward(first_events)

    padded_diff = float((first_padded_a - first_padded_b).abs().max().item())
    cropped_diff = float((first_cropped_a - first_cropped_b).abs().max().item())
    if padded_diff > args.reset_tolerance or cropped_diff > args.reset_tolerance:
        raise RuntimeError(f"fresh-reset replay differs: padded={padded_diff}, cropped={cropped_diff}")
    elapsed = time.monotonic() - wall_start
    result = {
        "schema": "e2vid_checkpoint_state_padding_audit_v1",
        "status": "synthetic_checkpoint_forward_state_and_padding_pass_no_scoring",
        "started_utc": started,
        "finished_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "host": os.uname().nodename,
        "device": "cpu",
        "threads": args.threads,
        "generator": args.generator,
        "real_data_used": False,
        "real_scoring_enabled": False,
        "checkpoint_selection_allowed": False,
        "reference_pixels_used_as_model_input": False,
        "fixture": fixture_info,
        "synthetic_sequence": {
            "path": str(sequence_path),
            "resolution_hw": [height, width],
            "source_file_hashes": data_info["hashes"],
            "tested_sample_indices": [1, 2],
            "target_timestamps_s": [float(dataset.frame_ts[1]), float(dataset.frame_ts[2])],
            "event_counts": [int(dataset[1]["event_count"]), int(dataset[2]["event_count"])],
            "event_window_semantics": "EVREAL between_frames; event-only intervals ending at each target timestamp",
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
            "generator_metadata": checkpoint.get("generator"),
            "model_args": model_args,
        },
        "registered_model_config": {
            "path": str(model_config_path),
            "sha256": model_config_sha256,
            "model_args_source": "reconstruction.model_args",
        },
        "pinned_sources": {"EVREAL": evreal_info, "event_cnn_minimal": model_info},
        "pad_crop_coordinate_checks": geometry,
        "recurrent_state": {
            "model_eval": not model.training,
            "parameter_gradients_disabled": all(not p.requires_grad for p in model.parameters()),
            "reset_initial": reset_initial,
            "state_after_first_window": after_first,
            "state_after_second_window": after_second,
            "reset_after_activity": reset_after_second,
            "first_window_replay_max_abs_padded": padded_diff,
            "first_window_replay_max_abs_native": cropped_diff,
            "replay_tolerance": args.reset_tolerance,
            "reset_replay_pass": True,
        },
        "candidate_output": {
            "padded_shape": list(first_padded_a.shape),
            "native_shape": list(first_cropped_a.shape),
            "first_native_raw_min": float(first_cropped_a.min().item()),
            "first_native_raw_max": float(first_cropped_a.max().item()),
            "second_native_raw_min": float(second_cropped.min().item()),
            "second_native_raw_max": float(second_cropped.max().item()),
            "all_finite": True,
        },
        "interpretation": "checks ordinary eval forward, recurrent reset, and crop geometry on two synthetic event windows; does not diagnose HQF quality or establish scene generalization",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "generator": args.generator,
                      "checkpoint_sha256": result["checkpoint"]["sha256"],
                      "pad_crop": geometry, "reset_replay_max_abs": cropped_diff,
                      "output": str(args.output), "elapsed_seconds": elapsed}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("eventgenesis", "v2e"), required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--evreal-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True,
                        help="registered Stage-5 config used because these checkpoints omit model_args")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--reset-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.threads < 1 or args.reset_tolerance < 0:
        parser.error("threads must be positive and reset tolerance nonnegative")
    run(args)


if __name__ == "__main__":
    main()
