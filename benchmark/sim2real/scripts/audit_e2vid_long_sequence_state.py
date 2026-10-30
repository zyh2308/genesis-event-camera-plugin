#!/usr/bin/env python3
"""Trace a registered E2VID candidate over all 200 synthetic event windows.

This is a CPU-only interface/state stability audit. Target images are discarded
and never passed to the model or scored. It checks finite outputs and recurrent
state over a complete fixed synthetic rollout, then checks that reset+replay
reproduces the first output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from audit_e2vid_checkpoint_state_padding import (
    checkpoint_to_model,
    fixture_hashes,
    load_e2vid_class,
    load_evreal_dataset_class,
    sha256_file,
    validate_sequence_files,
)
from verify_pinned_source import verify_pinned_source


def tensor_digest(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def collect_state(model: torch.nn.Module) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []

    def visit(value) -> None:
        if torch.is_tensor(value):
            result.append(value.detach())
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                visit(nested)

    # The pinned model stores recurrent tensors in a nested ``model.states``
    # mapping; it does not expose them as module.state attributes.
    visit(getattr(model, "states", {}))
    return result


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    fixture = args.fixture_manifest.resolve()
    fixture_info = fixture_hashes(fixture, args.generator)
    config_path = args.model_config.resolve()
    config = json.loads(config_path.read_text())
    model_args = config["reconstruction"]["model_args"]
    sequence_path = fixture.parent / f"s5q_02_train_{args.generator}"
    sequence_info = validate_sequence_files(sequence_path)
    if sequence_info["num_frames"] != 201:
        raise RuntimeError(f"expected 201 endpoints, got {sequence_info['num_frames']}")

    evreal_info = verify_pinned_source(
        args.evreal_root, "EVREAL", "398f03551b7f5d150ae10aa89e4e731557fce240",
        args.repo_root / "benchmark/sim2real/manifests/downstream_source_bundle.json",
    )
    model_info = verify_pinned_source(
        args.model_root, "event_cnn_minimal", "78aa471a66593f55e349303563997297fbc816e8",
        args.repo_root / "benchmark/sim2real/manifests/downstream_source_bundle.json",
    )
    dataset_cls = load_evreal_dataset_class(args.evreal_root)
    model_cls, crop_cls = load_e2vid_class(args.model_root)
    h, w = sequence_info["sensor_resolution_hw"]
    dataset = dataset_cls(str(sequence_path), sensor_resolution=(h, w), num_bins=5,
                          voxel_method={"method": "between_frames"}, keep_ratio=1)
    source_loader_length = len(dataset)
    if source_loader_length != dataset.num_frames - 1 or len(dataset.frames_to_use) != dataset.num_frames:
        raise RuntimeError("unexpected pinned EVREAL between_frames length/index contract")
    # Follow the registered HQF adapter: expose the final index N-1 while
    # skipping index 0, which is the unobserved initial boundary image.
    dataset.length = dataset.num_frames
    if len(dataset) != 201:
        raise RuntimeError(f"expected 201 addressable endpoints after registered extension, got {len(dataset)}")
    model, checkpoint, model_args_used = checkpoint_to_model(
        args.checkpoint.resolve(), torch.device("cpu"), model_cls,
        model_args_override=model_args,
    )
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("model is not frozen ordinary eval")
    cropper = crop_cls(w, h, model.num_encoders)

    output_rows = []
    event_counts = []
    previous_output = None
    first_event = None
    first_output = None
    with torch.inference_mode():
        for index in range(1, len(dataset)):
            item = dataset[index]
            event = item["events"]
            item.pop("frame", None)
            if not torch.is_tensor(event):
                event = torch.as_tensor(event)
            event = event.to(dtype=torch.float32).unsqueeze(0)
            if tuple(event.shape) != (1, 5, h, w) or not torch.isfinite(event).all():
                raise RuntimeError(f"invalid event tensor at interval {index}: {tuple(event.shape)}")
            event_counts.append(int(torch.count_nonzero(event).item()))
            if index == 1:
                first_event = event.clone()
            prediction = model(cropper.pad(event))
            if not isinstance(prediction, dict) or "image" not in prediction:
                raise RuntimeError(f"invalid model result at interval {index}")
            output = cropper.crop(prediction["image"]).detach()
            if tuple(output.shape) != (1, 1, h, w) or not torch.isfinite(output).all():
                raise RuntimeError(f"invalid output at interval {index}: {tuple(output.shape)}")
            if first_output is None:
                first_output = output.clone()
            diff_rms = None if previous_output is None else float((output - previous_output).square().mean().sqrt())
            output_rows.append({
                "interval_index": index,
                "output_min": float(output.min()),
                "output_max": float(output.max()),
                "output_mean": float(output.mean()),
                "output_std": float(output.std(unbiased=False)),
                "consecutive_output_delta_rms": diff_rms,
                "state_tensor_count": len(collect_state(model)),
            })
            previous_output = output

    final_state = collect_state(model)
    if not final_state or not all(torch.isfinite(s).all() for s in final_state):
        raise RuntimeError("final recurrent state is missing or nonfinite")
    final_state_hash = tensor_digest(final_state)
    model.reset_states()
    with torch.inference_mode():
        replay = cropper.crop(model(cropper.pad(first_event))["image"]).detach()
    replay_max_abs = float((replay - first_output).abs().max())
    if replay_max_abs > args.replay_tolerance:
        raise RuntimeError(f"reset/replay mismatch: {replay_max_abs}")

    result = {
        "schema": "e2vid_long_synthetic_state_audit_v1",
        "status": "passed_200_window_eval_state_and_reset_replay_no_scoring",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "generator": args.generator,
        "device": "cpu",
        "real_data_used": False,
        "real_scoring_enabled": False,
        "target_pixels_used_as_input_or_scored": False,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256_file(args.checkpoint.resolve()),
                       "model_args_source": str(config_path), "model_config_sha256": sha256_file(config_path)},
        "fixture": fixture_info,
        "sequence": {"path": str(sequence_path), "files": sequence_info,
                     "source_loader_length_before_registered_extension": source_loader_length,
                     "window_count": len(output_rows),
                     "event_voxel_nonzero_count_min": min(event_counts),
                     "event_voxel_nonzero_count_max": max(event_counts),
                     "event_voxel_nonzero_count_median": float(np.median(event_counts))},
        "pinned_sources": {"EVREAL": evreal_info, "event_cnn_minimal": model_info},
        "model": {"training": model.training, "model_args": model_args_used,
                  "output_hw": [h, w], "all_200_outputs_finite": True,
                  "recurrent_state_tensor_count_after_200": len(final_state),
                  "recurrent_state_sha256_after_200": final_state_hash,
                  "reset_replay_max_abs": replay_max_abs,
                  "replay_tolerance": args.replay_tolerance},
        "per_window_numeric_trace": output_rows,
        "interpretation": "This audits numerical stability and reset behavior over one fixed synthetic rollout only; it is not an image-quality score or explanation of HQF artifacts.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "windows": len(output_rows),
                      "final_state_sha256": final_state_hash,
                      "reset_replay_max_abs": replay_max_abs,
                      "elapsed_s": time.monotonic() - args.started_monotonic,
                      "output": str(args.output)}, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generator", choices=("eventgenesis", "v2e"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--fixture-manifest", type=Path, required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--evreal-root", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--replay-tolerance", type=float, default=1e-6)
    args = p.parse_args()
    args.started_monotonic = time.monotonic()
    if args.threads < 1 or args.replay_tolerance <= 0:
        p.error("threads and replay tolerance must be positive")
    run(args)


if __name__ == "__main__":
    main()
