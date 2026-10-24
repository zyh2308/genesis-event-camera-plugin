#!/usr/bin/env python3
"""Fail-closed paired scorer for the frozen 14-sequence HQF v3 protocol.

Normal mode accepts only the exact frozen data manifest, both registered
synthetic-best checkpoints, and all 28 complete event-only inference manifests.
``--self-test`` creates its own synthetic artifacts in a temporary directory;
it never accepts real data, checkpoints, or remote paths.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SIM2REAL = ROOT / "benchmark/sim2real"
DEFAULT_PROTOCOL = SIM2REAL / "protocol/hqf_14_reconstruction_test_frozen_20261024_v1.json"
DEFAULT_DATA_MANIFEST = SIM2REAL / "evidence/S5/20261021-recon-flow-localization/20261021-hqf14-manifest-validation-v3/hqf_14_file_manifest_v3.json"
DEFAULT_TRAINING_MANIFEST = SIM2REAL / "evidence/S5/20261021-recon-flow-localization/reconstruction_training_artifact_manifest.json"
EXPECTED_ARMS = ("eventgenesis", "v2e")
EXPECTED_MEMMAP_FILES = (
    "events_ts.npy", "events_xy.npy", "events_p.npy", "images.npy",
    "images_ts.npy", "image_event_indices.npy", "metadata.json", "conversion_manifest.json",
)
INFERABLE_STATUS = "fixed_sample_event_only_inference_completed_no_GT_scoring"

sys.path.insert(0, str(ROOT))
from benchmark.sim2real.scripts.evaluate_synthetic_reconstruction_fixture import (  # noqa: E402
    classical_metrics,
    load_lpips_alex,
    lpips_input,
    validate_pair,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_frozen_protocol(protocol_path: Path) -> tuple[dict[str, Any], str]:
    protocol_path = protocol_path.resolve()
    protocol = read_json(protocol_path)
    require(protocol.get("schema") == "hqf_14_reconstruction_test_frozen_v1", "protocol is not the frozen HQF v3 execution schema")
    require(protocol.get("status") == "frozen_execution_protocol_unscored", "protocol is not in the frozen unscored state")
    require(protocol.get("real_scoring_authorized") is True, "protocol does not record the current task authorization")
    lock = protocol.get("protocol_freeze", {})
    require(lock.get("required_real_run_flag") == "--enable-real-scoring", "protocol real-score gate is missing")
    sidecar = protocol_path.with_suffix(protocol_path.suffix + ".sha256")
    require(sidecar.is_file(), f"protocol SHA256 sidecar is missing: {sidecar}")
    sidecar_digest = sidecar.read_text().split()[0]
    protocol_digest = sha256_file(protocol_path)
    require(protocol_digest == sidecar_digest, "frozen protocol SHA256 sidecar mismatch")
    required_sources = (
        lock.get("source_candidate"), lock.get("data_manifest"),
        lock.get("training_artifact_manifest"), lock.get("model_config"),
        lock.get("training_config"),
        lock.get("source_bundle"),
    )
    for record in required_sources:
        require(isinstance(record, dict) and record.get("path") and record.get("sha256"),
                "frozen protocol has an incomplete source binding")
        path = ROOT / record["path"]
        require(path.is_file() and sha256_file(path) == record["sha256"],
                f"frozen source changed or is missing: {record['path']}")
    code = lock.get("execution_code", {})
    expected_code = {
        "benchmark/sim2real/scripts/infer_hqf_e2vid_candidate.py",
        "benchmark/sim2real/scripts/score_hqf14_reconstruction.py",
        "benchmark/sim2real/scripts/run_hqf14_reconstruction_eval.py",
        "benchmark/sim2real/scripts/test_hqf14_reconstruction_eval.py",
        "benchmark/sim2real/scripts/freeze_hqf_v3_protocol.py",
    }
    require(set(code) == expected_code, "frozen code fingerprint inventory is incomplete or unexpected")
    for rel, digest in code.items():
        path = ROOT / rel
        require(path.is_file() and sha256_file(path) == digest, f"frozen evaluation code changed or missing: {rel}")
    require(protocol.get("sequences") and len(protocol["sequences"]) == 14, "frozen sequence inventory is not 14 entries")
    require(lock.get("checkpoint_bindings", {}).keys() >= set(EXPECTED_ARMS), "both frozen checkpoint bindings are required")
    return protocol, protocol_digest


def validate_data_manifest(
    protocol: dict[str, Any], data_manifest_path: Path, hqf_root: Path,
    *, self_test: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    data_manifest_path = data_manifest_path.resolve()
    hqf_root = hqf_root.resolve()
    manifest = read_json(data_manifest_path)
    expected_sequences = list(protocol["sequences"])
    rows = manifest.get("sequences", [])
    require(manifest.get("status") == "ready_unscored", "HQF manifest must be ready_unscored")
    require(manifest.get("expected_sequence_count") == 14 and manifest.get("ready_unscored_sequence_count") == 14,
            "HQF data manifest does not report 14/14 ready sequences")
    require(manifest.get("real_scoring_enabled") is False and manifest.get("real_gt_metrics_computed") is False,
            "input data manifest is not a pre-score artifact")
    require([row.get("sequence") for row in rows] == expected_sequences, "HQF sequence list is incomplete, duplicated, or reordered")
    lock = protocol["protocol_freeze"]["data_manifest"]
    if not self_test:
        require(data_manifest_path == (ROOT / lock["path"]).resolve(), "data manifest path differs from frozen protocol")
        require(sha256_file(data_manifest_path) == lock["sha256"], "HQF data manifest hash differs from frozen protocol")
    by_name = {row["sequence"]: row for row in rows}
    for name in expected_sequences:
        row = by_name[name]
        memmap = row.get("evreal_memmap", {})
        require(row.get("status") == "ready_unscored" and memmap.get("exists") is True,
                f"HQF sequence {name} is not ready_unscored")
        require(not memmap.get("missing_files"), f"HQF sequence {name} manifest lists missing files")
        registered_path = Path(memmap.get("path", "")).resolve()
        actual_path = (hqf_root / name).resolve()
        require(registered_path == actual_path,
                f"HQF sequence {name} path differs from frozen manifest: {registered_path} != {actual_path}")
        file_records = memmap.get("files", {})
        require(set(file_records) == set(EXPECTED_MEMMAP_FILES), f"HQF sequence {name} file inventory is not complete")
        for filename in EXPECTED_MEMMAP_FILES:
            path = actual_path / filename
            record = file_records[filename]
            require(path.is_file(), f"HQF sequence {name} missing file: {filename}")
            require(path.stat().st_size == record.get("size_bytes"), f"HQF sequence {name} size mismatch: {filename}")
            require(sha256_file(path) == record.get("sha256"), f"HQF sequence {name} SHA256 mismatch: {filename}")
        manifest_hashes = memmap.get("schema_checks", {}).get("hashes", {})
        for filename in EXPECTED_MEMMAP_FILES[:-1]:
            if filename in manifest_hashes:
                require(manifest_hashes[filename] == file_records[filename]["sha256"],
                        f"HQF sequence {name} file hash fields disagree: {filename}")
    return manifest, by_name


def validate_sequence_contract(sequence_root: Path, manifest_row: dict[str, Any]) -> dict[str, Any]:
    """Independently check data shapes, timestamps, endpoint offsets and raw window bounds."""
    sequence_root = sequence_root.resolve()
    arrays = {
        "events_ts": np.load(sequence_root / "events_ts.npy", mmap_mode="r").reshape(-1),
        "events_xy": np.load(sequence_root / "events_xy.npy", mmap_mode="r"),
        "events_p": np.load(sequence_root / "events_p.npy", mmap_mode="r").reshape(-1),
        "images": np.load(sequence_root / "images.npy", mmap_mode="r"),
        "images_ts": np.load(sequence_root / "images_ts.npy", mmap_mode="r").reshape(-1),
        "image_event_indices": np.load(sequence_root / "image_event_indices.npy", mmap_mode="r").reshape(-1),
    }
    ev_t = arrays["events_ts"]
    image_t = arrays["images_ts"]
    offsets = arrays["image_event_indices"].astype(np.int64, copy=False)
    images = arrays["images"]
    h, w = map(int, images.shape[1:3])
    require(arrays["events_xy"].ndim == 2 and arrays["events_xy"].shape == (len(ev_t), 2), "events_xy must be [N,2]")
    require(arrays["events_p"].shape == ev_t.shape, "events_p/events_ts length mismatch")
    require(images.ndim == 4 and images.shape[-1] == 1 and images.dtype == np.uint8, "images must be uint8 [N,H,W,1]")
    require((h, w) == tuple(manifest_row["evreal_memmap"]["schema_checks"]["sensor_resolution_hw"]),
            "image resolution differs from frozen data manifest")
    require(len(image_t) == len(images) == len(offsets) and len(image_t) >= 2, "image timestamps, images and offsets length mismatch")
    require(ev_t.dtype == np.float64 and image_t.dtype == np.float64, "registered timestamps must be float64 seconds")
    require(np.isfinite(ev_t).all() and np.isfinite(image_t).all(), "event/image timestamp contains NaN/Inf")
    require(np.all(np.diff(ev_t) >= 0) and np.all(np.diff(image_t) > 0), "events must be nondecreasing and retained images strictly increasing")
    require(np.isin(np.unique(arrays["events_p"]), (0, 1)).all(), "raw event polarity must be encoded 0/1")
    require(np.all(offsets >= 0) and np.all(offsets <= len(ev_t)), "image event offsets out of range")
    independent = np.searchsorted(ev_t, image_t, side="right").astype(np.int64)
    require(np.array_equal(offsets, independent), "image_event_indices differ from independent searchsorted(right)")
    if len(ev_t):
        has_prev = offsets > 0
        require(np.all(ev_t[offsets[has_prev] - 1] <= image_t[has_prev]), "event end offset includes a future event")
        has_next = offsets < len(ev_t)
        require(np.all(ev_t[offsets[has_next]] > image_t[has_next]), "event end offset omits event at/before endpoint")
    row_shapes = manifest_row["evreal_memmap"]["schema_checks"].get("shapes", {})
    require(list(images.shape) == row_shapes.get("images"), "images.npy shape differs from manifest")
    require(list(ev_t.shape) == row_shapes.get("events_ts"), "events_ts.npy shape differs from manifest")
    return {"arrays": arrays, "height": h, "width": w, "independent_offsets": independent}


def validate_run_manifest(
    protocol: dict[str, Any], protocol_sha256: str, run_manifest_path: Path,
    inference_root: Path, checkpoint_root_override: Path | None,
    *, self_test: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    run_manifest_path = run_manifest_path.resolve()
    inference_root = inference_root.resolve()
    run_manifest = read_json(run_manifest_path)
    require(run_manifest.get("schema") == "hqf14_paired_inference_run_v1", "run manifest schema mismatch")
    require(run_manifest.get("status") == "all_28_inferences_complete_unscored", "pair run is incomplete or already scored")
    require(run_manifest.get("protocol_sha256") == protocol_sha256, "run manifest is not bound to this frozen protocol")
    require(run_manifest.get("inference_root") == str(inference_root), "run manifest inference root differs from CLI path")
    expected_names = list(protocol["sequences"])
    manifests: dict[str, dict[str, dict[str, Any]]] = {}
    checkpoint_bindings = protocol["protocol_freeze"]["checkpoint_bindings"]
    for arm in EXPECTED_ARMS:
        arm_record = run_manifest.get("arms", {}).get(arm, {})
        expected_cp = checkpoint_bindings[arm]
        if checkpoint_root_override is None:
            cp_path = Path(expected_cp["checkpoint_path"])
        else:
            # A staging-root override preserves relative layout under the registered run directory.
            registered = Path(expected_cp["checkpoint_path"])
            marker = "/runs/sim2real/"
            if marker not in str(registered):
                raise RuntimeError("registered checkpoint path lacks the expected runs/sim2real layout")
            cp_path = checkpoint_root_override / str(registered).split(marker, 1)[1]
        cp_path = cp_path.resolve()
        require(arm_record.get("checkpoint_path") == str(cp_path), f"{arm} checkpoint path differs from run record")
        require(arm_record.get("checkpoint_sha256") == expected_cp["checkpoint_sha256"], f"{arm} checkpoint hash differs from frozen synthetic-best")
        if not self_test:
            require(cp_path.is_file(), f"{arm} synthetic-best checkpoint missing: {cp_path}")
            require(cp_path.stat().st_size == expected_cp["checkpoint_size_bytes"], f"{arm} checkpoint size mismatch")
            require(sha256_file(cp_path) == expected_cp["checkpoint_sha256"], f"{arm} checkpoint SHA256 mismatch")
        seq_hashes = arm_record.get("inference_manifest_sha256", {})
        require(set(seq_hashes) == set(expected_names), f"{arm} must have exactly 14 inference outputs")
        manifests[arm] = {}
        for name in expected_names:
            manifest_path = inference_root / arm / name / "inference_manifest.json"
            require(manifest_path.is_file(), f"{arm}/{name} inference manifest is missing")
            require(sha256_file(manifest_path) == seq_hashes[name], f"{arm}/{name} inference manifest hash differs from run record")
            inference = read_json(manifest_path)
            require(inference.get("status") == INFERABLE_STATUS, f"{arm}/{name} is not a full HQF candidate inference")
            require(inference.get("self_test", {}).get("is_synthetic_fixture") is False, f"{arm}/{name} inference is a fixture, not HQF")
            model = inference.get("model", {})
            require(model.get("checkpoint_sha256") == expected_cp["checkpoint_sha256"], f"{arm}/{name} checkpoint SHA mismatch")
            require(model.get("checkpoint_path") == str(cp_path), f"{arm}/{name} checkpoint path mismatch")
            require(model.get("public_EVREAL_weights_loaded") is False, f"{arm}/{name} substituted public pretrained weights")
            require(model.get("reference_frame_pixels_used_as_input") is False and model.get("real_GT_scoring_performed") is False,
                    f"{arm}/{name} inference used reference GT or scored real data inside model runner")
            require(model.get("input_contract") == "event-only; [1,5,H,W]", f"{arm}/{name} model input is not registered event-only")
            require(model.get("source_verification", {}).get("EVREAL", {}).get("commit") == protocol["sources"]["official_evreal_commit"],
                    f"{arm}/{name} EVREAL source commit mismatch")
            require(model.get("source_verification", {}).get("event_cnn_minimal", {}).get("commit") == "78aa471a66593f55e349303563997297fbc816e8",
                    f"{arm}/{name} E2VID source commit mismatch")
            seq = inference.get("sequence", {})
            require(seq.get("name") == name, f"inference manifest sequence name mismatch for {arm}/{name}")
            require(seq.get("initial_anchor_endpoint_excluded_from_model_and_metrics") is True,
                    f"{arm}/{name} did not exclude initial anchor")
            require(seq.get("timestamps_strictly_increasing") is True, f"{arm}/{name} timestamps not monotonic")
            require(seq.get("num_inferable_event_endpoints") == int(seq.get("num_frames_in_memmap", 0)) - 1,
                    f"{arm}/{name} inferable sample count is inconsistent")
            require(model.get("recurrent_reset_calls") == 1, f"{arm}/{name} recurrent state was not reset exactly once")
            require(model.get("reset_policy") == "one reset at the initial anchor; first model call uses events from (t0,t1]; no reset inside sequence",
                    f"{arm}/{name} recurrent reset policy mismatch")
            require(model.get("output_contract_after_crop") == [1, 1, 180, 240],
                    f"{arm}/{name} native-resolution crop contract mismatch")
            registered = inference.get("registered_inference", {})
            require(registered.get("save_every_n_samples") == 1 and registered.get("save_float_arrays") is True,
                    f"{arm}/{name} must save every endpoint as float32")
            require(registered.get("saved_sample_indices") == list(range(1, int(seq.get("num_frames_in_memmap", 0)))),
                    f"{arm}/{name} saved sample inventory is incomplete")
            data_file_rows = inference.get("provenance", {}).get("data_files", {})
            registered_files = protocol.get("expected_converted_files_per_sequence", [])
            for filename in registered_files:
                if filename == "conversion_manifest.json":
                    continue
                expected_record = protocol["protocol_freeze"]["data_manifest"].get("sequence_file_hashes", {}).get(name, {}).get(filename)
                if expected_record:
                    actual_record = data_file_rows.get(filename, {})
                    require(actual_record.get("sha256") == expected_record,
                            f"{arm}/{name} inference data hash mismatch for {filename}")
            manifests[arm][name] = inference
    return run_manifest, manifests


def _psnr_summary(values: list[float]) -> dict[str, Any]:
    if any(math.isinf(x) and x > 0 for x in values):
        return {"mean_db": None, "mean_is_positive_infinity": True,
                "exact_match_frame_count": sum(math.isinf(x) and x > 0 for x in values)}
    return {"mean_db": float(np.mean(values)), "mean_is_positive_infinity": False,
            "exact_match_frame_count": 0}


def _metric_from_model(metric_model: Any, pred: np.ndarray, target01: np.ndarray, device: torch.device) -> float:
    pred_rgb = lpips_input(pred, device)
    target_rgb = lpips_input(target01, device)
    value = metric_model.forward(pred_rgb, target_rgb, normalize=False)
    result = float(value.detach().float().mean().cpu())
    if not math.isfinite(result):
        raise RuntimeError("LPIPS-Alex returned NaN/Inf")
    return result


def score_pair(
    *, protocol: dict[str, Any], protocol_sha256: str,
    data_manifest: dict[str, Any], data_rows: dict[str, dict[str, Any]],
    hqf_root: Path, inference_manifests: dict[str, dict[str, dict[str, Any]]],
    run_manifest_path: Path, output_path: Path, metric_model: Any,
    device: torch.device, metric_provenance: dict[str, Any] | None = None,
    self_test: bool = False,
) -> dict[str, Any]:
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite score artifact: {output_path}")
    sequence_names = list(protocol["sequences"])
    preview_registry = protocol["protocol_freeze"]["pre_registered_preview_indices"]
    all_frame_rows: list[dict[str, Any]] = []
    per_arm: dict[str, Any] = {}
    for arm in EXPECTED_ARMS:
        seq_summaries = {}
        for name in sequence_names:
            sequence_root = (hqf_root / name).resolve()
            contract = validate_sequence_contract(sequence_root, data_rows[name])
            arrays = contract["arrays"]
            n = int(len(arrays["images_ts"]))
            expected_indices = list(range(1, n))
            inference = inference_manifests[arm][name]
            seq_info = inference["sequence"]
            require(seq_info.get("path") == str(sequence_root), f"{arm}/{name} inference input path differs from frozen HQF root")
            require(seq_info.get("num_frames_in_memmap") == n and seq_info.get("num_samples_processed") == n - 1,
                    f"{arm}/{name} inference sample count does not cover every inferable endpoint")
            sequence_samples = seq_info.get("samples", [])
            saved_rows = inference["registered_inference"].get("saved_images", [])
            require([int(r.get("sample_index", -1)) for r in sequence_samples] == expected_indices,
                    f"{arm}/{name} sequence sample rows are incomplete or reordered")
            require([int(r.get("sample_index", -1)) for r in saved_rows] == expected_indices,
                    f"{arm}/{name} saved prediction rows are incomplete or reordered")
            preview_indices = preview_registry[name]["sample_indices"]
            require(preview_indices and all(1 <= int(i) < n for i in preview_indices),
                    f"{name} pre-registered preview indices are invalid")

            frame_values: dict[str, list[float]] = {"lpips_alex": [], "ssim": [], "psnr_db": [], "mse": []}
            preview_records = []
            for sample, saved in zip(sequence_samples, saved_rows):
                index = int(sample["sample_index"])
                prev_end = int(contract["independent_offsets"][index - 1])
                curr_end = int(contract["independent_offsets"][index])
                target_t = float(arrays["images_ts"][index])
                require(float(sample.get("target_timestamp_s", math.nan)) == target_t,
                        f"{arm}/{name}/{index} label timestamp mismatch")
                require(float(sample.get("voxel_timestamp_s", math.nan)) == target_t,
                        f"{arm}/{name}/{index} event voxel timestamp mismatch")
                require(int(sample.get("event_start_index_inclusive", -1)) == prev_end,
                        f"{arm}/{name}/{index} event start offset mismatch")
                require(int(sample.get("event_end_index_exclusive", -1)) == curr_end,
                        f"{arm}/{name}/{index} event end offset mismatch")
                require(int(sample.get("event_count", -1)) == curr_end - prev_end,
                        f"{arm}/{name}/{index} event count mismatch")
                require(sample.get("event_tensor_shape") == [5, contract["height"], contract["width"]]
                        and sample.get("model_input_shape") == [1, 5, contract["height"], contract["width"]],
                        f"{arm}/{name}/{index} event tensor/model input shape mismatch")
                require(sample.get("cropped_output_shape") == [1, 1, contract["height"], contract["width"]],
                        f"{arm}/{name}/{index} cropped output shape mismatch")
                interval_t = arrays["events_ts"][prev_end:curr_end]
                if interval_t.size:
                    require(float(interval_t[0]) > float(arrays["images_ts"][index - 1])
                            and float(interval_t[-1]) <= target_t,
                            f"{arm}/{name}/{index} event times are outside (previous,target]")
                    require(not np.any(interval_t > target_t), f"{arm}/{name}/{index} contains future events")
                pred_path = Path(saved.get("float_array_path", "")).resolve()
                require(pred_path.is_file(), f"{arm}/{name}/{index} float32 prediction is missing")
                require(sha256_file(pred_path) == saved.get("float_array_sha256"),
                        f"{arm}/{name}/{index} prediction SHA256 mismatch")
                pred = np.load(pred_path, allow_pickle=False)
                target_u8 = np.asarray(arrays["images"][index, :, :, 0])
                target01 = validate_pair(pred, target_u8, (contract["height"], contract["width"]))
                classical = classical_metrics(pred, target01)
                lpips = _metric_from_model(metric_model, pred, target01, device)
                vals = {"lpips_alex": lpips, **classical}
                if not all(math.isfinite(v) for k, v in vals.items() if k != "psnr_db"):
                    raise RuntimeError(f"{arm}/{name}/{index} has nonfinite metric: {vals}")
                if not (math.isfinite(vals["psnr_db"]) or (math.isinf(vals["psnr_db"]) and vals["psnr_db"] > 0)):
                    raise RuntimeError(f"{arm}/{name}/{index} has invalid PSNR")
                for metric_name in frame_values:
                    frame_values[metric_name].append(float(vals[metric_name]))
                row = {
                    "arm": arm, "sequence": name, "sample_index": index,
                    "target_timestamp_s": target_t,
                    "event_start_index_inclusive": prev_end,
                    "event_end_index_exclusive": curr_end,
                    "event_count": curr_end - prev_end,
                    "event_timestamp_first_s": float(interval_t[0]) if interval_t.size else None,
                    "event_timestamp_last_s": float(interval_t[-1]) if interval_t.size else None,
                    "prediction_path": str(pred_path),
                    "prediction_sha256": saved["float_array_sha256"],
                    "target_uint8_sha256": sha256_array(target_u8),
                    "prediction_shape": list(pred.shape),
                    "prediction_dtype": str(pred.dtype),
                    "prediction_range": [float(pred.min()), float(pred.max())],
                    "target_range": [float(target01.min()), float(target01.max())],
                    "is_preregistered_preview": index in preview_indices,
                    "metrics": {
                        "lpips_alex": lpips, "ssim": vals["ssim"], "mse": vals["mse"],
                        "psnr_db": None if math.isinf(vals["psnr_db"]) else vals["psnr_db"],
                        "psnr_exact_match": bool(math.isinf(vals["psnr_db"])),
                    },
                }
                all_frame_rows.append(row)
                if index in preview_indices:
                    preview_records.append({
                        "sample_index": index, "target_timestamp_s": target_t,
                        "prediction_path": str(pred_path), "prediction_sha256": saved["float_array_sha256"],
                        "target_uint8_sha256": row["target_uint8_sha256"],
                    })
            require(len(frame_values["mse"]) == n - 1, f"{arm}/{name} scored sample count mismatch")
            seq_summaries[name] = {
                "scored_frame_count": n - 1,
                "num_frames_including_anchor": n,
                "initial_anchor_excluded": True,
                "metrics": {
                    "lpips_alex_mean": float(np.mean(frame_values["lpips_alex"])),
                    "ssim_mean": float(np.mean(frame_values["ssim"])),
                    "mse_mean": float(np.mean(frame_values["mse"])),
                    "psnr": _psnr_summary(frame_values["psnr_db"]),
                },
                "pre_registered_preview_indices": preview_indices,
                "pre_registered_preview_artifacts": preview_records,
            }
        macro = {}
        for metric_name in ("lpips_alex_mean", "ssim_mean", "mse_mean"):
            macro[metric_name] = float(np.mean([seq_summaries[name]["metrics"][metric_name] for name in sequence_names]))
        sequence_psnr = [seq_summaries[name]["metrics"]["psnr"] for name in sequence_names]
        if any(row["mean_is_positive_infinity"] for row in sequence_psnr):
            macro["psnr"] = {"mean_db": None, "mean_is_positive_infinity": True,
                              "sequence_positive_infinity_count": sum(row["mean_is_positive_infinity"] for row in sequence_psnr)}
        else:
            macro["psnr"] = {"mean_db": float(np.mean([row["mean_db"] for row in sequence_psnr])),
                              "mean_is_positive_infinity": False, "sequence_positive_infinity_count": 0}
        per_arm[arm] = {"per_sequence": seq_summaries, "equal_sequence_mean": macro,
                        "sequence_count": len(seq_summaries),
                        "total_scored_frames": sum(x["scored_frame_count"] for x in seq_summaries.values())}

    run_record = read_json(run_manifest_path)
    output = {
        "schema": "hqf14_reconstruction_paired_real_score_v1" if not self_test else "hqf14_reconstruction_scorer_synthetic_selftest_v1",
        "status": "real_hqf14_paired_scoring_completed" if not self_test else "synthetic_fixture_scorer_contract_pass_not_real_score",
        "created_utc": utc_now(),
        "real_data_used": not self_test,
        "real_scoring_enabled": not self_test,
        "real_scores_computed": not self_test,
        "checkpoint_selection_allowed": False,
        "protocol": {"sha256": protocol_sha256, "schema": protocol.get("schema"),
                     "path": str(DEFAULT_PROTOCOL) if not self_test else "synthetic fixture protocol"},
        "data_manifest": {"path": str(data_manifest.get("__path__", "synthetic fixture")),
                           "sha256": data_manifest.get("__sha256__"), "sequence_count": 14},
        "run_manifest": {"path": str(run_manifest_path), "sha256": sha256_file(run_manifest_path)},
        "metric_provenance": metric_provenance or {"model": "deterministic synthetic self-test surrogate; not LPIPS"},
        "metric_contract": protocol.get("photometry_and_metrics"),
        "aggregation": "frame arithmetic mean within sequence; equal arithmetic mean over all 14 sequence means",
        "per_arm": per_arm,
        "checkpoint_bindings": {
            arm: {"path": run_record["arms"][arm]["checkpoint_path"],
                  "sha256": run_record["arms"][arm]["checkpoint_sha256"]}
            for arm in EXPECTED_ARMS
        },
        "per_frame": all_frame_rows,
        "safety_note": "Real scores, when present, are report-only and must not be used to tune generators, training, scenes, or select checkpoints." if not self_test else "No real files, weights, or score data were opened.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    output["output_path"] = str(output_path)
    output["output_sha256"] = sha256_file(output_path)
    return output


class _FixtureMetric:
    """Deterministic metric surrogate used only inside --self-test."""
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        if normalize:
            raise AssertionError("fixture inputs are already normalized")
        return (prediction - target).abs().mean(dim=(1, 2, 3), keepdim=True)


def run_self_test(output_path: Path) -> dict[str, Any]:
    """Exercise all 14 data rows, event windows, every-frame metrics and macro aggregation."""
    ssim_shim_used = False
    try:
        import skimage.metrics  # noqa: F401
    except ImportError:
        # The execution host may not have scikit-image installed. This shim is
        # confined to the synthetic self-test process and is never accepted by
        # the real-score CLI; the score artifact labels it as a fixture metric.
        import types
        metrics_module = types.ModuleType("skimage.metrics")
        def fixture_ssim(a, b, data_range=1.0):
            mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
            return float(1.0 / (1.0 + mse / max(float(data_range) ** 2, 1e-12)))
        metrics_module.structural_similarity = fixture_ssim
        skimage_module = types.ModuleType("skimage")
        skimage_module.metrics = metrics_module
        sys.modules["skimage"] = skimage_module
        sys.modules["skimage.metrics"] = metrics_module
        ssim_shim_used = True
    from benchmark.sim2real.scripts.infer_hqf_e2vid_candidate import REGISTERED_SEQUENCES
    with tempfile.TemporaryDirectory(prefix="hqf14-score-selftest-") as temp:
        temp_root = Path(temp)
        hqf_root = temp_root / "data"
        inf_root = temp_root / "inference"
        hqf_root.mkdir(); inf_root.mkdir()
        fake_checkpoint = temp_root / "synthetic_only_fake_checkpoint.pt"
        fake_checkpoint.write_bytes(b"fixture checkpoint; never a trained model")
        cp_sha = sha256_file(fake_checkpoint)
        sequences = []
        inference_hashes: dict[str, dict[str, str]] = {arm: {} for arm in EXPECTED_ARMS}
        checkpoint_bindings = {}
        for arm in EXPECTED_ARMS:
            checkpoint_bindings[arm] = {"checkpoint_path": str(fake_checkpoint), "checkpoint_size_bytes": fake_checkpoint.stat().st_size,
                                        "checkpoint_sha256": cp_sha}
        preview = {name: {"sample_indices": [1, 2]} for name in REGISTERED_SEQUENCES}
        for seq_no, name in enumerate(REGISTERED_SEQUENCES):
            seq_root = hqf_root / name; seq_root.mkdir()
            h, w = 180, 240
            yy, xx = np.mgrid[:h, :w]
            base = ((xx + 2 * yy + seq_no * 3) % 240).astype(np.uint8)
            images = np.stack([base, base.copy(), base.copy()], axis=0)[..., None]
            images[2, 10:20, 12:24, 0] = np.minimum(images[2, 10:20, 12:24, 0].astype(np.int16) + 10, 255).astype(np.uint8)
            image_ts = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
            if seq_no == 0:
                # Empty second endpoint window, while retaining a prior endpoint and exact boundary events.
                event_ts = np.asarray([0.05, 0.1, 0.1, 0.2], dtype=np.float64)
            else:
                event_ts = np.asarray([0.05, 0.1, 0.1, 0.2, 0.3, 0.3], dtype=np.float64)
            event_xy = np.asarray([[i % w, (i * 7) % h] for i in range(len(event_ts))], dtype=np.int32)
            event_p = np.asarray([i % 2 for i in range(len(event_ts))], dtype=np.uint8)
            offsets = np.searchsorted(event_ts, image_ts, side="right").astype(np.int64)
            metadata = {"sensor_resolution": [h, w]}
            arrays = {"events_ts.npy": event_ts, "events_xy.npy": event_xy, "events_p.npy": event_p,
                      "images.npy": images, "images_ts.npy": image_ts, "image_event_indices.npy": offsets}
            for filename, arr in arrays.items():
                np.save(seq_root / filename, arr, allow_pickle=False)
            (seq_root / "metadata.json").write_text(json.dumps(metadata) + "\n")
            (seq_root / "conversion_manifest.json").write_text(json.dumps({"source_bag": {"bag_sha256": "fixture-only"}}) + "\n")
            files = {filename: {"size_bytes": (seq_root / filename).stat().st_size,
                                "sha256": sha256_file(seq_root / filename)} for filename in EXPECTED_MEMMAP_FILES}
            sequences.append({"sequence": name, "status": "ready_unscored", "evreal_memmap": {
                "exists": True, "path": str(seq_root), "missing_files": [], "files": files,
                "schema_checks": {"hashes": {key: value["sha256"] for key, value in files.items()},
                    "sensor_resolution_hw": [h, w], "shapes": {"images": list(images.shape), "events_ts": list(event_ts.shape)}}}})
            for arm in EXPECTED_ARMS:
                arm_seq_root = inf_root / arm / name; arm_seq_root.mkdir(parents=True)
                saved, sample_rows = [], []
                for idx in (1, 2):
                    pred = images[idx, :, :, 0].astype(np.float32) / np.float32(255.0)
                    if idx == 2 and arm == "v2e":
                        pred = np.clip(pred + np.float32(5.0 / 255.0), 0, 1).astype(np.float32)
                    pred_path = arm_seq_root / f"reconstruction_{idx:06d}.npy"
                    np.save(pred_path, pred, allow_pickle=False)
                    start, end = int(offsets[idx - 1]), int(offsets[idx])
                    sample_rows.append({"sample_index": idx, "target_timestamp_s": float(image_ts[idx]),
                        "voxel_timestamp_s": float(image_ts[idx]),
                        "event_start_index_inclusive": start, "event_end_index_exclusive": end, "event_count": end - start,
                        "event_tensor_shape": [5, h, w], "model_input_shape": [1, 5, h, w],
                        "cropped_output_shape": [1, 1, h, w]})
                    saved.append({"sample_index": idx, "float_array_path": str(pred_path), "float_array_sha256": sha256_file(pred_path)})
                inference = {
                    "status": INFERABLE_STATUS,
                    "self_test": {"is_synthetic_fixture": False},
                    "sequence": {"name": name, "path": str(seq_root), "num_frames_in_memmap": 3,
                        "num_samples_processed": 2, "num_inferable_event_endpoints": 2,
                        "initial_anchor_endpoint_excluded_from_model_and_metrics": True,
                        "timestamps_strictly_increasing": True, "samples": sample_rows},
                    "registered_inference": {"save_every_n_samples": 1, "save_float_arrays": True,
                        "saved_sample_indices": [1, 2], "saved_images": saved},
                    "model": {"checkpoint_sha256": cp_sha, "checkpoint_path": str(fake_checkpoint),
                        "public_EVREAL_weights_loaded": False, "reference_frame_pixels_used_as_input": False,
                        "real_GT_scoring_performed": False, "input_contract": "event-only; [1,5,H,W]",
                        "recurrent_reset_calls": 1,
                        "reset_policy": "one reset at the initial anchor; first model call uses events from (t0,t1]; no reset inside sequence",
                        "output_contract_after_crop": [1, 1, h, w],
                        "source_verification": {"EVREAL": {"commit": "398f03551b7f5d150ae10aa89e4e731557fce240"},
                            "event_cnn_minimal": {"commit": "78aa471a66593f55e349303563997297fbc816e8"}}},
                }
                inf_path = arm_seq_root / "inference_manifest.json"
                inf_path.write_text(json.dumps(inference, sort_keys=True) + "\n")
                inference_hashes[arm][name] = sha256_file(inf_path)
        data_manifest = {"schema": "hqf_14_file_manifest_v1", "status": "ready_unscored", "expected_sequence_count": 14,
            "ready_unscored_sequence_count": 14, "real_scoring_enabled": False,
            "real_gt_metrics_computed": False, "sequences": sequences,
            "__path__": "synthetic fixture", "__sha256__": "fixture-only"}
        data_manifest_path = temp_root / "data_manifest.json"
        data_manifest_path.write_text(json.dumps(data_manifest, sort_keys=True) + "\n")
        run_manifest_path = temp_root / "run_manifest.json"
        run_manifest = {"schema": "hqf14_paired_inference_run_v1", "status": "all_28_inferences_complete_unscored",
            "protocol_sha256": "synthetic-protocol", "inference_root": str(inf_root),
            "arms": {arm: {"checkpoint_path": str(fake_checkpoint), "checkpoint_sha256": cp_sha,
                           "inference_manifest_sha256": inference_hashes[arm]} for arm in EXPECTED_ARMS}}
        run_manifest_path.write_text(json.dumps(run_manifest, sort_keys=True) + "\n")
        protocol = {
            "schema": "synthetic-fixture-protocol", "sequences": list(REGISTERED_SEQUENCES),
            "sources": {"official_evreal_commit": "398f03551b7f5d150ae10aa89e4e731557fce240"},
            "protocol_freeze": {"checkpoint_bindings": checkpoint_bindings,
                "pre_registered_preview_indices": preview, "data_manifest": {"path": "fixture", "sha256": "fixture"}},
            "photometry_and_metrics": {"primary_metric": "fixture only; not a scientific result"},
        }
        checked_data, checked_rows = validate_data_manifest(protocol, data_manifest_path, hqf_root, self_test=True)
        checked_data["__path__"] = "synthetic fixture"
        checked_data["__sha256__"] = "fixture-only"
        _, checked_inferences = validate_run_manifest(protocol, "synthetic-protocol", run_manifest_path,
                                                       inf_root, None, self_test=True)
        result = score_pair(protocol=protocol, protocol_sha256="synthetic-protocol",
            data_manifest=checked_data, data_rows=checked_rows,
            hqf_root=hqf_root, inference_manifests={
                arm: checked_inferences[arm] for arm in EXPECTED_ARMS}, run_manifest_path=run_manifest_path,
            output_path=temp_root / "fixture_score.json", metric_model=_FixtureMetric(), device=torch.device("cpu"),
            metric_provenance={"model": "deterministic synthetic self-test surrogate; not LPIPS",
                               "ssim": "synthetic-only analytic test shim" if ssim_shim_used else "local scikit-image; synthetic-only"},
            self_test=True)
        require(result["status"] == "synthetic_fixture_scorer_contract_pass_not_real_score", "self-test status mismatch")
        require(result["real_data_used"] is False and result["real_scoring_enabled"] is False, "self-test unexpectedly enabled real scoring")
        for arm in EXPECTED_ARMS:
            require(result["per_arm"][arm]["sequence_count"] == 14, f"{arm} did not aggregate 14 sequences")
            require(result["per_arm"][arm]["total_scored_frames"] == 28, f"{arm} did not score all synthetic fixture frames")
        require(len(result["per_frame"]) == 56, "paired frame result count is not 56")
        require(result["per_arm"]["eventgenesis"]["equal_sequence_mean"]["mse_mean"] == 0.0,
                "controlled exact-match EventGenesis fixture should have zero MSE")
        require(result["per_arm"]["v2e"]["equal_sequence_mean"]["mse_mean"] > 0.0,
                "controlled V2E fixture perturbation should produce nonzero MSE")
        require(all(len(result["per_arm"][arm]["per_sequence"][name]["pre_registered_preview_artifacts"]) == 2
                    for arm in EXPECTED_ARMS for name in REGISTERED_SEQUENCES), "fixed preview rows are incomplete")

        negative_checks = []
        def expect_rejection(label: str, function: Callable[[], Any]) -> None:
            try:
                function()
            except (RuntimeError, ValueError, FileNotFoundError):
                negative_checks.append(label)
                return
            raise AssertionError(f"fail-closed self-test did not reject {label}")

        incomplete_data = copy.deepcopy(checked_data)
        incomplete_data["sequences"] = incomplete_data["sequences"][:-1]
        incomplete_path = temp_root / "incomplete_data_manifest.json"
        incomplete_path.write_text(json.dumps(incomplete_data, sort_keys=True) + "\n")
        expect_rejection("13-of-14 dataset", lambda: validate_data_manifest(protocol, incomplete_path, hqf_root, self_test=True))

        bad_time = copy.deepcopy(checked_inferences)
        bad_time["eventgenesis"][REGISTERED_SEQUENCES[0]]["sequence"]["samples"][0]["event_start_index_inclusive"] += 1
        expect_rejection("misaligned event window", lambda: score_pair(protocol=protocol, protocol_sha256="synthetic-protocol",
            data_manifest=checked_data, data_rows=checked_rows, hqf_root=hqf_root,
            inference_manifests=bad_time, run_manifest_path=run_manifest_path,
            output_path=temp_root / "bad_time_score.json", metric_model=_FixtureMetric(), device=torch.device("cpu"), self_test=True))
        require(not (temp_root / "bad_time_score.json").exists(), "failed score validation wrote a partial result")

        bad_hash = copy.deepcopy(checked_inferences)
        bad_hash["v2e"][REGISTERED_SEQUENCES[0]]["registered_inference"]["saved_images"][0]["float_array_sha256"] = "0" * 64
        expect_rejection("prediction artifact hash mismatch", lambda: score_pair(protocol=protocol, protocol_sha256="synthetic-protocol",
            data_manifest=checked_data, data_rows=checked_rows, hqf_root=hqf_root,
            inference_manifests=bad_hash, run_manifest_path=run_manifest_path,
            output_path=temp_root / "bad_hash_score.json", metric_model=_FixtureMetric(), device=torch.device("cpu"), self_test=True))
        require(not (temp_root / "bad_hash_score.json").exists(), "failed hash validation wrote a partial result")

        bad_run = copy.deepcopy(run_manifest)
        bad_run["arms"]["eventgenesis"]["checkpoint_sha256"] = "f" * 64
        bad_run_path = temp_root / "bad_checkpoint_run_manifest.json"
        bad_run_path.write_text(json.dumps(bad_run, sort_keys=True) + "\n")
        expect_rejection("wrong checkpoint binding", lambda: validate_run_manifest(protocol, "synthetic-protocol",
            bad_run_path, inf_root, None, self_test=True))
        require(len(negative_checks) == 4, "not all fail-closed negative checks ran")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_path = output_path.resolve()
        if final_path.exists():
            raise FileExistsError(final_path)
        final_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        return {"status": result["status"], "arms": list(EXPECTED_ARMS), "sequences_per_arm": 14,
                "scored_frames_per_arm": 28, "paired_frame_rows": 56,
                "eventgenesis_macro_mse": result["per_arm"]["eventgenesis"]["equal_sequence_mean"]["mse_mean"],
                "v2e_macro_mse": result["per_arm"]["v2e"]["equal_sequence_mean"]["mse_mean"],
                "negative_checks_rejected": negative_checks,
                "synthetic_ssim_test_shim_used": ssim_shim_used,
                "real_data_used": False, "real_scoring_enabled": False,
                "output": str(final_path), "sha256": sha256_file(final_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--hqf-root", type=Path)
    parser.add_argument("--inference-root", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--enable-real-scoring", action="store_true",
                        help="explicitly execute the already authorized frozen one-time HQF evaluation")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-output", type=Path)
    args = parser.parse_args()

    if args.self_test:
        if any((args.hqf_root, args.inference_root, args.run_manifest, args.model_root, args.output, args.enable_real_scoring)):
            parser.error("--self-test rejects real data/run inputs; specify only --self-test-output")
        if args.self_test_output is None:
            parser.error("--self-test requires --self-test-output")
        print(json.dumps(run_self_test(args.self_test_output), indent=2, ensure_ascii=False))
        return 0
    if not args.enable_real_scoring:
        parser.error("real HQF scoring is disabled unless --enable-real-scoring is explicitly supplied")
    for value, name in ((args.hqf_root, "--hqf-root"), (args.inference_root, "--inference-root"),
                        (args.run_manifest, "--run-manifest"), (args.model_root, "--model-root"), (args.output, "--output")):
        if value is None:
            parser.error(f"real scoring requires {name}")
    protocol_path = args.protocol.resolve()
    protocol, protocol_sha = verify_frozen_protocol(protocol_path)
    data_manifest_path = args.data_manifest.resolve()
    data_manifest, data_rows = validate_data_manifest(protocol, data_manifest_path, args.hqf_root)
    data_manifest["__path__"] = str(data_manifest_path)
    data_manifest["__sha256__"] = sha256_file(data_manifest_path)
    run_manifest, inference_manifests = validate_run_manifest(
        protocol, protocol_sha, args.run_manifest, args.inference_root, None)
    if args.output.resolve().exists():
        raise FileExistsError(f"refusing to overwrite score: {args.output}")
    device = torch.device(args.device)
    metric, metric_source, metric_weights, backbone = load_lpips_alex(args.model_root, device)
    provenance = {
        "model": "pinned PerceptualSimilarity v0.1 AlexNet with torchvision ImageNet backbone",
        "input": "grayscale repeated to RGB and mapped linearly from [0,1] to [-1,1]",
        "source_path": str(metric_source), "source_sha256": sha256_file(metric_source),
        "metric_weights_path": str(metric_weights), "metric_weights_sha256": sha256_file(metric_weights),
        "backbone_weights_path": str(backbone), "backbone_weights_sha256": sha256_file(backbone),
        "device": str(device),
    }
    result = score_pair(protocol=protocol, protocol_sha256=protocol_sha, data_manifest=data_manifest,
        data_rows=data_rows, hqf_root=args.hqf_root, inference_manifests=inference_manifests,
        run_manifest_path=args.run_manifest, output_path=args.output, metric_model=metric,
        device=device, metric_provenance=provenance)
    print(json.dumps({"status": result["status"], "protocol_sha256": protocol_sha,
                      "sequence_count": 14, "scored_frame_rows": len(result["per_frame"]),
                      "per_arm": {arm: result["per_arm"][arm]["equal_sequence_mean"] for arm in EXPECTED_ARMS},
                      "output": str(args.output.resolve()), "output_sha256": result["output_sha256"]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
