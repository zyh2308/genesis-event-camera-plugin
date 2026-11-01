#!/usr/bin/env python3
"""Generate diagnostic coverage-candidate events, retrain, and replay R1 refs.

All Genesis frames/states come from the existing matched rollout cache. The
frozen plugin core and the Direct-HDR evaluation source remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Mapping

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.r1_5_coverage_candidate import candidate_frame
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
from genesis_event_plugin.dvs_emulator import DvsEmulator
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator
from genesis_event_plugin.physics_warp import (
    compute_se3_flow,
    endpoint_radiance_residual,
    project_endpoint_to_alpha,
    projected_target_visibility,
)
from genesis_event_plugin.radiometry import prepare_linear_radiance


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _code_commit() -> str:
    value = os.environ.get("ROBOT_EXECUTION_CODE_COMMIT")
    if value:
        return value
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        result[name] = digest
    return result


def _write_hash_manifest(root: Path):
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n")
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(entries), encoding="utf-8")
    return manifest


def _write_source_contract(path: Path, contract: Dict[str, Any]) -> None:
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")


def _run_fit(*, runner_python: Path, scene_config: Path, source_root: Path,
             out: Path, train_ids, validation_ids, training: Dict[str, Any],
             source_label: str, log_path: Path):
    command = [
        str(runner_python), "-m", "benchmark.robot_closed_loop.diagnostics.train_event_state_development",
        "--config", str(scene_config), "--source-results", str(source_root), "--out", str(out),
        "--train-episodes", ",".join(map(str, train_ids)),
        "--validation-episodes", ",".join(map(str, validation_ids)),
        "--epochs", str(int(training["max_epochs"])),
        "--patience", str(int(training["patience"])),
        "--batch-size", str(int(training["batch_size"])),
        "--seed", str(int(training["seed"])),
        "--threads", str(int(training["threads"])),
        "--architecture", str(training["network"]),
        "--source-label", source_label,
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(training["threads"]),
        "MKL_NUM_THREADS": str(training["threads"]),
        "OPENBLAS_NUM_THREADS": str(training["threads"]),
        "PYTHONPATH": str(ROOT),
    }
    process = subprocess.run(command, cwd=ROOT, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.write_text(process.stdout, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"candidate model fit failed: {process.stdout[-3000:]}")
    summary_path = out / "development_summary.json"
    checkpoint_path = out / "spatial_event_state.pt"
    return {
        "checkpoint_sha256": _sha256(checkpoint_path),
        "summary_sha256": _sha256(summary_path),
        "source_manifest_sha256": _sha256(source_root / "SHA256SUMS"),
        "validation_metrics": json.loads(summary_path.read_text(encoding="utf-8"))["validation_metrics"],
    }


def _timestamp_hash(cache_path: Path) -> str:
    with h5py.File(cache_path, "r") as handle:
        timestamps = np.asarray(handle["timestamps_s"][:], dtype=np.float64)
    return hashlib.sha256(timestamps.tobytes()).hexdigest()


def _camera_intrinsics(width: int, height: int, vertical_fov_deg: float) -> np.ndarray:
    focal = height / (2.0 * np.tan(np.deg2rad(vertical_fov_deg) / 2.0))
    return np.asarray([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)


def _object_poses(handle: h5py.File, index: int, ids: np.ndarray) -> Dict[int, np.ndarray]:
    poses = handle["object_poses"][index]
    return {int(object_id): poses[slot] for slot, object_id in enumerate(ids)}


def _legacy_sample_schedule(
    interpolator: PhysicsInterpolator,
    image0: np.ndarray,
    depth0: np.ndarray,
    seg0: np.ndarray,
    image1: np.ndarray,
    depth1: np.ndarray,
    seg1: np.ndarray,
    camera0: np.ndarray,
    camera1: np.ndarray,
    t0: float,
    t1: float,
    object_world0: Mapping[int, np.ndarray],
    object_world1: Mapping[int, np.ndarray],
):
    """Reuse the exact frozen sampler inputs without constructing old frames."""
    I0 = PhysicsInterpolator._luminance(image0, "linear")
    I1 = PhysicsInterpolator._luminance(image1, "linear")
    dt = float(t1 - t0)
    object_delta = {
        int(key): np.asarray(object_world1[key], dtype=np.float64)
        @ np.linalg.inv(np.asarray(object_world0[key], dtype=np.float64))
        for key in object_world0.keys() & object_world1.keys()
    }
    camera_flow01, _ = compute_se3_flow(
        depth0, seg0, interpolator.K, camera0, camera1, object_world_deltas=None,
    )
    flow01, _ = compute_se3_flow(
        depth0, seg0, interpolator.K, camera0, camera1,
        object_world_deltas=object_delta,
    )
    articulation_flow01 = flow01 - camera_flow01
    endpoint0 = project_endpoint_to_alpha(
        depth0, seg0, interpolator.K, camera0, camera1, 1.0, 0,
        object_world0, object_world1, object_delta,
    )
    _, visibility01, _ = projected_target_visibility(depth0, endpoint0[0], endpoint0[1], endpoint0[2])
    target_valid1 = np.isfinite(depth1) & (np.asarray(depth1) > 1e-6)
    unknown01 = (~visibility01) | (~target_valid1)
    residual = endpoint_radiance_residual(
        I0, I1, depth0, endpoint0[0], endpoint0[1], endpoint0[2], eps=1e-6,
    )
    correspondence_relative_depth = np.full_like(depth0, np.nan, dtype=np.float32)
    depth_support = endpoint0[2] & np.isfinite(depth0) & (np.asarray(depth0) > 1e-6)
    correspondence_relative_depth[depth_support] = (
        np.abs(endpoint0[1][depth_support] - np.asarray(depth0)[depth_support])
        / np.maximum(np.asarray(depth0)[depth_support], 1e-6)
    )
    sampling = interpolator.sampler.compute(
        I0, flow01, dt,
        depth0=depth0,
        depth1=depth1,
        seg0=seg0,
        seg1=seg1,
        camera_flow=camera_flow01,
        articulation_flow=articulation_flow01,
        endpoint_radiance_residual=residual,
        disocclusion_mask=unknown01,
        correspondence_relative_depth=correspondence_relative_depth,
    )
    alphas = np.arange(1, sampling.samples + 1, dtype=np.float32) / float(sampling.samples)
    times = [float(t0 + float(alpha) * dt) for alpha in alphas]
    return I0, I1, object_delta, alphas, times, sampling


def _event_emulator(width: int, height: int, seed: int) -> DvsEmulator:
    response = DvsNoiseConfig(
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        leak_jitter_fraction=0.0,
        shot_noise_rate_hz=0.0,
        noise_rate_cov_decades=0.0,
        refractory_period_s=0.0,
        log_input=False,
        radiance_input=True,
        radiance_scale=1.0,
        radiance_white_level=1.0,
        radiance_log_threshold=20.0,
        response_mode="physical_log",
        radiometric_gain=1.0,
        radiometric_offset=0.0,
        radiometric_floor=1e-6,
        radiometric_calibration_status="uncalibrated",
        readout_model="ideal_readout",
        seed=int(seed),
    )
    return DvsEmulator(res=(height, width), config=response, device="cpu")


def _generate_episode_events(
    cache_path: Path,
    output_path: Path,
    max_intervals: int | None = None,
):
    chunks = []
    interval_rows = []
    with h5py.File(cache_path, "r") as handle:
        timestamps = np.asarray(handle["timestamps_s"][:], dtype=np.float64)
        camera_poses = np.asarray(handle["camera_to_world_cv"][:], dtype=np.float64)
        ids = np.asarray(handle["segmentation_pose_ids"][:], dtype=np.int32)
        width, height = (int(value) for value in handle.attrs["camera_resolution_wh"])
        intrinsics = _camera_intrinsics(width, height, float(handle.attrs["vertical_fov_deg"]))
        interpolator = PhysicsInterpolator(
            intrinsics,
            sampler_config=AdaptiveSamplingConfig(),
            disocclusion_policy="hold",
            composite="next_primary",
            warp_backend="numpy",
        )
        first_luma = prepare_linear_radiance(handle["radiance"][0], input_space="linear")
        dvs = _event_emulator(width, height, seed=42)
        dvs.initialize(first_luma, float(timestamps[0]))
        previous = {
            "image": handle["radiance"][0],
            "luma": first_luma,
            "depth": handle["depth"][0],
            "seg": handle["segmentation"][0],
            "objects": _object_poses(handle, 0, ids),
        }
        total_intervals = len(timestamps) - 1
        run_intervals = total_intervals if max_intervals is None else min(total_intervals, max_intervals)
        for index in range(run_intervals):
            next_index = index + 1
            current = {
                "image": handle["radiance"][next_index],
                "luma": prepare_linear_radiance(handle["radiance"][next_index], input_space="linear"),
                "depth": handle["depth"][next_index],
                "seg": handle["segmentation"][next_index],
                "objects": _object_poses(handle, next_index, ids),
            }
            t0, t1 = float(timestamps[index]), float(timestamps[next_index])
            I0, I1, object_delta, alphas, sample_times, sampling = _legacy_sample_schedule(
                interpolator,
                previous["image"], previous["depth"], previous["seg"],
                current["image"], current["depth"], current["seg"],
                camera_poses[index], camera_poses[next_index], t0, t1,
                previous["objects"], current["objects"],
            )
            emitted = 0
            for alpha, sample_time in zip(alphas, sample_times):
                frame = candidate_frame(
                    I0, previous["depth"], previous["seg"],
                    I1, current["depth"], current["seg"],
                    camera_poses[index], camera_poses[next_index], intrinsics,
                    float(alpha), previous["objects"], current["objects"],
                )
                event_chunk = np.asarray(dvs.generate_events(frame, sample_time), dtype=np.float64)
                if event_chunk.size:
                    chunks.append(event_chunk)
                    emitted += int(len(event_chunk))
            interval_rows.append({
                "interval_index": index,
                "t0_s": t0,
                "t1_s": t1,
                "sample_count": int(sampling.samples),
                "sample_times_s": sample_times,
                "max_boundary_flow_px": float(sampling.max_boundary_flow_px),
                "max_radiance_residual": float(sampling.max_radiance_residual),
                "generated_events": emitted,
            })
            previous = current
            if (index + 1) % 20 == 0 or index + 1 == run_intervals:
                print(json.dumps({
                    "cache": cache_path.name,
                    "intervals_done": index + 1,
                    "intervals_total": run_intervals,
                    "events_so_far": int(sum(len(chunk) for chunk in chunks)),
                }), flush=True)

    events = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)
    if len(events):
        events = events[np.argsort(events[:, 0], kind="stable")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, events_2000hz=events)
    return events, interval_rows, total_intervals, run_intervals


def _run_evaluation(*, python: Path, scene_config: Path, control_config: Path,
                    reference_root: Path, ours_checkpoint: Path,
                    v2e_checkpoint: Path, output: Path, log: Path,
                    threads: int, deceleration: float):
    command = [
        str(python), "-m", "benchmark.robot_closed_loop.diagnostics.evaluate_r1_common_direct_hdr",
        "--scene-config", str(scene_config), "--control-config", str(control_config),
        "--reference-root", str(reference_root), "--ours-checkpoint", str(ours_checkpoint),
        "--v2e-checkpoint", str(v2e_checkpoint), "--output", str(output),
        "--deceleration", str(deceleration),
    ]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "PYTHONPATH": str(ROOT),
    }
    started = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log.write_text(process.stdout, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"common Direct-HDR evaluation failed: {process.stdout[-3000:]}")
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--v2e-training-source", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--baseline-ours-checkpoint", type=Path, required=True)
    parser.add_argument("--v2e-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path,
                        default=ROOT / "benchmark/robot_closed_loop/configs/r1_5_coverage_candidate_retrain_20261101a.json")
    parser.add_argument("--scene-config", type=Path,
                        default=ROOT / "benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json")
    parser.add_argument("--control-config", type=Path,
                        default=ROOT / "benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--episode-ids", default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--max-intervals", type=int, default=None,
                        help="bounded generation smoke only; never trains/evaluates a partial run")
    parser.add_argument("--skip-fit", action="store_true",
                        help="generate-only smoke; never fits/evaluates partial event streams")
    args = parser.parse_args()

    config_path = args.experiment_config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training = config["training"]
    train_ids = [int(value) for value in training["train_episode_ids"]]
    validation_ids = [int(value) for value in training["validation_episode_ids"]]
    episode_ids = [int(value) for value in args.episode_ids.split(",") if value.strip()]
    expected_ids = set(train_ids) | set(validation_ids)
    if not set(episode_ids).issubset(expected_ids):
        raise ValueError("episode IDs must be a subset of the locked train/validation episodes")
    partial = args.max_intervals is not None or set(episode_ids) != expected_ids
    if partial and not args.skip_fit:
        raise ValueError("partial generation is smoke-only; pass --skip-fit and do not train")
    if args.max_intervals is not None and args.max_intervals <= 0:
        raise ValueError("--max-intervals must be positive")

    cache_root = args.cache_root.resolve()
    v2e_root = args.v2e_training_source.resolve()
    reference_root = args.reference_root.resolve()
    baseline_ours = args.baseline_ours_checkpoint.resolve()
    v2e_checkpoint = args.v2e_checkpoint.resolve()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite R1.5 candidate run: {out}")
    for required in (v2e_root / "SHA256SUMS", v2e_root / "episode_split_contract.json",
                     reference_root / "SHA256SUMS", baseline_ours, v2e_checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)
    experiment = json.loads(config_path.read_text(encoding="utf-8"))
    v2e_manifest_sha = _sha256(v2e_root / "SHA256SUMS")
    if v2e_manifest_sha != experiment["v2e_input_endpoint"]["source_manifest_sha256"]:
        raise ValueError("V2E training source manifest differs from the locked paired small9 batch")
    v2e_manifest = _manifest(v2e_root / "SHA256SUMS")
    v2e_split = json.loads((v2e_root / "episode_split_contract.json").read_text(encoding="utf-8"))
    if (v2e_split["train_episode_ids"] != train_ids
            or v2e_split["validation_episode_ids"] != validation_ids
            or v2e_split["training_seed"] != int(training["seed"])
            or v2e_split["architecture"] != training["network"]
            or v2e_split["fixed_budget"]["max_epochs"] != int(training["max_epochs"])
            or v2e_split["fixed_budget"]["patience"] != int(training["patience"])
            or v2e_split["fixed_budget"]["batch_size"] != int(training["batch_size"])
            or v2e_split["fixed_budget"]["threads"] != int(training["threads"])):
        raise ValueError("V2E source episode split or training budget differs from the locked protocol")

    out.mkdir(parents=True)
    candidate_root = out / "ours_coverage_candidate_training"
    candidate_root.mkdir()
    split_contract = {
        "train_episode_ids": train_ids,
        "validation_episode_ids": validation_ids,
        "split_unit": "episode",
        "training_seed": int(training["seed"]),
        "architecture": training["network"],
        "fixed_budget": {
            "max_epochs": int(training["max_epochs"]),
            "patience": int(training["patience"]),
            "batch_size": int(training["batch_size"]),
            "threads": int(training["threads"]),
        },
        "purpose": "coverage-splat candidate event retrain and same Direct-HDR development replay; not a ranking",
        "candidate_method": "front-to-back bilinear footprint resolve; frozen sampler timestamps and physical_log DVS response",
    }
    split_sha = hashlib.sha256(json.dumps(split_contract, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
    _write_source_contract(candidate_root / "episode_split_contract.json",
                           {**split_contract, "sha256": split_sha})

    scene_config = args.scene_config.resolve()
    source_cache_hashes: Dict[str, str] = {}
    episode_records = []
    generation_started = time.perf_counter()
    for episode_id in episode_ids:
        episode_name = f"episode_{episode_id:03d}"
        cache_dir = cache_root / episode_name
        cache_path = cache_dir / "trajectory_cache.h5"
        result_path = cache_dir / "episode_result.json"
        trace_path = cache_dir / "oracle_trace.jsonl"
        frame_manifest = cache_dir / "frame_manifest.json"
        for path in (cache_path, result_path, trace_path, frame_manifest):
            if not path.is_file():
                raise FileNotFoundError(path)
        source_cache_hashes[str(cache_path)] = _sha256(cache_path)
        v2e_episode = v2e_root / "episodes" / episode_name
        v2e_event_path = v2e_episode / "direct_hdr_events_1khz_2khz.npz"
        pair_path = v2e_episode / "pair_provenance.json"
        for path in (v2e_event_path, pair_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        provenance = {
            "shared_cache_sha256": _sha256(cache_path),
            "frame_manifest_sha256": _sha256(frame_manifest),
            "episode_result_sha256": _sha256(result_path),
            "trace_sha256": _sha256(trace_path),
        }
        mismatch = {key: {"v2e": pair.get(key), "cache": value}
                    for key, value in provenance.items() if pair.get(key) != value}
        if mismatch:
            raise ValueError(f"V2E paired rollout provenance mismatch for {episode_name}: {mismatch}")
        if v2e_manifest.get(f"episodes/{episode_name}/direct_hdr_events_1khz_2khz.npz") != _sha256(v2e_event_path):
            raise ValueError(f"V2E event archive hash mismatch for {episode_name}")
        expected_split = "train" if episode_id in train_ids else "validation"
        if pair.get("split") != expected_split:
            raise ValueError(f"episode split mismatch for {episode_name}")

        event_path = candidate_root / "episodes" / episode_name / "direct_hdr_events_1khz_2khz.npz"
        started = time.perf_counter()
        events, interval_rows, total_intervals, run_intervals = _generate_episode_events(
            cache_path, event_path, max_intervals=args.max_intervals,
        )
        generation_wall_s = time.perf_counter() - started
        episode_result = json.loads(result_path.read_text(encoding="utf-8"))
        trace = _jsonl(trace_path)
        episode_dir = event_path.parent
        episode_dir.mkdir(parents=True, exist_ok=True)
        fields = ["episode_id", "seed", "launch_speed_m_s", "start_y_m", "first_contact_s",
                  "lift_start_s", "sim_duration_s"]
        with (episode_dir / "episode.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow({key: episode_result.get(key) for key in fields})
        with (episode_dir / "state_control_trace.jsonl").open("w", encoding="utf-8") as stream:
            for row in trace:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        (episode_dir / "pair_provenance.json").write_text(json.dumps({
            "episode_id": int(episode_id),
            "seed": int(episode_result["seed"]),
            "speed_m_s": float(episode_result["launch_speed_m_s"]),
            "launch_y_m": float(episode_result["start_y_m"]),
            **provenance,
            "v2e_event_archive_sha256": _sha256(v2e_event_path),
            "candidate_generator": "R1.5 diagnostic front-to-back coverage splat",
            "same_physics_cache_camera_timestamps_and_labels_as_paired_v2e": True,
            "split": expected_split,
        }, indent=2), encoding="utf-8")
        (episode_dir / "sampling_intervals.json").write_text(json.dumps({
            "complete": run_intervals == total_intervals,
            "interval_count": run_intervals,
            "cache_interval_count": total_intervals,
            "sample_count_total": int(sum(row["sample_count"] for row in interval_rows)),
            "intervals": interval_rows,
        }, separators=(",", ":")), encoding="utf-8")
        episode_records.append({
            "episode_id": episode_id,
            "seed": int(episode_result["seed"]),
            "speed_m_s": float(episode_result["launch_speed_m_s"]),
            "launch_y_m": float(episode_result["start_y_m"]),
            "cache_sha256": _sha256(cache_path),
            "timestamps_sha256": _timestamp_hash(cache_path),
            "candidate_event_count": int(len(events)),
            "candidate_event_archive_sha256": _sha256(event_path),
            "complete": run_intervals == total_intervals,
            "sampling_interval_count": run_intervals,
            "sampling_frame_count": int(sum(row["sample_count"] for row in interval_rows)),
            "generation_wall_s": generation_wall_s,
        })
        print(json.dumps({"episode_id": episode_id, "candidate_event_count": len(events),
                          "complete": run_intervals == total_intervals,
                          "generation_wall_s": generation_wall_s}), flush=True)

    _write_hash_manifest(candidate_root)
    if args.skip_fit:
        summary = {
            "status": "GENERATION_SMOKE_ONLY",
            "code_commit": _code_commit(),
            "candidate_generator_sha256": _sha256(Path(__file__).with_name("r1_5_coverage_candidate.py")),
            "candidate_event_source_manifest_sha256": _sha256(candidate_root / "SHA256SUMS"),
            "episodes": episode_records,
            "partial_generation": partial,
            "frozen_core_modified": False,
        }
        (out / "generation_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_hash_manifest(out)
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    if partial or set(episode_ids) != expected_ids:
        raise ValueError("a training run requires complete candidate events for all nine locked episodes")
    generation_wall = time.perf_counter() - generation_started

    training = config["training"]
    fit_started = time.perf_counter()
    runner_python = Path(args.python).resolve()
    candidate_fit = _run_fit(
        runner_python=runner_python,
        scene_config=scene_config,
        source_root=candidate_root,
        out=out / "candidate_fit",
        train_ids=train_ids,
        validation_ids=validation_ids,
        training=training,
        source_label="R1.5 coverage-candidate Genesis scene-linear HDR events",
        log_path=out / "candidate_fit.log",
    )
    fit_wall = time.perf_counter() - fit_started

    control_config = args.control_config.resolve()
    deceleration = float(json.loads(scene_config.read_text(encoding="utf-8"))["controller"]
                         ["initial_deceleration_assumption_m_s2"])
    eval_root = out / "common_reference_eval"
    candidate_eval_out = eval_root / "candidate_ours_vs_v2e"
    baseline_eval_out = eval_root / "previous_native_ours_vs_v2e"
    candidate_eval_wall = _run_evaluation(
        python=runner_python, scene_config=scene_config, control_config=control_config,
        reference_root=reference_root,
        ours_checkpoint=out / "candidate_fit" / "spatial_event_state.pt",
        v2e_checkpoint=v2e_checkpoint, output=candidate_eval_out,
        log=out / "candidate_common_reference_eval.log",
        threads=int(training["threads"]), deceleration=deceleration,
    )
    baseline_eval_wall = _run_evaluation(
        python=runner_python, scene_config=scene_config, control_config=control_config,
        reference_root=reference_root,
        ours_checkpoint=baseline_ours, v2e_checkpoint=v2e_checkpoint,
        output=baseline_eval_out, log=out / "baseline_common_reference_eval.log",
        threads=int(training["threads"]), deceleration=deceleration,
    )
    candidate_summary = json.loads((candidate_eval_out / "r1_common_direct_hdr_summary.json").read_text())
    baseline_summary = json.loads((baseline_eval_out / "r1_common_direct_hdr_summary.json").read_text())
    candidate_manifest = candidate_summary["reference_manifest_sha256"]
    baseline_manifest = baseline_summary["reference_manifest_sha256"]
    if candidate_manifest != baseline_manifest:
        raise RuntimeError("candidate and prior-model evaluations did not use the same reference manifest")

    summary = {
        "status": "PASS",
        "experiment_id": experiment["experiment_id"],
        "purpose": "candidate-event development retrain on paired Genesis caches and same Direct-HDR reference replay; not ranking",
        "code_commit": _code_commit(),
        "experiment_config_sha256": _sha256(config_path),
        "scene_config_sha256": _sha256(scene_config),
        "control_config_sha256": _sha256(control_config),
        "candidate_generator_sha256": _sha256(Path(__file__).with_name("r1_5_coverage_candidate.py")),
        "candidate_source_manifest_sha256": _sha256(candidate_root / "SHA256SUMS"),
        "candidate_source_root": str(candidate_root),
        "cache_root": str(cache_root),
        "cache_input_sha256": source_cache_hashes,
        "v2e_training_source_manifest_sha256": v2e_manifest_sha,
        "episode_split": {**split_contract, "sha256": split_sha},
        "episodes": episode_records,
        "training": {
            "candidate_checkpoint_sha256": candidate_fit["checkpoint_sha256"],
            "validation_metrics": candidate_fit["validation_metrics"],
            "previous_native_ours_checkpoint_sha256": _sha256(baseline_ours),
            "v2e_checkpoint_sha256": _sha256(v2e_checkpoint),
            "same_architecture_budget_split_seed_as_previous_native_hdr_retrain": True,
        },
        "common_reference": {
            "root": str(reference_root),
            "manifest_sha256": candidate_manifest,
            "candidate_evaluation_summary_sha256": _sha256(candidate_eval_out / "r1_common_direct_hdr_summary.json"),
            "candidate_evaluation_polls_sha256": _sha256(candidate_eval_out / "r1_common_direct_hdr_polls.csv"),
            "previous_model_evaluation_summary_sha256": _sha256(baseline_eval_out / "r1_common_direct_hdr_summary.json"),
            "previous_model_evaluation_polls_sha256": _sha256(baseline_eval_out / "r1_common_direct_hdr_polls.csv"),
            "same_reference_files_timestamps_and_evaluator": True,
        },
        "evaluation": {
            "candidate_ours": candidate_summary["overall"],
            "previous_native_ours": baseline_summary["overall"],
            "same_v2e_control_metrics": candidate_summary["overall"].get("v2e"),
        },
        "cost_s": {
            "candidate_event_generation": generation_wall,
            "candidate_fit": fit_wall,
            "candidate_common_reference_evaluation": candidate_eval_wall,
            "previous_checkpoint_rerun": baseline_eval_wall,
            "end_to_end": generation_wall + fit_wall + candidate_eval_wall + baseline_eval_wall,
            "excludes": "original physics/cache capture, Direct-HDR reference generation, and archived V2E generation",
        },
        "boundaries": {
            "candidate_uses_cached_GT_geometry_only_inside_event_generator": True,
            "learned_model_inputs_only_causal_five_bin_events": True,
            "GT_used_for_training_labels_and_post_inference_scoring_only": True,
            "evaluation_set_is_the_previously_inspected_nine_episode_development_set": True,
            "formal_held_out_test_or_real_event_truth": False,
            "frozen_core_modified": False,
        },
        "interpretation": "Candidate utility is measured on a common Direct-HDR development reference. This does not establish physical event-camera truth or held-out generalization.",
    }
    (out / "candidate_retrain_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_hash_manifest(out)
    print(json.dumps({
        "status": "PASS",
        "output": str(out),
        "generation_s": generation_wall,
        "candidate_checkpoint_sha256": candidate_fit["checkpoint_sha256"],
        "candidate_overall": candidate_summary["overall"],
        "previous_native_ours_overall": baseline_summary["overall"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
