#!/usr/bin/env python3
"""Bounded CUDA/CPU parity probe for the isolated R1.5 core candidate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.validate_r1_5_core_candidate import (  # noqa: E402
    event_metrics,
    load_cached_episode,
    sha256,
)
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig  # noqa: E402
from genesis_event_plugin.dvs_emulator import DvsEmulator  # noqa: E402
from genesis_event_plugin.noise_model import DvsNoiseConfig  # noqa: E402
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator  # noqa: E402


DEFAULT_CACHE = ROOT / "benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059"
DEFAULT_CONFIG = ROOT / "benchmark/robot_closed_loop/configs/r1_5_core_cuda_parity_20261030a.json"


def luma_error(a: np.ndarray, b: np.ndarray) -> dict:
    eps = 1e-6
    aa = np.log(np.maximum(np.asarray(a, dtype=np.float32), eps))
    bb = np.log(np.maximum(np.asarray(b, dtype=np.float32), eps))
    delta = np.abs(aa - bb)
    return {
        "log_luma_abs_mean": float(delta.mean()),
        "log_luma_abs_p95": float(np.quantile(delta, 0.95)),
        "log_luma_abs_max": float(delta.max()),
        "radiance_abs_max": float(np.max(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))),
    }


def _dvs_config() -> DvsNoiseConfig:
    return DvsNoiseConfig(
        pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0,
        cutoff_hz=0.0, leak_rate_hz=0.0, leak_jitter_fraction=0.0,
        shot_noise_rate_hz=0.0, noise_rate_cov_decades=0.0,
        refractory_period_s=0.0, radiance_input=True,
        response_mode="physical_log", readout_model="ideal_readout", seed=42,
    )


def run_dvs(frames: list[np.ndarray], times: list[float], initial: np.ndarray, t0: float, device: str):
    h, w = initial.shape
    emulator = DvsEmulator(res=(h, w), config=_dvs_config(), device=device)
    emulator.initialize(initial, t0)
    chunks = [emulator.generate_events(frame, t) for frame, t in zip(frames, times)]
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)


def run_threshold_probes() -> list[dict]:
    rows = []
    for polarity in (1, -1):
        for magnitude in (0.1999, 0.2, 0.2001, 0.3999, 0.4, 0.4001):
            delta = float(polarity * magnitude)
            initial = np.ones((2, 3), dtype=np.float64)
            endpoint = np.full_like(initial, math.exp(delta))
            cpu = run_dvs([endpoint], [0.01], initial, 0.0, "cpu")
            gpu = run_dvs([endpoint], [0.01], initial, 0.0, "cuda:0")
            exact = cpu.shape == gpu.shape and np.array_equal(cpu, gpu)
            rows.append({
                "log_contrast": delta,
                "cpu_event_count": int(len(cpu)),
                "cuda_event_count": int(len(gpu)),
                "cpu_polarities": {str(int(p)): int(np.count_nonzero(cpu[:, 3] == p)) for p in (-1, 1)},
                "cuda_polarities": {str(int(p)): int(np.count_nonzero(gpu[:, 3] == p)) for p in (-1, 1)},
                "raw_event_arrays_exact": bool(exact),
                "cpu_events": cpu,
                "cuda_events": gpu,
            })
    return rows


def run_episode(cache_root: Path, episode: str) -> dict:
    source, dense_times, dense_luma, _rgb, _seg, width, height, fov, cache_path, evidence_path = load_cached_episode(cache_root, episode)
    t0, t1 = 0.5805, 0.5905
    timestamps = source["timestamps_s"]
    left = int(np.flatnonzero(np.isclose(timestamps, t0, atol=1e-9, rtol=0))[0])
    right = int(np.flatnonzero(np.isclose(timestamps, t1, atol=1e-9, rtol=0))[0])
    pose_ids = source["segmentation_pose_ids"]
    object0 = {int(sid): source["object_poses"][left, j] for j, sid in enumerate(pose_ids)}
    object1 = {int(sid): source["object_poses"][right, j] for j, sid in enumerate(pose_ids)}
    focal = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    K = np.asarray([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    common = dict(
        sampler_config=AdaptiveSamplingConfig(min_samples=2, max_samples=2),
        disocclusion_policy="hold", composite="same_surface_log_linear",
        strict_fidelity=False, max_photometric_interval_s=0.005,
    )
    outputs = {}
    timings = {}
    diagnostics = {}
    masks = {}
    for name, backend, device in (("numpy", "numpy", "cpu"), ("cuda", "torch", "cuda:0")):
        interpolator = PhysicsInterpolator(K, warp_backend=backend, device=device, **common)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        frames, frame_times = interpolator.interpolate(
            source["radiance_luma"][left], source["depth"][left], source["segmentation"][left],
            source["radiance_luma"][right], source["depth"][right], source["segmentation"][right],
            source["camera_to_world_cv"][left], source["camera_to_world_cv"][right], None,
            t0, t1, input_space="linear", object_world0=object0, object_world1=object1,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings[name] = float(time.perf_counter() - started)
        if len(frames) != 2 or not np.allclose(frame_times, [0.5855, t1], atol=1e-8, rtol=0):
            raise RuntimeError(f"episode {episode}: unexpected candidate timestamps {frame_times}")
        outputs[name] = np.stack([np.asarray(frame, dtype=np.float32) for frame in frames])
        diagnostics[name] = {
            "valid_fraction_by_output": [float(np.asarray(mask, dtype=bool).mean()) for mask in interpolator.last_visibility["intermediate_visibility"]],
            "unknown_fraction_by_output": [float(np.asarray(mask, dtype=bool).mean()) for mask in interpolator.last_visibility["intermediate_unknown"]],
            "uncertainty": interpolator.last_uncertainty,
            "output_times_s": [float(x) for x in frame_times],
        }
        masks[name] = np.asarray(interpolator.last_visibility["intermediate_visibility"], dtype=bool)

    pair = luma_error(outputs["numpy"][0], outputs["cuda"][0])
    event_outputs = {}
    direct_rows = (dense_times > t0 + 1e-9) & (dense_times <= t1 + 1e-9)
    dense_clip = dense_luma[direct_rows]
    direct_clip_times = dense_times[direct_rows]
    source_ref = source["radiance_luma"][left]
    event_outputs["direct"] = run_dvs([*dense_clip], [float(x) for x in direct_clip_times], source_ref, t0, "cpu")
    for name, backend, device in (("numpy", "numpy", "cpu"), ("cuda", "torch", "cuda:0")):
        event_outputs[name] = run_dvs(
            [outputs[name][0], outputs[name][1]], [0.5855, t1], source_ref, t0, "cpu",
        )
    support_mismatch = int(np.count_nonzero(masks["numpy"][0] != masks["cuda"][0]))
    common_support = masks["numpy"][0] & masks["cuda"][0]
    if not common_support.any():
        raise RuntimeError(f"episode {episode}: no common supported midpoint pixels")
    supported_error = luma_error(outputs["numpy"][0][common_support], outputs["cuda"][0][common_support])
    events_pair = event_metrics(event_outputs["numpy"], event_outputs["cuda"], width)
    events_pair["event_count_relative_delta"] = float(
        abs(events_pair["candidate_event_count"] - events_pair["reference_event_count"])
        / max(1, events_pair["reference_event_count"])
    )
    events_vs_direct_numpy = event_metrics(event_outputs["direct"], event_outputs["numpy"], width)
    events_vs_direct_cuda = event_metrics(event_outputs["direct"], event_outputs["cuda"], width)
    return {
        "episode": episode,
        "input_sha256": {"state_cache": sha256(cache_path), "direct_hdr_evidence": sha256(evidence_path)},
        "source_shape": list(source["radiance_luma"][left].shape),
        "timestamps_s": [t0, 0.5855, t1],
        "wall_s_full_interpolator": timings,
        "backend_diagnostics": diagnostics,
        "numpy_cuda_midpoint_image": pair,
        "numpy_cuda_midpoint_supported_pixels": supported_error,
        "numpy_cuda_supported_mask_mismatch_pixels": support_mismatch,
        "numpy_cuda_support_mismatch_checked_exactly": True,
        "events": {
            "numpy_vs_cuda": events_pair,
            "numpy_vs_archived_1ms_direct": events_vs_direct_numpy,
            "cuda_vs_archived_1ms_direct": events_vs_direct_cuda,
            "claim_boundary": "two cached 10 ms robot intervals; the candidate midpoint warp is not a direct-render event reference",
        },
        "arrays": {
            "numpy_midpoint_luma": outputs["numpy"][0],
            "cuda_midpoint_luma": outputs["cuda"][0],
            "numpy_support": masks["numpy"][0].astype(np.uint8),
            "cuda_support": masks["cuda"][0].astype(np.uint8),
            "archived_direct_events_1ms": event_outputs["direct"],
            "numpy_events": event_outputs["numpy"],
            "cuda_events": event_outputs["cuda"],
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["episode", "backend", "wall_s", "mean_log_error", "p95_log_error", "max_log_error",
              "valid_fraction_midpoint", "unknown_fraction_midpoint", "candidate_events", "direct_events",
              "event_count_l1", "event_time_p95_s"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.cuda.init()
    torch.cuda.synchronize()
    device = torch.cuda.get_device_properties(0)
    started = time.perf_counter()
    episodes = [run_episode(args.cache_root.resolve(), ep) for ep in ("004", "005")]
    threshold_rows = run_threshold_probes()
    threshold_summary = [{k: v for k, v in row.items() if k not in {"cpu_events", "cuda_events"}} for row in threshold_rows]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    criteria = config["acceptance"]
    threshold_exact = all(row["raw_event_arrays_exact"] for row in threshold_rows)
    episode_gates = {
        episode["episode"]: {
            "support_mask_exact": episode["numpy_cuda_supported_mask_mismatch_pixels"] == 0,
            "supported_log_luma_max_within_limit": episode["numpy_cuda_midpoint_supported_pixels"]["log_luma_abs_max"] <= criteria["max_abs_log_luminance_delta_supported_pixels"],
            "cached_event_count_relative_delta_within_limit": episode["events"]["numpy_vs_cuda"]["event_count_relative_delta"] <= criteria["cached_clip_event_count_relative_delta_max"],
        }
        for episode in episodes
    }
    acceptance = {
        "threshold_dvs_cpu_cuda_exact": threshold_exact,
        "episodes": episode_gates,
        "all_gates_pass": threshold_exact and all(all(gates.values()) for gates in episode_gates.values()),
        "note": "backend parity only; this is not a renderer-fidelity or closed-loop acceptance decision",
    }
    summary = {
        "run_id": "r1_5_core_cuda_parity_20261030a",
        "scope": "isolated candidate CUDA/CPU backend parity on cached tracks and threshold-adjacent noiseless DVS controls; no Genesis renderer, physics advancement, training, or ranking",
        "candidate_branch": "exp/r1_5-core-candidate-20261030",
        "candidate_commit": __import__("subprocess").check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": {"name": device.name, "total_memory_bytes": int(device.total_memory), "capability": list(torch.cuda.get_device_capability(0))},
        "config_sha256": sha256(args.config.resolve()),
        "source_module_sha256": {
            name: sha256(ROOT / "genesis_event_plugin" / name)
            for name in ("physics_warp.py", "torch_physics.py", "physics_interpolator.py", "dvs_emulator.py")
        },
        "diagnostic_script_sha256": sha256(Path(__file__).resolve()),
        "threshold_probe": threshold_summary,
        "episodes": [{k: v for k, v in episode.items() if k != "arrays"} for episode in episodes],
        "wall_s_total": float(time.perf_counter() - started),
        "acceptance_config": config,
        "acceptance_result": acceptance,
        "scope_limit": "backend parity does not validate the renderer, photometric sampling fidelity, or real-time end-to-end cost",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "threshold_dvs.json").write_text(json.dumps(threshold_summary, indent=2) + "\n", encoding="utf-8")
    flat = []
    for episode in episodes:
        for backend in ("numpy", "cuda"):
            err = luma_error(episode["arrays"][f"{backend}_midpoint_luma"], episode["arrays"]["numpy_midpoint_luma"])
            evt = episode["events"]["numpy_vs_archived_1ms_direct"] if backend == "numpy" else episode["events"]["cuda_vs_archived_1ms_direct"]
            flat.append({
                "episode": episode["episode"], "backend": backend,
                "wall_s": episode["wall_s_full_interpolator"][backend],
                "mean_log_error": err["log_luma_abs_mean"], "p95_log_error": err["log_luma_abs_p95"],
                "max_log_error": err["log_luma_abs_max"],
                "valid_fraction_midpoint": episode["backend_diagnostics"][backend]["valid_fraction_by_output"][0],
                "unknown_fraction_midpoint": episode["backend_diagnostics"][backend]["unknown_fraction_by_output"][0],
                "candidate_events": evt["candidate_event_count"], "direct_events": evt["reference_event_count"],
                "event_count_l1": evt["pixel_polarity_count_l1"],
                "event_time_p95_s": evt["matched_event_time_abs_error_p95_s"],
            })
    write_csv(args.output / "cached_episode_metrics.csv", flat)
    arrays = {}
    for episode in episodes:
        for key, value in episode["arrays"].items():
            arrays[f"ep{episode['episode']}_{key}"] = value
    arrays["threshold_log_contrasts"] = np.asarray([r["log_contrast"] for r in threshold_rows], dtype=np.float64)
    for i, row in enumerate(threshold_rows):
        arrays[f"threshold_{i}_cpu_events"] = row["cpu_events"]
        arrays[f"threshold_{i}_cuda_events"] = row["cuda_events"]
    np.savez_compressed(args.output / "cuda_parity_arrays.npz", **arrays)
    files = sorted(p for p in args.output.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(args.output).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "gpu": device.name, "summary": summary,
                      "acceptance_result": acceptance,
                      "manifest_sha256": sha256(args.output / "SHA256SUMS")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
