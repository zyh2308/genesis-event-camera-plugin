#!/usr/bin/env python3
"""Measure strict 1 ms intermediate HDR samples on exact archived states.

This is an offline renderer-cost replay. It does not advance physics or create
state rows. Every rendered pose must match an exact timestamp in the archived
trace. Scene-linear HDR and segmentation are compared with the stored direct
frames; CPU DVS is run over both sequences using the fixed R1.5 response.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from benchmark.robot_closed_loop.robot_env import build_robot_scene
from genesis_event_plugin.dvs_emulator import DvsEmulator
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.radiometry import linear_luminance


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hdr_runtime_preflight() -> dict:
    """Fail before scene setup if this interpreter lacks the verified HDR API."""
    import genesis as gs
    from genesis.vis.camera import Camera

    signature = inspect.signature(Camera.render)
    if "radiance" not in signature.parameters:
        raise RuntimeError(
            "This Genesis runtime does not expose Camera.render(radiance=...). "
            "Use the verified .venv-robot interpreter; never substitute display RGB for HDR. "
            f"python={sys.executable}; genesis={gs.__file__}; signature={signature}"
        )

    paths = {
        "genesis": Path(gs.__file__).resolve(),
        "camera": Path(inspect.getsourcefile(Camera)).resolve(),
    }
    try:
        from genesis.ext.pyrender import offscreen, renderer
        from genesis.vis import rasterizer
        paths.update({
            "rasterizer": Path(inspect.getsourcefile(rasterizer)).resolve(),
            "renderer": Path(inspect.getsourcefile(renderer)).resolve(),
            "offscreen": Path(inspect.getsourcefile(offscreen)).resolve(),
        })
    except (ImportError, TypeError):
        pass
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "genesis_module": str(paths["genesis"]),
        "camera_render_signature": str(signature),
        "camera_radiance_parameter": True,
        "renderer_module_paths": {name: str(path) for name, path in paths.items()},
        "renderer_module_sha256": {name: sha256(path) for name, path in paths.items() if path.is_file()},
        "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
        "egl_device_id": os.environ.get("EGL_DEVICE_ID"),
    }


def materialize_image(value) -> tuple[np.ndarray, float]:
    """Measure any device-to-host conversion plus an owned host array copy."""
    started = time.perf_counter()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True), time.perf_counter() - started


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def exact_state(row: dict, env) -> None:
    env.robot.set_qpos(np.asarray(row["robot_qpos"], dtype=np.float32), zero_velocity=True)
    obj = row["object"]
    qpos = np.asarray(list(obj["position_m"]) + list(obj["orientation_wxyz"]), dtype=np.float32)
    env.target.set_qpos(qpos, zero_velocity=True)


def response_config() -> DvsNoiseConfig:
    return DvsNoiseConfig(
        pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0, cutoff_hz=0.0,
        leak_rate_hz=0.0, leak_jitter_fraction=0.0, shot_noise_rate_hz=0.0,
        noise_rate_cov_decades=0.0, refractory_period_s=0.0, log_input=False,
        radiance_input=True, response_mode="physical_log", radiometric_gain=1.0,
        radiometric_offset=0.0, radiometric_floor=1e-6,
        radiometric_calibration_status="uncalibrated", readout_model="ideal_readout", seed=42,
    )


def events_from_sequence(luma: list[np.ndarray], times: list[float]) -> tuple[np.ndarray, float]:
    h, w = luma[0].shape
    dvs = DvsEmulator(res=(h, w), config=response_config(), device="cpu")
    started = time.perf_counter()
    dvs.initialize(luma[0], times[0])
    chunks = []
    for image, stamp in zip(luma[1:], times[1:]):
        events = dvs.generate_events(image, stamp)
        if len(events):
            chunks.append(np.asarray(events, dtype=np.float64))
    elapsed = time.perf_counter() - started
    result = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)
    if len(result):
        result = result[np.argsort(result[:, 0], kind="stable")]
    return result, elapsed


def event_delta(reference: np.ndarray, candidate: np.ndarray) -> dict:
    def keys(events):
        return [(int(round(float(e[0]) * 1e9)), int(e[1]), int(e[2]), int(np.sign(e[3]))) for e in events]
    from collections import Counter
    a, b = Counter(keys(reference)), Counter(keys(candidate))
    matched = sum(min(n, b.get(k, 0)) for k, n in a.items())
    def grouped_times(events):
        result = {}
        for event in events:
            key = (int(event[1]), int(event[2]), int(np.sign(event[3])))
            result.setdefault(key, []).append(float(event[0]))
        for values in result.values():
            values.sort()
        return result
    ref_groups, cand_groups = grouped_times(reference), grouped_times(candidate)
    time_errors = []
    for key in set(ref_groups) & set(cand_groups):
        n = min(len(ref_groups[key]), len(cand_groups[key]))
        time_errors.extend(abs(x - y) for x, y in zip(ref_groups[key][:n], cand_groups[key][:n]))
    return {
        "reference_count": len(reference), "candidate_count": len(candidate),
        "reference_on_off": [int(np.count_nonzero(reference[:, 3] > 0)), int(np.count_nonzero(reference[:, 3] < 0))],
        "candidate_on_off": [int(np.count_nonzero(candidate[:, 3] > 0)), int(np.count_nonzero(candidate[:, 3] < 0))],
        "pixel_polarity_l1": int(sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))),
        "exact_time_pixel_polarity_matches_1ns": int(matched),
        "matched_pixel_polarity_event_time_abs_error_median_s": float(np.median(time_errors)) if time_errors else None,
        "matched_pixel_polarity_event_time_abs_error_p95_s": float(np.percentile(time_errors, 95)) if time_errors else None,
        "matched_pixel_polarity_event_time_abs_error_max_s": float(np.max(time_errors)) if time_errors else None,
        "timestamp_quantization_s": 1e-9,
    }


def main() -> int:
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true", help="render and save one exact-state HDR frame, then exit")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    scene_cfg = json.loads(args.scene_config.read_text(encoding="utf-8"))
    runtime = hdr_runtime_preflight()
    scene_config_hash = sha256(args.scene_config.resolve())
    expected_scene_hash = cfg.get("expected_scene_config_sha256")
    if expected_scene_hash is not None and scene_config_hash != expected_scene_hash:
        raise RuntimeError(f"scene config hash mismatch: expected={expected_scene_hash}, actual={scene_config_hash}")
    t0, t1 = map(float, cfg["interval_s"])
    dt = float(cfg["observed_frame_spacing_s"])
    episodes = [int(x) for x in cfg["episodes"]]
    if args.smoke_only:
        episodes = episodes[:1]
    cache_root = args.cache_root.resolve()

    input_load_started = time.perf_counter()
    loaded = {}
    source_hashes = {}
    for episode in episodes:
        ep_dir = cache_root / f"episode_{episode:03d}"
        trace_path = ep_dir / "trace_window.jsonl"
        evidence_path = ep_dir / "rendered_frame_evidence.h5"
        source_path = ep_dir / "same_state_source_cache.h5"
        rows = read_rows(trace_path)
        row_by_time = {round(float(row["time_s"]), 9): row for row in rows}
        with h5py.File(evidence_path, "r") as cache:
            dense_times = np.asarray(cache["dense_times_s"][:], dtype=np.float64)
            dense_hdr = np.asarray(cache["dense_direct_hdr_rgb"][:], dtype=np.float32)
            dense_seg = np.asarray(cache["dense_direct_segmentation"][:], dtype=np.int32)
        with h5py.File(source_path, "r") as cache:
            source_times = np.asarray(cache["timestamps_s"][:], dtype=np.float64)
            source_luma = np.asarray(cache["radiance_luma"][:], dtype=np.float32)
        start_matches = np.flatnonzero(np.isclose(source_times, t0, atol=1e-9, rtol=0))
        if len(start_matches) != 1:
            raise RuntimeError(f"episode {episode}: expected one archived caller frame at {t0}, found {len(start_matches)}")
        start_luma = source_luma[int(start_matches[0])]
        select = (dense_times > t0 + 1e-9) & (dense_times <= t1 + 1e-9)
        times = np.round(dense_times[select], 9)
        hdr = dense_hdr[select]
        seg = dense_seg[select]
        expected = np.round(np.arange(t0 + dt, t1 + dt * 0.5, dt), 9)
        if not np.array_equal(times, expected):
            raise RuntimeError(f"episode {episode}: archived samples do not form the configured exact schedule: {times.tolist()}")
        missing = [float(t) for t in [t0, *times] if float(t) not in row_by_time]
        if missing:
            raise RuntimeError(f"episode {episode}: no exact saved physical state at requested time(s): {missing}")
        loaded[episode] = {
            "rows": [row_by_time[float(t)] for t in times], "times": times,
            "hdr": hdr, "seg": seg, "start_luma": start_luma,
            "trace_path": trace_path, "evidence_path": evidence_path, "source_path": source_path,
        }
        source_hashes[f"episode_{episode:03d}/trace_window.jsonl"] = sha256(trace_path)
        source_hashes[f"episode_{episode:03d}/rendered_frame_evidence.h5"] = sha256(evidence_path)
        source_hashes[f"episode_{episode:03d}/same_state_source_cache.h5"] = sha256(source_path)

    for relpath, expected_hash in cfg.get("expected_input_sha256", {}).items():
        actual_hash = source_hashes.get(relpath)
        if actual_hash != expected_hash:
            raise RuntimeError(f"input hash mismatch for {relpath}: expected={expected_hash}, actual={actual_hash}")

    input_load_wall = time.perf_counter() - input_load_started
    args.output.mkdir(parents=True, exist_ok=False)

    import genesis as gs
    startup_started = time.perf_counter()
    gs.init(backend=gs.cpu, logging_level="warning")
    genesis_init_wall = time.perf_counter() - startup_started
    env = build_robot_scene(scene_cfg, dt_s=float(scene_cfg["physics"]["dt_s"]), segmentation_level="link")
    scene_build_wall = time.perf_counter() - startup_started - genesis_init_wall
    renderer_backend = (
        "rasterizer" if getattr(env.camera, "_rasterizer", None) is not None else
        "batch" if getattr(env.camera, "_batch_renderer", None) is not None else
        "raytracer" if getattr(env.camera, "_raytracer", None) is not None else "unknown"
    )
    if renderer_backend != "rasterizer":
        raise RuntimeError(f"HDR run requires the rasterizer backend; selected backend={renderer_backend}")
    first = next(row for row in read_rows(loaded[episodes[0]]["trace_path"])
                 if round(float(row["time_s"]), 9) == round(t0, 9))
    exact_state(first, env)
    warmup_started = time.perf_counter()
    warmup_outputs = env.camera.render(rgb=True, depth=True, segmentation=True, radiance=True, force_render=True)
    warmup_materialize_wall = 0.0
    warmup_materialized = {}
    for name, value in zip(("hdr", "depth", "segmentation"), warmup_outputs[:3]):
        if value is not None:
            array, copy_wall = materialize_image(value)
            warmup_materialize_wall += copy_wall
            warmup_materialized[name] = array
    warmup_wall = time.perf_counter() - warmup_started

    if args.smoke_only:
        hdr = np.asarray(warmup_materialized["hdr"], dtype=np.float32)
        finite_fraction = float(np.isfinite(hdr).mean())
        if hdr.ndim != 3 or hdr.shape[-1] not in (3, 4) or not np.isfinite(hdr).any():
            raise RuntimeError(f"HDR smoke returned invalid image shape/data: shape={hdr.shape}, finite={finite_fraction}")
        smoke = {
            "run_id": cfg["run_id"], "smoke_only": True, "source_commit": os.environ.get("GENESIS_EVENT_PLUGIN_SOURCE_COMMIT", cfg.get("source_commit", "unknown")),
            "state_timestamp_s": t0, "physics_steps": 0, "hdr_shape": list(hdr.shape),
            "hdr_dtype": str(hdr.dtype), "hdr_finite_fraction": finite_fraction,
            "hdr_min": float(np.nanmin(hdr)), "hdr_max": float(np.nanmax(hdr)),
            "camera_render_call_wall_s_including_gl_readback": warmup_wall - warmup_materialize_wall,
            "returned_array_materialization_wall_s": warmup_materialize_wall,
            "genesis_initialization_wall_s": genesis_init_wall, "scene_build_wall_s": scene_build_wall,
            "runtime": runtime, "renderer_backend": renderer_backend,
            "host": __import__("socket").gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "config_sha256": sha256(args.experiment_config.resolve()),
            "scene_config_sha256": scene_config_hash,
            "script_sha256": sha256(Path(__file__).resolve()), "input_hashes": source_hashes,
            "total_process_wall_s_before_artifact_hashing": time.perf_counter() - run_started,
        }
        np.savez_compressed(args.output / "single_frame_hdr_smoke.npz", timestamp_s=np.asarray([t0]), hdr_rgb=hdr)
        (args.output / "smoke_summary.json").write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil = __import__("shutil")
        shutil.copy2(args.experiment_config, args.output / "config.json")
        (args.output / "run_manifest.json").write_text(json.dumps({
            "run_id": cfg["run_id"], "host": smoke["host"], "source_commit": smoke["source_commit"],
            "config_sha256": smoke["config_sha256"], "scene_config_sha256": smoke["scene_config_sha256"],
            "script_sha256": smoke["script_sha256"], "input_hashes": source_hashes,
            "renderer_runtime": runtime, "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES", "not-set"),
            "egl_device_id": os.environ.get("EGL_DEVICE_ID", "not-set"), "smoke_only": True,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.output / "SHA256SUMS").write_text(
            "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(args.output.iterdir()) if path.is_file() and path.name != "SHA256SUMS"),
            encoding="utf-8",
        )
        print(json.dumps({"run_id": cfg["run_id"], "smoke_only": True, "output": str(args.output), "hdr_shape": list(hdr.shape), "finite_fraction": finite_fraction}, sort_keys=True), flush=True)
        return 0

    source_commit = os.environ.get("GENESIS_EVENT_PLUGIN_SOURCE_COMMIT", cfg.get("source_commit", "unknown"))
    summary = {
        "run_id": cfg["run_id"], "source_commit": source_commit,
        "execution": {"platform": platform.platform(), "python": platform.python_version(), "runtime": runtime,
                      "renderer_backend": renderer_backend},
        "config_sha256": sha256(args.experiment_config.resolve()),
        "scene_config_sha256": scene_config_hash,
        "script_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": source_hashes,
        "interval_s": [t0, t1], "observed_frame_spacing_s": dt,
        "render_policy": "offline replay only; no physics steps; every row must match an exact archived trace timestamp; no interpolated state rows",
        "input_load_wall_s": input_load_wall,
        "genesis_initialization_wall_s": genesis_init_wall,
        "scene_build_wall_s": scene_build_wall,
        "warmup_render_wall_s_excluded_from_steady_state": warmup_wall,
        "warmup_array_materialization_wall_s": warmup_materialize_wall,
        "episodes": {},
    }
    saved_hdr, saved_times, saved_episode_ids = [], [], []
    saved_reference_hdr, saved_reference_times, saved_reference_episode_ids = [], [], []
    raw_events = {}
    for episode in episodes:
        data = loaded[episode]
        rerendered_hdr, rerendered_luma, rerendered_seg = [], [], []
        frame_records = []
        total_apply_wall, total_render_wall, total_materialize_wall = 0.0, 0.0, 0.0
        # The caller supplies the right endpoint; only nine interior direct
        # observations are new strict samples for this 10 ms bracket.
        interior = data["times"] < t1 - 1e-9
        interior_rows = [row for row, keep in zip(data["rows"], interior) if keep]
        if len(interior_rows) != int(np.count_nonzero(interior)):
            raise RuntimeError(f"episode {episode}: trace row count does not match archived frame schedule")
        for i, (stamp, row, ref_hdr, ref_seg) in enumerate(zip(
            data["times"][interior], interior_rows, data["hdr"][interior], data["seg"][interior],
        )):
            apply_started = time.perf_counter()
            exact_state(row, env)
            apply_wall = time.perf_counter() - apply_started
            render_started = time.perf_counter()
            hdr_device, depth_device, seg_device, _ = env.camera.render(
                rgb=True, depth=True, segmentation=True, radiance=True, force_render=True,
            )
            render_wall = time.perf_counter() - render_started
            materialize_started = time.perf_counter()
            hdr, hdr_copy_wall = materialize_image(hdr_device)
            depth, depth_copy_wall = materialize_image(depth_device)
            seg, seg_copy_wall = materialize_image(seg_device)
            materialize_wall = time.perf_counter() - materialize_started
            hdr = np.asarray(hdr, dtype=np.float32)
            seg = np.asarray(seg, dtype=np.int32)
            luma = linear_luminance(hdr)
            ref_luma = linear_luminance(ref_hdr)
            finite = np.isfinite(luma) & np.isfinite(ref_luma)
            delta = np.abs(luma[finite] - ref_luma[finite])
            record = {
                "episode": episode, "timestamp_s": float(stamp), "trace_state_exact": True,
                "state_apply_wall_s": apply_wall,
                "camera_render_call_wall_s_including_gl_readback": render_wall,
                "returned_array_materialization_wall_s": materialize_wall,
                "hdr_array_materialization_wall_s": hdr_copy_wall,
                "depth_array_materialization_wall_s": depth_copy_wall,
                "segmentation_array_materialization_wall_s": seg_copy_wall,
                "radiance_luma_mae": float(delta.mean()) if delta.size else None,
                "radiance_luma_p99_abs": float(np.percentile(delta, 99)) if delta.size else None,
                "radiance_luma_max_abs": float(delta.max()) if delta.size else None,
                "segmentation_mismatch_pixels": int(np.count_nonzero(seg != ref_seg)),
                "depth_finite_fraction": float(np.isfinite(depth).mean()),
            }
            frame_records.append(record)
            total_apply_wall += apply_wall
            total_render_wall += render_wall
            total_materialize_wall += materialize_wall
            rerendered_hdr.append(hdr)
            rerendered_luma.append(luma)
            rerendered_seg.append(seg)
            saved_hdr.append(hdr)
            saved_times.append(float(stamp))
            saved_episode_ids.append(episode)

        for stamp, frame in zip(data["times"], data["hdr"]):
            saved_reference_hdr.append(np.asarray(frame, dtype=np.float32))
            saved_reference_times.append(float(stamp))
            saved_reference_episode_ids.append(episode)

        ref_luma = [data["start_luma"]] + [linear_luminance(frame) for frame in data["hdr"]]
        sequence_times = [t0] + [float(t) for t in data["times"]]
        rerun_luma_sequence = [data["start_luma"]] + rerendered_luma + [linear_luminance(data["hdr"][-1])]
        ref_events, ref_dvs_wall = events_from_sequence(ref_luma, sequence_times)
        rerun_events, rerun_dvs_wall = events_from_sequence(rerun_luma_sequence, sequence_times)
        raw_events[f"episode_{episode:03d}_reference"] = ref_events
        raw_events[f"episode_{episode:03d}_rerender"] = rerun_events
        interior_count = len(frame_records)
        render_interior = float(sum(row["camera_render_call_wall_s_including_gl_readback"] for row in frame_records))
        apply_interior = float(sum(row["state_apply_wall_s"] for row in frame_records))
        summary["episodes"][str(episode)] = {
            "frame_count_including_caller_endpoints": len(frame_records) + 2,
            "strict_extra_sample_count_between_caller_endpoints": interior_count,
            "timestamps_s": sequence_times,
            "mean_camera_render_call_wall_s_per_interior_sample_including_gl_readback": render_interior / max(interior_count, 1),
            "sum_camera_render_call_wall_s_interior_samples_including_gl_readback": render_interior,
            "sum_returned_array_materialization_wall_s_interior_samples": total_materialize_wall,
            "sum_state_apply_wall_s_interior_samples": apply_interior,
            "sum_render_materialize_and_state_apply_wall_s_interior_samples": render_interior + total_materialize_wall + apply_interior,
            "sum_render_wall_s_new_samples": total_render_wall,
            "sum_state_apply_wall_s_new_samples": total_apply_wall,
            "cached_sequence_dvs_wall_s": ref_dvs_wall,
            "rerender_sequence_dvs_wall_s": rerun_dvs_wall,
            "event_comparison": event_delta(ref_events, rerun_events),
            "frame_records": frame_records,
            "availability_contract": {
                "sample_state_available_at_sim_time_s": "each rendered timestamp; real wall delay reported separately above",
                "event_sequence_available_no_earlier_than_sim_time_s": t1,
                "online_closed_loop_validated": False,
                "offline_replay_uses_archived_future_rows_sequentially": True,
            },
        }

    summary["aggregate"] = {
        "strict_extra_sample_count": sum(x["strict_extra_sample_count_between_caller_endpoints"] for x in summary["episodes"].values()),
        "sum_camera_render_call_wall_s_including_gl_readback": sum(x["sum_camera_render_call_wall_s_interior_samples_including_gl_readback"] for x in summary["episodes"].values()),
        "sum_returned_array_materialization_wall_s": sum(x["sum_returned_array_materialization_wall_s_interior_samples"] for x in summary["episodes"].values()),
        "sum_state_restore_wall_s_for_strict_extra_samples": sum(x["sum_state_apply_wall_s_interior_samples"] for x in summary["episodes"].values()),
        "sum_reference_and_rerender_dvs_wall_s": sum(x["cached_sequence_dvs_wall_s"] + x["rerender_sequence_dvs_wall_s"] for x in summary["episodes"].values()),
        "interpretation": "measured camera render call (including GL readback), returned-array materialization, exact-state restore, and DVS for 9 existing 1ms intermediate samples per 10ms interval; offline replay, not a live closed-loop latency test",
    }
    summary["output_available_at_semantics"] = "simulated right bracket endpoint; measured wall-clock sample render latency is separate and must be added for online claims"
    summary["total_process_wall_s_before_artifact_hashing"] = time.perf_counter() - run_started
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(next(iter(summary["episodes"].values()))["frame_records"][0]))
        writer.writeheader()
        for episode in episodes:
            writer.writerows(summary["episodes"][str(episode)]["frame_records"])
    np.savez_compressed(
        args.output / "rerendered_hdr_frames.npz",
        timestamps_s=np.asarray(saved_times, dtype=np.float64),
        hdr_rgb=np.stack(saved_hdr).astype(np.float32),
        episode_ids=np.asarray(saved_episode_ids, dtype=np.int32),
    )
    np.savez_compressed(
        args.output / "reference_hdr_frames.npz",
        timestamps_s=np.asarray(saved_reference_times, dtype=np.float64),
        hdr_rgb=np.stack(saved_reference_hdr).astype(np.float32),
        episode_ids=np.asarray(saved_reference_episode_ids, dtype=np.int32),
    )
    np.savez_compressed(args.output / "raw_events.npz", **raw_events)
    metadata = {
        "run_id": cfg["run_id"], "host": __import__("socket").gethostname(),
        "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES", "not-set"),
        "egl_device_id": os.environ.get("EGL_DEVICE_ID", "not-set"),
        "source_commit": source_commit,
        "experiment_config_sha256": summary["config_sha256"],
        "source_hashes": source_hashes,
        "finished_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "runtime": runtime,
        "renderer_backend": renderer_backend,
        "total_process_wall_s_before_artifact_hashing": summary["total_process_wall_s_before_artifact_hashing"],
    }
    (args.output / "run_manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    import shutil
    shutil.copy2(args.experiment_config, args.output / "config.json")
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(args.output.iterdir()) if path.is_file() and path.name != "SHA256SUMS"),
        encoding="utf-8",
    )
    print(json.dumps({"run_id": cfg["run_id"], "aggregate": summary["aggregate"], "output": str(args.output)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
