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
import json
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    scene_cfg = json.loads(args.scene_config.read_text(encoding="utf-8"))
    t0, t1 = map(float, cfg["interval_s"])
    dt = float(cfg["observed_frame_spacing_s"])
    episodes = [int(x) for x in cfg["episodes"]]
    cache_root = args.cache_root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)

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

    import genesis as gs
    startup_started = time.perf_counter()
    gs.init(backend=gs.cpu, logging_level="warning")
    env = build_robot_scene(scene_cfg, dt_s=float(scene_cfg["physics"]["dt_s"]), segmentation_level="link")
    first = next(row for row in read_rows(loaded[episodes[0]]["trace_path"])
                 if round(float(row["time_s"]), 9) == round(t0, 9))
    exact_state(first, env)
    warmup_started = time.perf_counter()
    env.camera.render(rgb=True, depth=True, segmentation=True, radiance=True, force_render=True)
    warmup_wall = time.perf_counter() - warmup_started
    scene_setup_wall = time.perf_counter() - startup_started

    source_commit = __import__("os").environ.get("GENESIS_EVENT_PLUGIN_SOURCE_COMMIT", cfg.get("source_commit", "unknown"))
    summary = {
        "run_id": cfg["run_id"], "source_commit": source_commit,
        "execution": {"platform": __import__("platform").platform(), "python": __import__("platform").python_version()},
        "config_sha256": sha256(args.experiment_config.resolve()),
        "scene_config_sha256": sha256(args.scene_config.resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": source_hashes,
        "interval_s": [t0, t1], "observed_frame_spacing_s": dt,
        "render_policy": "offline replay only; no physics steps; every row must match an exact archived trace timestamp; no interpolated state rows",
        "scene_setup_wall_s_including_initialization": scene_setup_wall,
        "warmup_render_wall_s_excluded_from_steady_state": warmup_wall,
        "episodes": {},
    }
    saved_hdr, saved_times = [], []
    all_frames = []
    for episode in episodes:
        data = loaded[episode]
        rerendered_hdr, rerendered_luma, rerendered_seg = [], [], []
        frame_records = []
        total_apply_wall, total_render_wall = 0.0, 0.0
        # The caller supplies the right endpoint; only nine interior direct
        # observations are new strict samples for this 10 ms bracket.
        interior = data["times"] < t1 - 1e-9
        for i, (stamp, row, ref_hdr, ref_seg) in enumerate(zip(
            data["times"][interior], data["rows"][interior], data["hdr"][interior], data["seg"][interior],
        )):
            apply_started = time.perf_counter()
            exact_state(row, env)
            apply_wall = time.perf_counter() - apply_started
            render_started = time.perf_counter()
            hdr, depth, seg, _ = env.camera.render(
                rgb=True, depth=True, segmentation=True, radiance=True, force_render=True,
            )
            render_wall = time.perf_counter() - render_started
            hdr = np.asarray(hdr, dtype=np.float32).copy()
            seg = np.asarray(seg, dtype=np.int32).copy()
            luma = linear_luminance(hdr)
            ref_luma = linear_luminance(ref_hdr)
            finite = np.isfinite(luma) & np.isfinite(ref_luma)
            delta = np.abs(luma[finite] - ref_luma[finite])
            record = {
                "episode": episode, "timestamp_s": float(stamp), "trace_state_exact": True,
                "state_apply_wall_s": apply_wall, "render_and_readback_wall_s": render_wall,
                "radiance_luma_mae": float(delta.mean()) if delta.size else None,
                "radiance_luma_p99_abs": float(np.percentile(delta, 99)) if delta.size else None,
                "radiance_luma_max_abs": float(delta.max()) if delta.size else None,
                "segmentation_mismatch_pixels": int(np.count_nonzero(seg != ref_seg)),
                "depth_finite_fraction": float(np.isfinite(depth).mean()),
            }
            frame_records.append(record)
            total_apply_wall += apply_wall
            total_render_wall += render_wall
            rerendered_hdr.append(hdr)
            rerendered_luma.append(luma)
            rerendered_seg.append(seg)
            saved_hdr.append(hdr)
            saved_times.append(float(stamp))
            all_frames.append((episode, float(stamp), luma))

        ref_luma = [data["start_luma"]] + [linear_luminance(frame) for frame in data["hdr"]]
        sequence_times = [t0] + [float(t) for t in data["times"]]
        rerun_luma_sequence = [data["start_luma"]] + rerendered_luma + [linear_luminance(data["hdr"][-1])]
        ref_events, ref_dvs_wall = events_from_sequence(ref_luma, sequence_times)
        rerun_events, rerun_dvs_wall = events_from_sequence(rerun_luma_sequence, sequence_times)
        interior_count = len(frame_records)
        render_interior = float(sum(row["render_and_readback_wall_s"] for row in frame_records))
        apply_interior = float(sum(row["state_apply_wall_s"] for row in frame_records))
        summary["episodes"][str(episode)] = {
            "frame_count_including_caller_endpoints": len(frame_records) + 2,
            "strict_extra_sample_count_between_caller_endpoints": interior_count,
            "timestamps_s": sequence_times,
            "mean_render_and_readback_wall_s_per_interior_sample": render_interior / max(interior_count, 1),
            "sum_render_and_readback_wall_s_interior_samples": render_interior,
            "sum_state_apply_wall_s_interior_samples": apply_interior,
            "sum_render_and_state_apply_wall_s_interior_samples": render_interior + apply_interior,
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
        "sum_render_wall_s_for_strict_extra_samples": sum(x["sum_render_and_readback_wall_s_interior_samples"] for x in summary["episodes"].values()),
        "sum_state_restore_wall_s_for_strict_extra_samples": sum(x["sum_state_apply_wall_s_interior_samples"] for x in summary["episodes"].values()),
        "interpretation": "measured renderer/readback and exact-state restore cost for 9 existing 1ms intermediate samples per 10ms interval; not a live closed-loop latency test",
    }
    summary["output_available_at_semantics"] = "simulated right bracket endpoint; measured wall-clock sample render latency is separate and must be added for online claims"
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
        episode_ids=np.asarray([e for e, _, _ in all_frames], dtype=np.int32),
        rerendered_luma=np.stack([x[2] for x in all_frames]).astype(np.float32),
    )
    metadata = {
        "run_id": cfg["run_id"], "host": __import__("socket").gethostname(),
        "gpu_uuid": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "not-set"),
        "egl_device_id": __import__("os").environ.get("EGL_DEVICE_ID", "not-set"),
        "source_commit": source_commit,
        "experiment_config_sha256": summary["config_sha256"],
        "source_hashes": source_hashes,
        "finished_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
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
