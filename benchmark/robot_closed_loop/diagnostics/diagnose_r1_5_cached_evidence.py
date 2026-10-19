#!/usr/bin/env python3
"""CPU-only R1.5 replay using already saved exact-state HDR evidence.

This runner does not initialize Genesis or render new images. It reconstructs
candidate images from saved 100 Hz HDR/depth/segmentation/pose endpoints at
the actual timestamps in the immutable physics trace, then uses the ten
previously saved 1 kHz direct HDR renders to score image errors and fill only
those candidate pixels that explicitly request a render at a matching saved
timestamp. All other requested pixels are held from the nearer endpoint and
counted as unresolved; results using that path are diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.diagnose_r1_native_hdr_replay import (  # noqa: E402
    infer_event_state,
    interpolate_trace_label,
    load_event_models,
    project_world,
    read_trace,
    verify_source_hashes,
)
from benchmark.robot_closed_loop.diagnostics.diagnose_r1_ours_localization_cadence import (  # noqa: E402
    generate_sparse_plugin,
    response_config,
)
from benchmark.robot_closed_loop.diagnostics.r1_5_consistency_candidates import (  # noqa: E402
    project_endpoint_candidate,
    reconstruct_pair_candidate,
)
from benchmark.robot_closed_loop.event_state_model import five_bin_tensor  # noqa: E402
from genesis_event_plugin.dvs_emulator import DvsEmulator  # noqa: E402
from genesis_event_plugin.radiometry import linear_luminance  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_for_time(time_s: float, source_times: np.ndarray) -> tuple[int, float]:
    exact = np.flatnonzero(np.isclose(source_times, time_s, atol=1e-9, rtol=0.0))
    if exact.size:
        idx = int(exact[0])
        return (max(0, idx - 1), 0.0 if idx == 0 else 1.0)
    right = int(np.searchsorted(source_times, time_s, side="right"))
    left = right - 1
    if left < 0 or right >= len(source_times):
        raise ValueError(f"time {time_s:.9f} is outside cached endpoint support")
    alpha = (float(time_s) - float(source_times[left])) / (float(source_times[right]) - float(source_times[left]))
    return left, float(alpha)


def _events_from_frames(frames: list[np.ndarray], times: list[float]) -> tuple[np.ndarray, float]:
    h, w = frames[0].shape
    emulator = DvsEmulator(res=(h, w), config=response_config(), device="cpu")
    started = time.perf_counter()
    emulator.initialize(np.asarray(frames[0], dtype=np.float32), t=float(times[0]))
    chunks = []
    for frame, timestamp in zip(frames[1:], times[1:]):
        part = emulator.generate_events(np.asarray(frame, dtype=np.float32), float(timestamp))
        if len(part):
            chunks.append(np.asarray(part, dtype=np.float64))
    elapsed = time.perf_counter() - started
    events = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)
    if len(events):
        events = events[np.argsort(events[:, 0], kind="stable")]
    return events, elapsed


def _image_error(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    ref, est = np.asarray(reference, dtype=np.float32), np.asarray(estimate, dtype=np.float32)
    valid = np.isfinite(ref) & np.isfinite(est) & (ref >= 0.0) & (est >= 0.0) & np.asarray(mask, dtype=bool)
    if not valid.any():
        return {"pixels": 0, "log_abs_mean": None, "log_abs_p95": None, "log_abs_max": None,
                "pixels_over_0_2": 0}
    diff = np.abs(np.log(np.maximum(ref[valid], 1e-6)) - np.log(np.maximum(est[valid], 1e-6)))
    return {"pixels": int(diff.size), "log_abs_mean": float(diff.mean()),
            "log_abs_p95": float(np.percentile(diff, 95)), "log_abs_max": float(diff.max()),
            "pixels_over_0_2": int(np.count_nonzero(diff > 0.2))}


def _write_hashes(root: Path) -> None:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        rows.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True,
                        help="native_hdr_cutoff059 directory containing both episode caches")
    parser.add_argument("--ours-checkpoint", type=Path, required=True)
    parser.add_argument("--v2e-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    scene = json.loads(args.scene_config.read_text(encoding="utf-8"))
    ref_root, evidence_root, output = args.reference_root.resolve(), args.evidence_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    width, height = (int(v) for v in scene["camera"]["resolution_wh"])
    fov = float(scene["camera"]["vertical_fov_deg"])
    focal = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    intrinsics = np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=np.float64)
    K = np.asarray([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    trace_start, trace_end = (float(x) for x in config["trace_time_s"])
    query_start, query_end = (float(x) for x in config["query_time_s"])
    query_step = float(config["query_step_s"])
    queries = [round(float(x), 9) for x in np.arange(query_start, query_end + query_step / 2.0, query_step)]
    if not queries or abs(queries[-1] - query_end) > 1e-8:
        raise ValueError("query interval must divide evenly by query_step_s")

    output.mkdir(parents=True)
    all_queries: list[dict[str, Any]] = []
    episode_summaries: dict[str, Any] = {}
    episode_ids = {int(v) for v in config["episodes"].values()}
    model_records: dict[int, tuple[dict, dict[str, str]]] = {}
    global_start = time.perf_counter()

    for ep_id in sorted(episode_ids):
        label = next(k for k, v in config["episodes"].items() if int(v) == ep_id)
        ev_dir = evidence_root / f"episode_{ep_id:03d}"
        ref_dir = ref_root / "episodes" / f"episode_{ep_id:03d}"
        cache_path = ev_dir / "same_state_source_cache.h5"
        dense_path = ev_dir / "rendered_frame_evidence.h5"
        trace_path = ref_dir / "state_control_trace.jsonl"
        event_path = ref_dir / "direct_hdr_events_1khz_2khz.npz"
        episode_path = ref_dir / "episode.csv"
        verified = verify_source_hashes(ref_root, [trace_path, event_path, episode_path])
        trace = read_trace(trace_path)
        trace_rows = [r for r in trace if trace_start - 1e-9 <= float(r["time_s"]) <= trace_end + 1e-9]
        times = np.asarray([round(float(r["time_s"]), 9) for r in trace_rows], dtype=np.float64)
        if len(times) < 3 or abs(times[0] - trace_start) > 1e-8 or abs(times[-1] - trace_end) > 1e-8:
            raise RuntimeError(f"episode_{ep_id:03d}: trace lacks configured endpoints")
        if np.any(np.diff(times) <= 0):
            raise RuntimeError(f"episode_{ep_id:03d}: nonmonotonic timestamps")

        with h5py.File(cache_path, "r") as cache:
            source_times = np.asarray(cache["timestamps_s"][:], dtype=np.float64)
            pose_ids = np.asarray(cache["segmentation_pose_ids"][:], dtype=np.int32)
            camera_pose = np.asarray(cache["camera_to_world_cv"][0], dtype=np.float64)
            if tuple(int(v) for v in cache.attrs["camera_resolution_wh"]) != (width, height):
                raise RuntimeError("source cache resolution mismatch")
            if not np.isclose(float(cache.attrs["vertical_fov_deg"]), fov, atol=1e-8, rtol=0):
                raise RuntimeError("source cache FOV mismatch")
            sources = []
            for i, t in enumerate(source_times):
                sources.append({
                    "time_s": round(float(t), 9),
                    "hdr_rgb": np.asarray(cache["radiance"][i], dtype=np.float32),
                    "radiance_luma": np.asarray(cache["radiance_luma"][i], dtype=np.float32),
                    "display_rgb": np.asarray(cache["display_rgb"][i], dtype=np.uint8),
                    "depth": np.asarray(cache["depth"][i], dtype=np.float32),
                    "segmentation": np.asarray(cache["segmentation"][i], dtype=np.int32),
                    "poses": {int(seg): np.asarray(cache["object_poses"][i, j], dtype=np.float64)
                              for j, seg in enumerate(pose_ids)},
                })
        if not np.allclose(np.asarray([s["time_s"] for s in sources]), source_times, atol=1e-9, rtol=0):
            raise RuntimeError("source timestamp contract mismatch")
        # Identify the target segmentation label from its saved world pose.
        pose_distance = {}
        for j, seg_id in enumerate(pose_ids):
            diffs = []
            for src in sources:
                row = min(trace, key=lambda r: abs(float(r["time_s"]) - src["time_s"]))
                gt = np.asarray(row["object"]["position_m"], dtype=np.float64)
                diffs.append(float(np.linalg.norm(src["poses"][int(seg_id)][:3, 3] - gt)))
            pose_distance[int(seg_id)] = float(np.median(diffs))
        target_seg_id = min(pose_distance, key=pose_distance.get)
        if pose_distance[target_seg_id] > 1e-3:
            raise RuntimeError(f"cannot identify target segmentation from cached pose; best residual {pose_distance[target_seg_id]:.6g} m")

        with h5py.File(dense_path, "r") as dense:
            dense_times = np.asarray(dense["dense_times_s"][:], dtype=np.float64)
            dense_lumas = np.asarray(dense["dense_direct_hdr_luminance"][:], dtype=np.float32)
            dense_segs = np.asarray(dense["dense_direct_segmentation"][:], dtype=np.int32)
            if dense_lumas.shape[1:] != (height, width):
                raise RuntimeError("dense HDR evidence resolution mismatch")
            dense_by_time = {round(float(t), 9): {"luma": dense_lumas[i], "seg": dense_segs[i]}
                             for i, t in enumerate(dense_times)}
        # Restore exact dense renders to the corresponding actual trace rows.
        trace_time_lookup = {round(float(r["time_s"]), 9): r for r in trace_rows}
        if any(t not in trace_time_lookup for t in dense_by_time):
            raise RuntimeError("a saved direct HDR frame does not match an exact trace timestamp")

        episode_out = output / f"episode_{ep_id:03d}"
        episode_out.mkdir()
        if ep_id not in model_records:
            model_records[ep_id] = load_event_models(
                {"ours_native_hdr": args.ours_checkpoint.resolve(), "v2e": args.v2e_checkpoint.resolve()},
                camera_pose, intrinsics, (width, height))
        models, checkpoint_hashes = model_records[ep_id]

        mode_map = {
            "frozen_reconstruction": "frozen",
            "endpoint_projection_identity": "endpoint_projection_identity",
            "endpoint_exact": "endpoint_exact",
            "log_radiance_cached_render_requests": "log_radiance_with_render_request",
        }
        frames_by_source: dict[str, list[np.ndarray]] = {k: [] for k in mode_map}
        unresolved_requests = {k: 0 for k in mode_map}
        cached_fill_pixels = {k: 0 for k in mode_map}
        total_request_pixels = {k: 0 for k in mode_map}
        image_rows: list[dict[str, Any]] = []
        endpoint_projection_rows: list[dict[str, Any]] = []
        reconstruction_started = time.perf_counter()
        for row in trace_rows:
            t = round(float(row["time_s"]), 9)
            exact_idxes = np.flatnonzero(np.isclose(source_times, t, atol=1e-9, rtol=0))
            if exact_idxes.size and int(exact_idxes[0]) == 0:
                for name in mode_map:
                    frames_by_source[name].append(sources[0]["radiance_luma"].copy())
                continue
            pair_idx, alpha = _pair_for_time(t, source_times)
            s0, s1 = sources[pair_idx], sources[pair_idx + 1]
            if abs(alpha - 1.0) <= 1e-9:
                for exact_mode, label_name in ((False, "frozen_projection"), (True, "identity_projection")):
                    projected = project_endpoint_candidate(
                        s1["depth"], s1["segmentation"], K, camera_pose, camera_pose, 1.0, 1,
                        s0["poses"], s1["poses"], exact_endpoints=exact_mode,
                    )
                    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
                    grid = np.stack((xx, yy), axis=0).astype(np.float32)
                    uv_error = np.abs(projected[0] - grid)[:, projected[2]]
                    endpoint_projection_rows.append({
                        "episode_id": ep_id, "time_s": t, "projection": label_name,
                        "valid_source_pixels": int(projected[2].sum()),
                        "max_uv_error_px": float(uv_error.max()) if uv_error.size else None,
                        "p99_uv_error_px": float(np.percentile(uv_error, 99)) if uv_error.size else None,
                    })
            for name, mode in mode_map.items():
                result = reconstruct_pair_candidate(
                    s0["radiance_luma"], s0["depth"], s0["segmentation"], camera_pose,
                    s1["radiance_luma"], s1["depth"], s1["segmentation"], camera_pose,
                    K, s0["poses"], s1["poses"], alpha, mode=mode,
                )
                image = np.asarray(result["frame"], dtype=np.float32).copy()
                request = np.asarray(result["request_render"], dtype=bool)
                total_request_pixels[name] += int(request.sum())
                direct = dense_by_time.get(t)
                if name == "log_radiance_cached_render_requests" and direct is not None and request.any():
                    image[request] = direct["luma"][request]
                    cached_fill_pixels[name] += int(request.sum())
                    request[:] = False
                remaining = request | ~np.isfinite(image)
                if remaining.any():
                    fallback = s0["radiance_luma"] if alpha < 0.5 else s1["radiance_luma"]
                    image[remaining] = fallback[remaining]
                    unresolved_requests[name] += int(remaining.sum())
                frames_by_source[name].append(image)
        reconstruction_wall = time.perf_counter() - reconstruction_started

        # Run the same ideal pixel response on actual logged timestamps only.
        event_streams: dict[str, np.ndarray] = {}
        event_wall: dict[str, float] = {}
        source_rows = [{k: v for k, v in frame.items()} for frame in sources]
        sparse_events, _, sparse_wall = generate_sparse_plugin(
            source_rows, pose_ids, camera_pose, width, height, fov)
        event_streams["original_sparse_100hz_plugin"] = sparse_events
        event_wall["original_sparse_100hz_plugin"] = sparse_wall
        for name, frames in frames_by_source.items():
            event_streams[name], event_wall[name] = _events_from_frames(frames, times.tolist())
        # The exact 1 kHz direct-render evidence is a short local interval; it
        # is useful for photometry comparison but is not padded to 25 ms.
        direct_times = [float(t) for t in dense_times]
        direct_events, direct_event_wall = _events_from_frames(
            [dense_by_time[round(t, 9)]["luma"] for t in direct_times], direct_times)
        event_streams["cached_direct_HDR_1kHz_short"] = direct_events
        event_wall["cached_direct_HDR_1kHz_short"] = direct_event_wall
        with np.load(event_path, allow_pickle=False) as archive:
            if "events_2000hz" not in archive:
                raise RuntimeError("direct reference archive lacks events_2000hz")
            archived_events = np.asarray(archive["events_2000hz"], dtype=np.float64)
        event_streams["archived_direct_HDR_2kHz"] = archived_events

        # Save every candidate raw event stream before model scoring.
        for name, stream in event_streams.items():
            np.savez_compressed(episode_out / f"{name}_events.npz", events=stream)

        # Compare reconstructed luminance only at the ten exact saved direct renders.
        mode_names = list(mode_map)
        row_pos = {round(float(t), 9): i for i, t in enumerate(times)}
        for t, direct in dense_by_time.items():
            idx = row_pos[t]
            gt = interpolate_trace_label(trace, t)
            uv = project_world(np.asarray(gt["position_m"], dtype=np.float64), camera_pose, intrinsics)
            x, y = int(np.clip(round(uv[0]), 0, width - 1)), int(np.clip(round(uv[1]), 0, height - 1))
            label_patch = direct["seg"][max(0, y - 4):y + 5, max(0, x - 4):x + 5]
            labels, counts = np.unique(label_patch[label_patch > 0], return_counts=True)
            target_mask = direct["seg"] == target_seg_id
            for name in mode_names:
                err = {
                    "episode_id": ep_id, "episode": label, "time_s": t, "source": name,
                    "target_seg_id": target_seg_id, "target_pose_residual_m": pose_distance[target_seg_id],
                    "target_pixels_direct": int(target_mask.sum()),
                    **{f"all_{k}": v for k, v in _image_error(direct["luma"], frames_by_source[name][idx], np.ones_like(target_mask)).items()},
                    **{f"target_{k}": v for k, v in _image_error(direct["luma"], frames_by_source[name][idx], target_mask).items()},
                }
                image_rows.append(err)

        # Sample all 0.5 ms decision phases. The short direct event source is
        # intentionally omitted from state inference because it cannot supply
        # a causal 25 ms window.
        query_rows: list[dict[str, Any]] = []
        evaluation_sources = [k for k in event_streams if k != "cached_direct_HDR_1kHz_short"]
        for q in queries:
            gt = interpolate_trace_label(trace, q)
            gt_pos = np.asarray(gt["position_m"], dtype=np.float64)
            gt_vel = np.asarray(gt["velocity_m_s"], dtype=np.float64)
            gt_uv = project_world(gt_pos, camera_pose, intrinsics)
            for source_name in evaluation_sources:
                stream = event_streams[source_name]
                voxel = five_bin_tensor(stream, q, width, height)
                for model_name, model in models.items():
                    estimate, _ = infer_event_state(stream, model, q, width, height, camera_pose,
                                                    intrinsics, fixed_height_m=float(scene["target"]["size_m"][2]) / 2.0)
                    item = {
                        "episode_id": ep_id, "episode": label, "query_s": q,
                        "model": model_name, "source": source_name,
                        "event_count_25ms": estimate["event_count_25ms"],
                        "nonempty_5ms_bins": estimate["nonempty_5ms_bins"],
                        "state_available": estimate["state_available"],
                        "position_error_xy_mm": None, "velocity_error_xy_m_s": None,
                        "heatmap_uv_error_px": None,
                        "unresolved_render_request_pixels_by_trace": unresolved_requests.get(source_name),
                        "input_tensor_sha256": estimate["input_tensor_sha256"],
                    }
                    if estimate.get("state_available"):
                        pos = np.asarray(estimate["position_xyz_m"], dtype=np.float64)
                        vel = np.asarray(estimate["velocity_xy_m_s"], dtype=np.float64)
                        pred_uv = np.asarray(estimate["heatmap_uv_px"], dtype=np.float64)
                        item.update({
                            "position_error_xy_mm": float(np.linalg.norm(pos[:2] - gt_pos[:2]) * 1000.0),
                            "velocity_error_xy_m_s": float(np.linalg.norm(vel - gt_vel[:2])),
                            "heatmap_uv_error_px": float(np.linalg.norm(pred_uv - gt_uv)),
                        })
                    query_rows.append(item)
                    all_queries.append(item)

        # Compare the local 1 kHz direct-render event path to archived 2 kHz
        # events only over their exact common time interval.
        lo, hi = min(direct_times), max(direct_times)
        local = direct_events[(direct_events[:, 0] >= lo) & (direct_events[:, 0] <= hi)]
        archived_local = archived_events[(archived_events[:, 0] >= lo) & (archived_events[:, 0] <= hi)]
        local_counts = []
        for left, right in zip(direct_times[:-1], direct_times[1:]):
            local_counts.append({
                "interval_start_s": left, "interval_end_s": right,
                "cached_direct_1khz_count": int(np.count_nonzero((local[:, 0] >= left) & (local[:, 0] < right))),
                "archived_direct_2khz_count": int(np.count_nonzero((archived_local[:, 0] >= left) & (archived_local[:, 0] < right))),
            })

        with (episode_out / "image_errors_at_cached_direct_rows.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(image_rows[0]))
            writer.writeheader(); writer.writerows(image_rows)
        with (episode_out / "queries.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(query_rows[0]))
            writer.writeheader(); writer.writerows(query_rows)
        with (episode_out / "direct_event_count_by_1ms_interval.json").open("w", encoding="utf-8") as stream:
            json.dump(local_counts, stream, indent=2)
        with (episode_out / "endpoint_projection_errors.json").open("w", encoding="utf-8") as stream:
            json.dump(endpoint_projection_rows, stream, indent=2)

        ep_summary = {
            "episode": label, "episode_id": ep_id,
            "trace_row_count": len(trace_rows),
            "trace_interval_s": {"min": float(np.diff(times).min()), "median": float(np.median(np.diff(times))),
                                  "max": float(np.diff(times).max()), "effective_mean_rate_hz": float(1.0 / np.diff(times).mean())},
            "saved_direct_render_times_s": dense_times.tolist(),
            "source_cache_sha256": _sha256(cache_path), "dense_render_evidence_sha256": _sha256(dense_path),
            "source_trace_sha256": verified[f"episodes/episode_{ep_id:03d}/state_control_trace.jsonl"],
            "direct_event_archive_sha256": verified[f"episodes/episode_{ep_id:03d}/direct_hdr_events_1khz_2khz.npz"],
            "target_segmentation_id": target_seg_id, "target_pose_median_residual_m": pose_distance[target_seg_id],
            "candidate_reconstruction_wall_s": reconstruction_wall,
            "candidate_event_emulation_wall_s": event_wall,
            "candidate_event_count_trace_interval": {k: int(len(v)) for k, v in event_streams.items()},
            "render_requests": {
                name: {"requested_pixels_sum": total_request_pixels[name],
                       "pixels_filled_from_saved_exact_render": cached_fill_pixels[name],
                       "unresolved_pixels_filled_from_nearest_endpoint": unresolved_requests[name],
                       "render_request_fulfillment_fraction": (cached_fill_pixels[name] / total_request_pixels[name]
                           if total_request_pixels[name] else 1.0)}
                for name in mode_map
            },
            "direct_event_common_interval_s": [lo, hi],
            "direct_event_common_interval_count_1khz": int(len(local)),
            "direct_event_common_interval_count_archived_2khz": int(len(archived_local)),
            "endpoint_projection_diagnostics": endpoint_projection_rows,
            "checkpoint_sha256": checkpoint_hashes,
            "boundaries": {
                "physics": "no physics was advanced; only exact immutable logged states were consumed",
                "rendering": "no new render was run; ten exact direct HDR images were read from the prior hashed evidence cache",
                "unresolved_pixels": "only exact timestamp matches could fill a render request; other requests used nearer endpoint for event emulation and remain counted as unresolved",
                "model_inputs": "causal five-bin event tensors only; GT labels are joined after inference",
                "inference_validity": "cached_direct_HDR_1kHz_short excluded from model inference because it cannot supply a full causal 25 ms history",
                "frozen_core_modified": False,
            },
        }
        episode_summaries[f"episode_{ep_id:03d}"] = ep_summary
        (episode_out / "summary.json").write_text(json.dumps(ep_summary, indent=2, allow_nan=False), encoding="utf-8")

    with (output / "queries.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_queries[0]))
        writer.writeheader(); writer.writerows(all_queries)
    run_manifest = {
        "experiment_id": "r1_5_cached_evidence_cpu_20261019a",
        "code_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": _sha256(args.config.resolve()),
        "scene_config_sha256": _sha256(args.scene_config.resolve()),
        "checkpoint_sha256": {"ours": _sha256(args.ours_checkpoint.resolve()), "v2e": _sha256(args.v2e_checkpoint.resolve())},
        "episodes": episode_summaries,
        "elapsed_wall_s": time.perf_counter() - global_start,
        "compute": "CPU only; torch threads constrained by reused loader to one; no Genesis renderer, EGL, CUDA, or new physics",
        "status": "bounded cached-evidence diagnostic; not a replacement for the planned all-row exact direct-render run",
    }
    (output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, allow_nan=False), encoding="utf-8")
    (output / "README.md").write_text(
        "# R1.5 cached evidence CPU replay\n\n"
        "This run replays the saved center and +55 mm episodes on CPU. It compares the frozen 100 Hz path, endpoint identity/exact candidates, and log-radiance with exact cached-render fills where timestamps match. Ten previously saved 1 kHz direct HDR images provide image references; candidate event streams use only actual state rows from the saved trace. Unmatched render requests are counted and filled from the nearer endpoint solely to keep the event emulator defined.\n\n"
        "The query table includes the 0.5 ms read phases. These are learned-model diagnostics on candidate event streams; rows using unresolved render requests are not evidence that the full render-request path is accurate. The cached direct event sequence is only 9 ms long and is excluded from 25 ms model inference. No simulator core, physics, or renderer was changed or initialized.\n",
        encoding="utf-8")
    _write_hashes(output)
    print(json.dumps({"output": str(output), "queries": len(all_queries), "episodes": sorted(episode_summaries),
                      "elapsed_wall_s": run_manifest["elapsed_wall_s"], "sha256sums": _sha256(output / "SHA256SUMS")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
