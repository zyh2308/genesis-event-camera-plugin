#!/usr/bin/env python3
"""Offline diagnostic ablation of archived vs matched direct events by scene region.

GT segmentation is used only to assign event pixels to broad regions and remove
events for evaluator-side counterfactuals. No segmentation, depth, or GT enters
the unchanged event-state network. This is not a deployable perception path.
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

from benchmark.robot_closed_loop.diagnostics.diagnose_r1_5_shadow_perception_causal_replay import (  # noqa: E402
    camera_to_world_cv_from_config,
)
from benchmark.robot_closed_loop.diagnostics.diagnose_r1_native_hdr_replay import (  # noqa: E402
    infer_event_state,
    interpolate_trace_label,
    load_event_models,
    read_trace,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_events(events: np.ndarray, times: np.ndarray, seg: np.ndarray,
                    target_id: int, robot_ids: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """Assign events by nearest saved evaluator mask (10 ms cadence)."""
    if not len(events):
        return np.empty((0,), dtype=np.int16), np.empty((0,), dtype=np.float64)
    frame_idx = np.abs(events[:, 0, None] - times[None, :]).argmin(axis=1)
    x = np.rint(events[:, 1]).astype(np.int64)
    y = np.rint(events[:, 2]).astype(np.int64)
    h, w = seg.shape[1:]
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    label = np.full(len(events), -1, dtype=np.int32)
    label[valid] = seg[frame_idx[valid], y[valid], x[valid]]
    region = np.full(len(events), 3, dtype=np.int16)  # other/unknown
    region[label == 1] = 2  # static support plane
    region[np.isin(label, list(robot_ids))] = 1
    region[label == target_id] = 0
    return region, frame_idx.astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--direct-root", type=Path, required=True,
                    help="E shadow-on raw event files")
    ap.add_argument("--cache-root", type=Path, required=True,
                    help="read-only source cache containing masks and exact trace")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--scene-config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    scene = json.loads(args.scene_config.read_text())
    width, height = map(int, scene["camera"]["resolution_wh"])
    fov = np.deg2rad(float(scene["camera"]["vertical_fov_deg"]))
    focal = height / (2.0 * np.tan(fov / 2.0))
    intr = np.asarray([focal, focal, width / 2.0, height / 2.0])
    camera_pose = camera_to_world_cv_from_config(scene)
    center_height = float(scene["target"]["size_m"][2]) / 2.0
    models, checkpoint_hashes = load_event_models(
        {"ours_old_checkpoint": args.checkpoint.resolve()}, camera_pose, intr, (width, height)
    )
    model = models["ours_old_checkpoint"]
    region_names = ["target", "robot", "floor", "other_or_unknown"]
    variants = {
        "all": lambda r: np.ones(len(r), dtype=bool),
        "drop_target": lambda r: r != 0,
        "drop_robot": lambda r: r != 1,
        "drop_floor": lambda r: r != 2,
        "target_only": lambda r: r == 0,
        "robot_only": lambda r: r == 1,
        "floor_only": lambda r: r == 2,
    }
    result_rows: list[dict[str, Any]] = []
    region_totals: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    queries = [0.5855, 0.5860, 0.5865, 0.5875, 0.5885, 0.5895, 0.5905]
    started = time.perf_counter()

    for ep in ("004", "005"):
        cache_dir = args.cache_root / f"episode_{ep}"
        with h5py.File(cache_dir / "same_state_source_cache.h5", "r") as f:
            mask_times = np.asarray(f["timestamps_s"][:], dtype=np.float64)
            masks = np.asarray(f["segmentation"][:], dtype=np.int32)
            pose_ids = np.asarray(f["segmentation_pose_ids"][:], dtype=np.int32)
            poses = np.asarray(f["object_poses"][:], dtype=np.float64)
        # The single moving body in the saved pose table is the cube (id 2 in
        # this cache). Verify this from motion instead of assuming silently.
        translation = poses[:, :, :3, 3]
        displacement = np.linalg.norm(translation[-1] - translation[0], axis=1)
        moving = np.argsort(displacement)[::-1]
        if (displacement[moving[0]] < 0.010 or
                (len(moving) > 1 and displacement[moving[0]] < 2.0 * displacement[moving[1]])):
            raise RuntimeError(
                f"episode {ep}: cached target motion is not separated from articulation: "
                f"top displacements {displacement[moving[:4]].tolist()}"
            )
        target_id = int(pose_ids[moving[0]])
        robot_ids = set(int(x) for x in pose_ids if int(x) != target_id)
        unique = set(int(x) for x in np.unique(masks))
        if target_id not in unique:
            raise RuntimeError(f"episode {ep}: cache segmentation labels do not match pose IDs")
        trace_path = cache_dir / "trace_window.jsonl"
        trace = read_trace(trace_path)
        sources = {
            "archived": args.reference_root / f"episode_{ep}" / "archived_direct_hdr_2khz_reference_events.npz",
            "fresh_shadow_on": args.direct_root / f"episode_{ep}_shadow_on_raw_events.npz",
        }
        for source_name, path in sources.items():
            source_hashes[str(path.resolve())] = sha256(path)
        source_data = {}
        for source_name, path in sources.items():
            arr = np.load(path, allow_pickle=False)["events"].astype(np.float64, copy=False)
            in_span = (arr[:, 0] >= queries[0] - 0.025 - 1e-9) & (arr[:, 0] <= queries[-1] + 1e-9)
            arr = arr[in_span]
            regions, frame_idx = classify_events(arr, mask_times, masks, target_id, robot_ids)
            source_data[source_name] = (arr, regions)
            region_totals[f"{ep}_{source_name}"] = {
                "events_in_query_union": int(len(arr)),
                "events_by_region": {name: int(np.count_nonzero(regions == i)) for i, name in enumerate(region_names)},
                "event_time_min_s": float(arr[:, 0].min()) if len(arr) else None,
                "event_time_max_s": float(arr[:, 0].max()) if len(arr) else None,
                "mask_assignment": "nearest saved 10 ms segmentation frame",
            }
        trace_for_queries = [interpolate_trace_label(trace, t) for t in queries]
        for query, label in zip(queries, trace_for_queries):
            gt = np.asarray(label["position_m"], dtype=np.float64)[:2]
            for source_name, (events, regions) in source_data.items():
                for variant, select in variants.items():
                    candidate = events[select(regions)]
                    est, voxel = infer_event_state(candidate, model, query, width, height,
                                                   camera_pose, intr, center_height)
                    pred = est.get("position_xyz_m")
                    err = None if pred is None else float(np.linalg.norm(np.asarray(pred)[:2] - gt) * 1000)
                    result_rows.append({
                        "episode_id": ep, "query_s": query, "source": source_name,
                        "region_ablation": variant,
                        "event_count_25ms": int(est["event_count_25ms"]),
                        "nonempty_5ms_bins": int(est["nonempty_5ms_bins"]),
                        "state_available": bool(est["state_available"]),
                        "state_status": est.get("state_status"),
                        "xy_error_mm": err,
                        "voxel_sha256": hashlib.sha256(np.ascontiguousarray(voxel).tobytes()).hexdigest(),
                    })
            # Cross-source region swaps test whether the target-local events
            # or the rest of the image carries the source-specific failure.
            # The masks are evaluator-only and are not an estimator input.
            archived, archived_regions = source_data["archived"]
            fresh, fresh_regions = source_data["fresh_shadow_on"]
            hybrids = {
                "archive_background_fresh_target": np.concatenate([
                    archived[archived_regions != 0], fresh[fresh_regions == 0]], axis=0),
                "fresh_background_archive_target": np.concatenate([
                    fresh[fresh_regions != 0], archived[archived_regions == 0]], axis=0),
            }
            for hybrid_name, candidate in hybrids.items():
                est, voxel = infer_event_state(candidate, model, query, width, height,
                                               camera_pose, intr, center_height)
                pred = est.get("position_xyz_m")
                err = None if pred is None else float(np.linalg.norm(np.asarray(pred)[:2] - gt) * 1000)
                result_rows.append({
                    "episode_id": ep, "query_s": query, "source": hybrid_name,
                    "region_ablation": "cross_source_swap_target_vs_background",
                    "event_count_25ms": int(est["event_count_25ms"]),
                    "nonempty_5ms_bins": int(est["nonempty_5ms_bins"]),
                    "state_available": bool(est["state_available"]),
                    "state_status": est.get("state_status"),
                    "xy_error_mm": err,
                    "voxel_sha256": hashlib.sha256(np.ascontiguousarray(voxel).tobytes()).hexdigest(),
                })
    csv_path = out / "per_query_region_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0]))
        writer.writeheader(); writer.writerows(result_rows)
    grouped = {}
    for row in result_rows:
        key = (row["episode_id"], row["source"], row["region_ablation"])
        grouped.setdefault(key, []).append(row)
    summary_groups = []
    for (ep, source, variant), rows in grouped.items():
        errors = [r["xy_error_mm"] for r in rows if r["xy_error_mm"] is not None]
        summary_groups.append({
            "episode_id": ep, "source": source, "region_ablation": variant,
            "queries": len(rows), "available": sum(r["state_available"] for r in rows),
            "mean_xy_error_mm": None if not errors else float(np.mean(errors)),
            "median_xy_error_mm": None if not errors else float(np.median(errors)),
            "max_xy_error_mm": None if not errors else float(np.max(errors)),
            "mean_event_count_25ms": float(np.mean([r["event_count_25ms"] for r in rows])),
        })
    payload = {
        "status": "PASS", "experiment_id": out.name,
        "scope": "offline evaluator-only region ablation of archived and fresh direct event streams; no physics or rendering",
        "code_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_sha256": checkpoint_hashes,
        "scene_config_sha256": sha256(args.scene_config),
        "input_hashes": source_hashes,
        "cache_hashes": {str((args.cache_root / f"episode_{ep}" / name).resolve()): sha256(args.cache_root / f"episode_{ep}" / name)
                          for ep in ("004", "005") for name in ("same_state_source_cache.h5", "trace_window.jsonl")},
        "query_times_s": queries,
        "region_definition": {"target": "unique moving segmentation pose ID, verified from cached pose displacement",
                              "robot": "other segmentation pose IDs", "floor": "segmentation label 1",
                              "other_or_unknown": "remaining labels", "mask_cadence": "nearest of cached 10 ms masks"},
        "boundary": "segmentation is used only to filter events in an offline diagnostic ablation; it is never supplied to the network or proposed as a perception input",
        "event_region_totals": region_totals,
        "summary": summary_groups,
        "wall_s": time.perf_counter() - started,
        "per_query_csv": csv_path.name,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "wall_s": payload["wall_s"], "groups": summary_groups}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
