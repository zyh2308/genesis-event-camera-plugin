#!/usr/bin/env python3
"""Measure paired synthetic event support against shared GT in a Stage-5 bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from prepare_stage0_flow_bank import gaussian_count_representation

from stage4_v2e_artifacts import locate_shard_artifacts

THRESHOLDS = (0.25, 0.5, 1.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_events(events: np.ndarray, rollout: str, generator: str) -> np.ndarray:
    if events.ndim != 2 or events.shape[1] != 4:
        raise RuntimeError(f"{generator}/{rollout}: expected [N,4] events, got {events.shape}")
    if len(events):
        if not np.isfinite(events).all() or np.any(np.diff(events[:, 0]) < 0):
            raise RuntimeError(f"{generator}/{rollout}: non-finite or unsorted events")
        if events[:, 1].min() < 0 or events[:, 1].max() >= 346 or events[:, 2].min() < 0 or events[:, 2].max() >= 260:
            raise RuntimeError(f"{generator}/{rollout}: event coordinates out of bounds")
    return events


def active_mask(chunk: np.ndarray, left_us: float) -> np.ndarray:
    """Use the registered equal-count Gaussian event-frame encoder's support."""
    if not len(chunk):
        return np.zeros((260, 346), dtype=bool)
    local = chunk.copy()
    local[:, 0] = (local[:, 0] - left_us) * 1e-6
    local[:, 3] = np.where(local[:, 3] > 0, 1.0, -1.0)
    representation = gaussian_count_representation(local, 260, 346)
    return np.any(representation != 0, axis=(0, 3))


def summarize_rows(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, row.get("factors", {}).get(k)) for k in group_keys)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        record = {name: value for name, value in zip(group_keys, key)}
        record["rollout_count"] = len({r["rollout_id"] for r in group})
        record["window_count"] = len(group)
        record["mean_event_count_per_window"] = {
            arm: float(np.mean([r["event_count"][arm] for r in group]))
            for arm in ("eventgenesis", "v2e")
        }
        record["event_active_valid_coverage_macro"] = {
            arm: float(np.mean([r["event_active_valid_coverage"][arm] for r in group]))
            for arm in ("eventgenesis", "v2e")
        }
        record["common_union_valid_coverage_macro"] = float(
            np.mean([r["common_union_valid_coverage"] for r in group]))
        record["gt_valid_coverage"] = {
            str(threshold): float(np.mean([r["gt_valid_coverage"][str(threshold)] for r in group]))
            for threshold in THRESHOLDS
        }
        record["common_union_intersection_gt_coverage"] = {
            str(threshold): float(np.mean([r["common_union_intersection_gt_coverage"][str(threshold)] for r in group]))
            for threshold in THRESHOLDS
        }
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--v2e-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bank, v2e = args.bank_root.resolve(), args.v2e_root.resolve()
    bank_path, bank_manifest_path = bank / "shared_trajectory.h5", bank / "bank_manifest.json"
    split_path, v2e_manifest_path = bank / "registered_split.json", v2e / "v2e_complete_manifest.json"
    if not all(path.is_file() for path in (bank_path, bank_manifest_path, split_path, v2e_manifest_path)):
        raise RuntimeError("statistics require a complete shared bank and V2E manifest")
    bank_manifest, split, v2e_manifest = (
        json.loads(bank_manifest_path.read_text()),
        json.loads(split_path.read_text()),
        json.loads(v2e_manifest_path.read_text()),
    )
    if bank_manifest.get("schema") not in {
        "stage5_flow_scene_coverage_shared_bank_v2",
        "stage5_flow_scene_coverage_shared_bank_v3",
    }:
        raise RuntimeError("unexpected shared physical-bank schema")
    if v2e_manifest.get("schema") not in {
        "stage5_v2e_coverage_pilot_raw_events_v2",
        "stage5_v2e_coverage_pilot_raw_events_v3",
    }:
        raise RuntimeError("unexpected V2E manifest schema")
    if sha256(bank_path) != bank_manifest.get("bank_sha256"):
        raise RuntimeError("shared physical bank hash mismatch")
    if sha256(bank_manifest_path) != v2e_manifest.get("bank_manifest_sha256"):
        raise RuntimeError("V2E raw events were not produced from this physical bank manifest")
    v2e_rows = {row["rollout_id"]: row for row in v2e_manifest["episodes"]}
    if len(split) != 8 or set(v2e_rows) != {row["id"] for row in split}:
        raise RuntimeError("split and complete V2E manifest do not contain the same eight rollout IDs")

    started = datetime.now(timezone.utc).isoformat()
    window_rows = []
    rollout_rows = []
    with h5py.File(bank_path, "r") as shared:
        if set(shared["episodes"].keys()) != {row["id"] for row in split}:
            raise RuntimeError("HDF5 episode groups differ from registered split")
        for item in split:
            rollout = item["id"]
            group = shared[f"episodes/{rollout}"]
            times = np.asarray(group["frame_timestamps_s"], dtype=np.float64)
            if times.shape != (201,) or not np.allclose(np.diff(times), .01, atol=1e-7, rtol=0):
                raise RuntimeError(f"{rollout}: expected 201 uniform 100 Hz endpoints")
            gt = np.asarray(group["flow_forward_30ms_px"], dtype=np.float32)
            valid = np.asarray(group["flow_valid_30ms"], dtype=np.uint8).astype(bool)
            if gt.shape != (66, 2, 260, 346) or valid.shape != (66, 260, 346):
                raise RuntimeError(f"{rollout}: shared label shape mismatch")
            if not np.isfinite(gt).all():
                raise RuntimeError(f"{rollout}: non-finite shared flow GT")
            eg = validate_events(np.asarray(group["eventgenesis_events_txyp"], dtype=np.float64), rollout, "eventgenesis")
            if eg.size:
                eg = eg.copy()
                eg[:, 0] *= 1e6
            v2e_path, status_path = locate_shard_artifacts(v2e, rollout)
            recorded = v2e_rows[rollout]
            status = json.loads(status_path.read_text())
            if (sha256(v2e_path) != recorded["raw_events_sha256"]
                    or sha256(v2e_path) != status.get("raw_h5_sha256")
                    or status.get("exit_code") != 0):
                raise RuntimeError(f"{rollout}: V2E status/manifest/raw artifact hash mismatch")
            with h5py.File(v2e_path, "r") as raw:
                v2 = np.asarray(raw["events"], dtype=np.float64)
            if v2.size:
                v2[:, 0] *= 1.0  # retain integer microseconds for strict endpoint slicing
                v2[:, 3] = np.where(v2[:, 3] > 0, 1.0, -1.0)
            v2 = validate_events(v2, rollout, "v2e")
            for arm, events in (("eventgenesis", eg), ("v2e", v2)):
                if len(events) and (events[:, 0].min() < 0 or events[:, 0].max() > 2_000_000):
                    raise RuntimeError(f"{arm}/{rollout}: timestamps outside the 2 s rollout")

            eg_ts = eg[:, 0] if len(eg) else np.empty((0,), dtype=np.float64)
            v2_ts = v2[:, 0] if len(v2) else np.empty((0,), dtype=np.float64)
            rollout_active = {"eventgenesis": 0, "v2e": 0, "common_union": 0, "valid": 0}
            rollout_events = {"eventgenesis": 0, "v2e": 0}
            for window_index, endpoint_index in enumerate(range(0, 198, 3)):
                left, right = float(times[endpoint_index]) * 1e6, float(times[endpoint_index + 3]) * 1e6
                active_masks = {}
                event_counts = {}
                for arm, events, event_ts in (("eventgenesis", eg, eg_ts), ("v2e", v2, v2_ts)):
                    lo = int(np.searchsorted(event_ts, left, side="right"))
                    hi = int(np.searchsorted(event_ts, right, side="right"))
                    chunk = events[lo:hi]
                    if len(chunk) and (np.any(chunk[:, 0] <= left) or np.any(chunk[:, 0] > right)):
                        raise RuntimeError(f"{arm}/{rollout}: event outside registered (t_i,t_i+3] window")
                    active_masks[arm] = active_mask(chunk, left)
                    event_counts[arm] = int(len(chunk))
                    rollout_events[arm] += len(chunk)
                common_union = active_masks["eventgenesis"] | active_masks["v2e"]
                valid_mask = valid[window_index]
                magnitude = np.linalg.norm(gt[window_index], axis=0)
                valid_pixels = int(valid_mask.sum())
                gt_counts = {str(threshold): int((valid_mask & (magnitude >= threshold)).sum())
                             for threshold in THRESHOLDS}
                active_valid = {
                    arm: int((valid_mask & active_masks[arm]).sum())
                    for arm in ("eventgenesis", "v2e")
                }
                union_valid = int((valid_mask & common_union).sum())
                union_gt_intersection = {
                    str(threshold): int((valid_mask & common_union & (magnitude >= threshold)).sum())
                    for threshold in THRESHOLDS
                }
                rollout_active["eventgenesis"] += active_valid["eventgenesis"]
                rollout_active["v2e"] += active_valid["v2e"]
                rollout_active["common_union"] += union_valid
                rollout_active["valid"] += valid_pixels
                window_rows.append({
                    "rollout_id": rollout,
                    "split": item["split"],
                    "stratum": item["stratum"],
                    "factors": {key: item[key] for key in (
                        "texture_pattern", "texture_contrast", "speed_class", "speed_scale",
                        "amplitude_class", "amplitude_scale", "size_class", "size_range_m", "layout_class",
                        "floor_tiles_per_side", "floor_extent_m", "object_top_panels_per_side", "tile_gap_fraction",
                        "occlusion_crossing_base_x_m")},
                    "window_index": window_index,
                    "sample_id": f"{rollout}/dt1_{endpoint_index:03d}",
                    "event_start_us": left,
                    "event_end_us": right,
                    "event_count": event_counts,
                    "valid_pixels": valid_pixels,
                    "event_active_valid_pixels": active_valid,
                    "event_active_valid_coverage": {
                        arm: active_valid[arm] / max(valid_pixels, 1)
                        for arm in ("eventgenesis", "v2e")
                    },
                    "common_union_valid_pixels": union_valid,
                    "common_union_valid_coverage": union_valid / max(valid_pixels, 1),
                    "gt_valid_coverage": {
                        threshold: gt_counts[threshold] / max(valid_pixels, 1)
                        for threshold in gt_counts
                    },
                    "common_union_intersection_gt_pixels": union_gt_intersection,
                    "common_union_intersection_gt_coverage": {
                        threshold: union_gt_intersection[threshold] / max(valid_pixels, 1)
                        for threshold in union_gt_intersection
                    },
                })
            rollout_rows.append({
                "rollout_id": rollout,
                "split": item["split"],
                "stratum": item["stratum"],
                "factors": {key: item[key] for key in (
                    "texture_pattern", "texture_contrast", "speed_class", "speed_scale",
                    "amplitude_class", "amplitude_scale", "size_class", "size_range_m", "layout_class",
                    "floor_tiles_per_side", "floor_extent_m", "object_top_panels_per_side", "tile_gap_fraction",
                    "occlusion_crossing_base_x_m")},
                "windows": 66,
                "events_in_windows": rollout_events,
                "raw_events_total": {"eventgenesis": int(len(eg)), "v2e": int(len(v2))},
                "events_outside_registered_66_windows": {
                    arm: int(len(events) - rollout_events[arm])
                    for arm, events in (("eventgenesis", eg), ("v2e", v2))
                },
                "valid_pixels": rollout_active["valid"],
                "event_active_valid_pixels": {arm: rollout_active[arm] for arm in ("eventgenesis", "v2e")},
                "event_active_valid_coverage": {
                    arm: rollout_active[arm] / max(rollout_active["valid"], 1)
                    for arm in ("eventgenesis", "v2e")
                },
                "common_union_valid_coverage": rollout_active["common_union"] / max(rollout_active["valid"], 1),
            })

    result = {
        "schema": "stage5_flow_scene_coverage_statistics_v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True).strip(),
        "bank": {"path": str(bank_path), "size_bytes": bank_path.stat().st_size, "sha256": sha256(bank_path),
                 "manifest_path": str(bank_manifest_path), "manifest_sha256": sha256(bank_manifest_path)},
        "v2e_manifest": {"path": str(v2e_manifest_path), "sha256": sha256(v2e_manifest_path)},
        "event_window_rule": "(t_i,t_i+3] in shared 100 Hz physical endpoint timestamps; left-exclusive/right-inclusive",
        "gt_motion_thresholds_px_per_30ms": list(THRESHOLDS),
        "coverage_definition": "valid shared GT pixels intersected with per-generator event-active pixels; common support is union of normal EventGenesis and V2E active pixels",
        "event_count_filtering": "none",
        "sensor_thresholds_changed": False,
        "real_data_used": False,
        "training_or_checkpoint_selection": False,
        "started_utc": started,
        "rollouts": rollout_rows,
        "stratum_summary": summarize_rows(window_rows, ("stratum",)),
        "split_summary": summarize_rows(window_rows, ("split",)),
        "factor_summary": {
            key: summarize_rows(window_rows, (key,))
            for key in ("texture_pattern", "texture_contrast", "speed_class", "amplitude_class", "size_class", "layout_class",
                        "floor_tiles_per_side", "object_top_panels_per_side")
        },
        "windows": window_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output), "rollouts": len(rollout_rows), "windows": len(window_rows)}, indent=2))


if __name__ == "__main__":
    main()
