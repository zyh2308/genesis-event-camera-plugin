#!/usr/bin/env python3
"""Score archived-vs-fresh-shadow events after an exact-seed source replay.

The compact JSON is generated on g1 from the immutable archived event file and
the new run. The fresh-shadow events come from the matched cached-state replay.
This evaluator feeds only causal five-bin events to the checkpoint and joins
saved GT labels after inference.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.event_state_model import (  # noqa: E402
    decode_heatmap_state,
    five_bin_tensor,
    load_checkpoint,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float | None]:
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def camera_to_world_cv(config: dict[str, Any]) -> np.ndarray:
    camera = config["camera"]
    position = np.asarray(camera["position_m"], dtype=np.float64)
    forward = np.asarray(camera["lookat_m"], dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    up = np.asarray(camera["up_m"], dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, -camera_up, forward))
    matrix[:3, 3] = position
    return matrix


def evaluate(events: np.ndarray, query: dict[str, Any], source: str,
             model_record, width: int, height: int, camera: np.ndarray,
             intrinsics: np.ndarray, fixed_height: float) -> dict[str, Any]:
    cutoff = float(query["time_s"])
    voxel = five_bin_tensor(events, cutoff, width, height)
    count = int(voxel.sum())
    bins = int(np.count_nonzero(voxel.sum(axis=(1, 2, 3))))
    valid = count >= int(model_record[3].get("validity_policy", {}).get("minimum_events", 12)) and bins >= int(
        model_record[3].get("validity_policy", {}).get("minimum_nonempty_5ms_bins", 3))
    row: dict[str, Any] = {
        "query_s": cutoff, "source": source, "event_count_25ms": count,
        "nonempty_5ms_bins": int(bins), "state_available": bool(valid),
        "label_method": query["label_method"],
        "gt_position_xy_m": query["position_m"][:2],
        "gt_velocity_xy_m_s": query["velocity_m_s"][:2],
        "input_voxel_sha256": hashlib.sha256(np.ascontiguousarray(voxel).tobytes()).hexdigest(),
        "five_bin_counts": voxel.sum(axis=(1, 2, 3)).astype(int).tolist(),
    }
    if not valid:
        row.update({"state_status": "insufficient_events_or_temporal_coverage",
                    "position_xy_error_mm": None, "velocity_xy_error_mm_s": None})
        return row
    model, label_mean, label_scale, _ = model_record
    tensor = torch.as_tensor(np.log1p(voxel).reshape(1, 10, 64, 64), dtype=torch.float32)
    with torch.no_grad():
        output = model(tensor)
        decoded = decode_heatmap_state(
            output, label_mean, label_scale,
            torch.as_tensor(camera, dtype=torch.float32),
            torch.as_tensor(intrinsics, dtype=torch.float32), (width, height),
            fixed_height_m=fixed_height,
        )[0].cpu().numpy().astype(np.float64)
    gt_position = np.asarray(query["position_m"], dtype=np.float64)
    gt_velocity = np.asarray(query["velocity_m_s"], dtype=np.float64)
    row.update({
        "state_status": "fresh", "pred_position_xyz_m": decoded[:3].tolist(),
        "pred_velocity_xy_m_s": decoded[3:5].tolist(),
        "position_xy_error_mm": float(np.linalg.norm(decoded[:2] - gt_position[:2]) * 1000),
        "velocity_xy_error_mm_s": float(np.linalg.norm(decoded[3:5] - gt_velocity[:2]) * 1000),
    })
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-evidence", type=Path, required=True)
    parser.add_argument("--fresh-shadow-events", type=Path, required=True)
    parser.add_argument("--fresh-source-label", default="fresh_shadow_on",
                        help="name recorded in output for the supplied fresh event stream")
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    evidence = json.loads(args.compact_evidence.read_text(encoding="utf-8"))
    payload = base64.b64decode(evidence["event_payload_npz_base64"])
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        archive_events = np.asarray(data["archive_events"], dtype=np.float64)
        rerun_events = np.asarray(data["rerun_events"], dtype=np.float64)
        query_times = np.asarray(data["query_times_s"], dtype=np.float64)
    if not np.array_equal(archive_events, rerun_events):
        raise RuntimeError("archived and exact-seed rerun causal event windows are not elementwise identical")
    with np.load(args.fresh_shadow_events, allow_pickle=False) as data:
        if "events" not in data:
            raise KeyError(f"expected events key, found {data.files}")
        fresh_events = np.asarray(data["events"], dtype=np.float64)

    scene = json.loads(args.scene_config.read_text(encoding="utf-8"))
    width, height = map(int, scene["camera"]["resolution_wh"])
    focal = height / (2.0 * np.tan(np.deg2rad(float(scene["camera"]["vertical_fov_deg"])) / 2.0))
    intrinsics = np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=np.float64)
    camera = camera_to_world_cv(scene)
    model_record = load_checkpoint(str(args.checkpoint.resolve()), device="cpu")
    fixed_height = float(scene["target"]["size_m"][2]) / 2.0
    archive_labels = evidence["post_inference_gt_labels"]["archive"]
    rerun_labels = evidence["post_inference_gt_labels"]["seed_matched_rerun"]
    rows: list[dict[str, Any]] = []
    for i, query_s in enumerate(query_times.tolist()):
        if abs(float(archive_labels[i]["time_s"]) - query_s) > 1e-9:
            raise RuntimeError("compact GT query timestamp mismatch")
        rows.append(evaluate(archive_events, archive_labels[i], "archive_and_exact_rerun",
                             model_record, width, height, camera, intrinsics, fixed_height))
        rows.append(evaluate(fresh_events, archive_labels[i], args.fresh_source_label,
                             model_record, width, height, camera, intrinsics, fixed_height))

    csv_path = output / "query_localization.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, Any] = {
        "status": "PASS",
        "experiment": "exact seed replay versus cached-state event stream; old checkpoint diagnostic only",
        "fresh_source_label": args.fresh_source_label,
        "inputs": {
            "compact_evidence_sha256": sha256(args.compact_evidence),
            "fresh_shadow_events_sha256": sha256(args.fresh_shadow_events),
            "scene_config_sha256": sha256(args.scene_config),
            "checkpoint_sha256": sha256(args.checkpoint),
        },
        "source_reproduction": {
            "state_trace_comparison": evidence["state_trace_comparison"],
            "event_source_match_5505_5905ms": evidence["event_source_match_5505_5905ms"],
            "causal_window_events_elementwise_identical": True,
            "causal_window_event_count": int(len(archive_events)),
        },
        "query_summary": {
            source: {
                "position_xy_error_mm": stats([float(r["position_xy_error_mm"]) for r in rows
                                                if r["source"] == source and r["position_xy_error_mm"] is not None]),
                "velocity_xy_error_mm_s": stats([float(r["velocity_xy_error_mm_s"]) for r in rows
                                                  if r["source"] == source and r["velocity_xy_error_mm_s"] is not None]),
                "valid_fraction": float(np.mean([r["state_available"] for r in rows if r["source"] == source])),
            } for source in ("archive_and_exact_rerun", args.fresh_source_label)
        },
        "queries": rows,
        "gt_boundary": evidence["boundary"],
        "query_csv": csv_path.name,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "source_reproduction": summary["source_reproduction"],
                      "query_summary": summary["query_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
