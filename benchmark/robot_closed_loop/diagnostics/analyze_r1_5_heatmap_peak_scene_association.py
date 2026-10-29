#!/usr/bin/env python3
"""Check whether wrong heatmap peaks coincide with the robot end effector.

All robot/object state is used only after event-only inference, for offline
diagnostics. It is never added to the model input or decoder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_projection_helpers(path: Path, code_root: Path):
    sys.path.insert(0, str(code_root.resolve()))
    spec = importlib.util.spec_from_file_location("r1_5_projection_helpers", path.resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load projection helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--decoder-csv", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--heatmap-helpers", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default="ours_native_hdr")
    parser.add_argument("--decoder", default="global_softargmax")
    parser.add_argument("--wrong-peak-threshold-px", type=float, default=20.0)
    parser.add_argument("--tcp-near-threshold-px", type=float, default=25.0)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    helper = load_projection_helpers(args.heatmap_helpers, args.code_root)
    scene = json.loads(args.scene_config.read_text(encoding="utf-8"))
    width, height = map(int, scene["camera"]["resolution_wh"])
    focal = height / (2.0 * np.tan(np.deg2rad(float(scene["camera"]["vertical_fov_deg"])) / 2.0))
    intrinsics = np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=np.float64)
    camera = helper.camera_to_world_cv(scene["camera"])

    with args.decoder_csv.open(newline="", encoding="utf-8") as stream:
        source_rows = [row for row in csv.DictReader(stream)
                       if row["checkpoint"] == args.checkpoint and row["decoder"] == args.decoder]
    if not source_rows:
        raise ValueError(f"no rows for {args.checkpoint}/{args.decoder}")

    trace_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, str]] = {}
    out_rows: list[dict[str, Any]] = []
    for row in source_rows:
        eid = int(row["episode_id"])
        if eid not in trace_cache:
            trace_path = args.reference_root / "episodes" / f"episode_{eid:03d}" / "state_control_trace.jsonl"
            trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            times = np.asarray([item["time_s"] for item in trace], dtype=np.float64)
            tcp = np.asarray([item["tcp_pad_center_m"] for item in trace], dtype=np.float64)
            hand = np.asarray([item["hand_link_position_m"] for item in trace], dtype=np.float64)
            trace_cache[eid] = (times, tcp, hand, sha256(trace_path))
        times, tcp_trace, hand_trace, trace_hash = trace_cache[eid]
        cutoff = float(row["cutoff_s"])
        tcp = np.asarray([np.interp(cutoff, times, tcp_trace[:, i]) for i in range(3)])
        hand = np.asarray([np.interp(cutoff, times, hand_trace[:, i]) for i in range(3)])
        tcp_uv = helper.project_world(tcp[None, :], camera, intrinsics)[0]
        hand_uv = helper.project_world(hand[None, :], camera, intrinsics)[0]
        gt_uv = np.asarray([float(row["gt_u_px"]), float(row["gt_v_px"])])
        peak_uv = np.asarray([float(row["peak_u_px"]), float(row["peak_v_px"])])
        pred_uv = np.asarray([float(row["pred_u_px"]), float(row["pred_v_px"])])
        peak_to_gt = float(np.linalg.norm(peak_uv - gt_uv))
        peak_to_tcp = float(np.linalg.norm(peak_uv - tcp_uv))
        peak_to_hand = float(np.linalg.norm(peak_uv - hand_uv))
        out_rows.append({
            "episode_id": eid,
            "seed": row["seed"],
            "speed_m_s": row["speed_m_s"],
            "launch_y_m": row["launch_y_m"],
            "cutoff_s": cutoff,
            "position_error_mm": float(row["position_xy_error_mm"]),
            "peak_target_distance_px": peak_to_gt,
            "peak_tcp_distance_px": peak_to_tcp,
            "peak_hand_distance_px": peak_to_hand,
            "softargmax_uv": pred_uv.tolist(),
            "peak_uv": peak_uv.tolist(),
            "target_uv": gt_uv.tolist(),
            "tcp_uv": tcp_uv.tolist(),
            "hand_uv": hand_uv.tolist(),
            "heatmap_target_roi_mass": float(row["heatmap_probability_in_target_roi"]),
            "trace_sha256": trace_hash,
        })

    wrong = [r for r in out_rows if r["peak_target_distance_px"] > args.wrong_peak_threshold_px]
    near_tcp_wrong = [r for r in wrong if r["peak_tcp_distance_px"] <= args.tcp_near_threshold_px]
    near_hand_wrong = [r for r in wrong if r["peak_hand_distance_px"] <= args.tcp_near_threshold_px]
    fields = list(out_rows[0])
    csv_path = output / "peak_scene_association.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)

    summary = {
        "status": "completed",
        "purpose": "post-inference spatial diagnostic; test whether wrong heatmap top modes overlap projected end-effector points",
        "boundary": "saved state trace, object labels, and projected tool landmarks are post-inference diagnostics only; no GT enters event/model input",
        "checkpoint": args.checkpoint,
        "decoder": args.decoder,
        "thresholds": {"wrong_peak_px": args.wrong_peak_threshold_px,
                       "near_tcp_or_hand_px": args.tcp_near_threshold_px},
        "counts": {
            "queries": len(out_rows),
            "top_peak_far_from_target": len(wrong),
            "far_target_peak_also_near_tcp": len(near_tcp_wrong),
            "far_target_peak_also_near_hand_link": len(near_hand_wrong),
            "fractions_among_far_target_peaks": {
                "near_tcp": len(near_tcp_wrong) / len(wrong) if wrong else None,
                "near_hand_link": len(near_hand_wrong) / len(wrong) if wrong else None,
            },
        },
        "inputs": {
            "decoder_csv_sha256": sha256(args.decoder_csv),
            "scene_config_sha256": sha256(args.scene_config),
            "trace_sha256_by_episode": {str(eid): value[3] for eid, value in sorted(trace_cache.items())},
        },
        "output_csv": csv_path.name,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in (csv_path, summary_path)),
        encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
