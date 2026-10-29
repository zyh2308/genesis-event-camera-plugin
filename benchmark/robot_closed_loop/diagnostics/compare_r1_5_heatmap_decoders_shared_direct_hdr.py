#!/usr/bin/env python3
"""Compare heatmap decoders on the shared nine-episode Direct-HDR reference.

The model sees only causal five-bin event tensors ending at each cutoff. Saved
state traces are interpolated at the cutoff after inference for scoring, matching
the existing common-reference evaluator's label convention. GT projection is
also used post-inference for approximate event/heatmap ROI diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values), q))


def metrics(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean_mm": None if not values else float(np.mean(values)),
        "p50_mm": percentile(values, 50),
        "p95_mm": percentile(values, 95),
        "max_mm": None if not values else float(np.max(values)),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_helpers(path: Path, code_root: Path):
    sys.path.insert(0, str(code_root.resolve()))
    spec = importlib.util.spec_from_file_location("r1_5_heatmap_helpers", path.resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load diagnostic helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_episode: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    by_speed: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    by_model: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["checkpoint"], row["decoder"], row["episode_id"])
        by_episode[key].append(float(row["position_xy_error_mm"]))
        by_speed[(row["checkpoint"], row["decoder"], row["speed_m_s"])].append(
            float(row["position_xy_error_mm"]))
        by_model[(row["checkpoint"], row["decoder"])].append(
            float(row["position_xy_error_mm"]))

    episodes = []
    for (checkpoint, decoder, episode_id), values in sorted(by_episode.items()):
        episodes.append({"checkpoint": checkpoint, "decoder": decoder,
                         "episode_id": int(episode_id), **metrics(values)})
    speeds = []
    for (checkpoint, decoder, speed), values in sorted(by_speed.items()):
        speeds.append({"checkpoint": checkpoint, "decoder": decoder,
                       "speed_m_s": float(speed), **metrics(values)})
    overall = []
    for (checkpoint, decoder), values in sorted(by_model.items()):
        overall.append({"checkpoint": checkpoint, "decoder": decoder, **metrics(values)})

    # Paired per-query comparison: each decoder is evaluated on the same
    # checkpoint, event tensor and causal cutoff.
    pivot: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        pivot[(row["checkpoint"], row["episode_id"], row["query_s"])][row["decoder"]] = float(
            row["position_xy_error_mm"])
    paired = defaultdict(list)
    for (checkpoint, _episode, _query), values in pivot.items():
        if "global_softargmax" in values and "local_softargmax_r2" in values:
            paired[checkpoint].append(values["local_softargmax_r2"] - values["global_softargmax"])
    paired_local_vs_global = {
        name: {
            "n": len(values),
            "local_lower_error_fraction": float(np.mean(np.asarray(values) < 0)) if values else None,
            "median_local_minus_global_error_mm": percentile(values, 50),
            "mean_local_minus_global_error_mm": float(np.mean(values)) if values else None,
        }
        for name, values in paired.items()
    }
    return {"overall": overall, "by_speed": speeds, "by_episode": episodes,
            "paired_local_vs_global": paired_local_vs_global}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--polls", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH, repeatable")
    parser.add_argument("--heatmap-helpers", type=Path, required=True,
                        help="path to inspect_r1_5_heatmap_evidence.py")
    parser.add_argument("--code-root", type=Path, required=True,
                        help="checkout containing benchmark.robot_closed_loop.event_state_model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--local-radius-cells", type=int, default=2)
    parser.add_argument("--roi-padding-px", type=float, default=6.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    torch.set_num_threads(args.torch_threads)
    np.set_printoptions(precision=7, suppress=True)

    helper = load_helpers(args.heatmap_helpers, args.code_root)
    reference_summary = json.loads(args.reference_summary.read_text(encoding="utf-8"))
    scene = json.loads(args.scene_config.read_text(encoding="utf-8"))
    width, height = map(int, scene["camera"]["resolution_wh"])
    focal = height / (2.0 * np.tan(np.deg2rad(float(scene["camera"]["vertical_fov_deg"])) / 2.0))
    intrinsics = np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=np.float64)
    camera = helper.camera_to_world_cv(scene["camera"])
    target_size = np.asarray(scene["target"]["size_m"], dtype=np.float64)
    plane_z = float(target_size[2] / 2.0)
    checkpoints = []
    for item in args.checkpoint:
        if "=" not in item:
            raise ValueError("--checkpoint must be NAME=PATH")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path).resolve()
        model, label_mean, label_scale, metadata = helper.checkpoint_record(path)
        model.eval()
        checkpoints.append((name, path, model, label_mean, label_scale, metadata, sha256(path)))

    provenance = {int(item["episode_id"]): item for item in reference_summary["provenance"]}
    poll_rows = list(csv.DictReader(args.polls.open(newline="", encoding="utf-8")))
    by_episode_queries: dict[int, list[dict[str, Any]]] = defaultdict(list)
    paired_cutoffs: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in poll_rows:
        if row["source"] not in ("ours", "v2e"):
            continue
        eid = int(row["episode_id"])
        paired_cutoffs[eid][row["source"]].append(float(row["cutoff_s"]))
        if row["source"] == "ours":
            by_episode_queries[eid].append(row)
    if not by_episode_queries:
        raise ValueError("no Ours poll rows found")
    for eid, sources in paired_cutoffs.items():
        if len(sources.get("ours", [])) != len(sources.get("v2e", [])) or not np.allclose(
                sources["ours"], sources["v2e"], rtol=0.0, atol=1e-12):
            raise ValueError(f"Ours/V2E poll cutoffs differ for episode {eid}")

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    episode_inputs = []
    decoder_names = ("global_softargmax", "heatmap_argmax", "local_softargmax_r2")
    for eid in sorted(by_episode_queries):
        info = provenance[eid]
        episode_dir = args.reference_root / "episodes" / f"episode_{eid:03d}"
        event_path = episode_dir / "direct_hdr_events_1khz_2khz.npz"
        trace_path = episode_dir / "state_control_trace.jsonl"
        event_hash, trace_hash = helper.sha256(event_path), helper.sha256(trace_path)
        if event_hash != info["events_sha256"] or trace_hash != info["trace_sha256"]:
            raise ValueError(f"reference hash mismatch in episode {eid}")
        archive = np.load(event_path, allow_pickle=False)
        if "events_2000hz" not in archive:
            raise KeyError(f"events_2000hz missing from {event_path}; keys={archive.files}")
        events = np.asarray(archive["events_2000hz"], dtype=np.float64).reshape(-1, 4)
        events = events[np.argsort(events[:, 0], kind="stable")]
        trace = read_jsonl(trace_path)
        trace_times = np.asarray([item["time_s"] for item in trace], dtype=np.float64)
        trace_positions = np.asarray([item["object"]["position_m"] for item in trace], dtype=np.float64)
        contact_s = float(info["first_contact_s"])
        episode_inputs.append({"episode_id": eid, "events_sha256": event_hash,
                               "trace_sha256": trace_hash, "event_count": int(len(events)),
                               "first_contact_s": contact_s,
                               "query_count_from_paired_polls": len(by_episode_queries[eid])})

        for query in by_episode_queries[eid]:
            cutoff = float(query["cutoff_s"])
            if cutoff >= contact_s:
                continue
            if cutoff < trace_times[0] or cutoff > trace_times[-1]:
                raise ValueError(f"cutoff outside state trace for episode {eid}: {cutoff}")
            # This is the same post-inference timestamp interpolation convention
            # used by evaluate_r1_common_direct_hdr.py.
            gt_position = np.asarray([
                np.interp(cutoff, trace_times, trace_positions[:, axis]) for axis in range(3)
            ], dtype=np.float64)
            gt_uv = helper.project_world(gt_position[None, :], camera, intrinsics)[0]
            roi = helper.cube_bounds(gt_position, target_size, camera, intrinsics,
                                     args.roi_padding_px, width, height)
            voxel = helper.five_bin_tensor(events, cutoff, width, height)
            window_mask = ((events[:, 0] >= cutoff - 0.025) & (events[:, 0] < cutoff))
            window_events = events[window_mask]
            x0, y0, x1, y1 = roi
            roi_mask = ((window_events[:, 1] >= x0) & (window_events[:, 1] < x1) &
                        (window_events[:, 2] >= y0) & (window_events[:, 2] < y1))
            tensor = torch.as_tensor(np.log1p(voxel).reshape(1, 10, 64, 64), dtype=torch.float32)
            for name, path, model, label_mean, label_scale, metadata, checkpoint_hash in checkpoints:
                with torch.no_grad():
                    output_pair = model(tensor)
                    logits = output_pair[0][0]
                    prob_t = torch.softmax(logits.flatten() / args.temperature, dim=0).reshape(64, 64)
                probability = prob_t.cpu().numpy().astype(np.float64)
                soft_xs = (np.arange(64, dtype=np.float64) + 0.5) * width / 64
                soft_ys = (np.arange(64, dtype=np.float64) + 0.5) * height / 64
                soft_uv = np.asarray([probability.sum(axis=0) @ soft_xs,
                                      probability.sum(axis=1) @ soft_ys], dtype=np.float64)
                soft_xyz = helper.unproject_to_plane(soft_uv, plane_z, camera, intrinsics)
                peak_y, peak_x = np.unravel_index(int(np.argmax(probability)), probability.shape)
                peak_uv = np.asarray([(peak_x + 0.5) * width / 64,
                                      (peak_y + 0.5) * height / 64], dtype=np.float64)
                peak_xyz = helper.unproject_to_plane(peak_uv, plane_z, camera, intrinsics)
                radius = max(0, int(args.local_radius_cells))
                y0p, y1p = max(0, peak_y - radius), min(64, peak_y + radius + 1)
                x0p, x1p = max(0, peak_x - radius), min(64, peak_x + radius + 1)
                patch = probability[y0p:y1p, x0p:x1p]
                patch_mass = float(patch.sum())
                patch = patch / max(patch_mass, 1e-30)
                xs = (np.arange(x0p, x1p, dtype=np.float64) + 0.5) * width / 64
                ys = (np.arange(y0p, y1p, dtype=np.float64) + 0.5) * height / 64
                local_uv = np.asarray([patch.sum(axis=0) @ xs, patch.sum(axis=1) @ ys])
                local_xyz = helper.unproject_to_plane(local_uv, plane_z, camera, intrinsics)
                entropy = float(-(probability * np.log(np.maximum(probability, 1e-30))).sum())
                decoded = {
                    "global_softargmax": (soft_xyz[:2], soft_uv),
                    "heatmap_argmax": (peak_xyz[:2], peak_uv),
                    "local_softargmax_r2": (local_xyz[:2], local_uv),
                }
                for decoder in decoder_names:
                    position_xy, uv = decoded[decoder]
                    rows.append({
                        "episode_id": eid,
                        "seed": int(info["seed"]),
                        "speed_m_s": float(info["speed_m_s"]),
                        "launch_y_m": float(info["launch_y_m"]),
                        "cutoff_s": cutoff,
                        "first_contact_s": contact_s,
                        "cutoff_to_contact_s": contact_s - cutoff,
                        "checkpoint": name,
                        "checkpoint_sha256": checkpoint_hash,
                        "decoder": decoder,
                        "gt_x_m": float(gt_position[0]), "gt_y_m": float(gt_position[1]),
                        "pred_x_m": float(position_xy[0]), "pred_y_m": float(position_xy[1]),
                        "position_xy_error_mm": float(np.linalg.norm(position_xy - gt_position[:2]) * 1000.0),
                        "pred_u_px": float(uv[0]), "pred_v_px": float(uv[1]),
                        "gt_u_px": float(gt_uv[0]), "gt_v_px": float(gt_uv[1]),
                        "peak_u_px": float(peak_uv[0]), "peak_v_px": float(peak_uv[1]),
                        "peak_pixel_distance_to_gt": float(np.linalg.norm(peak_uv - gt_uv)),
                        "heatmap_peak_probability": float(probability[peak_y, peak_x]),
                        "heatmap_entropy_normalized": entropy / math.log(probability.size),
                        "heatmap_effective_cells": math.exp(entropy),
                        "heatmap_probability_in_target_roi": helper.region_mass(probability, roi, width, height),
                        "local_patch_probability_mass": patch_mass,
                        "event_count_25ms": int(len(window_events)),
                        "target_roi_event_count_25ms": int(roi_mask.sum()),
                        "target_roi_event_fraction": float(roi_mask.mean()) if len(window_events) else 0.0,
                        "gt_label_method": "np.interp(saved_state_control_trace at causal cutoff; post-inference only)",
                    })

    if not rows:
        raise ValueError("no pre-contact rows evaluated")
    fields = list(rows[0])
    csv_path = output / "shared_direct_hdr_decoder_queries.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    try:
        run_code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.code_root, text=True).strip()
    except Exception:
        run_code_commit = "unknown"
    summary = {
        "status": "completed",
        "purpose": "paired decoder-only development diagnosis on the same nine shared Direct-HDR episodes; not a ranking or closed-loop result",
        "run_code_commit": run_code_commit,
        "diagnostic_script_sha256": sha256(Path(__file__).resolve()),
        "heatmap_helper_sha256": sha256(args.heatmap_helpers),
        "runtime_s": time.perf_counter() - started,
        "device": "cpu",
        "torch_threads": args.torch_threads,
        "query_contract": {
            "source_poll_rows": "Ours poll schedule, verified identical to V2E poll cutoffs episode by episode",
            "phase": "cutoff_s < episode first_contact_s",
            "cutoff": "existing common-reference poll cutoff (poll_s minus 10 ms latency)",
            "event_window": "causal [cutoff_s - 25 ms, cutoff_s), five 5 ms bins",
            "gt_label": "saved state trace linearly interpolated at cutoff after event-only inference, matching the common-reference evaluator",
            "fixed_height_m": plane_z,
            "gt_boundary": "GT and projected cube ROI are scoring/diagnostic only; neither enters the event tensor or decoder",
            "decoders": list(decoder_names),
            "local_decoder": f"soft-argmax restricted to {2 * args.local_radius_cells + 1}x{2 * args.local_radius_cells + 1} cells around global peak",
        },
        "inputs": {
            "reference_root": str(args.reference_root),
            "reference_summary_sha256": sha256(args.reference_summary),
            "polls_csv_sha256": sha256(args.polls),
            "scene_config_sha256": sha256(args.scene_config),
            "scene_config_path": str(args.scene_config),
            "episode_inputs": episode_inputs,
            "checkpoints": [{"name": name, "path": str(path), "sha256": checkpoint_hash,
                             "architecture": metadata.get("architecture")}
                            for name, path, _, _, _, metadata, checkpoint_hash in checkpoints],
        },
        "row_count": len(rows),
        "query_count": len({(r["episode_id"], r["cutoff_s"]) for r in rows}),
        "summary": summarize(rows),
        "output_csv": csv_path.name,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in (csv_path, summary_path)),
        encoding="utf-8")
    print(json.dumps({"status": "completed", "runtime_s": summary["runtime_s"],
                      "rows": len(rows), "queries": summary["query_count"],
                      "output": str(output), "summary": summary["summary"]["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
