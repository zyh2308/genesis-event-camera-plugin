#!/usr/bin/env python3
"""Compute the four frozen Phase 5 primary metrics for one completed run.

The script is intentionally boring: it reads the already archived Real,
Ours, and official V2E streams, clips only to the declared full-direction
interval, computes the four protocol metrics, writes descriptive LR/RL
tables, and creates qualitative (non-aggregate) 5 ms visualizations.  It does
not retune, align, resample, crop, or rerun either simulator.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.io.real_stream import discover_ecf_plugin, load_real_hdf5  # noqa: E402


CORE_COMMIT = "e79e656ac00f7847b98db9e12bf43b093f3afff1"
READINESS_COMMIT = "70db6c7117a4aa4d585d3a1882668eea82a18212"
RUNTIME_CHECK_COMMIT = "5bb3d079b64118de4d76570ef98ca37841c5acab"
PROTOCOL_SHA = "c6d0926147e3fb08b15892c1f9ee368c5d7380ad3326cca0b8d962ae1f78d763"
PAIR_SHA = "86c9596bb3c6e79788e1ae5aa627319ade886a64c88571f4ccc5e7ede6fbc175"
CHECKPOINT_SHA = "d4d3070431eff774a1c27038930121ffeae59ede3a7d184118c539b443ea27cb"
METRICS = (
    "event_rate_relative_error",
    "polarity_ratio_error",
    "active_pixel_ratio_error",
    "spatial_distribution_l1_8x8",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def read_hdf5_events(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "events" not in handle:
            raise RuntimeError(f"{path}: missing events dataset")
        dataset = handle["events"]
        raw = dataset[:]
        if raw.ndim != 2 or raw.shape[1] != 4:
            raise RuntimeError(f"{path}: unexpected events shape {raw.shape}")
    # Ours and the official V2E HDF5 outputs both use [t_us,x,y,p_01].
    return np.asarray(raw, dtype=np.float64)


def direction_stats(events_txyp_us: np.ndarray, duration_s: float, width: int, height: int):
    arr = np.asarray(events_txyp_us, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise RuntimeError(f"invalid event array shape: {arr.shape}")
    raw_count = int(len(arr))
    if raw_count:
        finite = np.isfinite(arr).all(axis=1)
        polarity_ok = np.isin(arr[:, 3], (0.0, 1.0))
        coord_ok = (arr[:, 1] >= 0) & (arr[:, 1] < width) & (arr[:, 2] >= 0) & (arr[:, 2] < height)
        timestamp_ok = arr[:, 0] >= 0
        invalid_mask = ~(finite & polarity_ok & coord_ok & timestamp_ok)
        invalid_count = int(invalid_mask.sum())
        monotonic = bool(np.all(np.diff(arr[:, 0]) >= 0))
        first_raw_s = float(arr[0, 0] / 1e6)
        last_raw_s = float(arr[-1, 0] / 1e6)
    else:
        invalid_count = 0
        monotonic = True
        first_raw_s = None
        last_raw_s = None
    require(invalid_count == 0, f"invalid event count is {invalid_count}")
    require(monotonic, "event timestamps are not monotonic")
    upper_us = float(duration_s) * 1e6
    in_interval = (arr[:, 0] >= 0) & (arr[:, 0] <= upper_us + 1e-9)
    tail = arr[arr[:, 0] > upper_us + 1e-9]
    clipped = arr[in_interval]
    clipped_count = int(len(clipped))
    on_count = int((clipped[:, 3] > 0).sum()) if clipped_count else 0
    off_count = int((clipped[:, 3] == 0).sum()) if clipped_count else 0
    if clipped_count:
        first_s = float(clipped[0, 0] / 1e6)
        last_s = float(clipped[-1, 0] / 1e6)
        active = int(len(np.unique(clipped[:, 1:3].astype(np.int64), axis=0)))
        grid = np.zeros((8, 8), dtype=np.float64)
        bx = np.minimum((clipped[:, 1].astype(np.int64) * 8) // width, 7)
        by = np.minimum((clipped[:, 2].astype(np.int64) * 8) // height, 7)
        np.add.at(grid, (by, bx), 1.0)
        spatial = (grid / grid.sum()).reshape(-1)
        polarity_ratio = on_count / clipped_count
        active_pixel_ratio = active / float(width * height)
    else:
        first_s = None
        last_s = None
        active = 0
        spatial = None
        polarity_ratio = None
        active_pixel_ratio = 0.0
    return {
        "raw_event_count": raw_count,
        "clipped_event_count": clipped_count,
        "padded_tail_event_count": int(len(tail)),
        "on_count": on_count,
        "off_count": off_count,
        "first_timestamp_s": first_s,
        "last_timestamp_s": last_s,
        "first_raw_timestamp_s": first_raw_s,
        "last_raw_timestamp_s": last_raw_s,
        "timestamps_monotonic": monotonic,
        "invalid_event_count": invalid_count,
        "coordinate_bounds_valid": True,
        "evaluation_interval_clip_count": raw_count - clipped_count,
        "event_rate_hz": clipped_count / float(duration_s),
        "polarity_ratio_on": polarity_ratio,
        "active_pixel_count": active,
        "active_pixel_ratio": active_pixel_ratio,
        "spatial_distribution_8x8": spatial.tolist() if spatial is not None else None,
    }


def errors(real: dict, sim: dict) -> dict:
    eps = 1e-12
    spatial = None
    if real["spatial_distribution_8x8"] is not None and sim["spatial_distribution_8x8"] is not None:
        spatial = float(np.abs(
            np.asarray(real["spatial_distribution_8x8"])
            - np.asarray(sim["spatial_distribution_8x8"])
        ).sum())
    return {
        "event_rate_relative_error": abs(sim["event_rate_hz"] - real["event_rate_hz"]) / max(real["event_rate_hz"], eps),
        "polarity_ratio_error": (
            abs(sim["polarity_ratio_on"] - real["polarity_ratio_on"])
            if sim["polarity_ratio_on"] is not None and real["polarity_ratio_on"] is not None else None
        ),
        "active_pixel_ratio_error": abs(sim["active_pixel_ratio"] - real["active_pixel_ratio"]),
        "spatial_distribution_l1_8x8": spatial,
    }


def _panel(events, duration_s, width, height, center_s, window_s=0.005):
    from PIL import Image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    lo = (center_s - window_s / 2.0) * 1e6
    hi = (center_s + window_s / 2.0) * 1e6
    mask = (events[:, 0] >= lo) & (events[:, 0] < hi)
    selected = events[mask]
    if len(selected):
        x = selected[:, 1].astype(np.int64)
        y = selected[:, 2].astype(np.int64)
        on = selected[:, 3] > 0
        image[y[on], x[on], 0] = 255
        image[y[on], x[on], 1] = 255
        image[y[~on], x[~on], 2] = 255
        image[y[~on], x[~on], 1] = 120
    return Image.fromarray(image, mode="RGB"), int(len(selected))


def qualitative_visualizations(root: Path, direction: str, duration_s: float, streams: dict, width: int, height: int):
    from PIL import Image, ImageDraw
    output = root / "visualizations" / direction
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for fraction in (0.25, 0.50, 0.75):
        center = fraction * duration_s
        panels = []
        for method in ("Real", "Ours", "V2E"):
            panel, count = _panel(streams[method], duration_s, width, height, center)
            panel = panel.resize((width // 2, height // 2), Image.Resampling.NEAREST)
            panels.append(panel)
            counts[f"{fraction:.2f}_{method}"] = count
        canvas = Image.new("RGB", (width // 2 * 3, height // 2 + 28), "black")
        draw = ImageDraw.Draw(canvas)
        for i, (method, panel) in enumerate(zip(("Real", "Ours", "V2E"), panels)):
            canvas.paste(panel, (i * width // 2, 28))
            draw.text((i * width // 2 + 4, 6), f"{method} | qualitative only", fill="white")
        canvas.save(output / f"fraction_{fraction:.2f}.png")
    return {"path": str(output), "window_duration_s": 0.005, "not_synchronized": True, "event_counts": counts}


def file_manifest(root: Path):
    entries = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"formal_run_manifest.json", "file_manifest.json"}):
        entries[str(path.relative_to(root))] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--real-lr", type=Path, required=True)
    parser.add_argument("--real-rl", type=Path, required=True)
    parser.add_argument("--v2e-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    approval_path = ROOT / "benchmark/FORMAL_RUN_APPROVAL_PHASE5.yaml"
    approval = yaml.safe_load(approval_path.read_text(encoding="utf-8"))
    require(approval.get("approved") is True and approval.get("formal_execution_authorized") is True, "Phase 5 approval is not active")
    require(sha256(ROOT / "benchmark/protocol_v3.yaml") == PROTOCOL_SHA, "protocol_v3 changed")
    require(sha256(ROOT / "benchmark/evaluation_pairs_v3.yaml") == PAIR_SHA, "pair manifest changed")
    require(git("status", "--porcelain") == "", "repository must be clean before formal metric creation")
    require(subprocess.run(["git", "-C", str(ROOT), "diff", "--exit-code", CORE_COMMIT, "--", "genesis_event_plugin/"]).returncode == 0, "frozen core changed")
    require(args.checkpoint.is_file() and sha256(args.checkpoint) == CHECKPOINT_SHA, "official checkpoint hash mismatch")
    require(args.real_lr.is_file() and args.real_rl.is_file(), "declared real sources are missing")

    protocol = yaml.safe_load((ROOT / "benchmark/protocol_v3.yaml").read_text(encoding="utf-8"))
    durations = {direction: float(protocol["real_sequences"][direction]["duration_s"]) for direction in ("LR", "RL")}
    paths = {
        "LR": {"real": args.real_lr, "genesis": run_root / "LR" / "ours" / "aer_events.h5", "v2e": run_root / "LR" / "v2e" / "aer_events.h5"},
        "RL": {"real": args.real_rl, "genesis": run_root / "RL" / "ours" / "aer_events.h5", "v2e": run_root / "RL" / "v2e" / "aer_events.h5"},
    }
    all_results = {}
    all_visuals = {}
    plugin_path = discover_ecf_plugin()
    require(plugin_path == "/usr/lib/x86_64-linux-gnu/hdf5/serial/plugins", f"unexpected ECF plugin path: {plugin_path}")
    for direction in ("LR", "RL"):
        duration_s = durations[direction]
        real_sequence = load_real_hdf5(paths[direction]["real"], direction=direction, sequence_start_us=0, duration_s_declared=duration_s)
        real_events = np.asarray(real_sequence.events_txyp_us, dtype=np.float64)
        ours_events = read_hdf5_events(paths[direction]["genesis"])
        v2e_events = read_hdf5_events(paths[direction]["v2e"])
        stream_stats = {
            "Real": direction_stats(real_events, duration_s, 1280, 720),
            "Ours": direction_stats(ours_events, duration_s, 1280, 720),
            "V2E": direction_stats(v2e_events, duration_s, 1280, 720),
        }
        all_results[direction] = {
            "duration_s": duration_s,
            "streams": stream_stats,
            "errors": {method: errors(stream_stats["Real"], stream_stats[method]) for method in ("Ours", "V2E")},
        }
        all_visuals[direction] = qualitative_visualizations(run_root, direction, duration_s, {"Real": real_events, "Ours": ours_events, "V2E": v2e_events}, 1280, 720)

    raw_dir = ROOT / "benchmark/results/raw"
    sequence_dir = ROOT / "benchmark/results/per_sequence"
    aggregate_dir = ROOT / "benchmark/results/aggregate"
    table_dir = ROOT / "benchmark/results/tables"
    for directory in (raw_dir, sequence_dir, aggregate_dir, table_dir):
        directory.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"formal_metrics_{args.run_id}.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["direction", "method"] + list(next(iter(next(iter(all_results.values()))["streams"].values())).keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for direction, result in all_results.items():
            for method, stats in result["streams"].items():
                row = {"direction": direction, "method": method, **stats}
                row["spatial_distribution_8x8"] = json.dumps(row["spatial_distribution_8x8"])
                writer.writerow(row)
    sequence_path = sequence_dir / f"formal_per_direction_{args.run_id}.csv"
    with sequence_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["direction", "metric", "real", "ours", "v2e", "ours_error", "v2e_error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for direction, result in all_results.items():
            for metric in METRICS:
                writer.writerow({
                    "direction": direction,
                    "metric": metric,
                    "real": result["streams"]["Real"][metric.replace("event_rate_relative_error", "event_rate_hz").replace("polarity_ratio_error", "polarity_ratio_on").replace("active_pixel_ratio_error", "active_pixel_ratio").replace("spatial_distribution_l1_8x8", "spatial_distribution_8x8")],
                    "ours": result["streams"]["Ours"][metric.replace("event_rate_relative_error", "event_rate_hz").replace("polarity_ratio_error", "polarity_ratio_on").replace("active_pixel_ratio_error", "active_pixel_ratio").replace("spatial_distribution_l1_8x8", "spatial_distribution_8x8")],
                    "v2e": result["streams"]["V2E"][metric.replace("event_rate_relative_error", "event_rate_hz").replace("polarity_ratio_error", "polarity_ratio_on").replace("active_pixel_ratio_error", "active_pixel_ratio").replace("spatial_distribution_l1_8x8", "spatial_distribution_8x8")],
                    "ours_error": result["errors"]["Ours"][metric],
                    "v2e_error": result["errors"]["V2E"][metric],
                })
    aggregate_path = aggregate_dir / f"formal_aggregate_{args.run_id}.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["metric", "ours_mean", "ours_std", "v2e_mean", "v2e_std"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in METRICS:
            ours_values = np.asarray([all_results[d]["errors"]["Ours"][metric] for d in ("LR", "RL")], dtype=np.float64)
            v2e_values = np.asarray([all_results[d]["errors"]["V2E"][metric] for d in ("LR", "RL")], dtype=np.float64)
            writer.writerow({"metric": metric, "ours_mean": float(np.mean(ours_values)), "ours_std": float(np.std(ours_values, ddof=1)), "v2e_mean": float(np.mean(v2e_values)), "v2e_std": float(np.std(v2e_values, ddof=1))})

    file_manifest_path = run_root / "file_manifest.json"
    file_manifest_path.write_text(json.dumps(file_manifest(run_root), indent=2), encoding="utf-8")
    manifest = {
        "run_id": args.run_id,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_head": git("rev-parse", "HEAD"),
        "frozen_core_commit": CORE_COMMIT,
        "runtime_check_commit": RUNTIME_CHECK_COMMIT,
        "phase4_7_final_readiness_status_commit": READINESS_COMMIT,
        "protocol": "benchmark/protocol_v3.yaml",
        "protocol_sha256": PROTOCOL_SHA,
        "pair_manifest": "benchmark/evaluation_pairs_v3.yaml",
        "pair_sha256": PAIR_SHA,
        "approval_record": str(approval_path),
        "config_hashes": json.loads((ROOT / "benchmark/configs/CONFIG_HASHES.json").read_text(encoding="utf-8")),
        "replay_sha256": sha256(ROOT / "benchmark/replays/checkerboard_translation_v3.yaml"),
        "real_ecf_plugin_path": plugin_path,
        "superslomo_checkpoint_sha256": CHECKPOINT_SHA,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "genesis_version": "recorded by formal Genesis replay process",
        "v2e_version": "1.5.1",
        "source_cadence_hz": 100.0,
        "source_dt_s": SOURCE_DT_S,
        "seeds": {"ours": 42, "v2e": 42},
        "evaluation": {d: {"evaluation_duration_s": durations[d], "generated_duration_s": all_results[d]["duration_s"] + (math.ceil(durations[d] / SOURCE_DT_S) * SOURCE_DT_S - durations[d]), "padding_duration_s": math.ceil(durations[d] / SOURCE_DT_S) * SOURCE_DT_S - durations[d]} for d in ("LR", "RL")},
        "formal_metrics": list(METRICS),
        "cd": "NO",
        "gd": "NO",
        "temporal_offset_search": "NO",
        "display_rgb_used_for_ours": "NO",
        "retuning": "NO",
        "results": {
            "raw": str(raw_path),
            "per_direction": str(sequence_path),
            "aggregate": str(aggregate_path),
            "visualizations": all_visuals,
        },
        "file_manifest": str(file_manifest_path),
        "formal_benchmark_executed": True,
    }
    (run_root / "formal_run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    status_path = table_dir / "FORMAL_BENCHMARK_STATUS.md"
    lines = [
        "# Phase 5 — Frozen Formal Quality Benchmark Status",
        "",
        "This is the first formal benchmark result. The run used the frozen protocol, pair manifest, Ours core, V2E parameters, replay, lighting, cadence, and seed. No retuning or alternate run was performed.",
        "",
        f"- run_id: `{args.run_id}`",
        f"- formal git commit: `{manifest['git_head']}`",
        f"- protocol SHA256: `{PROTOCOL_SHA}`",
        f"- pair SHA256: `{PAIR_SHA}`",
        f"- frozen core: `{CORE_COMMIT}`; integrity: PASS",
        "- formal benchmark executed: YES",
        "- CD/GD: NO; temporal offset search: NO; display RGB for Ours: NO; retuning: NO",
        "",
        "## Per-direction metrics",
        "",
        "| Direction | Metric | Ours error | V2E error | Lower error |",
        "|---|---|---:|---:|---|",
    ]
    for direction in ("LR", "RL"):
        for metric in METRICS:
            ours_error = all_results[direction]["errors"]["Ours"][metric]
            v2e_error = all_results[direction]["errors"]["V2E"][metric]
            lower = "Ours" if ours_error < v2e_error else "V2E" if v2e_error < ours_error else "tie"
            lines.append(f"| {direction} | {metric} | {ours_error:.9g} | {v2e_error:.9g} | {lower} |")
    lines.extend([
        "",
        "## Limitations and raw-data policy",
        "",
        "- LR and RL are one acquisition family; aggregate mean/std is descriptive only.",
        "- The real trajectory is not hardware-synchronized to Genesis; these are coarse full-direction distributional comparisons, not pixelwise fidelity claims.",
        "- The 5 ms fraction views are qualitative-only and not included in any aggregate.",
        "- All archived replay, Ours, V2E, logs, CLI/config records, and hashes are preserved under the external formal run root.",
        "- Any NA values, runtime anomalies, or padded-tail counts are recorded in the raw CSV; no post-result correction was applied.",
    ])
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "formal_metrics_executed": True, "run_id": args.run_id, "manifest": str(run_root / "formal_run_manifest.json"), "raw": str(raw_path), "per_direction": str(sequence_path), "aggregate": str(aggregate_path), "status_report": str(status_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
