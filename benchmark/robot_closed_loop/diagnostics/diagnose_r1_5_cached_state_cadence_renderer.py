#!/usr/bin/env python3
"""Matched 2 kHz vs archived sparse-state replay with an MSAA toggle.

Uses the same saved physical state trajectory for all conditions and does not
advance physics. It asks whether the cached-state renderer can reproduce the
archived online event source at 2 kHz, whether the archived 47-row cadence
reproduces the saved shadow-on event file, and whether the GL multisample state
changes either result. Segmentation and depth are evaluator-only controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

ARCHIVE_EVENTS_SHA256 = "b698817a8347ca48ad75015f0cb81c3bfc711958a60d3387d468c70456a7b2b6"
ARCHIVE_TRACE_SHA256 = "310837ea07ab2f819f590ccce91b3952dd9e85ea5005537bc1f4d1a339bff28c"
E_TRACE_SHA256 = "f0f556842029cd54b3fc082ee0e2848dd194f3322a8a9c0769c2c889a6f6ae5d"
E_EVENTS_SHA256 = "c46f7bc28504ed9f43e76c334533534291d0de64b0c0f9bd429fa1a49eab58d8"
SCENE_SHA256 = "aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a"
WINDOW_START_S = 0.5505
WINDOW_END_S = 0.5905


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stats(values: list[float] | np.ndarray) -> dict[str, float | None]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"mean": None, "median": None, "p95": None, "p99": None, "max": None}
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


def load_events(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive:
            raise KeyError(f"{path} is missing {key}; keys={archive.files}")
        events = np.asarray(archive[key], dtype=np.float64).reshape(-1, 4)
    return events


def event_match(a: np.ndarray, b: np.ndarray, tolerance_s: float = 0.0005) -> dict[str, Any]:
    groups_a: dict[tuple[int, int, int], list[float]] = {}
    groups_b: dict[tuple[int, int, int], list[float]] = {}
    for target, events in ((groups_a, a), (groups_b, b)):
        for t, x, y, p in events:
            target.setdefault((int(x), int(y), int(p)), []).append(float(t))
    matched = 0
    deltas: list[float] = []
    for key in groups_a.keys() & groups_b.keys():
        aa, bb = sorted(groups_a[key]), sorted(groups_b[key])
        i = j = 0
        while i < len(aa) and j < len(bb):
            d = aa[i] - bb[j]
            if abs(d) <= tolerance_s:
                matched += 1
                deltas.append(abs(d))
                i += 1
                j += 1
            elif d < 0:
                i += 1
            else:
                j += 1
    return {
        "events_a": int(len(a)), "events_b": int(len(b)), "matched": matched,
        "fraction_a_matched": None if not len(a) else matched / len(a),
        "fraction_b_matched": None if not len(b) else matched / len(b),
        "pixel_polarity_match_tolerance_s": tolerance_s,
        "matched_abs_time_error_ms": stats(np.asarray(deltas) * 1000.0),
    }


def crop_events(events: np.ndarray) -> np.ndarray:
    return events[(events[:, 0] >= WINDOW_START_S) & (events[:, 0] <= WINDOW_END_S)]


def gl_audit_summary(condition: dict[str, Any]) -> dict[str, Any]:
    records = condition["hdr_pass_gl_audit"]
    readbacks = condition["radiance_readback_audit"]
    first = records[0]
    return {
        "pass_count": len(records), "radiance_readback_count": len(readbacks),
        "actual_gl_multisample_enabled_values": sorted({bool(r["actual_gl_multisample_enabled"]) for r in records}),
        "sample_buffers_values": sorted({int(r["sample_buffers"]) for r in records}),
        "sample_count_values": sorted({int(r["samples"]) for r in records}),
        "draw_is_genesis_multisample_framebuffer_values": sorted({bool(r["draw_is_genesis_multisample_framebuffer"]) for r in records}),
        "first_pass_record": first,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-trace", type=Path, required=True)
    parser.add_argument("--e-trace-window", type=Path, required=True)
    parser.add_argument("--archive-events", type=Path, required=True)
    parser.add_argument("--e-shadow-on-events", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    for path in (args.archive_trace, args.e_trace_window, args.archive_events,
                 args.e_shadow_on_events, args.scene_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_hashes = {
        args.archive_trace: ARCHIVE_TRACE_SHA256,
        args.e_trace_window: E_TRACE_SHA256,
        args.archive_events: ARCHIVE_EVENTS_SHA256,
        args.e_shadow_on_events: E_EVENTS_SHA256,
        args.scene_config: SCENE_SHA256,
    }
    actual_hashes = {str(path.resolve()): sha256(path) for path in expected_hashes}
    for path, expected in expected_hashes.items():
        if sha256(path) != expected:
            raise RuntimeError(f"input hash mismatch: {path}; expected={expected}; actual={sha256(path)}")

    full_trace = read_trace(args.archive_trace)
    e_trace = read_trace(args.e_trace_window)
    dense_rows = [row for row in full_trace
                  if WINDOW_START_S - 1e-10 <= float(row["time_s"]) <= WINDOW_END_S + 1e-10]
    dense_rows.sort(key=lambda row: float(row["time_s"]))
    dense_times = [round(float(row["time_s"]), 9) for row in dense_rows]
    e_times = [round(float(row["time_s"]), 9) for row in e_trace]
    if len(dense_rows) != 81 or any(abs((b - a) - 0.0005) > 1e-9 for a, b in zip(dense_times, dense_times[1:])):
        raise RuntimeError(f"expected 81 exact 0.5ms saved states; got {len(dense_rows)}")
    row_map = {round(float(row["time_s"]), 9): row for row in dense_rows}
    if not e_times or e_times[0] != round(WINDOW_START_S, 9) or e_times[-1] != round(WINDOW_END_S, 9):
        raise RuntimeError("E trace does not span the same exact frame window")
    if any(t not in row_map for t in e_times):
        raise RuntimeError("an E cached-state time is absent from the exact 2kHz physical trace")
    sparse_rows = [row_map[t] for t in e_times]

    archive_events = crop_events(load_events(args.archive_events, "events_2000hz"))
    e_events = crop_events(load_events(args.e_shadow_on_events, "events"))
    import genesis as gs
    from benchmark.robot_closed_loop.diagnostics.diagnose_r1_5_shadow_multisample_cached_states import render_condition
    from benchmark.robot_closed_loop.diagnostics.diagnose_r1_ours_localization_cadence import generate_dense_direct

    scene_config = json.loads(args.scene_config.read_text(encoding="utf-8"))
    conditions: dict[str, dict[str, Any]] = {}
    condition_events: dict[str, np.ndarray] = {}
    timing: dict[str, Any] = {}
    started = time.perf_counter()
    for name, msaa, render_rows in (
        ("msaa_on_sparse47", True, sparse_rows),
        ("msaa_on_dense2khz", True, dense_rows),
        ("msaa_off_dense2khz", False, dense_rows),
    ):
        gs.init(backend=gs.cpu, logging_level="warning")
        conditions[name] = render_condition(
            gs, scene_config, render_rows, shadow=True, multisample=msaa,
            capture_repeats=2, warmup_passes=1,
        )
        render_times = [round(float(row["time_s"]), 9) for row in render_rows]
        frames = [{"time_s": t, "hdr_rgb": hdr}
                  for t, hdr in zip(render_times, conditions[name]["hdr_rgb"])]
        capture_t0 = time.perf_counter()
        condition_events[name], dvs_wall = generate_dense_direct(frames)
        dense_dvs_wall = time.perf_counter() - capture_t0
        timing[name] = {
            "scene_build_wall_s": conditions[name]["scene_build_wall_s"],
            "render_wall_s": conditions[name]["captured_render_wall_s"],
            "dvs_wall_s": dense_dvs_wall,
            "dvs_reported_wall_s": dvs_wall,
            "frames": len(render_rows),
            "frame_time_range_s": [render_times[0], render_times[-1]],
            "gl_audit": gl_audit_summary(conditions[name]),
        }
        print(json.dumps({"condition": name, "frames": len(render_rows),
                          "events_window": int(len(crop_events(condition_events[name]))),
                          "wall_s": timing[name]}, separators=(",", ":")), flush=True)

    segmentation_identical = bool(np.array_equal(conditions["msaa_on_dense2khz"]["segmentation"],
                                                  conditions["msaa_off_dense2khz"]["segmentation"]))
    depth_delta = np.abs(conditions["msaa_on_dense2khz"]["depth"].astype(np.float64) -
                         conditions["msaa_off_dense2khz"]["depth"].astype(np.float64))
    log_luma_on = np.stack([np.log(np.maximum(
        np.asarray(frame, dtype=np.float64)[..., 0] * 0.2126 +
        np.asarray(frame, dtype=np.float64)[..., 1] * 0.7152 +
        np.asarray(frame, dtype=np.float64)[..., 2] * 0.0722, 1e-6))
        for frame in conditions["msaa_on_dense2khz"]["hdr_rgb"]])
    log_luma_off = np.stack([np.log(np.maximum(
        np.asarray(frame, dtype=np.float64)[..., 0] * 0.2126 +
        np.asarray(frame, dtype=np.float64)[..., 1] * 0.7152 +
        np.asarray(frame, dtype=np.float64)[..., 2] * 0.0722, 1e-6))
        for frame in conditions["msaa_off_dense2khz"]["hdr_rgb"]])
    log_luma_delta = np.abs(log_luma_on - log_luma_off)

    comparisons: dict[str, Any] = {}
    sources = {"archive_online_2khz": archive_events, "E_cached_sparse": e_events}
    for generated_name, generated in condition_events.items():
        generated_window = crop_events(generated)
        comparisons[generated_name] = {
            source: event_match(reference, generated_window)
            for source, reference in sources.items()
        }

    for name, condition in conditions.items():
        frame_times = e_times if "sparse47" in name else dense_times
        np.savez_compressed(out / f"frames_{name}.npz",
                            times_s=np.asarray(frame_times),
                            hdr_rgb=np.asarray(condition["hdr_rgb"], dtype=np.float32))
    for name, events in condition_events.items():
        np.savez_compressed(out / f"events_{name}.npz", events=events)
    summary = {
        "status": "PASS",
        "experiment_id": out.name,
        "question": "Does cached-state rendering reproduce online 2kHz events, and is the archived-to-E mismatch caused by cadence or GL multisampling?",
        "physical_trace": {
            "source_full_trace_sha256": actual_hashes[str(args.archive_trace.resolve())],
            "full_exact_rows": len(full_trace), "window_exact_rows_2khz": len(dense_rows),
            "window_times_s": [dense_times[0], dense_times[-1]],
            "window_interval_s": stats(np.diff(np.asarray(dense_times)).tolist()),
            "physics_advanced": False,
            "E_sparse_rows": len(sparse_rows),
            "E_sparse_interval_s": stats(np.diff(np.asarray(e_times)).tolist()),
            "E_sparse_rows_subset_of_dense_state_rows": True,
        },
        "renderer": {
            "scene_config_sha256": actual_hashes[str(args.scene_config.resolve())],
            "shadow": True, "capture_repeats_per_state": 2, "warmup_passes": 1,
            "conditions": {name: timing[name]["gl_audit"] for name in timing},
            "segmentation_identical": segmentation_identical,
            "depth_abs_delta": stats(depth_delta.ravel()),
            "log_luma_abs_delta": stats(log_luma_delta.ravel()),
            "log_luma_fraction_over_0_2": float(np.mean(log_luma_delta > 0.2)),
        },
        "event_sources": {
            "archive_online_sha256": actual_hashes[str(args.archive_events.resolve())],
            "E_cached_shadow_on_sha256": actual_hashes[str(args.e_shadow_on_events.resolve())],
            "generated_artifact_hashes": {},
            "comparisons_same_pixel_polarity_500us": comparisons,
        },
        "cost": {"wall_s": time.perf_counter() - started, "by_condition": timing},
        "limitations": [
            "Saved-state replay holds renderer geometry to logged qpos; no new physics was stepped.",
            "The two cadence streams share the same dense HDR captures within each MSAA condition.",
            "Segmentation and depth are used only to audit renderer equivalence, never as model input.",
        ],
    }
    for path in sorted(out.glob("*.npz")):
        summary["event_sources"]["generated_artifact_hashes"][path.name] = {
            "sha256": sha256(path), "bytes": path.stat().st_size,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "physical_trace": summary["physical_trace"],
                      "comparisons": comparisons, "renderer": summary["renderer"],
                      "cost": summary["cost"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
