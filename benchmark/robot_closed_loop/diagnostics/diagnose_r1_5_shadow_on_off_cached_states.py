#!/usr/bin/env python3
"""Matched shadow-on/off HDR render at immutable cached robot states.

This is a renderer ablation only: it restores already logged robot and target
poses, never advances physics, and runs the same ideal DVS response on both
rendered image sequences. The public VisOptions shadow flag is changed only by
the benchmark scene builder; the frozen event plugin is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.analyze_r1_5_cached_achromatic_shadow_signature import (  # noqa: E402
    floor_mask,
    run_ideal_dvs,
    sha256,
)
from benchmark.robot_closed_loop.robot_env import build_robot_scene  # noqa: E402
from genesis_event_plugin.radiometry import linear_luminance  # noqa: E402

DEFAULT_CACHE = ROOT / "benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059"
DEFAULT_CONFIG = ROOT / "benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"
DEFAULT_EXPERIMENT_CONFIG = ROOT / "benchmark/robot_closed_loop/configs/r1_5_shadow_on_off_cached_states_20261024h.json"
DEFAULT_OUTPUT = ROOT / "benchmark/robot_closed_loop/results/r1_5_shadow_on_off_cached_states_20261024h"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def exact_row(rows: list[dict[str, Any]], requested_time: float) -> dict[str, Any]:
    row = min(rows, key=lambda item: abs(float(item["time_s"]) - requested_time))
    actual = float(row["time_s"])
    if abs(actual - requested_time) > 1e-8:
        raise RuntimeError(f"requested cached time {requested_time:.9f}s missing; nearest is {actual:.9f}s")
    return row


def state_rows(cache_dir: Path, state_specs: list[dict[str, Any]]) -> tuple[list[str], np.ndarray, list[dict[str, Any]]]:
    with h5py.File(cache_dir / "same_state_source_cache.h5", "r") as f:
        source_times = np.asarray(f["timestamps_s"][:], dtype=np.float64)
    with h5py.File(cache_dir / "rendered_frame_evidence.h5", "r") as f:
        dense_times = np.asarray(f["dense_times_s"][:], dtype=np.float64)
    trace = read_jsonl(cache_dir / "trace_window.jsonl")
    labels = [str(spec["label"]) for spec in state_specs]
    clock_times = {"source": source_times, "dense": dense_times}
    times = np.asarray([
        clock_times[str(spec["clock"])][np.argmin(np.abs(
            clock_times[str(spec["clock"])] - float(spec["time_s"])
        ))]
        for spec in state_specs
    ], dtype=np.float64)
    rows = [exact_row(trace, float(t)) for t in times]
    return labels, times, rows


def set_cached_state(env: Any, row: dict[str, Any]) -> None:
    env.robot.set_qpos(np.asarray(row["robot_qpos"], dtype=np.float32), zero_velocity=True)
    obj = row["object"]
    target_qpos = np.asarray(list(obj["position_m"]) + list(obj["orientation_wxyz"]), dtype=np.float32)
    env.target.set_qpos(target_qpos, zero_velocity=True)


def render_current_state(env: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hdr, depth, seg, _ = env.camera.render(
        rgb=True, depth=True, segmentation=True, radiance=True, force_render=True,
    )
    hdr = np.asarray(hdr, dtype=np.float32).copy()
    depth = np.asarray(depth, dtype=np.float32).copy()
    seg = np.asarray(seg, dtype=np.int32).copy()
    if not np.isfinite(hdr).all() or np.any(hdr < 0):
        raise RuntimeError("renderer returned non-finite or negative HDR radiance")
    return hdr, depth, seg


def render_condition(gs: Any, config: dict[str, Any], rows: list[dict[str, Any]],
                     shadow: bool, capture_repeats: int, warmup_passes: int) -> dict[str, Any]:
    build_started = time.perf_counter()
    env = build_robot_scene(
        config, dt_s=float(config["physics"]["dt_s"]),
        segmentation_level="link", diagnostic_shadow=shadow,
    )
    build_s = time.perf_counter() - build_started
    hdr_frames, depth_frames, seg_frames, repeat_metrics = [], [], [], []
    render_wall = 0.0
    try:
        for idx, row in enumerate(rows):
            set_cached_state(env, row)
            if idx == 0:
                # Initialize shadow maps and shader programs at a real cached state.
                for _ in range(warmup_passes):
                    render_current_state(env)
            t0 = time.perf_counter()
            captures = [render_current_state(env) for _ in range(capture_repeats)]
            render_wall += time.perf_counter() - t0
            hdr_frames.append(captures[-1][0])
            depth_frames.append(captures[-1][1])
            seg_frames.append(captures[-1][2])
            repeat_delta = np.abs(captures[0][0] - captures[-1][0])
            repeat_metrics.append({
                "state_time_s": float(row["time_s"]),
                "hdr_rgb_repeat_mae": float(repeat_delta.mean()),
                "hdr_rgb_repeat_p99_abs": float(np.percentile(repeat_delta, 99)),
                "hdr_rgb_repeat_max_abs": float(repeat_delta.max()),
            })
    finally:
        gs.destroy()
    return {
        "shadow": bool(shadow),
        "scene_build_wall_s": build_s,
        "captured_render_wall_s": render_wall,
        "render_calls_included_in_wall": len(rows) * capture_repeats,
        "same_state_repeat": repeat_metrics,
        "hdr_rgb": np.stack(hdr_frames),
        "depth": np.stack(depth_frames),
        "segmentation": np.stack(seg_frames),
    }


def log_luma(hdr_rgb: np.ndarray) -> np.ndarray:
    luma = np.stack([linear_luminance(frame) for frame in hdr_rgb]).astype(np.float64)
    return np.log(np.maximum(luma, 1e-6))


def stats(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean_abs": float(np.mean(np.abs(x))),
        "median_abs": float(np.median(np.abs(x))),
        "p95_abs": float(np.percentile(np.abs(x), 95)),
        "p99_abs": float(np.percentile(np.abs(x), 99)),
        "fraction_abs_gt_0_02_log": float(np.mean(np.abs(x) > 0.02)),
        "fraction_abs_gt_0_20_log": float(np.mean(np.abs(x) > 0.20)),
    }


def renderer_provenance(gs: Any) -> dict[str, Any]:
    package = Path(gs.__file__).resolve().parent
    candidates = [
        Path(gs.__file__).resolve(),
        Path(inspect.getsourcefile(gs.options.VisOptions)).resolve(),
        package / "vis/rasterizer.py",
        package / "vis/camera.py",
        package / "ext/pyrender/renderer.py",
        package / "ext/pyrender/offscreen.py",
    ]
    files = {}
    for path in candidates:
        if path.is_file():
            files[str(path)] = sha256(path)
    return {"genesis_package_root": str(package), "source_files_sha256": files}


def preview(path: Path, shadow_on: np.ndarray, shadow_off: np.ndarray,
            floor: np.ndarray, label: str) -> None:
    l_on = log_luma(shadow_on)
    l_off = log_luma(shadow_off)
    lo = float(np.percentile(np.concatenate([l_on[floor], l_off[floor]]), 2))
    hi = float(np.percentile(np.concatenate([l_on[floor], l_off[floor]]), 98))
    h, w = floor.shape
    panel = Image.new("RGB", (3 * w, h + 26), (20, 20, 20))
    draw = ImageDraw.Draw(panel)
    for i, (title, data) in enumerate((("shadow on", shadow_on), ("shadow off", shadow_off))):
        luma = log_luma(data)
        gray = np.clip((luma - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=-1)
        rgb[~floor] = np.asarray([40, 40, 40], dtype=np.uint8)
        panel.paste(Image.fromarray(rgb), (i * w, 26))
        draw.text((i * w + 5, 6), title, fill=(245, 245, 245))
    diff = np.abs(l_on - l_off)
    heat = np.clip(diff / 0.20 * 255, 0, 255).astype(np.uint8)
    rgb_diff = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_diff[..., 0] = heat
    rgb_diff[..., 1] = (heat * 0.15).astype(np.uint8)
    rgb_diff[~floor] = 40
    panel.paste(Image.fromarray(rgb_diff), (2 * w, 26))
    draw.text((2 * w + 5, 6), f"abs log-luma delta {label}", fill=(245, 245, 245))
    panel.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    config = json.loads(args.scene_config.read_text(encoding="utf-8"))
    experiment = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    episode_ids = [str(item) for item in experiment["episode_ids"]]
    if not episode_ids or any(item not in {"004", "005"} for item in episode_ids):
        raise ValueError("the diagnostic is restricted to cached development episodes 004 and 005")
    state_specs = experiment["states"]
    shadow_conditions = [bool(value) for value in experiment["shadow_conditions"]]
    capture_repeats = int(experiment["capture_repeats_per_state"])
    warmup_passes = int(experiment["warmup_passes_at_first_state"])
    if capture_repeats < 2 or warmup_passes < 1:
        raise ValueError("need at least two repeat captures and one renderer warmup")
    expected_response = {
        "mode": "physical_log", "positive_threshold": 0.2, "negative_threshold": 0.2,
        "noise": False, "refractory": False, "bandwidth_limit": False,
        "evaluation_floor_mask_applied_after_event_generation": True,
    }
    if experiment.get("event_response") != expected_response:
        raise ValueError("experiment event_response must match the fixed ideal DVS replay implementation")
    if int(experiment.get("physics_steps", -1)) != 0 or experiment.get("render_channels") != [
        "scene_linear_hdr", "depth", "segmentation",
    ]:
        raise ValueError("unexpected physics-step or render-channel contract")
    if shadow_conditions != [True, False]:
        raise ValueError("conditions must be ordered as shadow on then shadow off")

    args.output.mkdir(parents=True)
    per_episode = {}
    import genesis as gs

    try:
        for ep in episode_ids:
            cache_dir = args.cache_root / f"episode_{ep}"
            labels, times, rows = state_rows(cache_dir, state_specs)
            conditions = {}
            for shadow in shadow_conditions:
                gs.init(backend=gs.cpu, logging_level="warning")
                conditions["shadow_on" if shadow else "shadow_off"] = render_condition(
                    gs, config, rows, shadow=shadow,
                    capture_repeats=capture_repeats,
                    warmup_passes=warmup_passes,
                )
            on, off = conditions["shadow_on"], conditions["shadow_off"]
            hdr = np.stack([on["hdr_rgb"], off["hdr_rgb"]])
            depth = np.stack([on["depth"], off["depth"]])
            seg = np.stack([on["segmentation"], off["segmentation"]])
            if not np.array_equal(seg[0], seg[1]):
                raise RuntimeError(f"episode {ep}: shadow toggle changed segmentation")
            finite_depth = np.isfinite(depth[0]) & np.isfinite(depth[1])
            depth_delta = np.abs(depth[0][finite_depth] - depth[1][finite_depth])
            luma_log = np.stack([log_luma(hdr[0]), log_luma(hdr[1])])
            with h5py.File(cache_dir / "same_state_source_cache.h5", "r") as f:
                source_seg = f["segmentation"][:]
                source_depth = f["depth"][:]
            floor = floor_mask(source_seg, source_depth, seg.reshape((-1, *seg.shape[-2:])))
            floor_delta = luma_log[0] - luma_log[1]
            luma_by_condition = np.stack([
                np.stack([linear_luminance(frame) for frame in hdr[condition]])
                for condition in range(2)
            ]).astype(np.float64)
            all_pixels = np.ones(floor.shape, dtype=bool)
            events_on, counts_on = run_ideal_dvs(
                luma_by_condition[0], times, all_pixels,
                luma_by_condition[0, 0], float(times[0]),
            )
            events_off, counts_off = run_ideal_dvs(
                luma_by_condition[1], times, all_pixels,
                luma_by_condition[1, 0], float(times[0]),
            )
            def floor_events(events: np.ndarray) -> np.ndarray:
                if not len(events):
                    return events
                x = events[:, 1].astype(np.int64)
                y = events[:, 2].astype(np.int64)
                return events[floor[y, x]]

            floor_events_on = floor_events(events_on)
            floor_events_off = floor_events(events_off)
            np.savez_compressed(
                args.output / f"episode_{ep}_shadow_on_off_frames_and_events.npz",
                times_s=times, state_labels=np.asarray(labels), hdr_rgb=hdr,
                depth=depth, segmentation=seg, floor_mask=floor,
                shadow_on_events_full_frame=events_on,
                shadow_off_events_full_frame=events_off,
                shadow_on_floor_events=floor_events_on,
                shadow_off_floor_events=floor_events_off,
                shadow_on_event_counts=np.asarray(counts_on, dtype=np.int32),
                shadow_off_event_counts=np.asarray(counts_off, dtype=np.int32),
            )
            for i, label in enumerate(labels):
                preview(
                    args.output / f"episode_{ep}_{label}_shadow_on_off.png",
                    hdr[0, i], hdr[1, i], floor, label,
                )
            per_episode[ep] = {
                "input_sha256": {
                    name: sha256(cache_dir / name)
                    for name in ("same_state_source_cache.h5", "rendered_frame_evidence.h5", "trace_window.jsonl")
                },
                "state_labels": labels,
                "state_times_s": times.tolist(),
                "physics_advanced": False,
                "shadow_on_off_segmentation_identical": True,
                "shadow_on_off_depth_abs_max": float(depth_delta.max()) if len(depth_delta) else 0.0,
                "shadow_on_off_depth_abs_p99": float(np.percentile(depth_delta, 99)) if len(depth_delta) else 0.0,
                "common_fixed_floor_pixels": int(floor.sum()),
                "shadow_on_vs_off_log_luma_difference_by_state": {
                    label: stats(floor_delta[i][floor]) for i, label in enumerate(labels)
                },
                "fixed_floor_log_luma_temporal_change_by_condition": {
                    "shadow_on": [stats(v[floor]) for v in np.diff(luma_log[0], axis=0)],
                    "shadow_off": [stats(v[floor]) for v in np.diff(luma_log[1], axis=0)],
                },
                "ideal_dvs": {
                    "response": "same physical_log +/-0.2 threshold, zero noise/refractory/bandwidth limit",
                    "calls": len(times),
                    "shadow_on_events_per_call": counts_on,
                    "shadow_off_events_per_call": counts_off,
                    "shadow_on_total_full_frame_events": int(len(events_on)),
                    "shadow_off_total_full_frame_events": int(len(events_off)),
                    "shadow_on_fixed_floor_events": int(len(floor_events_on)),
                    "shadow_off_fixed_floor_events": int(len(floor_events_off)),
                },
                "shadow_on_repeat_metrics": on["same_state_repeat"],
                "shadow_off_repeat_metrics": off["same_state_repeat"],
                "shadow_on_scene_build_s": on["scene_build_wall_s"],
                "shadow_off_scene_build_s": off["scene_build_wall_s"],
                "shadow_on_captured_render_s": on["captured_render_wall_s"],
                "shadow_off_captured_render_s": off["captured_render_wall_s"],
            }
        report = {
            "purpose": "causal benchmark-level shadow toggle on the same immutable robot/target states",
            "scope": "same cached states, camera, geometry, materials, and HDR/DVS paths; only VisOptions.shadow differs",
            "execution_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "robot_env_sha256": sha256(ROOT / "benchmark/robot_closed_loop/robot_env.py"),
            "event_response_code_sha256": {
                "dvs_emulator": sha256(ROOT / "genesis_event_plugin/dvs_emulator.py"),
                "noise_model": sha256(ROOT / "genesis_event_plugin/noise_model.py"),
                "radiometry": sha256(ROOT / "genesis_event_plugin/radiometry.py"),
                "diagnostic_replay_helper": sha256(ROOT / "benchmark/robot_closed_loop/diagnostics/analyze_r1_5_cached_achromatic_shadow_signature.py"),
            },
            "scene_config_sha256": sha256(args.scene_config.resolve()),
            "experiment_config_sha256": sha256(args.experiment_config.resolve()),
            "experiment_config": str(args.experiment_config.resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "renderer_provenance": renderer_provenance(gs),
            "geometry_source": "only already logged qpos/object pose restored for offline rendering; no target GT, mask, or depth enters the DVS model",
            "limitations": [
                "Three actual cached times test endpoint/midpoint radiometry; the result is not the full ten-bin 1 kHz event trace.",
                "The archival renderer runtime identity is not reconstructed by this new run; it is a matched rerender with the pinned current overlay.",
                "A shadow effect in these two cached episodes does not establish that shadowing is the only cause of learned localization failure.",
            ],
            "episodes": per_episode,
        }
        (args.output / "shadow_on_off_summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        readme = """# R1.5 matched shadow-on/off cached-state render\n\nThis benchmark-layer diagnostic restores the exact logged robot and target states for episodes 004/005 and renders them with the same camera/material/physics config under Genesis `VisOptions.shadow=True` and `False`. Physics is not stepped. Each state is rendered twice after shader/shadow initialization to measure repeatability. Segmentation and depth are checked across conditions. The same ideal physical-log DVS response is run on the three actual start/middle/end HDR frames; this is an endpoint/midpoint diagnostic, not the full 1 kHz event trajectory.\n\nThe expected causal signature is reduced fixed-floor log-radiance variation and fewer/moved floor events when shadows are disabled, with identical segmentation/depth and target silhouette. DVS processes the full image with the same response in both conditions; the fixed-floor segmentation mask filters events only after generation for evaluator-only statistics. This does not identify the historical renderer binary or prove the cause of learned localization failure outside these cached clips. Renderer provenance hashes are recorded in the JSON.\n\nRun on the verified g1 robot runtime after checking `nvidia-smi` and selecting a free GPU. From the repository root:\n\n```bash\nROBOT_PYTHON="$HOME/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python"\nGPU_ID=${GPU_ID:?set to a confirmed free GPU ID}\nPYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES= EGL_DEVICE_ID="$GPU_ID" PYOPENGL_PLATFORM=egl \\\n  "$ROBOT_PYTHON" benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_shadow_on_off_cached_states.py \\\n  --cache-root benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059 \\\n  --scene-config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \\\n  --output benchmark/robot_closed_loop/results/r1_5_shadow_on_off_cached_states_20261024h\n```\n"""
        readme = readme.replace(
            "Run on the verified g1 robot runtime",
            "Experiment settings are pinned in `benchmark/robot_closed_loop/configs/r1_5_shadow_on_off_cached_states_20261024h.json` (the runner default). Run on the verified g1 robot runtime",
        )
        (args.output / "README.md").write_text(readme, encoding="utf-8")
        sums = []
        for path in sorted(args.output.iterdir()):
            if path.name == "SHA256SUMS":
                continue
            sums.append(f"{sha256(path)}  {path.name}")
        (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "episodes": per_episode}, indent=2))
    finally:
        if getattr(gs, "_initialized", False):
            gs.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
