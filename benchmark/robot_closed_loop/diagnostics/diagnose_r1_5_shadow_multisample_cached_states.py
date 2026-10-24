#!/usr/bin/env python3
"""Matched shadow x multisample HDR render at immutable cached robot states.

This is a diagnostic-only renderer factorial. It restores logged Panda and
target poses, never advances physics, and applies the same ideal DVS response
to every direct-render condition. The benchmark temporarily overrides the
Genesis pyrender module's GL_MULTISAMPLE enable/disable calls only while it
renders this isolated camera; Genesis' framebuffer allocation and the frozen
event-camera core are unchanged.
"""

from __future__ import annotations

import argparse
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
from benchmark.robot_closed_loop.diagnostics.diagnose_r1_5_shadow_on_off_cached_states import (  # noqa: E402
    log_luma,
    renderer_provenance,
    set_cached_state,
    state_rows,
    stats,
)
from benchmark.robot_closed_loop.robot_env import build_robot_scene  # noqa: E402
from genesis_event_plugin.radiometry import linear_luminance  # noqa: E402

DEFAULT_CACHE = ROOT / "benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059"
DEFAULT_CONFIG = ROOT / "benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"
DEFAULT_EXPERIMENT_CONFIG = ROOT / "benchmark/robot_closed_loop/configs/r1_5_shadow_multisample_cached_states_20261025a.json"
DEFAULT_OUTPUT = ROOT / "benchmark/robot_closed_loop/results/r1_5_shadow_multisample_cached_states_20261025a"


def segmentation_contour_band(segmentation: np.ndarray) -> np.ndarray:
    """One-pixel evaluator band around all categorical silhouette edges."""
    masks = np.asarray(segmentation, dtype=np.int32)
    if masks.ndim != 3:
        raise ValueError("segmentation must have shape (states, height, width)")
    h, w = masks.shape[-2:]
    boundary = np.zeros((h, w), dtype=bool)
    for image in masks:
        boundary[:, 1:] |= image[:, 1:] != image[:, :-1]
        boundary[:, :-1] |= image[:, 1:] != image[:, :-1]
        boundary[1:, :] |= image[1:, :] != image[:-1, :]
        boundary[:-1, :] |= image[1:, :] != image[:-1, :]
    padded = np.pad(boundary, 1, constant_values=False)
    band = np.zeros_like(boundary)
    for dy in range(3):
        for dx in range(3):
            band |= padded[dy:dy + h, dx:dx + w]
    return band


def event_region_summary(events: np.ndarray, region: np.ndarray) -> dict[str, Any]:
    if not len(events):
        return {"events_total": 0, "events_in_region": 0, "event_fraction_in_region": None}
    x = events[:, 1].astype(np.int64)
    y = events[:, 2].astype(np.int64)
    valid = (x >= 0) & (x < region.shape[1]) & (y >= 0) & (y < region.shape[0])
    in_region = np.zeros(len(events), dtype=bool)
    in_region[valid] = region[y[valid], x[valid]]
    count = int(in_region.sum())
    return {
        "events_total": int(len(events)),
        "events_in_region": count,
        "event_fraction_in_region": float(count / len(events)),
    }


def _sample_positions(GL: Any, sample_count: int) -> list[list[float]]:
    positions = []
    for index in range(max(0, min(int(sample_count), 16))):
        try:
            value = GL.glGetMultisamplefv(GL.GL_SAMPLE_POSITION, index)
        except TypeError:
            value = None
        array = np.asarray(value, dtype=np.float64).reshape(-1) if value is not None else np.empty(0)
        if array.size < 2:
            output = np.zeros(2, dtype=np.float32)
            GL.glGetMultisamplefv(GL.GL_SAMPLE_POSITION, index, output)
            array = output.astype(np.float64)
        positions.append(array[:2].tolist())
    return positions


class MultisamplePassAudit:
    """Force a requested multisample state and record the actual HDR pass."""

    def __init__(self, gs: Any, camera: Any, enabled: bool):
        self.camera = camera
        self.enabled = bool(enabled)
        self.records: list[dict[str, Any]] = []
        self.readback_records: list[dict[str, Any]] = []
        self.blit_records: list[dict[str, Any]] = []
        self._renderer = None
        self._target = None
        self._original_enable = None
        self._original_disable = None
        self._original_forward = None

    def __enter__(self) -> "MultisamplePassAudit":
        from genesis.ext.pyrender import renderer as pyrender_renderer
        from genesis.ext.pyrender.constants import RenderFlags
        from OpenGL import GL

        rasterizer = getattr(self.camera, "_rasterizer", None)
        if rasterizer is None:
            raise RuntimeError("camera has no Genesis rasterizer")
        target = rasterizer._camera_targets.get(self.camera.uid)
        if target is None:
            raise RuntimeError("camera has no registered Genesis rasterizer target")
        self._renderer = pyrender_renderer
        self._target = target
        self._original_enable = pyrender_renderer.glEnable
        self._original_disable = pyrender_renderer.glDisable
        self._original_blit = pyrender_renderer.glBlitFramebuffer
        self._original_forward = target._forward_pass
        self._original_readback = target._read_main_framebuffer
        enabled = self.enabled

        def set_multisample(cap: Any, *args: Any, **kwargs: Any) -> Any:
            if int(cap) == int(GL.GL_MULTISAMPLE):
                if enabled:
                    return self._original_enable(cap, *args, **kwargs)
                return self._original_disable(cap, *args, **kwargs)
            return self._original_enable(cap, *args, **kwargs)

        def clear_multisample(cap: Any, *args: Any, **kwargs: Any) -> Any:
            if int(cap) == int(GL.GL_MULTISAMPLE):
                if enabled:
                    return self._original_enable(cap, *args, **kwargs)
                return self._original_disable(cap, *args, **kwargs)
            return self._original_disable(cap, *args, **kwargs)

        original_forward = self._original_forward
        original_readback = self._original_readback

        def audited_blit(*args: Any, **kwargs: Any) -> Any:
            record = {
                "read_framebuffer_before_blit": int(GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING)),
                "draw_framebuffer_before_blit": int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING)),
                "arguments": [int(value) for value in args],
            }
            result = self._original_blit(*args, **kwargs)
            self.blit_records.append(record)
            return result

        def audited_readback(scene: Any, flags: Any):
            first_blit = len(self.blit_records)
            result = original_readback(scene, flags)
            if flags & RenderFlags.RADIANCE:
                self.readback_records.append({
                    "render_flags_int": int(flags),
                    "radiance_readback": True,
                    "expected_msaa_source_fbo": int(target._main_fb_ms),
                    "expected_resolve_destination_fbo": int(target._main_fb),
                    "blit_calls_during_readback": self.blit_records[first_blit:],
                    "returned_attachment_shapes": [list(np.asarray(value).shape) for value in result],
                })
            return result

        def audited_forward(scene: Any, flags: Any, seg_node_map: Any = None, env_idx: int = -1):
            result = original_forward(scene, flags, seg_node_map=seg_node_map, env_idx=env_idx)
            if flags & RenderFlags.RADIANCE:
                draw_fbo = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
                sample_buffers = int(GL.glGetIntegerv(GL.GL_SAMPLE_BUFFERS))
                sample_count = int(GL.glGetIntegerv(GL.GL_SAMPLES))
                max_samples = int(GL.glGetIntegerv(GL.GL_MAX_SAMPLES))
                self.records.append({
                    "requested_multisample_enabled": enabled,
                    "actual_gl_multisample_enabled": bool(GL.glIsEnabled(GL.GL_MULTISAMPLE)),
                    "render_flags_int": int(flags),
                    "draw_framebuffer": draw_fbo,
                    "multisample_framebuffer": int(target._main_fb_ms),
                    "draw_is_genesis_multisample_framebuffer": draw_fbo == int(target._main_fb_ms),
                    "sample_buffers": sample_buffers,
                    "samples": sample_count,
                    "max_samples_supported": max_samples,
                    "sample_positions": _sample_positions(GL, sample_count),
                })
            return result

        pyrender_renderer.glEnable = set_multisample
        pyrender_renderer.glDisable = clear_multisample
        pyrender_renderer.glBlitFramebuffer = audited_blit
        target._forward_pass = audited_forward
        target._read_main_framebuffer = audited_readback
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._target is not None and self._original_forward is not None:
            self._target._forward_pass = self._original_forward
            self._target._read_main_framebuffer = self._original_readback
        if self._renderer is not None:
            self._renderer.glEnable = self._original_enable
            self._renderer.glDisable = self._original_disable
            self._renderer.glBlitFramebuffer = self._original_blit


def render_condition(
    gs: Any,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    shadow: bool,
    multisample: bool,
    capture_repeats: int,
    warmup_passes: int,
) -> dict[str, Any]:
    condition_started = time.perf_counter()
    build_started = time.perf_counter()
    env = build_robot_scene(
        config, dt_s=float(config["physics"]["dt_s"]),
        segmentation_level="link", diagnostic_shadow=shadow,
    )
    build_s = time.perf_counter() - build_started
    hdr_frames, depth_frames, seg_frames, repeat_metrics = [], [], [], []
    render_wall = 0.0
    warmup_wall = 0.0
    audit = MultisamplePassAudit(gs, env.camera, enabled=multisample)
    try:
        with audit:
            for idx, row in enumerate(rows):
                set_cached_state(env, row)
                if idx == 0:
                    warmup_started = time.perf_counter()
                    for _ in range(warmup_passes):
                        env.camera.render(rgb=True, depth=True, segmentation=True, radiance=True, force_render=True)
                    warmup_wall += time.perf_counter() - warmup_started
                t0 = time.perf_counter()
                captures = [
                    env.camera.render(rgb=True, depth=True, segmentation=True, radiance=True, force_render=True)
                    for _ in range(capture_repeats)
                ]
                render_wall += time.perf_counter() - t0
                hdr_captures = [np.asarray(capture[0], dtype=np.float32).copy() for capture in captures]
                depth_captures = [np.asarray(capture[1], dtype=np.float32).copy() for capture in captures]
                seg_captures = [np.asarray(capture[2], dtype=np.int32).copy() for capture in captures]
                for hdr in hdr_captures:
                    if not np.isfinite(hdr).all() or np.any(hdr < 0):
                        raise RuntimeError("renderer returned non-finite or negative HDR radiance")
                hdr_frames.append(hdr_captures[-1])
                depth_frames.append(depth_captures[-1])
                seg_frames.append(seg_captures[-1])
                repeat_delta = np.abs(hdr_captures[0] - hdr_captures[-1])
                repeat_metrics.append({
                    "state_time_s": float(row["time_s"]),
                    "hdr_rgb_repeat_mae": float(repeat_delta.mean()),
                    "hdr_rgb_repeat_p99_abs": float(np.percentile(repeat_delta, 99)),
                    "hdr_rgb_repeat_max_abs": float(repeat_delta.max()),
                })
    finally:
        gs.destroy()
    condition_wall = time.perf_counter() - condition_started

    expected_passes = len(rows) * capture_repeats + warmup_passes
    if len(audit.records) != expected_passes:
        raise RuntimeError(f"captured {len(audit.records)} HDR pass audits, expected {expected_passes}")
    if len(audit.readback_records) != expected_passes:
        raise RuntimeError(f"captured {len(audit.readback_records)} radiance readbacks, expected {expected_passes}")
    if any(len(item["blit_calls_during_readback"]) != 1 for item in audit.readback_records):
        raise RuntimeError("expected one multisample-to-single-sample resolve blit per radiance readback")
    if any(item["actual_gl_multisample_enabled"] != bool(multisample) for item in audit.records):
        raise RuntimeError("requested multisample mode did not match the actual GL state")
    if any(not item["draw_is_genesis_multisample_framebuffer"] for item in audit.records):
        raise RuntimeError("HDR pass did not draw to Genesis' multisample framebuffer")
    if any(item["sample_buffers"] != 1 or item["samples"] < 1 for item in audit.records):
        raise RuntimeError("HDR pass did not use a multisample framebuffer")
    return {
        "shadow": bool(shadow),
        "requested_multisample_enabled": bool(multisample),
        "scene_build_wall_s": build_s,
        "warmup_render_wall_s": warmup_wall,
        "captured_render_wall_s": render_wall,
        "condition_wall_s_including_scene_build_warmup_and_destroy": condition_wall,
        "render_calls_included_in_wall": len(rows) * capture_repeats,
        "same_state_repeat": repeat_metrics,
        "hdr_pass_gl_audit": audit.records,
        "radiance_readback_audit": audit.readback_records,
        "framebuffer_blit_audit": audit.blit_records,
        "hdr_rgb": np.stack(hdr_frames),
        "depth": np.stack(depth_frames),
        "segmentation": np.stack(seg_frames),
    }


def condition_key(shadow: bool, multisample: bool) -> str:
    return f"shadow_{'on' if shadow else 'off'}_multisample_{'on' if multisample else 'off'}"


def factorial_preview(path: Path, frames: list[np.ndarray], floor: np.ndarray,
                      keys: list[str], label: str) -> None:
    logs = [log_luma(frame) for frame in frames]
    values = np.concatenate([image[floor] for image in logs])
    lo, hi = float(np.percentile(values, 2)), float(np.percentile(values, 98))
    h, w = floor.shape
    panel = Image.new("RGB", (2 * w, 2 * (h + 26)), (20, 20, 20))
    draw = ImageDraw.Draw(panel)
    for index, (key, luma) in enumerate(zip(keys, logs)):
        gray = np.clip((luma - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=-1)
        rgb[~floor] = np.asarray([40, 40, 40], dtype=np.uint8)
        x0 = (index % 2) * w
        y0 = (index // 2) * (h + 26)
        panel.paste(Image.fromarray(rgb), (x0, y0 + 26))
        draw.text((x0 + 5, y0 + 6), f"{label}: {key}", fill=(245, 245, 245))
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
    multisample_conditions = [bool(value) for value in experiment["multisample_conditions"]]
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
        raise ValueError("event_response must match the fixed ideal DVS replay implementation")
    if int(experiment.get("physics_steps", -1)) != 0 or experiment.get("render_channels") != [
        "scene_linear_hdr", "depth", "segmentation",
    ]:
        raise ValueError("unexpected physics-step or render-channel contract")
    if shadow_conditions != [True, False] or multisample_conditions != [True, False]:
        raise ValueError("factor levels must be ordered [on, off]")
    if experiment.get("multisample_contract", {}).get("allocation") != "leave Genesis framebuffer sample allocation unchanged":
        raise ValueError("must keep Genesis framebuffer sample allocation unchanged")

    experiment_started = time.perf_counter()
    args.output.mkdir(parents=True)
    per_episode: dict[str, Any] = {}
    import genesis as gs
    try:
        for ep in episode_ids:
            cache_dir = args.cache_root / f"episode_{ep}"
            labels, times, rows = state_rows(cache_dir, state_specs)
            conditions: dict[str, dict[str, Any]] = {}
            for shadow in shadow_conditions:
                for multisample in multisample_conditions:
                    gs.init(backend=gs.cpu, logging_level="warning")
                    key = condition_key(shadow, multisample)
                    conditions[key] = render_condition(
                        gs, config, rows, shadow=shadow, multisample=multisample,
                        capture_repeats=capture_repeats, warmup_passes=warmup_passes,
                    )

            keys = [condition_key(s, m) for s in shadow_conditions for m in multisample_conditions]
            hdr = np.stack([conditions[key]["hdr_rgb"] for key in keys])
            depth = np.stack([conditions[key]["depth"] for key in keys])
            seg = np.stack([conditions[key]["segmentation"] for key in keys])
            if any(not np.array_equal(seg[0], seg[index]) for index in range(1, len(keys))):
                raise RuntimeError(f"episode {ep}: shadow/MSAA toggle changed segmentation")
            contour_band = segmentation_contour_band(seg[0])
            finite_depth = np.all(np.isfinite(depth), axis=0)
            depth_delta_by_condition = {
                key: float(np.abs(depth[idx][finite_depth] - depth[0][finite_depth]).max())
                for idx, key in enumerate(keys)
            }
            with h5py.File(cache_dir / "same_state_source_cache.h5", "r") as f:
                source_seg = f["segmentation"][:]
                source_depth = f["depth"][:]
            floor = floor_mask(source_seg, source_depth, seg.reshape((-1, *seg.shape[-2:])))
            log_luma_by_condition = np.stack([log_luma(hdr[idx]) for idx in range(len(keys))])
            luma_by_condition = np.stack([
                np.stack([linear_luminance(frame) for frame in hdr[idx]]) for idx in range(len(keys))
            ]).astype(np.float64)
            all_pixels = np.ones(floor.shape, dtype=bool)
            event_outputs: dict[str, dict[str, Any]] = {}
            for idx, key in enumerate(keys):
                events, counts = run_ideal_dvs(
                    luma_by_condition[idx], times, all_pixels,
                    luma_by_condition[idx, 0], float(times[0]),
                )
                x = events[:, 1].astype(np.int64) if len(events) else np.empty(0, dtype=np.int64)
                y = events[:, 2].astype(np.int64) if len(events) else np.empty(0, dtype=np.int64)
                floor_events = events[floor[y, x]] if len(events) else events
                event_outputs[key] = {
                    "events": events,
                    "floor_events": floor_events,
                    "counts": counts,
                    "contour_event_summary": event_region_summary(events, contour_band),
                }

            npz_fields: dict[str, Any] = {
                "times_s": times,
                "state_labels": np.asarray(labels),
                "condition_keys": np.asarray(keys),
                "hdr_rgb": hdr,
                "depth": depth,
                "segmentation": seg,
                "floor_mask": floor,
            }
            for key in keys:
                npz_fields[f"{key}_events_full_frame"] = event_outputs[key]["events"]
                npz_fields[f"{key}_floor_events"] = event_outputs[key]["floor_events"]
                npz_fields[f"{key}_event_counts"] = np.asarray(event_outputs[key]["counts"], dtype=np.int32)
                npz_fields[f"{key}_contour_band"] = contour_band
            np.savez_compressed(args.output / f"episode_{ep}_shadow_multisample_frames_and_events.npz", **npz_fields)

            condition_metrics = {}
            for idx, key in enumerate(keys):
                condition_metrics[key] = {
                    "shadow": conditions[key]["shadow"],
                    "requested_multisample_enabled": conditions[key]["requested_multisample_enabled"],
                    "actual_gl_multisample_state_per_hdr_pass": [x["actual_gl_multisample_enabled"] for x in conditions[key]["hdr_pass_gl_audit"]],
                    "sample_buffers_per_hdr_pass": [x["sample_buffers"] for x in conditions[key]["hdr_pass_gl_audit"]],
                    "samples_per_hdr_pass": [x["samples"] for x in conditions[key]["hdr_pass_gl_audit"]],
                    "max_samples_supported_per_hdr_pass": [x["max_samples_supported"] for x in conditions[key]["hdr_pass_gl_audit"]],
                    "sample_positions_per_hdr_pass": [x["sample_positions"] for x in conditions[key]["hdr_pass_gl_audit"]],
                    "draw_fbo_is_genesis_multisample_fbo_per_hdr_pass": [x["draw_is_genesis_multisample_framebuffer"] for x in conditions[key]["hdr_pass_gl_audit"]],
                    "depth_abs_max_vs_first_condition": depth_delta_by_condition[key],
                    "fixed_floor_events": int(len(event_outputs[key]["floor_events"])),
                    "silhouette_contour_band_pixels": int(contour_band.sum()),
                    "silhouette_contour_events": event_outputs[key]["contour_event_summary"],
                    "full_frame_events": int(len(event_outputs[key]["events"])),
                    "events_per_actual_interval": [int(x) for x in event_outputs[key]["counts"]],
                    "scene_build_wall_s": conditions[key]["scene_build_wall_s"],
                    "warmup_render_wall_s": conditions[key]["warmup_render_wall_s"],
                    "captured_render_wall_s": conditions[key]["captured_render_wall_s"],
                    "condition_wall_s_including_scene_build_warmup_and_destroy": conditions[key]["condition_wall_s_including_scene_build_warmup_and_destroy"],
                    "render_calls_included_in_wall": conditions[key]["render_calls_included_in_wall"],
                    "same_state_repeat": conditions[key]["same_state_repeat"],
                    "floor_log_luma_by_state": [stats(v[floor]) for v in log_luma_by_condition[idx]],
                }
            for state_idx, label in enumerate(labels):
                factorial_preview(
                    args.output / f"episode_{ep}_{label}_shadow_multisample_factorial.png",
                    [hdr[idx, state_idx] for idx in range(len(keys))], floor, keys, label,
                )

            factorial = {"shadow_effect_at_fixed_multisample": {}, "multisample_effect_at_fixed_shadow": {}}
            for ms in (True, False):
                on_key, off_key = condition_key(True, ms), condition_key(False, ms)
                factorial["shadow_effect_at_fixed_multisample"]["on" if ms else "off"] = {
                    label: stats((log_luma_by_condition[keys.index(on_key), i] - log_luma_by_condition[keys.index(off_key), i])[floor])
                    for i, label in enumerate(labels)
                }
            for shadow in (True, False):
                on_key, off_key = condition_key(shadow, True), condition_key(shadow, False)
                factorial["multisample_effect_at_fixed_shadow"]["on" if shadow else "off"] = {
                    label: stats((log_luma_by_condition[keys.index(on_key), i] - log_luma_by_condition[keys.index(off_key), i])[floor])
                    for i, label in enumerate(labels)
                }

            per_episode[ep] = {
                "input_sha256": {
                    name: sha256(cache_dir / name)
                    for name in ("same_state_source_cache.h5", "rendered_frame_evidence.h5", "trace_window.jsonl")
                },
                "state_labels": labels,
                "state_times_s": times.tolist(),
                "physics_advanced": False,
                "segmentation_identical_across_factorial": True,
                "factorial_metrics": factorial,
                "conditions": condition_metrics,
            }

        report = {
            "purpose": "matched direct-HDR shadow x GL multisample factorial on immutable cached robot/target states",
            "scope": "same cached states, camera, geometry, material, DVS replay; factors are VisOptions.shadow and actual GL_MULTISAMPLE state on the existing Genesis HDR multisample FBO",
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
            "experiment_wall_s_through_render_analysis_and_manifest_preparation": time.perf_counter() - experiment_started,
            "multisample_interpretation": "Genesis allocates a multisample HDR FBO with min(GL_MAX_SAMPLES, 4) samples. This diagnostic toggles GL_MULTISAMPLE during the HDR draw but intentionally does not reallocate the FBO. Actual draw FBO, sample-buffer count, sample count, GL enable state, and implementation sample positions are logged for every radiance pass.",
            "contour_control": "one-pixel dilation of categorical segmentation boundaries unioned across the three actual cached states; evaluator-only for event-location summaries and never passed to DVS generation",
            "geometry_source": "only already logged qpos/object pose restored for offline rendering; no target GT, mask, or depth enters DVS generation",
            "limitations": [
                "Three actual cached times are an endpoint/midpoint radiometry control, not a full 1 kHz trace.",
                "This first factorial rerenders Direct-HDR states; candidate-versus-reference reconstruction comparisons must use the same factor settings and retain the official default reference unchanged.",
                "The archival renderer binary is not reconstructed; this is a matched rerender with the pinned g1 overlay.",
                "Only the two cached development episodes are used; no independent test or downstream learning result is implied.",
            ],
            "episodes": per_episode,
        }
        (args.output / "shadow_multisample_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        readme = """# R1.5 matched shadow x multisample cached-state render

This diagnostic restores exact logged Panda and target states from cached development episodes 004 and 005. Physics is not stepped. It renders actual start, middle, and end timestamps under the four `shadow on/off x GL_MULTISAMPLE on/off` conditions. The same ideal physical-log DVS response is applied to each condition.

Genesis allocates an HDR multisample framebuffer using `min(GL_MAX_SAMPLES, 4)`. The experiment changes only the actual `GL_MULTISAMPLE` capability during the HDR draw. It does not change framebuffer allocation or the radiance resolve/readback pass. Every HDR pass records actual GL enable state, draw FBO identity, sample-buffer count, sample count, and implementation sample positions. The segmentation/depth checks and fixed floor mask are evaluator-only; the DVS consumes the full rendered image.

This is a direct-render factorial to identify shadow and raster-multisample contributions. It is not yet candidate-versus-reference reconstruction fidelity, not a full 1 kHz trace, and not a change to the official Direct-HDR reference. Any later candidate/reference comparison must use matching shadow and multisample settings within each pair.

Run on g1 only after confirming an idle GPU and checking ownership. From the repository root:

```bash
ROBOT_PYTHON="$HOME/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python"
GPU_ID=${GPU_ID:?set to a confirmed free GPU ID}
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES= EGL_DEVICE_ID="$GPU_ID" PYOPENGL_PLATFORM=egl \\
  "$ROBOT_PYTHON" benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_shadow_multisample_cached_states.py \\
  --cache-root benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059 \\
  --scene-config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \\
  --experiment-config benchmark/robot_closed_loop/configs/r1_5_shadow_multisample_cached_states_20261025a.json \\
  --output benchmark/robot_closed_loop/results/r1_5_shadow_multisample_cached_states_20261025a
```
"""
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
