#!/usr/bin/env python3
"""CPU-only spatial/temporal analysis of archived stationary-floor radiance.

This reuses the already archived Direct-HDR frames and event NPZs from the
2026-10-24 structural diagnosis. It does not initialize Genesis, render,
advance physics, train, or change any simulator/core code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059"
DEFAULT_EVIDENCE = ROOT / "benchmark/robot_closed_loop/results/r1_5_photometric_sampling_contract_20261024a/evidence"
DEFAULT_OUTPUT = ROOT / "benchmark/robot_closed_loop/results/r1_5_photometric_rootcause_cpu_20261024b"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _fixed_floor_mask(seg0: np.ndarray, seg1: np.ndarray,
                      dense_seg: np.ndarray, depth0: np.ndarray,
                      depth1: np.ndarray) -> np.ndarray:
    mask = (seg0 == 1) & (seg1 == 1) & np.all(dense_seg == 1, axis=0)
    for _ in range(2):
        padded = np.pad(mask, 1, constant_values=False)
        mask = np.logical_and.reduce([
            padded[y:y + mask.shape[0], x:x + mask.shape[1]]
            for y in range(3) for x in range(3)
        ])
    return mask & np.isfinite(depth0) & np.isfinite(depth1) & (np.abs(depth0 - depth1) < 1e-7)


def _counts_by_tile(active: np.ndarray, mask: np.ndarray, ny: int = 8, nx: int = 8) -> dict:
    h, w = mask.shape
    counts, densities = [], []
    for iy in range(ny):
        for ix in range(nx):
            ys = slice(iy * h // ny, (iy + 1) * h // ny)
            xs = slice(ix * w // nx, (ix + 1) * w // nx)
            denom = int(mask[ys, xs].sum())
            count = int(active[ys, xs].sum())
            counts.append(count)
            densities.append(count / max(denom, 1))
    return {
        "grid": [ny, nx],
        "active_pixel_count_q50_q90_q99_max": [
            float(v) for v in (*np.quantile(counts, [0.5, 0.9, 0.99]), max(counts))
        ],
        "active_fraction_q50_q90_q99_max": [
            float(v) for v in (*np.quantile(densities, [0.5, 0.9, 0.99]), max(densities))
        ],
    }


def _heatmap(values: np.ndarray, mask: np.ndarray, scale: float) -> Image.Image:
    q = np.clip(values / max(scale, 1e-12), 0.0, 1.0)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.uint8(q * 255)
    rgb[..., 1] = np.uint8(np.sqrt(q) * 96)
    rgb[..., 2] = np.where(mask, 0, 34).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _make_panel(ep: str, mid_rgb: np.ndarray, residual: np.ndarray,
                on: np.ndarray, off: np.ndarray, mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    panel = Image.new("RGB", (w * 3, h + 26), (22, 22, 22))
    draw = ImageDraw.Draw(panel)
    draw.text((4, 5), f"episode {ep}: cached HDR at 0.5855 s", fill=(240, 240, 240))
    # Display image is for spatial orientation only; metric calculations use linear HDR.
    rgb = np.clip(mid_rgb, 0, 1)
    view = Image.fromarray(np.uint8(rgb * 255), mode="RGB")
    panel.paste(view, (0, 26))
    panel.paste(_heatmap(residual, mask, 0.2), (w, 26))
    event_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    event_rgb[..., 1] = np.uint8(np.clip(on / 3.0, 0, 1) * 255)
    event_rgb[..., 0] = np.uint8(np.clip(off / 3.0, 0, 1) * 255)
    event_rgb[..., 2] = np.where(mask, 0, 34).astype(np.uint8)
    panel.paste(Image.fromarray(event_rgb, mode="RGB"), (2 * w, 26))
    draw.text((4, 26 + h - 14), "linear RGB view", fill=(255, 255, 255))
    draw.text((w + 4, 26 + h - 14), "max |log residual|; red > 0.2", fill=(255, 255, 255))
    draw.text((2 * w + 4, 26 + h - 14), "event pixels; green ON / red OFF", fill=(255, 255, 255))
    return panel


def analyze_episode(ep: str, source: Path, evidence: Path, output: Path, reference_metrics: dict) -> dict:
    cache_path = source / f"episode_{ep}/same_state_source_cache.h5"
    direct_path = source / f"episode_{ep}/rendered_frame_evidence.h5"
    event_path = evidence / f"episode_{ep}_static_floor_temporal_events.npz"
    with h5py.File(cache_path, "r") as f:
        source_times = f["timestamps_s"][:]
        source_luma = f["radiance_luma"][:]
        source_seg = f["segmentation"][:]
        source_depth = f["depth"][:]
    with h5py.File(direct_path, "r") as f:
        times = f["dense_times_s"][:]
        direct_luma = f["dense_direct_hdr_luminance"][:]
        direct_rgb = f["dense_direct_hdr_rgb"][:]
        direct_seg = f["dense_direct_segmentation"][:]

    lo = int(np.searchsorted(source_times, times[0], side="right") - 1)
    hi = int(np.searchsorted(source_times, times[-1], side="left"))
    if hi != lo + 1 or abs(source_times[hi] - times[-1]) > 1e-8:
        raise RuntimeError(f"{ep}: expected the archived two-endpoint interval")
    mask = _fixed_floor_mask(source_seg[lo], source_seg[hi], direct_seg,
                             source_depth[lo], source_depth[hi])
    log_direct = np.log(np.maximum(direct_luma, 1e-6))
    log0 = np.log(np.maximum(source_luma[lo], 1e-6))
    log1 = np.log(np.maximum(source_luma[hi], 1e-6))
    alpha = (times - source_times[lo]) / (source_times[hi] - source_times[lo])
    endpoint_prediction = np.stack([(1.0 - a) * log0 + a * log1 for a in alpha])
    residual = np.where(mask[None], log_direct - endpoint_prediction, np.nan)
    abs_residual = np.max(np.where(mask[None], np.abs(residual), 0.0), axis=0)
    temporal_span = np.max(np.where(mask[None], log_direct, -np.inf), axis=0) - np.min(
        np.where(mask[None], log_direct, np.inf), axis=0
    )
    increments = np.diff(log_direct, axis=0)
    signs = np.sign(increments)
    sign_reversal_count = np.sum((signs[1:] * signs[:-1] < 0) & mask[None], axis=0)

    event_archive = np.load(event_path)
    events = np.asarray(event_archive["direct"], dtype=np.float64)
    on = np.zeros(mask.shape, dtype=np.uint16)
    off = np.zeros(mask.shape, dtype=np.uint16)
    polarities = np.zeros(mask.shape, dtype=np.uint8)
    for _, x_f, y_f, polarity_f in events:
        x, y, polarity = int(x_f), int(y_f), int(polarity_f)
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
            continue
        if polarity > 0:
            on[y, x] += 1
            polarities[y, x] |= 1
        else:
            off[y, x] += 1
            polarities[y, x] |= 2
    active = (on + off) > 0
    event_y = events[:, 2].astype(np.int64) if len(events) else np.empty(0, dtype=np.int64)
    event_x = events[:, 1].astype(np.int64) if len(events) else np.empty(0, dtype=np.int64)
    event_residual = abs_residual[event_y, event_x] if len(events) else np.empty(0, dtype=np.float32)
    event_residual_overlap = {}
    for threshold in (0.05, 0.20):
        selected = event_residual > threshold
        unique_selected = (np.unique(np.stack([event_y[selected], event_x[selected]], axis=1), axis=0)
                           if np.any(selected) else np.empty((0, 2), dtype=np.int64))
        event_residual_overlap[f"greater_than_{threshold:.2f}_log"] = {
            "event_count": int(selected.sum()),
            "event_fraction": float(selected.mean()) if len(events) else None,
            "unique_active_pixel_count": int(len(unique_selected)),
            "active_pixel_fraction": float(len(unique_selected) / max(int(active.sum()), 1)),
        }
    both_polarities = (polarities == 3) & mask
    connected, n_components = ndimage.label(active & mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(connected.ravel())[1:]
    sizes = np.sort(sizes)[::-1]
    top3 = int(sizes[:3].sum()) if sizes.size else 0
    q_levels = np.quantile(log_direct[:, mask], [0.05, 0.5, 0.95], axis=1)
    medians = q_levels[1]
    mid = int(np.argmin(np.abs(times - 0.5855)))
    y_active, x_active = np.where(active & mask)

    panel = _make_panel(ep, direct_rgb[mid], abs_residual, on, off, mask)
    panel.save(output / f"episode_{ep}_floor_photometric_map.png")
    event_by_interval = reference_metrics[ep]["frame_event_counts"]["direct_cached"]
    return {
        "episode": ep,
        "source_interval_s": [float(source_times[lo]), float(source_times[hi])],
        "direct_sample_times_s": [float(t) for t in times],
        "fixed_mask_pixels": int(mask.sum()),
        "camera_unchanged_and_floor_depth_identical": True,
        "global_median_log_luminance_range": float(np.ptp(medians)),
        "per_sample_floor_log_luminance_quantiles_05_50_95": q_levels.T.tolist(),
        "endpoint_model_residual_pixels_gt_0_05": int(((abs_residual > 0.05) & mask).sum()),
        "endpoint_model_residual_pixels_gt_0_20": int(((abs_residual > 0.20) & mask).sum()),
        "direct_temporal_span_pixels_gt_0_20": int(((temporal_span > 0.20) & mask).sum()),
        "pixels_with_at_least_one_brightness_direction_reversal": int(((sign_reversal_count > 0) & mask).sum()),
        "pixels_with_two_or_more_direction_reversals": int(((sign_reversal_count >= 2) & mask).sum()),
        "direct_event_count": int(len(events)),
        "direct_event_pixels": int(active.sum()),
        "direct_events_on_endpoint_model_residual_pixels": event_residual_overlap,
        "pixels_with_both_event_polarities": int(both_polarities.sum()),
        "direct_event_count_by_saved_generator_call": event_by_interval,
        "active_event_pixel_bbox_xy": [int(x_active.min()), int(y_active.min()), int(x_active.max()), int(y_active.max())],
        "active_event_pixel_connected_components_8conn": int(n_components),
        "largest_active_component_pixel_sizes": [int(v) for v in sizes[:10]],
        "active_pixels_in_largest_3_components_fraction": float(top3 / max(int(active.sum()), 1)),
        "event_spatial_distribution": _counts_by_tile(active, mask),
        "source_cache_sha256": sha256(cache_path),
        "direct_frame_cache_sha256": sha256(direct_path),
        "event_npz_sha256": sha256(event_path),
        "map_png": f"episode_{ep}_floor_photometric_map.png",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    reference_metrics = json.loads((args.evidence / "static_surface_time.json").read_text(encoding="utf-8"))
    records = [analyze_episode(ep, args.source.resolve(), args.evidence.resolve(), args.output.resolve(), reference_metrics)
               for ep in ("004", "005")]
    manifest = {
        "purpose": "localize endpoint-radiance residuals and direct cached event activity on fixed stationary floor pixels",
        "scope": "two archived development episodes; CPU-only post hoc analysis; no new render, physics, training, checkpoint selection, or core edit",
        "data_source": str(args.source.resolve()),
        "evidence_source": str(args.evidence.resolve()),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "execution_commit": __import__("subprocess").check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "episodes": records,
        "interpretation_limit": "Localized, non-monotone floor radiance is consistent with moving shadows but cannot identify the cause without a matched shadow-off render intervention.",
    }
    report = args.output / "rootcause_analysis.json"
    report.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in args.output.iterdir() if p.is_file())
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in files), encoding="ascii"
    )
    print(json.dumps({"output": str(args.output.resolve()), "report": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
