#!/usr/bin/env python3
"""CPU spatial-translation null for the cached target-shadow/event association.

This reuses the same archived HDR state trace, floor mask, events, and analytic
shadow hypothesis as analyze_r1_5_cached_achromatic_shadow_signature.py. It
does not render, advance physics, train, or modify the event plugin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark/robot_closed_loop/diagnostics"))
import analyze_r1_5_cached_achromatic_shadow_signature as signature  # noqa: E402

DEFAULT_SOURCE = ROOT / "benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059"
DEFAULT_EVIDENCE = ROOT / "benchmark/robot_closed_loop/results/r1_5_photometric_sampling_contract_20261024a/evidence"
DEFAULT_SCENE = ROOT / "benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"
DEFAULT_OUTPUT = ROOT / "benchmark/robot_closed_loop/results/r1_5_shadow_edge_spatial_null_cpu_20261024g"


def digest(path: Path) -> str:
    return signature.sha256(path)


def one_episode(ep: str, source: Path, evidence: Path, target_size: np.ndarray,
                ) -> dict:
    cache_path = source / f"episode_{ep}/same_state_source_cache.h5"
    direct_path = source / f"episode_{ep}/rendered_frame_evidence.h5"
    event_path = evidence / f"episode_{ep}_static_floor_temporal_events.npz"
    with h5py.File(cache_path, "r") as f:
        source_seg = f["segmentation"][:]
        source_depth = f["depth"][:]
        pose_ids = f["segmentation_pose_ids"][:]
        object_poses = f["object_poses"][:]
        camera = f["camera_to_world_cv"][3]
        wh = tuple(int(v) for v in f.attrs["camera_resolution_wh"])
        fov = float(f.attrs["vertical_fov_deg"])
    with h5py.File(direct_path, "r") as f:
        times = f["dense_times_s"][:]
        dense_seg = f["dense_direct_segmentation"][:]
        dense_rows = [json.loads(v.decode() if isinstance(v, bytes) else v)
                      for v in f["dense_state_trace_rows_json"][:]]
    mask = signature.floor_mask(source_seg, source_depth, dense_seg)
    with np.load(event_path) as z:
        events = np.asarray(z["direct"], dtype=np.float64)
        if not np.array_equal(mask, z["mask"]):
            raise RuntimeError(f"episode {ep}: floor mask changed")
    if not len(events):
        raise RuntimeError(f"episode {ep}: empty event set")

    # Match the published union-of-time-varying-shadow-edge proximity measure.
    polygons = signature.predicted_target_shadow_edges(
        dense_rows, times, camera, target_size, wh, fov
    )
    edges = [signature.edge(m) for m in polygons]
    actual_union = np.logical_or.reduce(edges) & mask
    y = events[:, 2].astype(np.int64)
    x = events[:, 1].astype(np.int64)
    raw_union = np.logical_or.reduce(edges)
    actual_distance = ndimage.distance_transform_edt(~raw_union)
    observed = float(np.mean(actual_distance[y, x] <= 5.0))
    floor_base = float(np.mean(actual_distance[mask] <= 5.0))
    raw_pixels = int(raw_union.sum())
    actual_pixels = int(actual_union.sum())
    h, w = mask.shape
    yy, xx = np.where(raw_union)
    radius = 5
    dy_min = max(-h // 3, radius - int(yy.min()))
    dy_max = min(h // 3, h - radius - 1 - int(yy.max()))
    dx_min = max(-w // 3, radius - int(xx.min()))
    dx_max = min(w // 3, w - radius - 1 - int(xx.max()))
    if dy_min > dy_max or dx_min > dx_max:
        raise RuntimeError(f"episode {ep}: no in-frame contour translations")

    # Cross-correlations enumerate every integer translation in the declared
    # image-relative window. The dilated contour gives exact Euclidean <=5px
    # event proximity; floor correlation matches visible edge support.
    near_edge = ndimage.distance_transform_edt(~raw_union) <= radius
    event_count_map = np.bincount(y * w + x, minlength=h * w).reshape(h, w)

    def correlate(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        values = fftconvolve(first.astype(np.float64), second[::-1, ::-1].astype(np.float64), mode="full")
        return np.rint(values).astype(np.int64)

    hit_counts = correlate(event_count_map, near_edge)
    floor_support = correlate(mask, raw_union)
    actual_hits = int(hit_counts[h - 1, w - 1])
    if abs(actual_hits / len(events) - observed) > 1e-12:
        raise RuntimeError(f"episode {ep}: cross-correlation does not reproduce observed score")

    null_fractions: list[float] = []
    null_pixel_counts: list[int] = []
    for dy in range(dy_min, dy_max + 1):
        for dx in range(dx_min, dx_max + 1):
            if dx == 0 and dy == 0:
                continue
            iy, ix = dy + h - 1, dx + w - 1
            support_count = int(floor_support[iy, ix])
            count_ratio = support_count / max(actual_pixels, 1)
            if 0.8 <= count_ratio <= 1.2:
                null_fractions.append(int(hit_counts[iy, ix]) / len(events))
                null_pixel_counts.append(support_count)
    if len(null_fractions) < 100:
        raise RuntimeError(f"episode {ep}: only {len(null_fractions)} admissible exhaustive translations")
    null = np.asarray(null_fractions, dtype=np.float64)
    median = float(np.median(null))
    return {
        "episode": ep,
        "source_cache_sha256": digest(cache_path),
        "direct_frame_cache_sha256": digest(direct_path),
        "floor_event_archive_sha256": digest(event_path),
        "events": int(len(events)),
        "actual_shadow_edge_pixels_in_floor": actual_pixels,
        "unclipped_union_shadow_edge_pixels": raw_pixels,
        "actual_edge_fraction_within_5px_of_events": observed,
        "actual_edge_fraction_within_5px_of_floor_pixels": floor_base,
        "null_translation_count": int(len(null)),
        "null_translation_domain": {"dx": [dx_min, dx_max], "dy": [dy_min, dy_max], "integer_step": 1},
        "null_edge_pixel_count_range": [int(min(null_pixel_counts)), int(max(null_pixel_counts))],
        "null_event_fraction_within_5px_quantiles": {
            "q50": float(np.quantile(null, 0.50)),
            "q90": float(np.quantile(null, 0.90)),
            "q95": float(np.quantile(null, 0.95)),
            "q99": float(np.quantile(null, 0.99)),
            "max": float(np.max(null)),
        },
        "observed_minus_null_median": float(observed - median),
        "fractional_enrichment_vs_null_median": None if median == 0 else float(observed / median),
        "empirical_upper_tail_fraction": float(np.mean(null >= observed)),
        "null_definition": "exhaustive integer translations of the same time-unioned analytic shadow contour within +/- one third of each image dimension; retain only placements whose contour support inside the fixed floor ROI is 0.8-1.2x actual; contour plus 5px proximity neighborhood must remain inside image; score is fraction of archived floor events within 5px",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    scene = json.loads(args.scene_config.read_text(encoding="utf-8"))
    size = np.asarray(scene["target"]["size_m"], dtype=np.float64)
    args.output.mkdir(parents=True)
    records = [
        one_episode(ep, args.source.resolve(), args.evidence.resolve(), size,
                    )
        for ep in ("004", "005")
    ]
    report = {
        "purpose": "compare the cached event proximity to the analytic target-shadow edge against spatially translated same-shape contour controls",
        "scope": "CPU-only offline analysis of two existing development episodes; no renderer, physics, GPU, model, or frozen-core changes",
        "execution_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "analysis_script_sha256": digest(Path(__file__).resolve()),
        "scene_config_sha256": digest(args.scene_config.resolve()),
        "null_method": "exhaustive spatial translations in a fixed image-relative window, no sampling",
        "records": records,
        "interpretation_limit": "Translation nulls assess spatial specificity under this analytic contour and fixed ROI; they are not independent samples, do not establish shadow causality, and cannot identify the historical renderer binary.",
    }
    (args.output / "shadow_edge_spatial_null.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme = """# R1.5 target-shadow spatial translation null\n\nCPU-only post hoc analysis on the existing episode 004/005 cached direct-HDR events. The actual statistic is the fraction of fixed-floor event pixels within 5 px of the analytically projected, time-unioned target-shadow edge. The null exhaustively translates the same contour by every integer offset inside a fixed window of one third of each image dimension. It retains placements only when the number of contour pixels inside the fixed floor region is 0.8–1.2 times the actual visible support. This conditions on visible contour length because much of the projected edge falls outside the fixed floor mask or behind segmented objects. The contour and its 5 px neighborhood must remain inside the image.\n\nThe observed placement lies in the upper tail of these translations in both episodes, but some other placements can score similarly. This is a spatial specificity diagnostic, not a renderer intervention, an independent test set, a formal inferential p-value, or causal proof. The shadow polygon is an analytic hypothesis based on the documented default light direction. See the JSON for input/config/code hashes and exact translation ranges.\n\nReproduce from the repository root:\n\n```bash\npython benchmark/robot_closed_loop/diagnostics/check_r1_5_shadow_edge_spatial_null.py \\\n  --output benchmark/robot_closed_loop/results/r1_5_shadow_edge_spatial_null_cpu_20261024g\n```\n"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    sums = []
    for path in sorted(args.output.iterdir()):
        if path.name == "SHA256SUMS":
            continue
        sums.append(f"{digest(path)}  {path.name}")
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
