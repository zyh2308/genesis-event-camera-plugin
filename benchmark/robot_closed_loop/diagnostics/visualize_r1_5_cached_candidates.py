#!/usr/bin/env python3
"""Render small diagnostic montages and endpoint z-buffer evidence from caches.

No Genesis renderer or physics is initialized. The direct HDR panels come from
the archived exact-state evidence; candidate panels use the same cached
endpoints and projection code as the R1.5 replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.r1_5_consistency_candidates import (  # noqa: E402
    project_endpoint_candidate,
    reconstruct_pair_candidate,
)
from genesis_event_plugin.physics_warp import _splat_projected_diagnostics  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair(time_s: float, times: np.ndarray) -> tuple[int, float]:
    exact = np.flatnonzero(np.isclose(times, time_s, atol=1e-9, rtol=0))
    if exact.size:
        idx = int(exact[0])
        return (max(0, idx - 1), 0.0 if idx == 0 else 1.0)
    right = int(np.searchsorted(times, time_s, side="right"))
    left = right - 1
    if left < 0 or right >= len(times):
        raise ValueError(f"time {time_s} outside endpoint support")
    return left, float((time_s - times[left]) / (times[right] - times[left]))


def _safe_candidate(frame: np.ndarray, valid: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    out = np.asarray(frame, dtype=np.float32).copy()
    mask = ~np.asarray(valid, dtype=bool) | ~np.isfinite(out)
    out[mask] = np.asarray(fallback, dtype=np.float32)[mask]
    return out


def _log_gray(luma: np.ndarray, lo: float, hi: float) -> Image.Image:
    value = np.log(np.maximum(np.asarray(luma, dtype=np.float32), 1e-6))
    gray = np.clip((value - lo) / max(hi - lo, 1e-9), 0, 1)
    data = (gray * 255).astype(np.uint8)
    rgb = np.repeat(data[..., None], 3, axis=2)
    return Image.fromarray(rgb)


def _residual_image(reference: np.ndarray, candidate: np.ndarray) -> Image.Image:
    a = np.log(np.maximum(np.asarray(reference, dtype=np.float32), 1e-6))
    b = np.log(np.maximum(np.asarray(candidate, dtype=np.float32), 1e-6))
    residual = np.abs(a - b)
    intensity = np.clip(residual / 0.5, 0, 1)
    rgb = np.zeros((*residual.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * intensity).astype(np.uint8)
    rgb[..., 1] = (48 * intensity).astype(np.uint8)
    rgb[..., 2] = (16 * intensity).astype(np.uint8)
    return Image.fromarray(rgb)


def _make_montage(path: Path, direct: np.ndarray, candidates: dict[str, np.ndarray], time_s: float) -> None:
    all_logs = np.log(np.maximum(np.asarray(direct, dtype=np.float32), 1e-6))
    low, high = float(np.percentile(all_logs, 1)), float(np.percentile(all_logs, 99.5))
    labels = ["direct HDR", *candidates.keys()]
    images = [direct, *candidates.values()]
    w, h = direct.shape[1], direct.shape[0]
    canvas = Image.new("RGB", (w * len(labels), h * 2 + 48), (24, 26, 30))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(zip(labels, images)):
        x = idx * w
        draw.text((x + 5, 5), f"{label} @ {time_s:.4f}s", fill=(245, 245, 245))
        canvas.paste(_log_gray(image, low, high), (x, 24))
        if idx:
            residual = _residual_image(direct, image)
            draw.text((x + 5, h + 28), "abs log-luma residual (red > 0)", fill=(245, 245, 245))
            canvas.paste(residual, (x, h + 48))
    draw.text((5, h + 28), "log-luma display scale; no GT mask overlay", fill=(245, 245, 245))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _projection_error(uv: np.ndarray, valid: np.ndarray) -> dict:
    h, w = valid.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    grid = np.stack((xx, yy), axis=0).astype(np.float32)
    e = np.abs(np.asarray(uv, dtype=np.float32) - grid)[:, valid]
    return {"valid_source_pixels": int(valid.sum()),
            "max_uv_error_px": float(e.max()) if e.size else None,
            "p99_uv_error_px": float(np.percentile(e, 99)) if e.size else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", default="0.5855,0.5905")
    args = parser.parse_args()
    scene = json.loads(args.scene_config.read_text(encoding="utf-8"))
    evidence_root, output = args.evidence_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    width, height = (int(v) for v in scene["camera"]["resolution_wh"])
    fov = float(scene["camera"]["vertical_fov_deg"])
    focal = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    K = np.asarray([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    query_times = [float(x) for x in args.times.split(",")]
    started = time.perf_counter()
    summaries = {}

    for ep_id in (4, 5):
        ep_dir = evidence_root / f"episode_{ep_id:03d}"
        cache_path = ep_dir / "same_state_source_cache.h5"
        dense_path = ep_dir / "rendered_frame_evidence.h5"
        with h5py.File(cache_path, "r") as cache:
            source_times = np.asarray(cache["timestamps_s"][:], dtype=np.float64)
            pose_ids = np.asarray(cache["segmentation_pose_ids"][:], dtype=np.int32)
            camera = np.asarray(cache["camera_to_world_cv"][0], dtype=np.float64)
            sources = []
            for i, t in enumerate(source_times):
                sources.append({
                    "time_s": float(t),
                    "luma": np.asarray(cache["radiance_luma"][i], dtype=np.float32),
                    "depth": np.asarray(cache["depth"][i], dtype=np.float32),
                    "seg": np.asarray(cache["segmentation"][i], dtype=np.int32),
                    "poses": {int(seg): np.asarray(cache["object_poses"][i, j], dtype=np.float64)
                              for j, seg in enumerate(pose_ids)},
                })
        with h5py.File(dense_path, "r") as dense:
            dense_times = np.asarray(dense["dense_times_s"][:], dtype=np.float64)
            direct_lumas = np.asarray(dense["dense_direct_hdr_luminance"][:], dtype=np.float32)
            direct_by_time = {round(float(t), 9): direct_lumas[i] for i, t in enumerate(dense_times)}

        montage_meta = []
        for t in query_times:
            key = round(t, 9)
            if key not in direct_by_time:
                raise ValueError(f"no exact saved direct HDR image at requested montage time {t}")
            pair_idx, alpha = _pair(t, source_times)
            s0, s1 = sources[pair_idx], sources[pair_idx + 1]
            named = {}
            for name, mode in (
                ("frozen warp", "frozen"),
                ("endpoint identity", "endpoint_projection_identity"),
                ("exact endpoint", "endpoint_exact"),
                ("log-radiance + exact cached request fill", "log_radiance_with_render_request"),
            ):
                result = reconstruct_pair_candidate(
                    s0["luma"], s0["depth"], s0["seg"], camera,
                    s1["luma"], s1["depth"], s1["seg"], camera,
                    K, s0["poses"], s1["poses"], alpha, mode=mode)
                image = np.asarray(result["frame"], dtype=np.float32).copy()
                request = np.asarray(result["request_render"], dtype=bool)
                candidate_valid = np.asarray(result["valid"], dtype=bool).copy()
                if mode == "log_radiance_with_render_request" and request.any():
                    image[request] = direct_by_time[key][request]
                    candidate_valid[request] = True
                fallback = s0["luma"] if alpha < 0.5 else s1["luma"]
                image = _safe_candidate(image, candidate_valid, fallback)
                named[name] = image
            montage_path = output / f"episode_{ep_id:03d}_images_t{int(round(t * 10000)):04d}.png"
            _make_montage(montage_path, direct_by_time[key], named, t)
            montage_meta.append({"time_s": t, "path": montage_path.name,
                                 "direct_hdr_sha256": _sha256(dense_path)})

        # At alpha=1, measure the source-1 projection and the nearest-depth
        # rasterization consequences in the real saved cache.
        endpoint_time = 0.5905
        idx = int(np.flatnonzero(np.isclose(source_times, endpoint_time, atol=1e-9, rtol=0))[0])
        s0, s1 = sources[idx - 1], sources[idx]
        projections = {}
        splats = {}
        for name, exact in (("frozen_float_projection", False), ("identity_projection", True)):
            p = project_endpoint_candidate(
                s1["depth"], s1["seg"], K, camera, camera, 1.0, 1,
                s0["poses"], s1["poses"], exact_endpoints=exact)
            projections[name] = _projection_error(p[0], p[2])
            warped, diag = _splat_projected_diagnostics(s1["luma"], s1["depth"], p[0], p[1], p[2])
            splats[name] = (warped, diag)
        raw_warp, raw_diag = splats["frozen_float_projection"]
        exact_warp, exact_diag = splats["identity_projection"]
        direct = direct_by_time[round(endpoint_time, 9)]
        raw_finite = np.asarray(raw_diag["visibility"], dtype=bool) & np.isfinite(raw_warp) & np.isfinite(direct)
        exact_finite = np.asarray(exact_diag["visibility"], dtype=bool) & np.isfinite(exact_warp) & np.isfinite(direct)
        log_raw = np.abs(np.log(np.maximum(np.nan_to_num(raw_warp, nan=1e-6), 1e-6)) -
                         np.log(np.maximum(direct, 1e-6)))
        log_exact = np.abs(np.log(np.maximum(np.nan_to_num(exact_warp, nan=1e-6), 1e-6)) -
                           np.log(np.maximum(direct, 1e-6)))
        z_raw = np.asarray(raw_diag["z_buffer"], dtype=np.float32)
        z_exact = np.asarray(exact_diag["z_buffer"], dtype=np.float32)
        zvalid = np.isfinite(z_raw) & np.isfinite(z_exact)
        z_diff = np.abs(z_raw[zvalid] - z_exact[zvalid])
        endpoint_summary = {
            "time_s": endpoint_time,
            "projection": projections,
            "raw_versus_identity_raster": {
                "occupancy_mismatch_pixels": int(np.count_nonzero(raw_diag["occupancy"] != exact_diag["occupancy"])),
                "visibility_mismatch_pixels": int(np.count_nonzero(raw_diag["visibility"] != exact_diag["visibility"])),
                "zbuffer_compared_pixels": int(z_diff.size),
                "zbuffer_changed_over_1e-6_pixels": int(np.count_nonzero(z_diff > 1e-6)),
                "zbuffer_changed_over_1e-5_pixels": int(np.count_nonzero(z_diff > 1e-5)),
                "zbuffer_max_abs_delta": float(z_diff.max()) if z_diff.size else None,
                "raw_warp_direct_visible_pixels": int(raw_finite.sum()),
                "identity_warp_direct_visible_pixels": int(exact_finite.sum()),
                "raw_warp_vs_direct_pixels_log_error_over_0_2": int(np.count_nonzero(log_raw[raw_finite] > 0.2)),
                "identity_warp_vs_direct_pixels_log_error_over_0_2": int(np.count_nonzero(log_exact[exact_finite] > 0.2)),
                "raw_warp_vs_direct_log_error_max": float(log_raw[raw_finite].max()) if raw_finite.any() else None,
                "identity_warp_vs_direct_log_error_max": float(log_exact[exact_finite].max()) if exact_finite.any() else None,
            },
        }
        summaries[f"episode_{ep_id:03d}"] = {
            "source_cache_sha256": _sha256(cache_path), "dense_render_evidence_sha256": _sha256(dense_path),
            "montages": montage_meta, "endpoint_zbuffer": endpoint_summary,
        }

    manifest = {
        "experiment_id": "r1_5_cached_candidate_visuals_20261019a",
        "code_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scene_config_sha256": _sha256(args.scene_config.resolve()),
        "episodes": summaries, "elapsed_wall_s": time.perf_counter() - started,
        "compute": "CPU-only cache processing; no Genesis renderer, EGL, CUDA, or physics",
        "boundaries": "direct HDR frames are archived exact-state renders; GT geometry is used only for reconstruction and numerical diagnostics, not learned inputs",
    }
    (output / "summary.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    hashes = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        hashes.append(f"{_sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "elapsed_wall_s": manifest["elapsed_wall_s"],
                      "sha256sums": _sha256(output / "SHA256SUMS"), "episodes": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
