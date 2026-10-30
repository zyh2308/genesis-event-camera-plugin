#!/usr/bin/env python3
"""Create a read-only corrected E-POSE keypoint projection sidecar.

The selected adapter domain is raw sensor event coordinates with forward lens
distortion (K,D). Ideal pinhole coordinates are also emitted as a cross-check
for a future, explicitly undistorted event representation. Original MAT/HDF5
files are never changed; no image, mask, box, or real model score is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.io as sio


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_matlab_vector(source: str, variable: str) -> np.ndarray:
    match = re.search(rf"\b{re.escape(variable)}\s*=\s*\[([^\]]+)\]", source)
    if not match:
        raise ValueError(f"cannot locate {variable} in pinned E-POSE source")
    values = np.fromstring(match.group(1).replace(",", " "), sep=" ", dtype=np.float64)
    if values.shape != (5,):
        raise ValueError(f"{variable} must contain OpenCV k1,k2,p1,p2,k3, got {values}")
    return values


def source_domain_contract(source: str) -> dict[str, Any]:
    checks = {
        "raw_event_frames_are_preserved": "allData{i,1} = frames{i,1};" in source,
        "undistorted_event_images_are_separate": "allData{i,2} = undistorted_frames{i,1};" in source,
        "mat_events_use_raw_event_frames": "events =allData{i,1};" in source,
        "mat_keypoints_use_projected_points": "keypointprojections =allData{i,6}(1:8,:);" in source,
    }
    if not all(checks.values()):
        raise RuntimeError(f"official source event/label domain contract changed: {checks}")
    return checks


def distort_normalized(points_xy: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """OpenCV 5-coefficient Brown-Conrady forward distortion in normalized UV."""
    k1, k2, p1, p2, k3 = map(float, coeffs)
    x, y = points_xy[:, 0], points_xy[:, 1]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (3.0 * x * x + y * y)
    yd = y * radial + p1 * (x * x + 3.0 * y * y) + 2.0 * p2 * x * y
    return np.stack((xd, yd), axis=1)


def undistort_normalized_newton(
    distorted_xy: np.ndarray, coeffs: np.ndarray, tolerance: float = 1e-14, max_iterations: int = 50
) -> tuple[np.ndarray, int, float]:
    """Invert Brown-Conrady distortion with an explicit analytic Newton Jacobian."""
    k1, k2, p1, p2, k3 = map(float, coeffs)
    target = np.asarray(distorted_xy, dtype=np.float64)
    estimate = target.copy()
    for iteration in range(1, max_iterations + 1):
        x, y = estimate[:, 0], estimate[:, 1]
        r2 = x * x + y * y
        r4, r6 = r2 * r2, r2 * r2 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        radial_slope = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4
        dr_dx, dr_dy = 2.0 * x * radial_slope, 2.0 * y * radial_slope
        predicted = distort_normalized(estimate, coeffs)
        residual = predicted - target
        j11 = radial + x * dr_dx + 2.0 * p1 * y + 6.0 * p2 * x
        j12 = x * dr_dy + 2.0 * p1 * x + 2.0 * p2 * y
        j21 = y * dr_dx + 2.0 * p1 * x + 2.0 * p2 * y
        j22 = radial + y * dr_dy + 6.0 * p1 * y + 2.0 * p2 * x
        det = j11 * j22 - j12 * j21
        if np.any(np.abs(det) < 1e-14):
            raise RuntimeError("singular Brown-Conrady inverse Jacobian")
        delta_x = (j22 * residual[:, 0] - j12 * residual[:, 1]) / det
        delta_y = (-j21 * residual[:, 0] + j11 * residual[:, 1]) / det
        delta = np.stack((delta_x, delta_y), axis=1)
        estimate -= delta
        if float(np.max(np.linalg.norm(delta, axis=1))) <= tolerance:
            final_residual = distort_normalized(estimate, coeffs) - target
            return estimate, iteration, float(np.max(np.linalg.norm(final_residual, axis=1)))
    raise RuntimeError("Brown-Conrady Newton inverse did not converge")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite projection sidecar: {args.output}")
    source_path = args.epose_source_root / "CreateSaveDataset.m"
    mat_path = args.mat.resolve()
    audit_path = args.previous_audit.resolve()
    for path in (source_path, mat_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = source_path.read_text(encoding="utf-8", errors="replace")
    source_sha = sha256(source_path)
    previous = json.loads(audit_path.read_text())
    expected_source_sha = previous["inputs"]["source_file_sha256"]
    if source_sha != expected_source_sha:
        raise RuntimeError("local pinned E-POSE source differs from the previously audited source")
    expected_mat_path = Path(previous["inputs"]["mat_path_recorded_in_summary"]).resolve()
    expected_mat_sha = previous["inputs"]["mat_sha256_recorded_in_summary"]
    mat_sha = sha256(mat_path)
    if mat_path != expected_mat_path or mat_sha != expected_mat_sha:
        raise RuntimeError("MAT path/hash differs from the already audited original sample")

    source_domain = source_domain_contract(source)
    distortion = parse_matlab_vector(source, "distCoeffs")
    expected_distortion = np.asarray(
        previous["cad_and_pose_geometry"]["source_distortion_coefficients_k1_k2_p1_p2_k3"],
        dtype=np.float64,
    )
    if not np.array_equal(distortion, expected_distortion):
        raise RuntimeError("source distortion vector differs from the semantic audit")

    mat = sio.loadmat(mat_path)
    for key in ("keypointprojections", "intrinsicMatrix", "pose", "events"):
        if key not in mat:
            raise KeyError(f"original MAT lacks required field {key!r}")
    points = np.asarray(mat["keypointprojections"], dtype=np.float64)
    if points.shape == (5, 8):
        points = points.T
    if points.shape != (8, 5) or not np.isfinite(points).all():
        raise ValueError(f"keypointprojections must be finite [8,5] [u,v,X,Y,Z], got {points.shape}")
    stored_uv = points[:, :2]
    xyz_camera = points[:, 2:5]
    if np.any(xyz_camera[:, 2] <= 0):
        raise ValueError("all audited object keypoints must lie in front of the camera")

    K = np.asarray(mat["intrinsicMatrix"], dtype=np.float64)
    if K.shape != (3, 3) or not np.allclose(K[2], [0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError(f"MAT intrinsicMatrix is not in the pinned OpenCV row form: {K}")
    if not np.allclose(K, np.asarray(previous["cad_and_pose_geometry"]["K"], dtype=np.float64), rtol=0, atol=1e-10):
        raise RuntimeError("MAT K differs from the prior sample/CAD audit")
    prior_xyz = np.asarray(previous["cad_and_pose_geometry"]["mat_keypoint_xyz_camera_frame"], dtype=np.float64)
    xyz_check_max_abs = float(np.max(np.abs(xyz_camera - prior_xyz)))
    if xyz_check_max_abs > 1e-10:
        raise RuntimeError(f"MAT 3D keypoints differ from the CAD/pose-verified audit by {xyz_check_max_abs}")

    ideal_xy = xyz_camera[:, :2] / xyz_camera[:, 2:3]
    ideal_h = (K @ xyz_camera.T).T
    ideal_uv = ideal_h[:, :2] / ideal_h[:, 2:3]
    distorted_uv, _ = cv2.projectPoints(
        xyz_camera.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, distortion
    )
    distorted_uv = distorted_uv.reshape(-1, 2)
    opencv_default_undistorted_uv = cv2.undistortPoints(
        distorted_uv.reshape(-1, 1, 2), K, distortion, P=K
    ).reshape(-1, 2)
    cv2_default_error = np.linalg.norm(opencv_default_undistorted_uv - ideal_uv, axis=1)
    distorted_xy = np.stack(((distorted_uv[:, 0] - K[0, 2]) / K[0, 0],
                             (distorted_uv[:, 1] - K[1, 2]) / K[1, 1]), axis=1)
    recovered_xy, inverse_iterations, inverse_residual = undistort_normalized_newton(distorted_xy, distortion)
    precise_undistorted_uv = np.stack((K[0, 0] * recovered_xy[:, 0] + K[0, 2],
                                       K[1, 1] * recovered_xy[:, 1] + K[1, 2]), axis=1)
    precise_roundtrip_error = np.linalg.norm(precise_undistorted_uv - ideal_uv, axis=1)
    independent_forward_uv = np.stack((K[0, 0] * distort_normalized(ideal_xy, distortion)[:, 0] + K[0, 2],
                                       K[1, 1] * distort_normalized(ideal_xy, distortion)[:, 1] + K[1, 2]), axis=1)
    cv2_forward_error = np.linalg.norm(independent_forward_uv - distorted_uv, axis=1)
    if float(np.max(precise_roundtrip_error)) > args.roundtrip_tolerance_px:
        raise RuntimeError(f"precise distort→undistort did not recover pinhole UV: max={precise_roundtrip_error.max()} px")
    if float(np.max(cv2_forward_error)) > args.forward_tolerance_px:
        raise RuntimeError(f"independent Brown-Conrady forward map disagrees with cv2.projectPoints: max={cv2_forward_error.max()} px")

    raw_events = np.asarray(mat["events"])
    event_shape = list(raw_events.shape)
    expected_event_contract = "MAT raw event list retained before the separate event-image undistortion path"
    stored_vs_ideal = np.linalg.norm(stored_uv - ideal_uv, axis=1)
    stored_vs_raw_correct = np.linalg.norm(stored_uv - distorted_uv, axis=1)
    pose = np.asarray(mat["pose"], dtype=np.float64)
    if pose.shape not in ((3, 4), (4, 4)):
        raise ValueError(f"MAT pose has unexpected shape {pose.shape}")

    result = {
        "schema": "epose_projection_domain_sidecar_v1",
        "status": "corrected_projection_domains_verified_on_original_mat_sample_no_scoring",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "elapsed_seconds": time.monotonic() - args.start_time,
        "real_scoring_enabled": False,
        "training_or_model_selection_performed": False,
        "source_files": {
            "original_mat": {"path": str(mat_path), "size_bytes": mat_path.stat().st_size, "sha256": mat_sha},
            "official_source": {"path": str(source_path), "sha256": source_sha,
                                "commit": previous["inputs"]["epose_source_git_head"]},
            "previous_cad_pose_audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
        },
        "sample_contract": {
            "mat_pose_shape": list(pose.shape),
            "event_field_shape": event_shape,
            "raw_event_domain_source_checks": source_domain,
            "K": K.tolist(),
            "distortion_k1_k2_p1_p2_k3": distortion.tolist(),
            "camera_frame_xyz_source": "MAT keypointprojections columns 3:5; exactly matches previously CAD/pose-verified sample points",
            "camera_frame_xyz_max_abs_difference_from_verified_audit_m": xyz_check_max_abs,
        },
        "projection_domains": {
            "selected": "raw_distorted_sensor_pixels",
            "model_input_contract": "full-frame event representation built directly from raw event x,y coordinates; no GT crop/mask/box",
            "selected_label_formula": "OpenCV projectPoints(camera_frame_xyz, rvec=0, tvec=0, K, D)",
            "selected_keypoints_uv_raw_distorted_px": distorted_uv.tolist(),
            "ideal_alternative_uv_px": ideal_uv.tolist(),
            "ideal_alternative_requires": "apply the same pinned K,D undistortion map to event-derived input channels, then use ideal_uv_px",
            "distort_undistort_roundtrip_px_per_keypoint": precise_roundtrip_error.tolist(),
            "roundtrip_max_abs_px": float(precise_roundtrip_error.max()),
            "roundtrip_tolerance_px": args.roundtrip_tolerance_px,
            "precise_inverse": {
                "method": "analytic-Jacobian Newton inversion of the five-parameter OpenCV Brown-Conrady model",
                "max_iterations_used": inverse_iterations,
                "max_normalized_forward_residual": inverse_residual,
            },
            "opencv_default_undistort_points_error_px_per_keypoint": cv2_default_error.tolist(),
            "opencv_default_undistort_points_max_error_px": float(cv2_default_error.max()),
            "independent_forward_vs_cv2_project_points_max_error_px": float(cv2_forward_error.max()),
            "forward_tolerance_px": args.forward_tolerance_px,
            "stored_legacy_uv_px_preserved_for_audit_only": stored_uv.tolist(),
            "stored_legacy_vs_correct_ideal_error_px": {
                "mean": float(stored_vs_ideal.mean()), "max": float(stored_vs_ideal.max())
            },
            "stored_legacy_vs_correct_raw_distorted_error_px": {
                "mean": float(stored_vs_raw_correct.mean()), "max": float(stored_vs_raw_correct.max())
            },
            "negative_or_out_of_frame_labels_clamped": False,
        },
        "interpretation": "This creates a separately versioned corrected-label candidate for the raw-event coordinate domain. It does not overwrite official MAT labels, validate all E-POSE timestamps/cameraPose fields, or establish model accuracy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "selected_domain": result["projection_domains"]["selected"],
                      "roundtrip_max_px": result["projection_domains"]["roundtrip_max_abs_px"],
                      "stored_vs_ideal_mean_px": result["projection_domains"]["stored_legacy_vs_correct_ideal_error_px"]["mean"],
                      "output": str(args.output), "sha256": sha256(args.output)}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--epose-source-root", type=Path, required=True)
    parser.add_argument("--previous-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roundtrip-tolerance-px", type=float, default=1e-3)
    parser.add_argument("--forward-tolerance-px", type=float, default=1e-8)
    args = parser.parse_args()
    args.start_time = time.monotonic()
    if args.roundtrip_tolerance_px <= 0 or args.forward_tolerance_px <= 0:
        parser.error("projection tolerances must be positive")
    run(args)


if __name__ == "__main__":
    main()
