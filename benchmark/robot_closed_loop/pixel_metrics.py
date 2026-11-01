"""Post-inference localization scoring in original image coordinates."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence
import numpy as np


def project_world_to_pixel(position: Sequence[float], camera_to_world_cv: np.ndarray,
                           intrinsics: Sequence[float]) -> np.ndarray:
    """Project a physical target centre; reject invalid or behind-camera points."""
    point = np.asarray(position, dtype=np.float64)
    pose = np.asarray(camera_to_world_cv, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if point.shape != (3,) or pose.shape != (4, 4) or k.shape != (4,):
        raise ValueError("expected xyz, camera-to-world 4x4 and fx/fy/cx/cy")
    if not (np.isfinite(point).all() and np.isfinite(pose).all() and np.isfinite(k).all()):
        raise ValueError("non-finite projection input")
    if np.any(k[:2] <= 0):
        raise ValueError("focal lengths must be positive")
    cam = np.linalg.solve(pose, np.r_[point, 1.])
    if cam[2] <= 0:
        raise ValueError("point is not in front of the camera")
    return k[:2] * cam[:2] / cam[2] + k[2:]


def summarize_pixel_failures(rows: Iterable[Mapping], thresholds=(2., 5., 10., 20.)) -> dict:
    """Missing and invalid predictions count as failures at every threshold."""
    rows = list(rows)
    errors = [r.get("pixel_euclidean_error_px") for r in rows]
    finite = np.asarray([e is not None and np.isfinite(float(e)) for e in errors], dtype=bool)
    values = np.asarray([float(e) for e, ok in zip(errors, finite) if ok], dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("pixel error cannot be negative")
    n = len(rows)
    return {
        "all_polls": n, "finite_localizations": int(finite.sum()),
        "missing_or_invalid": int(n - finite.sum()),
        "thresholds_px": {
            str(float(t)): {
                "failed_or_missing": int(n - finite.sum() + np.sum(values > t)),
                "fraction_all_polls": None if not n else float((n - finite.sum() + np.sum(values > t)) / n),
            } for t in thresholds
        },
    }
