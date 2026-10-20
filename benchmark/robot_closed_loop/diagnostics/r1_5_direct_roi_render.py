"""Exact-ray, cropped direct rendering for R1.5 candidate diagnostics.

Genesis 1.2.2's public camera API only builds a centered perspective camera.
This diagnostic reuses its rasterizer and replaces one auxiliary camera's
Pyrender camera node with an IntrinsicsCamera whose principal point is shifted
by the integer crop origin. The camera pose and focal lengths remain identical
to the fixed reference camera, so each ROI pixel has the same ray as the
corresponding full-frame pixel. This is deliberately isolated from the frozen
simulator core and fails closed if the expected Genesis rasterizer internals
are unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def cropped_intrinsics(source_intrinsics: np.ndarray, crop_xyxy: Sequence[int]) -> np.ndarray:
    """Return pinhole intrinsics that preserve source-camera rays in a crop."""
    K = np.asarray(source_intrinsics, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("source_intrinsics must be a finite 3x3 matrix")
    if len(crop_xyxy) != 4:
        raise ValueError("crop_xyxy must be [x0, y0, x1, y1]")
    x0, y0, _, _ = (int(value) for value in crop_xyxy)
    out = K.copy()
    out[0, 2] -= x0
    out[1, 2] -= y0
    return out


def configure_intrinsics_crop_camera(
    source_camera: Any,
    crop_camera: Any,
    crop_xyxy: Sequence[int],
) -> Dict[str, Any]:
    """Give an auxiliary camera exact source-camera rays for an integer crop."""
    if len(crop_xyxy) != 4:
        raise ValueError("crop_xyxy must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (int(value) for value in crop_xyxy)
    source_w, source_h = (int(value) for value in source_camera.res)
    crop_w, crop_h = (int(value) for value in crop_camera.res)
    if not (0 <= x0 < x1 <= source_w and 0 <= y0 < y1 <= source_h):
        raise ValueError(f"crop {crop_xyxy} is outside source camera {(source_w, source_h)}")
    if (crop_w, crop_h) != (x1 - x0, y1 - y0):
        raise ValueError(
            f"crop camera resolution {(crop_w, crop_h)} does not match crop {(x1 - x0, y1 - y0)}"
        )
    if crop_camera._rasterizer is not source_camera._rasterizer:
        raise RuntimeError("source and crop cameras must use the same Genesis rasterizer")

    try:
        from genesis.ext import pyrender

        rasterizer = crop_camera._rasterizer
        camera_node = rasterizer._camera_nodes[crop_camera.uid]
    except (AttributeError, KeyError, ImportError) as exc:
        raise RuntimeError("Genesis 1.2.2 rasterizer camera-node API is unavailable") from exc

    if not hasattr(crop_camera, "_r1_5_original_camera_node_camera"):
        crop_camera._r1_5_original_camera_node_camera = camera_node.camera

    intrinsics = np.asarray(source_camera.intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise RuntimeError("source camera did not expose finite 3x3 pinhole intrinsics")
    crop_intrinsics = cropped_intrinsics(intrinsics, (x0, y0, x1, y1))

    # Keep the camera center and orientation exactly equal to the reference.
    crop_camera.set_pose(transform=np.asarray(source_camera.transform, dtype=np.float32))
    camera_node.camera = pyrender.IntrinsicsCamera(
        fx=float(crop_intrinsics[0, 0]),
        fy=float(crop_intrinsics[1, 1]),
        cx=float(crop_intrinsics[0, 2]),
        cy=float(crop_intrinsics[1, 2]),
        znear=float(crop_camera.near),
        zfar=float(crop_camera.far),
    )
    # Genesis Rasterizer.update_camera writes this field before each render.
    # It is irrelevant to an IntrinsicsCamera projection but accepting it
    # keeps the public Camera.render() path intact.
    camera_node.camera.yfov = np.deg2rad(float(crop_camera.fov))

    return {
        "source_resolution_wh": [source_w, source_h],
        "crop_xyxy": [x0, y0, x1, y1],
        "crop_resolution_wh": [crop_w, crop_h],
        "source_intrinsics": intrinsics.tolist(),
        "crop_intrinsics": crop_intrinsics.tolist(),
        "same_camera_pose": bool(np.array_equal(
            np.asarray(crop_camera.transform), np.asarray(source_camera.transform),
        )),
        "renderer_culling_fov_deg": float(crop_camera.fov),
        "projection_backend": "genesis.ext.pyrender.IntrinsicsCamera on an auxiliary Genesis Rasterizer camera node",
        "candidate_diagnostic_only": True,
        "frozen_simulator_core_modified": False,
    }


def restore_intrinsics_crop_camera(crop_camera: Any) -> None:
    """Restore Genesis' registered camera object before scene destruction.

    The diagnostic replaces ``Node.camera`` only to install exact off-axis
    intrinsics. Pyrender indexes nodes by the original camera object, so the
    original object must be restored before Genesis removes the node.
    """
    original = getattr(crop_camera, "_r1_5_original_camera_node_camera", None)
    if original is None:
        return
    try:
        node = crop_camera._rasterizer._camera_nodes[crop_camera.uid]
    except (AttributeError, KeyError) as exc:
        raise RuntimeError("Genesis camera node disappeared before restoration") from exc
    node.camera = original
    delattr(crop_camera, "_r1_5_original_camera_node_camera")


def crop_render_to_frame_dict(camera: Any, time_s: float) -> Dict[str, Any]:
    """Render HDR, depth and link labels from the configured crop camera."""
    from genesis_event_plugin.radiometry import linear_luminance

    hdr, depth, segmentation, _ = camera.render(
        rgb=True, depth=True, segmentation=True, radiance=True, force_render=True,
    )
    hdr = np.asarray(hdr, dtype=np.float32).copy()
    depth = np.asarray(depth, dtype=np.float32).copy()
    segmentation = np.asarray(segmentation, dtype=np.int32).copy()
    return {
        "time_s": float(time_s),
        "hdr_rgb": hdr,
        "radiance_luma": linear_luminance(hdr),
        "depth": depth,
        "segmentation": segmentation,
    }
