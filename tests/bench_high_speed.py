"""High-speed, renderer-free validation of the Genesis v2 physics path.

The benchmark uses a textured background plus one front object.  The object
or camera is moved by a known pixel displacement over one keyframe interval,
so the test can check both the physical flow decomposition and the generated
intermediate frames without requiring a Genesis installation.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "genesis_event_plugin"
_pkg = types.ModuleType("genesis_event_plugin")
_pkg.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("genesis_event_plugin", _pkg)

from genesis_event_plugin.dvs_emulator import DvsEmulator  # noqa: E402
from genesis_event_plugin.noise_model import DvsNoiseConfig  # noqa: E402
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator  # noqa: E402
from genesis_event_plugin.camera_margin import estimate_required_margin  # noqa: E402


def _shift_x(image: np.ndarray, shift: float) -> np.ndarray:
    """Subpixel horizontal translation with black outside the source view."""
    image = np.asarray(image, dtype=np.float32)
    H, W = image.shape
    x = np.arange(W, dtype=np.float32)
    source_x = x - float(shift)
    out = np.zeros_like(image)
    for y in range(H):
        out[y] = np.interp(source_x, x, image[y], left=0.0, right=0.0)
    return out


def _scene(H=96, W=128):
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    background = 0.12 + 0.28 * (xx / max(W - 1, 1))
    background += 0.05 * np.sin(xx / 3.0) * np.cos(yy / 7.0)
    object0 = np.zeros((H, W), dtype=np.float32)
    object0[H // 3: 2 * H // 3, W // 5: 2 * W // 5] = 0.55
    depth0 = np.ones((H, W), dtype=np.float32)
    depth0[object0 > 0] = 0.6
    seg0 = np.zeros((H, W), dtype=np.int32)
    seg0[object0 > 0] = 1
    return background.astype(np.float32), object0, depth0, seg0


def _case(kind: str, displacement_px: float):
    background, object0, depth0, seg0 = _scene()
    H, W = background.shape
    fx = 120.0
    K = np.array([[fx, 0.0, (W - 1) / 2], [0.0, fx, (H - 1) / 2], [0.0, 0.0, 1.0]])
    camera0 = np.eye(4, dtype=np.float64)
    camera1 = np.eye(4, dtype=np.float64)
    object_delta = {}

    if kind == "articulation":
        # The object is at z=0.6 m, hence this world translation yields the
        # requested image displacement under the pinhole model.
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = displacement_px * 0.6 / fx
        object_delta[1] = T
        frame0 = background + object0
        frame1 = background + _shift_x(object0, displacement_px)
        shifted_object = _shift_x((seg0 == 1).astype(np.float32), displacement_px) > 0.5
        depth1 = np.ones_like(depth0)
        depth1[shifted_object] = 0.6
        seg1 = np.zeros_like(seg0)
        seg1[shifted_object] = 1
    elif kind == "camera":
        # Moving the camera left makes static content move right in the image.
        camera1[0, 3] = -displacement_px / fx
        frame0 = background + object0
        frame1 = _shift_x(frame0, displacement_px)
        depth1 = _shift_x(depth0, displacement_px)
        seg1 = np.zeros_like(seg0)
        seg1[_shift_x((seg0 == 1).astype(np.float32), displacement_px) > 0.5] = 1
    else:
        raise ValueError(kind)
    return K, frame0, depth0, seg0, frame1, depth1, seg1, camera0, camera1, object_delta


def _run(kind: str, displacement_px: float) -> dict:
    values = _case(kind, displacement_px)
    K, frame0, depth0, seg0, frame1, depth1, seg1, camera0, camera1, object_delta = values
    interp = PhysicsInterpolator(K, disocclusion_policy="hold", composite="next_primary")
    start = time.perf_counter()
    frames, timestamps = interp.interpolate(
        frame0,
        depth0,
        seg0,
        frame1,
        depth1,
        seg1,
        camera0,
        camera1,
        object_delta,
        0.0,
        0.01,
        input_space="linear",
    )
    elapsed = time.perf_counter() - start

    # A clean deterministic DVS is used only to check that timestamps and
    # event generation remain well-formed at high speed.
    cfg = DvsNoiseConfig(
        pos_thres=0.05,
        neg_thres=0.05,
        sigma_thres=0.0,
        radiance_input=True,
        radiance_scale=1.0,
        radiance_white_level=1.0,
        seed=17,
    )
    def _event_stream(sequence):
        emulator = DvsEmulator(res=frame0.shape, config=cfg, device="cpu")
        emulator.initialize(frame0, 0.0)
        chunks = []
        for image, timestamp in sequence:
            chunk = emulator.generate_events(image, timestamp)
            if len(chunk):
                chunks.append(chunk)
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)

    events = _event_stream(zip(frames, timestamps))

    maes = []
    direct_frames = []
    for frame, timestamp in zip(frames, timestamps):
        alpha = timestamp / 0.01
        if kind == "articulation":
            target = _case(kind, displacement_px)[1] * 0.0  # shape only
            target = _case(kind, displacement_px)[1]  # frame0
            background, object0, *_ = _scene()
            target = background + _shift_x(object0, displacement_px * alpha)
        else:
            target = _shift_x(frame0, displacement_px * alpha)
        direct_frames.append(target.astype(np.float32))
        maes.append(float(np.mean(np.abs(frame - target))))

    direct_events = _event_stream(zip(direct_frames, timestamps))
    candidate_event_count = int(len(events))
    direct_event_count = int(len(direct_events))

    margin = estimate_required_margin(interp.last_flow01, safety_px=1.0)
    result = {
        "kind": kind,
        "displacement_px": float(displacement_px),
        "U": int(interp.last_sampling.samples),
        "U_contrast": int(interp.last_sampling.contrast_samples),
        "U_visibility": int(interp.last_sampling.visibility_samples),
        "U_camera": int(interp.last_sampling.camera_samples),
        "U_articulation": int(interp.last_sampling.articulation_samples),
        "U_uncapped": int(interp.last_sampling.uncapped_samples),
        "U_capped": bool(interp.last_sampling.capped),
        "max_camera_flow_px": float(interp.last_sampling.max_camera_flow_px),
        "max_articulation_flow_px": float(interp.last_sampling.max_articulation_flow_px),
        "suggested_symmetric_margin": {
            "left": margin.left,
            "right": margin.right,
            "top": margin.top,
            "bottom": margin.bottom,
        },
        "invalid_fraction": float(interp.last_invalid_fraction),
        "mean_frame_mae": float(np.mean(maes)),
        "max_frame_mae": float(np.max(maes)),
        "num_events": int(len(events)),
        "direct_num_events": direct_event_count,
        "event_count_abs_error": abs(candidate_event_count - direct_event_count),
        "event_count_relative_error": float(
            abs(candidate_event_count - direct_event_count) / max(direct_event_count, 1)
        ),
        "timestamps_monotonic": bool(
            len(events) < 2 or np.all(np.diff(events[:, 0]) >= -1e-12)
        ),
        "elapsed_ms": float(elapsed * 1000.0),
        "ms_per_synth_frame": float(elapsed * 1000.0 / max(len(frames), 1)),
    }
    return result


def main():
    results = []
    for kind in ("articulation", "camera"):
        for displacement in (4.0, 16.0, 32.0, 64.0):
            results.append(_run(kind, displacement))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
