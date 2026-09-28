"""Core v2 tests that do not require a Genesis installation or HDF5."""

import sys
import types
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "genesis_event_plugin"
_pkg = types.ModuleType("genesis_event_plugin")
_pkg.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("genesis_event_plugin", _pkg)

from genesis_event_plugin.adaptive_sampler import PhysicsAdaptiveSampler  # noqa: E402
from genesis_event_plugin.dvs_emulator import DvsEmulator  # noqa: E402
from genesis_event_plugin.noise_model import DvsNoiseConfig  # noqa: E402
from genesis_event_plugin.physics_warp import (  # noqa: E402
    compute_se3_flow,
    depth_aware_bidirectional_warp,
)
from genesis_event_plugin.radiometry import prepare_linear_radiance  # noqa: E402
from genesis_event_plugin.torch_physics import (  # noqa: E402
    compute_se3_flow_torch,
    depth_aware_bidirectional_warp_torch,
    prepare_linear_radiance_torch,
)
from genesis_event_plugin.camera_margin import (  # noqa: E402
    CameraMargin,
    crop_margin,
    estimate_required_margin,
    pad_intrinsics,
)


def test_srgb_is_linearized_without_display_gamma():
    rgb = np.array([[[0, 128, 255]]], dtype=np.uint8)
    y = prepare_linear_radiance(rgb, input_space="srgb")
    assert 0.20 < float(y[0, 0]) < 0.25


def test_se3_object_translation_produces_dense_flow():
    H = W = 20
    K = np.array([[15.0, 0, 9.5], [0, 15.0, 9.5], [0, 0, 1]])
    depth = np.ones((H, W), dtype=np.float32)
    seg = np.ones((H, W), dtype=np.int32)
    delta = np.eye(4)
    delta[0, 3] = 0.05
    flow, valid = compute_se3_flow(
        depth, seg, K, np.eye(4), np.eye(4), {1: delta}
    )
    assert valid.all()
    assert np.all(flow[0] > 0.0)


def test_sampler_increases_samples_at_log_contrast_edge():
    H = W = 32
    lum = np.ones((H, W), dtype=np.float32) * 0.5
    lum[:, W // 2 :] = 1.0
    flow = np.zeros((2, H, W), dtype=np.float32)
    flow[0] = 2.0
    result = PhysicsAdaptiveSampler().compute(lum, flow, 0.001)
    assert result.samples > result.contrast_samples - 1
    assert result.samples >= 2


def test_sampler_reports_camera_and_articulation_bounds_separately():
    H = W = 8
    lum = np.ones((H, W), dtype=np.float32)
    zero = np.zeros((2, H, W), dtype=np.float32)
    camera = np.zeros_like(zero)
    camera[0] = 5.0
    articulation = np.zeros_like(zero)
    articulation[1] = 3.0
    result = PhysicsAdaptiveSampler().compute(
        lum,
        zero,
        0.001,
        camera_flow=camera,
        articulation_flow=articulation,
    )
    assert result.camera_samples == 5
    assert result.articulation_samples == 3
    assert result.samples == 5


def test_depth_aware_warp_returns_finite_visible_frame():
    H = W = 24
    image = np.zeros((H, W), dtype=np.float32)
    image[:, W // 2 :] = 1.0
    depth = np.ones((H, W), dtype=np.float32)
    flow = np.zeros((2, H, W), dtype=np.float32)
    flow[0] = 1.0
    warped, valid = depth_aware_bidirectional_warp(
        image, depth, flow, image, depth, -flow, 0.5
    )
    assert warped.shape == image.shape
    assert valid.any()
    assert np.isfinite(warped[valid]).all()


def test_next_primary_warp_uses_next_keyframe_without_crossfade():
    H = W = 12
    frame0 = np.zeros((H, W), dtype=np.float32)
    frame1 = np.ones((H, W), dtype=np.float32)
    depth = np.ones((H, W), dtype=np.float32)
    flow = np.zeros((2, H, W), dtype=np.float32)
    warped, valid = depth_aware_bidirectional_warp(
        frame0,
        depth,
        flow,
        frame1,
        depth,
        flow,
        0.5,
        composite="next_primary",
    )
    assert valid.all()
    assert np.allclose(warped, frame1)


def test_physics_interpolator_torch_backend_matches_contract():
    from genesis_event_plugin.physics_interpolator import PhysicsInterpolator

    H = W = 12
    K = np.array([[10.0, 0, 5.5], [0, 10.0, 5.5], [0, 0, 1.0]])
    frame0 = np.zeros((H, W), dtype=np.float32)
    frame1 = np.ones((H, W), dtype=np.float32)
    depth = np.ones((H, W), dtype=np.float32)
    seg = np.zeros((H, W), dtype=np.int32)
    interp = PhysicsInterpolator(
        K,
        device="cuda" if __import__('torch').cuda.is_available() else "cpu",
        warp_backend="torch",
    )
    frames, timestamps = interp.interpolate(
        frame0, depth, seg, frame1, depth, seg,
        np.eye(4), np.eye(4), {}, 0.0, 0.01,
    )
    assert len(frames) == len(timestamps) >= 2
    assert all(frame.shape == (H, W) for frame in frames)


def test_dvs_accepts_unclipped_linear_radiance():
    cfg = DvsNoiseConfig(
        pos_thres=0.05,
        neg_thres=0.05,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        radiance_input=True,
        radiance_scale=100.0,
        radiance_white_level=100.0,
        seed=7,
    )
    emu = DvsEmulator(res=(8, 8), config=cfg, device="cpu")
    first = np.ones((8, 8), dtype=np.float32) * 0.5
    second = first * 2.0
    emu.initialize(first, 0.0)
    events = emu.generate_events(second, 0.01)
    assert events.ndim == 2 and events.shape[1] == 4
    if len(events):
        assert events[:, 0].min() >= 0.0
        assert events[:, 0].max() <= 0.01


def test_torch_physics_matches_shapes_and_radiance():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rgb = torch.tensor([[[0, 128, 255]]], device=device, dtype=torch.uint8)
    y = prepare_linear_radiance_torch(rgb, input_space="srgb")
    assert 0.20 < float(y.item()) < 0.25
    H = W = 16
    K = torch.tensor([[12.0, 0, 7.5], [0, 12.0, 7.5], [0, 0, 1]], device=device)
    depth = torch.ones((H, W), device=device)
    seg = torch.ones((H, W), device=device, dtype=torch.int32)
    delta = torch.eye(4, device=device)
    delta[0, 3] = 0.03
    flow, valid = compute_se3_flow_torch(
        depth, seg, K, torch.eye(4, device=device), torch.eye(4, device=device), {1: delta}
    )
    assert bool(valid.all())
    image = torch.zeros((H, W), device=device)
    image[:, W // 2:] = 1.0
    warped, warp_valid = depth_aware_bidirectional_warp_torch(
        image, depth, flow, image, depth, -flow, 0.5
    )
    assert warped.shape == image.shape
    assert bool(warp_valid.any())


def test_camera_margin_expands_intrinsics_and_crops_back():
    flow = np.zeros((2, 4, 5), dtype=np.float32)
    flow[0] = 3.2
    flow[1] = -1.1
    margin = estimate_required_margin(flow, safety_px=1.0)
    assert margin == CameraMargin(left=5, right=5, top=3, bottom=3)
    K = np.array([[10.0, 0.0, 2.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]])
    shifted = pad_intrinsics(K, margin)
    assert shifted[0, 2] == K[0, 2] + margin.left
    assert shifted[1, 2] == K[1, 2] + margin.top
    image = np.arange(12 * 14, dtype=np.float32).reshape(12, 14)
    cropped = crop_margin(image, margin)
    assert cropped.shape == (6, 4)
