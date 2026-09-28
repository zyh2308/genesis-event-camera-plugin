"""Dependency-light checks for the v3 physics mainline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from genesis_event_plugin.adaptive_sampler import PhysicsAdaptiveSampler
from genesis_event_plugin.calibration import CalibrationBundle
from genesis_event_plugin.dvs_emulator import DvsBatchEmulator, DvsEmulator
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.physics_interpolator import (
    PhysicsInterpolator,
    RenderRequiredError,
)
from genesis_event_plugin.physics_warp import (
    depth_aware_bidirectional_warp_projected,
    interpolate_se3,
    project_endpoint_to_alpha,
)
from genesis_event_plugin.radiometry import (
    RadiometricResponse,
    physical_log_response,
)
from genesis_event_plugin.torch_physics import (
    depth_aware_bidirectional_warp_projected_batch_torch,
)


def _K(size=8):
    return np.array([[10.0, 0, size / 2 - 0.5], [0, 10.0, size / 2 - 0.5], [0, 0, 1]], dtype=float)


def test_physical_log_is_explicitly_not_video_dn20():
    assert not np.allclose(
        physical_log_response(np.array([1.0], dtype=np.float32)),
        np.array([np.log(1.0 / 255.0)], dtype=np.float32),
    )
    response = RadiometricResponse(mode="physical_log", calibration_status="uncalibrated")
    response.validate()


def test_projected_alpha_uses_intermediate_z_for_axial_camera_motion():
    H = W = 8
    depth = np.ones((H, W), dtype=np.float32)
    seg = np.zeros((H, W), dtype=np.int32)
    T0, T1 = np.eye(4), np.eye(4)
    T1[2, 3] = 0.2
    T_alpha = interpolate_se3(T0, T1, 0.5)
    _, z, valid = project_endpoint_to_alpha(depth, seg, _K(), T0, T_alpha, 0.5)
    assert valid.all()
    assert np.allclose(z, 0.9, atol=1e-6)


def test_torch_projected_batch_preserves_all_environments():
    B, T, H, W = 2, 3, 5, 6
    frame0 = torch.arange(B * H * W, dtype=torch.float32).reshape(B, H, W)
    frame1 = frame0 + 1
    depth = torch.ones(B, H, W)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    uv = torch.zeros(B, T, 2, H, W)
    uv[:, :, 0], uv[:, :, 1] = xx, yy
    z = torch.ones(B, T, H, W)
    valid = torch.ones(B, T, H, W, dtype=torch.bool)
    output, mask = depth_aware_bidirectional_warp_projected_batch_torch(
        frame0, depth, uv, z, valid, frame1, depth, uv, z, valid
    )
    assert output.shape == (B, T, H, W)
    assert mask.shape == (B, T, H, W)
    assert torch.allclose(output[:, 0], frame1)


def test_numpy_and_torch_projected_warp_agree_for_b1():
    H = W = 7
    frame0 = np.linspace(0.0, 1.0, H * W, dtype=np.float32).reshape(H, W)
    frame1 = frame0 + 0.2
    depth = np.ones((H, W), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    uv0 = np.stack((xx + 0.25, yy), axis=0).astype(np.float32)
    uv1 = np.stack((xx - 0.25, yy), axis=0).astype(np.float32)
    z = np.ones((H, W), dtype=np.float32)
    valid = np.ones((H, W), dtype=bool)
    reference, reference_mask = depth_aware_bidirectional_warp_projected(
        frame0, depth, uv0, z, valid, frame1, depth, uv1, z, valid,
        composite="next_primary",
    )
    output, output_mask = depth_aware_bidirectional_warp_projected_batch_torch(
        torch.from_numpy(frame0)[None], torch.from_numpy(depth)[None],
        torch.from_numpy(uv0)[None, None], torch.from_numpy(z)[None, None],
        torch.from_numpy(valid)[None, None],
        torch.from_numpy(frame1)[None], torch.from_numpy(depth)[None],
        torch.from_numpy(uv1)[None, None], torch.from_numpy(z)[None, None],
        torch.from_numpy(valid)[None, None], composite="next_primary",
    )
    np.testing.assert_array_equal(reference_mask, output_mask[0, 0].numpy())
    np.testing.assert_allclose(reference, output[0, 0].numpy(), atol=1e-6, equal_nan=True)


def test_batch_dvs_returns_device_packed_offsets():
    cfg = DvsNoiseConfig(
        pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0,
        radiance_input=True, response_mode="physical_log", cutoff_hz=0.0,
    )
    emulator = DvsBatchEmulator((4, 5), cfg, device="cpu")
    first = torch.ones(2, 4, 5)
    emulator.initialize_batch(first, 0.0)
    events, offsets = emulator.generate_events_batch(first * 2.0, 0.01)
    assert events.ndim == 2 and events.shape[1] == 5
    assert offsets.tolist()[0] == 0 and offsets.numel() == 3
    assert offsets.tolist() == sorted(offsets.tolist())
    assert int(offsets[-1]) == events.shape[0]


def test_sampler_combines_radiance_sensor_and_disocclusion_guards():
    H = W = 6
    lum = np.ones((H, W), dtype=np.float32)
    flow = np.zeros((2, H, W), dtype=np.float32)
    cfg = __import__("genesis_event_plugin.adaptive_sampler", fromlist=["AdaptiveSamplingConfig"]).AdaptiveSamplingConfig(
        sensor_time_constant_s=0.001, sensor_eps_target=0.05,
    )
    result = PhysicsAdaptiveSampler(cfg).compute(
        lum, flow, 0.01,
        endpoint_radiance_residual=np.ones((H, W), dtype=np.float32),
        disocclusion_mask=np.pad(np.ones((1, 1), dtype=bool), ((0, 5), (0, 5))),
    )
    assert result.sensor_samples >= 4
    assert result.radiance_samples >= 10
    assert result.recommend_render


def test_strict_interpolator_fails_closed_on_invalid_projection():
    H = W = 8
    depth0 = np.ones((H, W), dtype=np.float32)
    depth1 = np.zeros((H, W), dtype=np.float32)
    seg = np.zeros((H, W), dtype=np.int32)
    interp = PhysicsInterpolator(_K(), disocclusion_policy="request_render", strict_fidelity=True)
    try:
        interp.interpolate(
            np.ones((H, W), dtype=np.float32), depth0, seg,
            np.ones((H, W), dtype=np.float32), depth1, seg,
            np.eye(4), np.eye(4), {}, 0.0, 0.01,
        )
    except RenderRequiredError as exc:
        assert exc.requests
    else:
        raise AssertionError("strict mode did not request a real subframe render")


def test_calibration_bundle_schema_is_fail_closed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "calibration" / "demo" / "v3"
        root.mkdir(parents=True)
        (root / "metadata.json").write_text(json.dumps({
            "schema_version": "genesis-event-calibration-v3",
            "camera_id": "demo", "source_status": "measured",
        }), encoding="utf-8")
        (root / "radiometry.json").write_text(json.dumps({
            "mode": "physical_log", "calibration_status": "effective",
        }), encoding="utf-8")
        bundle = CalibrationBundle.from_directory(str(root))
        assert bundle.radiometric_response().mode == "physical_log"
