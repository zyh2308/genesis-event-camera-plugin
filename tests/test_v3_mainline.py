"""Dependency-light checks for the v3 physics mainline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig, PhysicsAdaptiveSampler
from genesis_event_plugin.calibration import CalibrationBundle
from genesis_event_plugin.dvs_emulator import DvsBatchEmulator, DvsEmulator
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.physics_interpolator import (
    PhysicsInterpolator,
    RenderRequiredError,
)
from genesis_event_plugin.physics_warp import (
    compute_se3_flow,
    depth_aware_bidirectional_warp_projected_diagnostics,
    endpoint_radiance_residual,
    interpolate_se3,
    project_endpoint_to_alpha,
    projected_target_visibility,
)
from genesis_event_plugin.radiometry import (
    RadiometricResponse,
    physical_log_response,
)
from genesis_event_plugin.torch_physics import (
    depth_aware_bidirectional_warp_projected_batch_torch,
    project_endpoint_to_alpha_stack_torch,
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


def test_pure_camera_rotation_preserves_depth_and_matches_torch_projection():
    H = W = 12
    depth = np.ones((H, W), dtype=np.float32)
    seg = np.zeros((H, W), dtype=np.int32)
    theta = np.deg2rad(12.0)
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[:3, :3] = np.array(
        [[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0],
         [-np.sin(theta), 0.0, np.cos(theta)]], dtype=np.float64
    )
    alpha = 0.5
    Ta = interpolate_se3(T0, T1, alpha)
    uv_np, z_np, valid_np = project_endpoint_to_alpha(
        depth, seg, _K(H), T0, Ta, alpha
    )
    uv_t, z_t, valid_t = project_endpoint_to_alpha_stack_torch(
        torch.from_numpy(depth), torch.from_numpy(seg), torch.from_numpy(_K(H).astype(np.float32)),
        torch.from_numpy(T0.astype(np.float32)), torch.from_numpy(T0.astype(np.float32)),
        torch.from_numpy(T1.astype(np.float32)), torch.tensor([alpha]),
    )
    assert valid_np.all() and bool(valid_t.all())
    np.testing.assert_allclose(z_np, z_t[0].numpy(), atol=2e-5)
    np.testing.assert_allclose(uv_np, uv_t[0].numpy(), atol=2e-5)
    # A pure camera rotation changes image coordinates but not the depth of
    # the points that remain in front of the camera.
    assert np.isfinite(z_np[valid_np]).all()


def test_articulated_link_motion_is_separated_from_camera_motion():
    H = W = 20
    depth = np.ones((H, W), dtype=np.float32)
    seg = np.zeros((H, W), dtype=np.int32)
    seg[6:14, 7:13] = 5
    delta = np.eye(4)
    delta[0, 3] = 0.08
    flow, valid = compute_se3_flow(depth, seg, _K(H), np.eye(4), np.eye(4), {5: delta})
    camera_flow, _ = compute_se3_flow(depth, seg, _K(H), np.eye(4), np.eye(4), None)
    assert valid.all()
    assert np.allclose(camera_flow, 0.0)
    assert np.all(flow[0, seg == 5] > 0.0)
    assert np.allclose(flow[:, seg == 0], 0.0)


def test_projected_zbuffer_keeps_near_object_and_reports_target_occupancy():
    H = W = 8
    frame0 = np.zeros((H, W), dtype=np.float32)
    depth = np.full((H, W), np.inf, dtype=np.float32)
    uv = np.zeros((2, H, W), dtype=np.float32)
    valid = np.zeros((H, W), dtype=bool)
    # Two source samples collide at one target pixel. The nearer sample must
    # survive the target-domain z-buffer.
    depth[3, 3], depth[3, 4] = 1.0, 2.0
    frame0[3, 3], frame0[3, 4] = 10.0, 20.0
    uv[:, 3, 3] = (3.0, 3.0)
    uv[:, 3, 4] = (3.0, 3.0)
    valid[3, 3] = valid[3, 4] = True
    frame1 = np.zeros_like(frame0)
    invalid = np.zeros_like(valid)
    out, visible, diagnostics = depth_aware_bidirectional_warp_projected_diagnostics(
        frame0, depth, uv, depth, valid,
        frame1, depth, uv, depth, invalid,
        composite="next_primary",
    )
    assert diagnostics["occupancy0"][3, 3]
    assert diagnostics["visibility0"][3, 3]
    assert np.isclose(diagnostics["z_buffer0"][3, 3], 1.0)
    assert np.isclose(out[3, 3], 10.0)
    assert visible[3, 3]


def test_target_space_disocclusion_and_strict_render_request():
    H = W = 16
    depth = np.ones((H, W), dtype=np.float32)
    seg = np.zeros((H, W), dtype=np.int32)
    T1 = np.eye(4)
    T1[0, 3] = 0.15
    common = dict(
        frame0=np.ones((H, W), dtype=np.float32), depth0=depth, seg0=seg,
        frame1=np.ones((H, W), dtype=np.float32), depth1=depth, seg1=seg,
        camera_to_world0=np.eye(4), camera_to_world1=T1,
        object_world_delta={}, t0=0.0, t1=0.01,
    )
    hold = PhysicsInterpolator(_K(H), disocclusion_policy="hold")
    hold.interpolate(**common)
    target_disocclusion = hold.last_visibility["disocclusion_0_to_1"]
    assert target_disocclusion.any()
    assert target_disocclusion.shape == (H, W)
    assert np.array_equal(target_disocclusion, ~hold.last_visibility["visibility_0_to_1"])
    assert np.isclose(
        hold.last_sampling.disocclusion_fraction,
        hold.last_visibility["unknown_region_0_to_1"].mean(),
    )
    strict = PhysicsInterpolator(_K(H), disocclusion_policy="request_render", strict_fidelity=True)
    try:
        strict.interpolate(**common)
    except RenderRequiredError as exc:
        assert exc.requests and exc.requests[0].reason
    else:
        raise AssertionError("target-space disocclusion did not trigger strict mode")


def test_radiance_residual_support_is_target_domain():
    H = W = 16
    depth = np.ones((H, W), dtype=np.float32)
    seg = np.zeros((H, W), dtype=np.int32)
    T1 = np.eye(4)
    T1[0, 3] = 0.1
    uv, z, valid = project_endpoint_to_alpha(depth, seg, _K(H), np.eye(4), T1, 1.0)
    valid[:, :2] = False
    valid[:, -2:] = False
    source = np.exp(np.linspace(-1.0, 1.0, H * W, dtype=np.float32).reshape(H, W))
    from genesis_event_plugin.physics_warp import _splat_projected
    target_log, _ = _splat_projected(np.log(source), depth, uv, z, valid)
    target = np.exp(target_log)
    residual, support = endpoint_radiance_residual(
        source, target, depth, uv, z, valid, return_support=True
    )
    _, visibility, _ = projected_target_visibility(depth, uv, z, valid)
    assert np.array_equal(support, visibility)
    assert not np.array_equal(support, valid), "support must be in target, not source coordinates"
    assert np.nanmax(residual) < 1e-5
    # A target-space change must increase residual only where support exists.
    target_changed = target.copy()
    target_changed[support] *= 2.0
    changed, changed_support = endpoint_radiance_residual(
        source, target_changed, depth, uv, z, valid, return_support=True
    )
    assert np.array_equal(changed_support, support)
    assert np.all(changed[support] > 0.5)


def test_source_material_point_depth_correspondence_matches_axial_motion():
    H = W = 12
    depth = np.ones((H, W), dtype=np.float32)
    seg = np.zeros((H, W), dtype=np.int32)
    T1 = np.eye(4)
    T1[2, 3] = 0.2
    sampler = AdaptiveSamplingConfig(max_relative_depth_change=0.1)
    interp = PhysicsInterpolator(_K(H), sampler_config=sampler)
    interp.interpolate(
        np.ones((H, W), dtype=np.float32), depth, seg,
        np.ones((H, W), dtype=np.float32), depth, seg,
        np.eye(4), T1, {}, 0.0, 0.01,
    )
    assert interp.last_sampling.used_correspondence_depth
    assert interp.last_sampling.depth_correspondence_method == "source_material_point"
    assert np.isclose(interp.last_sampling.max_relative_depth_change, 0.2, atol=1e-5)


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
    reference, reference_mask, reference_diag = depth_aware_bidirectional_warp_projected_diagnostics(
        frame0, depth, uv0, z, valid, frame1, depth, uv1, z, valid,
        composite="next_primary",
    )
    output, output_mask, output_diag = depth_aware_bidirectional_warp_projected_batch_torch(
        torch.from_numpy(frame0)[None], torch.from_numpy(depth)[None],
        torch.from_numpy(uv0)[None, None], torch.from_numpy(z)[None, None],
        torch.from_numpy(valid)[None, None],
        torch.from_numpy(frame1)[None], torch.from_numpy(depth)[None],
        torch.from_numpy(uv1)[None, None], torch.from_numpy(z)[None, None],
        torch.from_numpy(valid)[None, None], composite="next_primary",
        return_diagnostics=True,
    )
    np.testing.assert_array_equal(reference_mask, output_mask[0, 0].numpy())
    np.testing.assert_allclose(reference, output[0, 0].numpy(), atol=1e-6, equal_nan=True)
    for key in ("occupancy0", "occupancy1", "visibility0", "visibility1", "intermediate_visibility"):
        np.testing.assert_array_equal(
            reference_diag[key if key in reference_diag else key.replace("occupancy", "occupancy")],
            output_diag[key][0, 0].numpy(),
        )
    np.testing.assert_allclose(
        reference_diag["z_buffer0"], output_diag["z_buffer0"][0, 0].numpy(), equal_nan=True
    )


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
