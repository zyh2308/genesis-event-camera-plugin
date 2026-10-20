#!/usr/bin/env python3
"""Small CPU controls for selective multi-reference scene composition."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from benchmark.robot_closed_loop.diagnostics.r1_5_multiref_surface_candidate import (
    compose_multireference_scene,
    moving_reference_indices,
    prepare_moving_projection_source,
    prepare_static_reference_composite,
    project_moving_source_to_target,
)
from benchmark.robot_closed_loop.diagnostics.r1_5_surface_raster_candidate import (
    prepare_surface_raster_source,
    surface_raster_warp,
)
from genesis_event_plugin.physics_warp import project_endpoint_to_alpha


def _ref(t, rad, depth, label, visible=None):
    rad = np.asarray(rad, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    label = np.asarray(label, dtype=np.int64)
    if visible is None:
        visible = np.isfinite(rad) & np.isfinite(depth) & (label >= 0)
    return {"time_s": float(t), "radiance": rad, "depth": depth,
            "labels": label, "visible": np.asarray(visible, dtype=bool)}


def run_controls() -> dict:
    started = time.perf_counter()
    tests = {}
    assert moving_reference_indices(1, 2, 5, 2) == [1, 2]
    assert moving_reference_indices(1, 2, 5, 5) == [0, 1, 2, 3, 4]
    tests["dynamic_reference_subset_is_bracketing_only"] = {
        "pass": True, "two_view_indices": [1, 2], "all_view_indices": [0, 1, 2, 3, 4],
    }
    # The moving-only projection must agree with the frozen general projector
    # on every target sample; this checks its camera/body transform order.
    h = w = 16
    K = np.asarray([[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]])
    depth = np.ones((h, w), dtype=np.float32)
    labels = np.zeros((h, w), dtype=np.int32)
    labels[5:11, 6:10] = 10
    body0 = np.eye(4)
    body1 = np.eye(4)
    body1[0, 3] = 0.025
    camera = np.eye(4)
    target_body = {10: body1}
    uv, z, valid = project_moving_source_to_target(
        depth, labels, K, camera, camera, {10: body0}, target_body, [10],
    )
    prepared_motion = prepare_moving_projection_source(depth, labels, K, camera, [10])
    prepared_uv, prepared_z, prepared_valid = project_moving_source_to_target(
        depth, labels, K, camera, camera, {10: body0}, target_body, [10],
        prepared_source=prepared_motion,
    )
    expected_uv, expected_z, expected_valid = project_endpoint_to_alpha(
        depth, labels, K, camera, camera, 1.0, 0, {10: body0}, target_body,
    )
    compare = valid & expected_valid
    assert valid.sum() == expected_valid[labels == 10].sum()
    assert np.max(np.abs(uv[:, compare] - expected_uv[:, compare])) < 1e-5
    assert np.max(np.abs(z[compare] - expected_z[compare])) < 1e-6
    assert np.array_equal(valid, prepared_valid)
    assert np.array_equal(uv, prepared_uv) and np.array_equal(z, prepared_z)
    raster_image = np.linspace(0.1, 1.0, h * w, dtype=np.float32).reshape(h, w)
    raster_unprepared = surface_raster_warp(
        raster_image, depth, labels, uv, z, valid, K, raster_labels={10},
    )
    prepared_raster = prepare_surface_raster_source(
        raster_image, depth, labels, K, raster_labels={10},
    )
    raster_prepared = surface_raster_warp(
        raster_image, depth, labels, uv, z, valid, K, raster_labels={10},
        prepared_source=prepared_raster,
    )
    assert np.array_equal(raster_unprepared[3], raster_prepared[3])
    assert np.allclose(raster_unprepared[0][raster_unprepared[3]],
                       raster_prepared[0][raster_prepared[3]], atol=0.0, rtol=0.0)
    tests["moving_projection_matches_frozen_transform"] = {
        "pass": True, "compared_samples": int(compare.sum()),
        "uv_max_abs_error_px": float(np.max(np.abs(uv[:, compare] - expected_uv[:, compare]))),
        "depth_max_abs_error_m": float(np.max(np.abs(z[compare] - expected_z[compare]))),
        "cached_projection_and_raster_bitwise_match": True,
    }

    # A background texel is hidden in one real source view and revealed in the
    # other. Fixed-camera static evidence should recover it without inpainting.
    static0 = _ref(0.0, [[0.2]], [[2.0]], [[3]])
    static1 = _ref(1.0, [[0.0]], [[0.5]], [[10]])
    no_moving = [_ref(0.0, [[np.nan]], [[np.inf]], [[-1]], [[False]]),
                 _ref(1.0, [[np.nan]], [[np.inf]], [[-1]], [[False]])]
    image, valid, request, diag = compose_multireference_scene(
        no_moving, [static0, static1], [10], 0.5,
    )
    json.dumps({key: value for key, value in diag.items()
                 if not isinstance(value, np.ndarray)})
    assert valid[0, 0] and not request[0, 0] and np.isclose(image[0, 0], 0.2)
    tests["complementary_static_surface_recovery"] = {
        "pass": True, "radiance": float(image[0, 0]),
        "support_fraction": diag["composited_supported_fraction"],
        "one_sided_static_sources": diag["static_surface_evidence"]["one_sided_photometric_pixels"],
    }

    # A nearer moving surface must win over a static background; reversing the
    # depth order must reveal the background. This checks both z-buffer orders.
    static_front = [_ref(0.0, [[0.1]], [[1.0]], [[3]])]
    moving_near = [_ref(0.0, [[0.8]], [[0.5]], [[10]])]
    image, valid, request, diag = compose_multireference_scene(
        moving_near, static_front, [10], 0.0,
    )
    assert valid[0, 0] and not request[0, 0] and np.isclose(image[0, 0], 0.8)
    tests["moving_surface_in_front"] = {"pass": True, "chosen_radiance": float(image[0, 0]),
                                         "dynamic_front_fraction": diag["dynamic_front_fraction"]}
    static_behind = [_ref(0.0, [[0.1]], [[2.0]], [[3]])]
    moving_far = [_ref(0.0, [[0.8]], [[3.0]], [[10]])]
    image, valid, request, diag = compose_multireference_scene(
        moving_far, static_behind, [10], 0.0,
    )
    assert valid[0, 0] and not request[0, 0] and np.isclose(image[0, 0], 0.1)
    tests["static_surface_in_front"] = {"pass": True, "chosen_radiance": float(image[0, 0]),
                                         "static_front_fraction": diag["static_front_fraction"]}

    # An equal-depth conflict between different labels is explicitly unknown.
    image, valid, request, diag = compose_multireference_scene(
        [_ref(0.0, [[0.8]], [[1.0]], [[10]])],
        [_ref(0.0, [[0.1]], [[1.0]], [[3]])], [10], 0.0,
    )
    assert not valid[0, 0] and request[0, 0] and np.isnan(image[0, 0])
    tests["equal_depth_label_conflict_requests_render"] = {
        "pass": True, "request_fraction": diag["render_request_fraction"],
    }

    # A query-time log-radiance ramp is interpolated only from the same
    # frontmost surface observed on both sides of the query.
    moving_ramp = [_ref(0.0, [[1.0]], [[0.5]], [[10]]),
                   _ref(1.0, [[np.exp(0.5)]], [[0.5]], [[10]])]
    static_background = [_ref(0.0, [[0.05]], [[2.0]], [[3]]),
                         _ref(1.0, [[0.05]], [[2.0]], [[3]])]
    image, valid, request, diag = compose_multireference_scene(
        moving_ramp, static_background, [10], 0.5,
    )
    assert valid[0, 0] and not request[0, 0] and abs(np.log(image[0, 0]) - 0.25) < 1e-6
    tests["same_surface_log_ramp"] = {"pass": True, "log_radiance": float(np.log(image[0, 0]))}

    # Truly unobserved pixels remain unknown; no spatial color guess is made.
    image, valid, request, diag = compose_multireference_scene(
        no_moving, [_ref(0.0, [[np.nan]], [[np.inf]], [[-1]], [[False]])], [10], 0.0,
    )
    assert not valid[0, 0] and request[0, 0] and np.isnan(image[0, 0])
    tests["unobserved_surface_requests_render"] = {"pass": True,
                                                    "request_fraction": diag["render_request_fraction"]}

    # Static geometry is fixed in this benchmark. The prepared visibility
    # cache must preserve the exact same support, depth, label, and radiance
    # decisions as recomputing the five-view z-selection for every query.
    static_refs = [
        _ref(0.0, [[0.10, 0.20], [0.30, 0.40]],
             [[2.0, 2.0], [2.0, 2.0]], [[3, 3], [3, 3]]),
        _ref(0.5, [[0.12, 0.22], [0.32, 0.42]],
             [[2.0, 2.0], [2.0, 2.0]], [[3, 3], [3, 3]]),
        _ref(1.0, [[0.14, 0.24], [0.34, 0.44]],
             [[2.0, 2.0], [2.0, 2.0]], [[3, 3], [3, 3]]),
    ]
    moving_refs = [
        _ref(0.0, [[np.nan, np.nan], [np.nan, np.nan]],
             [[np.inf, np.inf], [np.inf, np.inf]], [[-1, -1], [-1, -1]],
             [[False, False], [False, False]]),
        _ref(0.5, [[np.nan, 0.9], [np.nan, np.nan]],
             [[np.inf, 1.0], [np.inf, np.inf]], [[-1, 10], [-1, -1]],
             [[False, True], [False, False]]),
        _ref(1.0, [[np.nan, np.nan], [np.nan, np.nan]],
             [[np.inf, np.inf], [np.inf, np.inf]], [[-1, -1], [-1, -1]],
             [[False, False], [False, False]]),
    ]
    static_cache = prepare_static_reference_composite(
        static_refs, [10], [0.25, 0.5, 0.75],
    )
    for q in [0.25, 0.5, 0.75]:
        reference = compose_multireference_scene(moving_refs, static_refs, [10], q)
        cached = compose_multireference_scene(
            moving_refs, static_refs, [10], q, prepared_static=static_cache,
        )
        assert np.array_equal(reference[0], cached[0], equal_nan=True)
        assert np.array_equal(reference[1], cached[1])
        assert np.array_equal(reference[2], cached[2])
        assert np.array_equal(reference[3]["target_labels"], cached[3]["target_labels"])
        assert np.array_equal(reference[3]["target_depth"], cached[3]["target_depth"])
    tests["prepared_static_visibility_matches_uncached_bitwise"] = {
        "pass": True, "queries": 3, "evidence": "radiance/support/request/label/depth exact match",
    }
    return {"suite": "r1_5_multiref_surface_controls_20261021h",
            "tests": tests, "all_pass": all(x["pass"] for x in tests.values()),
            "wall_s": time.perf_counter() - started,
            "frozen_simulator_core_modified": False}


def main() -> int:
    result = run_controls()
    print(json.dumps(result, indent=2))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
