"""Read-only analytic probes of pinned old/new NumPy warp implementations.

These establish behavior on controlled inputs, not downstream task accuracy.
No rendering, training, public-real evaluation or simulator edits are performed.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load("old_warp", ROOT / "source/old/physics_warp.py")
new = load("new_warp", ROOT / "source/new/physics_warp.py")
results = {"old_commit": "e79e656ac00f7847b98db9e12bf43b093f3afff1",
           "new_commit": "62da4f35ce3c5c96f04268bd1687a27657af9238",
           "scope": "controlled analytic warp probes; not downstream scores"}

# A near sample covers 1% of target pixel 0 and a far sample can fill the rest.
# Declared front-to-back footprint expectation is 0.01*1 + 0.99*0.2 = 0.208.
image = np.array([[1., .2]], dtype=np.float32)
depth = np.array([[1., 2.]], dtype=np.float32)
uv = np.array([[[.99, 0.]], [[0., 0.]]], dtype=np.float32)
valid = np.ones((1, 2), bool)
legacy, legacy_diag = old._splat_projected_diagnostics(image, depth, uv, depth, valid)
fixed, fixed_diag = new._splat_projected_coverage_diagnostics(image, depth, uv, depth, valid)
expected = .208
assert abs(float(fixed[0, 0]) - expected) < 1e-6
assert abs(float(legacy[0, 0]) - 1.) < 1e-6
results["partial_coverage"] = {
    "front_fraction": .01, "near_radiance": 1., "far_radiance": .2,
    "expected_radiance": expected, "legacy_radiance": float(legacy[0, 0]),
    "new_radiance": float(fixed[0, 0]),
    "new_target_coverage": float(fixed_diag["coverage"][0, 0]),
}

h, w = 4, 5
yy, xx = np.indices((h, w))
uv = np.stack([xx, yy]).astype(np.float32)
depth = np.ones((h, w), np.float32)
valid = np.ones((h, w), bool)
first = np.ones((h, w), np.float32)
last = np.full((h, w), np.exp(.9), np.float32)
alpha = .25
args = (first, depth, uv, depth, valid, last, depth, uv, depth, valid)
legacy, _, _ = old.depth_aware_bidirectional_warp_projected_diagnostics(
    *args, composite="next_primary")
fixed, supported, _ = new.depth_aware_bidirectional_warp_projected_diagnostics(
    *args, composite="coverage_time_log_linear", alpha=alpha)
expected = float(np.exp(.9 * alpha))
assert supported.all() and np.allclose(fixed, expected, atol=1e-6)
results["temporal_ramp_quarter_interval"] = {
    "start_radiance": 1., "end_radiance": float(last[0, 0]),
    "alpha": alpha, "expected_radiance": expected,
    "legacy_radiance": float(legacy[0, 0]), "new_radiance": float(fixed[0, 0]),
    "threshold_time_ms_for_C_0_2_over_10ms": (np.array([.2, .4, .6, .8])/.9*10).tolist(),
}

# Even a perturbed geometric projection must not change known image endpoints.
perturbed = uv.copy(); perturbed[0] += .4
endpoint_results = []
for alpha, target in [(0., first), (1., last)]:
    x, support, _ = new.depth_aware_bidirectional_warp_projected_diagnostics(
        first, depth, perturbed, depth, valid, last, depth, perturbed, depth, valid,
        composite="coverage_time_log_linear", alpha=alpha)
    assert np.array_equal(x, target) and support.all()
    endpoint_results.append({"alpha": alpha, "exact": True})
results["exact_endpoints"] = endpoint_results

# With only 40% coverage the conditional value is finite but support is unknown.
small = np.array([[2.]], np.float32)
uv_small = np.array([[[.6]], [[0.]]], np.float32)
dd = np.ones((1, 1), np.float32)
value, diagnostics = new._splat_projected_coverage_diagnostics(small, dd, uv_small, dd)
assert np.isfinite(value).all() and not diagnostics["support"].any()
results["unknown_is_not_relabelled_observed"] = {
    "conditional_radiance": float(value[0, 0]),
    "coverage": float(diagnostics["coverage"][0, 0]),
    "supported": bool(diagnostics["support"][0, 0]),
}

# The fix need not change analytic GT flow for an unchanged rigid trajectory.
K = np.array([[15., 0., 2.], [0., 15., 1.5], [0., 0., 1.]])
T = np.eye(4); T[0, 3] = .05
seg = np.ones((h, w), np.int32)
f_old, v_old = old.compute_se3_flow(depth, seg, K, np.eye(4), np.eye(4), {1: T})
f_new, v_new = new.compute_se3_flow(depth, seg, K, np.eye(4), np.eye(4), {1: T})
assert np.array_equal(f_old, f_new) and np.array_equal(v_old, v_new)
results["unchanged_analytic_GT_flow"] = {
    "max_old_new_difference_px": float(np.max(np.abs(f_old-f_new))),
    "expected_horizontal_displacement_px": .75,
    "new_mean_horizontal_displacement_px": float(f_new[0].mean()),
}
results["checks_passed"] = 5
(ROOT / "controlled_review_result.json").write_text(json.dumps(results, indent=2)+"\n")
print(json.dumps(results, indent=2))
