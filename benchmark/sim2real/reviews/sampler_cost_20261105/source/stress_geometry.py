"""Small, strict CUDA regression suite for projected geometry optimizations.

Usage on the isolated GPU test worker::

    result = run_stress(reference_gpu_geometry, {
        "histfree": histfree_gpu_geometry,
        "deadmix": deadmix_gpu_geometry,
        "prune": prune_gpu_geometry,
    })

No release code is patched. Every returned tensor (including every diagnostic)
must have the same dtype, shape, and contiguous raw bytes. NaN payloads and the
sign of zero are consequently compared, rather than relaxed with equal_nan.
Invalid negative depth tolerances must raise the same ValueError on both sides.
Other exceptions fail the case. Nonfinite tolerance probes are observations of
the reference's existing bounded behavior, not assertions that they are valid
physical configuration values. This is correctness testing, not a benchmark.
"""

from __future__ import annotations

import sys
from typing import Any


_FUNCTION = "depth_aware_bidirectional_warp_projected_batch_torch"


def _resolve(module):
    function = getattr(module, _FUNCTION, None)
    if function is None and callable(module):
        function = module
    if not callable(function):
        raise TypeError(f"candidate must expose {_FUNCTION}, or be callable")
    return function


def _invoke(function, arguments, torch):
    """Retain failures, and bound accidental Python loops in NaN-tie probes.

    The old helper bounds its layer loop by its candidate count. A replacement
    while-loop must not hang when NaN tolerance prevents any candidate retiring.
    Trace only the geometry source file, with a deliberately generous line-event
    budget. This never converts an exception to a pass and is not used for timing.
    """
    previous_trace = sys.gettrace()
    source_file = getattr(getattr(function, "__code__", None), "co_filename", None)
    line_events = 0

    def trace(frame, event, arg):
        nonlocal line_events
        if event == "line" and frame.f_code.co_filename == source_file:
            line_events += 1
            if line_events > 100_000:
                raise RuntimeError("stress geometry Python line-event budget exceeded")
        return trace

    try:
        if source_file is not None:
            sys.settrace(trace)
        with torch.no_grad():
            output = function(**arguments)
        torch.cuda.synchronize()
        return {"ok": True, "output": output, "line_events": line_events}
    except Exception as exc:
        return {"ok": False, "exception_type": type(exc).__name__,
                "exception_message": str(exc), "line_events": line_events}
    finally:
        sys.settrace(previous_trace)


def _compare(reference, candidate, torch, path="return"):
    failures = []
    if torch.is_tensor(reference):
        if not torch.is_tensor(candidate):
            return [f"{path}: tensor replaced with {type(candidate).__name__}"]
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            return [f"{path}: shape/dtype {tuple(reference.shape)}/{reference.dtype} "
                    f"!= {tuple(candidate.shape)}/{candidate.dtype}"]
        rbytes = reference.detach().contiguous().reshape(-1).view(torch.uint8)
        cbytes = candidate.detach().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(rbytes, cbytes):
            byte_differences = int((rbytes != cbytes).sum().item())
            failures.append(f"{path}: {byte_differences} raw byte(s) differ")
        return failures
    if isinstance(reference, dict):
        if not isinstance(candidate, dict) or reference.keys() != candidate.keys():
            return [f"{path}: dictionary type/keys differ"]
        for key in reference:
            failures.extend(_compare(reference[key], candidate[key], torch, f"{path}.{key}"))
        return failures
    if isinstance(reference, (tuple, list)):
        if type(candidate) is not type(reference) or len(reference) != len(candidate):
            return [f"{path}: sequence type/length differ"]
        for index, (left, right) in enumerate(zip(reference, candidate)):
            failures.extend(_compare(left, right, torch, f"{path}[{index}]"))
        return failures
    if type(reference) is not type(candidate) or reference != candidate:
        return [f"{path}: non-tensor values differ"]
    return failures


def _clone_arguments(arguments, torch):
    # Isolate candidates from one another even if a candidate mutates its input.
    return {key: value.clone() if torch.is_tensor(value) else value
            for key, value in arguments.items()}


def _base(torch, rgb=False):
    device = torch.device("cuda")
    yy, xx = torch.meshgrid(torch.arange(8, device=device, dtype=torch.float32),
                            torch.arange(8, device=device, dtype=torch.float32), indexing="ij")
    frame = (0.125 + xx * 0.0625 + yy * 0.03125)[None]
    if rgb:
        frame = torch.stack((frame, frame * 2, frame * 4), dim=-1)
    uv = torch.stack((xx, yy), dim=0)[None, None].expand(1, 3, 2, 8, 8).clone()
    depth = torch.ones((1, 8, 8), device=device)
    z = torch.ones((1, 3, 8, 8), device=device)
    valid = torch.ones((1, 3, 8, 8), dtype=torch.bool, device=device)
    surface = (1 + (xx >= 4).long())[None]
    return dict(frame0=frame.clone(), depth0=depth.clone(), uv0=uv.clone(),
                z0=z.clone(), valid0=valid.clone(), frame1=frame.clone() * 2,
                depth1=depth.clone(), uv1=uv.clone(), z1=z.clone(), valid1=valid.clone(),
                composite="coverage_time_log_linear", return_diagnostics=True,
                alpha=torch.tensor([0.25, 0.5, 0.75], device=device, dtype=torch.float64),
                surface_id0=surface.clone(), surface_id1=surface.clone())


def _cases(torch):
    cases = []

    def add(name, arguments, *, expected_error=None, require=None):
        cases.append((name, arguments, expected_error, require))

    for rgb in (False, True):
        tag = "rgb" if rgb else "gray"
        base = _base(torch, rgb)
        add(f"{tag}/identity", _clone_arguments(base, torch))

        hdr = _clone_arguments(base, torch)
        hdr["frame0"] *= 16
        hdr["frame1"] *= 32
        add(f"{tag}/hdr", hdr)

        low = _clone_arguments(base, torch)
        low["frame0"] *= 2.0 ** -25
        low["frame1"] *= 2.0 ** -24
        add(f"{tag}/lowlight_log_floor", low)

        empty = _clone_arguments(base, torch)
        empty["valid0"].zero_()
        empty["valid1"].zero_()
        add(f"{tag}/empty_support", empty, require="empty")

        partial = _clone_arguments(base, torch)
        partial["valid0"].zero_()
        partial["valid1"].zero_()
        partial["valid0"][:, :, 3, 3] = True
        partial["valid1"][:, :, 3, 3] = True
        partial["uv0"][:, :, 0] += 0.25
        partial["uv0"][:, :, 1] += 0.5
        partial["uv1"][:, :, 0] += 0.5
        partial["uv1"][:, :, 1] += 0.25
        add(f"{tag}/partial_coverage", partial, require="partial")

        # Being above the support threshold (.995) is not the same as full
        # compositing coverage. A .99609375 front footprint must still admit
        # the remaining .00390625 of a deeper, differently lit contributor.
        almost_full = _clone_arguments(base, torch)
        for suffix in ("0", "1"):
            almost_full[f"valid{suffix}"].zero_()
            almost_full[f"valid{suffix}"][:, :, 3, 3:5] = True
            almost_full[f"uv{suffix}"][:, :, 0, 3, 3] = 3 + 1 / 256
            almost_full[f"uv{suffix}"][:, :, 0, 3, 4] = 3 + 255 / 256
            almost_full[f"uv{suffix}"][:, :, 1, 3, 3:5] = 3
            almost_full[f"z{suffix}"][:, :, 3, 3] = 1
            almost_full[f"z{suffix}"][:, :, 3, 4] = 2
            almost_full[f"frame{suffix}"][:, 3, 3] = 0.125
            almost_full[f"frame{suffix}"][:, 3, 4] = 16
        add(f"{tag}/support_threshold_is_not_full_coverage", almost_full)

        # At t0/t2 the surface maps agree. At t1 endpoint 1 moves half a
        # frame, so real target-domain coverage and surface matches change.
        crossing = _clone_arguments(base, torch)
        crossing["depth0"][:, :, 4:] = 2
        crossing["depth1"][:, :, 4:] = 2
        crossing["z0"][:, :, :, 4:] = 2
        crossing["z1"][:, :, :, 4:] = 2
        crossing["uv0"][:, 1, 0] += 2
        crossing["uv1"][:, 1, 0] -= 2
        add(f"{tag}/surface_crossing_unknown_visibility_flip", crossing, require="flip")

        # Two source surfaces really collide in one target column; the
        # front-to-back ordering switches in the middle temporal slice.
        layered = _clone_arguments(base, torch)
        for suffix in ("0", "1"):
            layered[f"uv{suffix}"][:, :, 0, :, 3] = 2
            layered[f"surface_id{suffix}"][:, :, 2] = 10
            layered[f"surface_id{suffix}"][:, :, 3] = 11
            layered[f"z{suffix}"][:, :, :, 2] = 1
            layered[f"z{suffix}"][:, :, :, 3] = 2
            layered[f"z{suffix}"][:, 1, :, 2] = 3
        add(f"{tag}/two_depth_occlusion_flip", layered, require="surface_flip")

        tied = _clone_arguments(layered, torch)
        tied["z0"][:, :, :, 3] = 1
        tied["z1"][:, :, :, 3] = 1
        tied["z0"][:, 1, :, 2] = 1
        tied["z1"][:, 1, :, 2] = 1
        add(f"{tag}/equal_depth_conflicting_surface_ids", tied)

        endpoints = _clone_arguments(crossing, torch)
        endpoints["alpha"] = torch.tensor([0.0, 0.5, 1.0], device="cuda", dtype=torch.float64)
        endpoints["depth0"][:, 0, :] = float("nan")
        endpoints["depth1"][:, 1, :] = float("inf")
        endpoints["valid0"][:, :, 2, :] = False
        endpoints["valid1"][:, :, 3, :] = False
        add(f"{tag}/observed_endpoint_override_invalid_depth", endpoints)

        nonfinite = _clone_arguments(base, torch)
        nonfinite["depth0"][0, 0, 0] = float("nan")
        nonfinite["depth1"][0, 0, 1] = float("inf")
        nonfinite["z0"][0, :, 0, 2] = -1
        nonfinite["uv1"][0, :, :, 0, 3] = float("nan")
        if rgb:
            nonfinite["frame0"][0, 0, 4, 1] = float("nan")
            nonfinite["frame1"][0, 0, 5, 2] = float("inf")
        else:
            nonfinite["frame0"][0, 0, 4] = float("nan")
            nonfinite["frame1"][0, 0, 5] = float("inf")
        add(f"{tag}/nonfinite_source_and_projection", nonfinite)

        signed = _clone_arguments(base, torch)
        signed["frame0"] *= -1
        signed["frame1"].zero_()
        signed["frame1"].reshape(-1)[::2] = -0.0
        add(f"{tag}/signed_and_negative_radiance_direct_api", signed)

        # A full front layer accepts one pixel, then multiple tied HDR back
        # values overflow layer_value. The reference still evaluates inf*0.
        # A pruning optimization must keep that behavior, or explicitly fall
        # back for this input; silently returning finite instead is a failure.
        extreme = _clone_arguments(base, torch)
        for suffix in ("0", "1"):
            extreme[f"valid{suffix}"].zero_()
            extreme[f"valid{suffix}"][:, :, 3, :4] = True
            extreme[f"uv{suffix}"][:, :, 0, 3, :4] = 3
            extreme[f"uv{suffix}"][:, :, 1, 3, :4] = 3
            extreme[f"z{suffix}"][:, :, 3, 0] = 1
            extreme[f"z{suffix}"][:, :, 3, 1:4] = 2
            extreme[f"frame{suffix}"][:, 3, 0] = 0.25
            extreme[f"frame{suffix}"][:, 3, 1:4] = 3.0e38
        add(f"{tag}/extreme_finite_hdr_hidden_layer_overflow", extreme)

        # Also test the shared splatter under its other coverage composite.
        same_surface = _clone_arguments(layered, torch)
        same_surface["composite"] = "same_surface_log_linear"
        add(f"{tag}/same_surface_composite", same_surface)

    # Probe a bounded reference behavior for nonfinite ties. These are not
    # declared legal sensor settings. Infinity may admit all candidates into
    # a tie, while NaN may retire none; neither may cause unbounded looping.
    tolerance_base = _base(torch, False)
    tolerance_base["uv0"][:, :, 0, :, 3] = 2
    tolerance_base["uv1"][:, :, 0, :, 3] = 2
    tolerance_base["z0"][:, :, :, 3] = 2
    tolerance_base["z1"][:, :, :, 3] = 2
    unlabeled = _clone_arguments(tolerance_base, torch)
    unlabeled["surface_id0"] = None
    unlabeled["surface_id1"] = None
    add("gray/missing_surface_labels", unlabeled)
    for parameter in ("depth_tie_abs_m", "depth_tie_rtol"):
        for value, label in ((float("nan"), "nan"), (float("inf"), "posinf")):
            arguments = _clone_arguments(tolerance_base, torch)
            arguments[parameter] = value
            add(f"tolerance/{parameter}/{label}", arguments)
        arguments = _clone_arguments(tolerance_base, torch)
        arguments[parameter] = -1.0
        add(f"tolerance/{parameter}/negative", arguments, expected_error="ValueError")
    return cases


def _reference_properties(output, require, torch):
    if require is None:
        return []
    diagnostics = output[2]
    visible = output[1]
    if require == "empty":
        if bool(visible.any()) or bool(diagnostics["occupancy0"].any()):
            return ["fixture did not create empty support"]
    elif require == "partial":
        coverage = diagnostics["coverage0"]
        if not bool(((coverage > 0) & (coverage < 0.995)).any()):
            return ["fixture did not create fractional incomplete coverage"]
    elif require == "flip":
        if not bool(diagnostics["intermediate_unknown"].any()):
            return ["fixture did not create real unknown pixels"]
        if not bool((visible[:, 0] != visible[:, 1]).any()):
            return ["fixture did not create temporal visibility changes"]
    elif require == "surface_flip":
        surface = diagnostics["surface_id0"]
        if not bool((surface[:, 0, :, 2] != surface[:, 1, :, 2]).any()):
            return ["fixture did not create a front-surface ordering flip"]
    return []


def run_stress(reference_module, candidates: dict) -> dict:
    """Run strict small CUDA cases and return a JSON-serializable result.

    No global deterministic settings, seeds, default devices, model state, or
    release files are changed. This function allocates only small test tensors
    on the current CUDA device. Caller selects that device before invoking it.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this helper does not silently substitute CPU")
    reference_function = _resolve(reference_module)
    candidate_functions = {name: _resolve(module) for name, module in candidates.items()}
    result: dict[str, Any] = {
        "schema": "geometry_byte_stress_v1", "device": str(torch.cuda.current_device()),
        "torch_version": torch.__version__, "all_passed": True,
        "comparison": "dtype/shape/contiguous raw uint8 equality, including NaN bits",
        "cases": [], "candidate_summary": {},
    }
    totals = {name: {"passed": 0, "failed": 0} for name in candidate_functions}
    for name, arguments, expected_error, require in _cases(torch):
        reference = _invoke(reference_function, _clone_arguments(arguments, torch), torch)
        row = {"name": name, "expected_error": expected_error,
               "reference_ok": reference["ok"], "candidates": {}, "passed": True}
        if not reference["ok"]:
            row["reference_exception"] = {key: reference[key] for key in
                                          ("exception_type", "exception_message")}
        reference_failures = []
        if expected_error is None:
            if not reference["ok"]:
                reference_failures.append("reference raised unexpectedly")
            else:
                reference_failures = _reference_properties(reference["output"], require, torch)
        elif reference["ok"] or reference.get("exception_type") != expected_error:
            reference_failures.append("reference did not raise expected exception type")
        if reference_failures:
            row["reference_failures"] = reference_failures
            row["passed"] = False
        for candidate_name, function in candidate_functions.items():
            candidate = _invoke(function, _clone_arguments(arguments, torch), torch)
            failures = list(reference_failures)
            if expected_error is not None:
                if candidate["ok"] or candidate.get("exception_type") != expected_error:
                    failures.append("candidate did not raise expected exception type")
                elif not reference["ok"] and (
                    candidate["exception_message"] != reference["exception_message"]
                ):
                    failures.append("negative-tolerance exception message differs")
            elif not candidate["ok"]:
                failures.append("candidate raised unexpectedly")
            elif reference["ok"]:
                failures.extend(_compare(reference["output"], candidate["output"], torch))
            entry = {"passed": not failures, "failures": failures,
                     "line_events": candidate["line_events"]}
            if not candidate["ok"]:
                entry["exception"] = {key: candidate[key] for key in
                                      ("exception_type", "exception_message")}
            row["candidates"][candidate_name] = entry
            totals[candidate_name]["failed" if failures else "passed"] += 1
            if failures:
                row["passed"] = False
        result["cases"].append(row)
        result["all_passed"] = result["all_passed"] and row["passed"]
    result["case_count"] = len(result["cases"])
    result["candidate_summary"] = totals
    return result
