"""Actual frozen CUDA sensor versus independent piecewise-constant RC oracles.

This isolated diagnostic imports the selected package without patching it. The
production 300 Hz bandwidth, .2 thresholds and .5 ms refractory are retained.
Temporal noise and spatial mismatch are explicitly disabled for this named
diagnostic, not for production or an end-to-end efficiency claim.

Example:
  python analytic_sensor_sweep.py --package-root /runs/qualified/candidate_release \
      --out /runs/new/analytic --device cuda:0

Input frames are float32 linear radiance; oracle levels are those SAME rounded
float32 values promoted to double. Sample grids remain unchanged. Query states
are linear interpolation of actual production LP states at the two surrounding
sample times, the same numerical trajectory used to localize signal events.
Only the needed states are captured by splitting calls at existing sample ticks.
A full one-call production run must match the split run bitwise, including RNG.
No timestamps are rounded, shifted, or normalized after generation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch


INTERVAL_S = 0.010
DEFAULT_N = [32, 63, 126, 189, 252, 377, 754, 1508]
SHAPE = (8, 8)
C = 0.2
REFRACTORY_S = 0.0005
FC = 300.0
TAU = 1.0 / (2.0 * math.pi * FC)
STATE_KEYS = ("lp_log_frame", "base_log_frame", "timestamp_mem", "diff_frame",
              "new_frame", "log_new_frame", "pos_thres", "neg_thres",
              "noise_rate_array", "photoreceptor_noise_arr")
BUDGETS = {"state_rmse_over_C": 0.025, "state_p99_over_C": 0.05,
           "event_matching_tolerance_s": 50e-6, "event_f1": 0.99,
           "matched_time_p95_s": 20e-6, "absolute_count_bias_fraction": 0.01}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def levels(initial, sign, magnitude):
    a = float(np.float32(initial))
    b = float(np.float32(a * math.exp(sign * magnitude)))
    return a, b


def make_cases():
    protected, sensitivity = [], []
    for light in (0.08, 0.4, 1.6):
        for sign in (1, -1):
            for magnitude in (0.24, 0.30, 0.36):
                for phase in (0.001137, 0.002911, 0.004073):
                    a, b = levels(light, sign, magnitude)
                    protected.append(dict(name=f"step_I{light:g}_p{sign:+d}_d{magnitude:g}_t{phase:.6f}",
                        group="single_crossing_step", protected=True, initial=a,
                        changes=[[phase, b]], illumination=light, sign=sign,
                        nominal_delta_log=magnitude, phase_s=phase))
        a = float(np.float32(light))
        protected.append(dict(name=f"constant_I{light:g}", group="zero_event_control",
                              protected=True, initial=a, changes=[], illumination=light,
                              sign=0, nominal_delta_log=0.0, phase_s=0.0))
    for light in (0.4, 1.6):
        for sign in (1, -1):
            for width in (80e-6, 400e-6, 1200e-6):
                a, b = levels(light, sign, 0.30)
                phase = 0.002911
                sensitivity.append(dict(name=f"pulse_I{light:g}_p{sign:+d}_w{width:.7f}",
                    group="short_pulse_sensitivity", protected=False, initial=a,
                    changes=[[phase, b], [phase + width, a]], illumination=light,
                    sign=sign, nominal_delta_log=0.30, phase_s=phase, pulse_width_s=width))
    for sign in (1, -1):
        for magnitude in (0.1999, 0.2001):
            for phase in (0.001137, 0.004073):
                a, b = levels(0.4, sign, magnitude)
                sensitivity.append(dict(name=f"near_p{sign:+d}_d{magnitude:g}_t{phase:.6f}",
                    group="near_threshold_sensitivity", protected=False, initial=a,
                    changes=[[phase, b]], illumination=0.4, sign=sign,
                    nominal_delta_log=magnitude, phase_s=phase))
    return {"protected": protected, "sensitivity": sensitivity}


def intensity(case, times):
    result = np.full(np.asarray(times).shape, case["initial"], dtype=np.float32)
    for edge, value in case["changes"]:
        result[np.asarray(times) >= edge] = np.float32(value)
    return result


def segment_end(state, linear_level, duration):
    log_level = math.log(max(linear_level, 1e-6))
    factor = min(max(linear_level, 0.0), 1.0) / TAU
    return log_level + (state - log_level) * math.exp(-factor * duration)


def oracle_states(case, queries):
    # Independent scalar closed-form RC evolution, including true physical edge
    # times that generally fall between the candidate's unchanged sample times.
    answer = []
    for query in queries:
        state, previous, level = math.log(case["initial"]), 0.0, case["initial"]
        for edge, next_level in case["changes"]:
            if edge > query:
                break
            state = segment_end(state, level, edge - previous)
            previous, level = edge, next_level
        answer.append(segment_end(state, level, float(query) - previous))
    return np.asarray(answer, dtype=np.float64)


def oracle_events(case, duration):
    """Exact threshold roots on monotonic constant-input RC segments.

    Cases are intentionally bounded to <2C excursions, so a segment has at most
    one accepted crossing. Any refractory ambiguity or >one crossing fails the
    fixture instead of silently inventing a continuous-time readout convention.
    Return-to-baseline asymptotes are NOT treated as finite threshold crossings.
    """
    state = initial_log = math.log(case["initial"])
    # Keep threshold offsets symbolic/integer. (L0+C)-C can round away from
    # L0 and turn a pulse's infinite-time return asymptote into a fake root.
    threshold_offset = 0
    last = -REFRACTORY_S
    previous, level = 0.0, case["initial"]
    events = []
    segments = list(case["changes"]) + [[duration, None]]
    for edge, next_level in segments:
        if edge > duration:
            edge, next_level = duration, None
        limit = math.log(max(level, 1e-6))
        rate = min(max(level, 0.0), 1.0) / TAU
        end = segment_end(state, level, edge - previous)
        direction = 1 if limit > state else -1 if limit < state else 0
        segment_count = 0
        if direction and rate > 0:
            target = initial_log + (threshold_offset + direction) * C
            # Strictly interior target: equality with the limiting level is an
            # infinite-time asymptote, even if exp() underflows at the horizon.
            interior = state < target < limit if direction > 0 else limit < target < state
            if interior:
                ratio = (target - limit) / (state - limit)
                root = previous - math.log(ratio) / rate
                if previous <= root <= edge:
                    if root - last < REFRACTORY_S:
                        raise RuntimeError(f"Oracle fixture has ambiguous refractory crossing: {case['name']}")
                    events.append([root, direction])
                    last, threshold_offset, segment_count = root, threshold_offset + direction, 1
            another = initial_log + (threshold_offset + direction) * C
            another_inside = (state < another < limit if direction > 0 else limit < another < state)
            if segment_count and another_inside:
                second = previous - math.log((another - limit) / (state - limit)) / rate
                if second <= edge:
                    raise RuntimeError(f"Oracle fixture has multiple segment crossings: {case['name']}")
        state, previous = end, edge
        if next_level is None or previous >= duration:
            break
        level = next_level
    return np.asarray(events, dtype=np.float64).reshape(-1, 2)


def validate_oracle(case, events, duration):
    checks = {"continuous_state_initial_error": abs(float(oracle_states(case, [0.0])[0]) - math.log(case["initial"]))}
    if case["group"] in {"single_crossing_step", "near_threshold_sensitivity"}:
        phase, final = case["changes"][0]
        delta = math.log(final) - math.log(case["initial"])
        if abs(delta) > C:
            # Independently rearranged single-crossing expression.
            direct = phase - TAU / min(final, 1.0) * math.log1p(-C / abs(delta))
            expected_count = int(direct <= duration)
        else:
            direct, expected_count = None, 0
        if len(events) != expected_count:
            raise AssertionError(f"Closed-form oracle event count disagrees: {case['name']}")
        checks["single_crossing_formula_time_s"] = direct
        if expected_count:
            checks["formula_time_error_s"] = abs(float(events[0, 0]) - direct)
            root_state = float(oracle_states(case, [direct])[0])
            checks["threshold_root_state_error"] = abs(root_state - (math.log(case["initial"]) + math.copysign(C, delta)))
            if checks["formula_time_error_s"] > 1e-12 or checks["threshold_root_state_error"] > 1e-12:
                raise AssertionError(f"Independent oracle root check failed: {case['name']}")
        if case["protected"] and expected_count != 1:
            raise AssertionError(f"Protected step does not complete one event: {case['name']}")
    if case["group"] == "zero_event_control" and len(events):
        raise AssertionError("Constant input oracle emitted events")
    return checks


def event_metrics(actual, expected, tolerance):
    matched_errors, nearest_errors = [], []
    missing_same_polarity_candidate = 0
    tp, fp, fn = 0, 0, 0
    for polarity in (1, -1):
        a = np.sort(actual[actual[:, 1] == polarity, 0])
        b = np.sort(expected[expected[:, 1] == polarity, 0])
        for stamp in b:
            if len(a):
                nearest_errors.append(float(np.min(np.abs(a - stamp))))
            else:
                # JSON null explicitly denotes failure / +infinite error for
                # a complete-reference CDF, not an omitted reference sample.
                nearest_errors.append(None)
                missing_same_polarity_candidate += 1
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i] < b[j] - tolerance:
                fp += 1
                i += 1
            elif b[j] < a[i] - tolerance:
                fn += 1
                j += 1
            else:
                tp += 1
                matched_errors.append(abs(float(a[i] - b[j])))
                i += 1
                j += 1
        fp += len(a) - i
        fn += len(b) - j
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    result = dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1,
                  actual_count=len(actual), reference_count=len(expected),
                  count_bias_fraction=(len(actual)-len(expected))/len(expected) if len(expected) else None,
                  zero_event_false_count=len(actual) if not len(expected) else None,
                  complete_reference_success=(fp == 0 and fn == 0),
                  nearest_error_reference_count=len(expected),
                  reference_events_without_same_polarity_candidate=missing_same_polarity_candidate,
                  nearest_error_null_means_failure_not_omission=True,
                  nearest_same_polarity_errors_s=nearest_errors)
    for p in (50, 95, 99):
        result[f"matched_time_p{p}_s"] = float(np.percentile(matched_errors, p)) if matched_errors else None
    return result


def snapshot(emu):
    value, none_keys = {}, []
    for key in STATE_KEYS:
        item = getattr(emu, key)
        if item is None:
            # diff_frame is legitimately None immediately after initialize().
            # Store that explicitly without object arrays / pickle loading.
            none_keys.append(key)
            value[key] = np.empty((0,), dtype=np.float64)
        else:
            value[key] = item.detach().cpu().numpy().copy()
    value["none_state_keys"] = np.asarray(none_keys, dtype="<U64")
    value["rng"] = emu.rng.get_state().detach().cpu().numpy().copy()
    return value


def raw_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def scalar_state(emu):
    return dict(t_previous=float(emu.t_previous), t_start=float(emu.t_start),
                frame_counter=int(emu.frame_counter), num_events_total=int(emu.num_events_total),
                num_events_on=int(emu.num_events_on), num_events_off=int(emu.num_events_off),
                fallback_count=int(getattr(emu, "_candidate_fallbacks", 0)),
                last_fallback_flags=int(getattr(emu, "_last_fallback_flags", 0)))


def flatten_chunks(chunks):
    nonempty = [np.asarray(value, dtype=np.float64) for value in chunks if len(value)]
    return np.concatenate(nonempty, axis=0) if nonempty else np.empty((0, 4), dtype=np.float64)


def make_config(config_class, profile):
    kwargs = dict(profile.get("noise_config_kwargs", {})) if profile else {}
    kwargs.update(pos_thres=C, neg_thres=C, sigma_thres=0.0, cutoff_hz=FC,
        leak_rate_hz=0.0, leak_jitter_fraction=0.0, shot_noise_rate_hz=0.0,
        noise_rate_cov_decades=0.0, refractory_period_s=REFRACTORY_S,
        photoreceptor_noise=False, log_input=False, radiance_input=True,
        radiance_scale=1.0, radiance_white_level=1.0, response_mode="physical_log",
        radiometric_gain=1.0, radiometric_offset=0.0, radiometric_floor=1e-6,
        readout_model="ideal_readout", seed=42)
    config = config_class(**kwargs)
    config.validate()
    if getattr(config, "pixel_noise_map_path", None):
        raise ValueError("Analytic diagnostic must not load an external calibrated noise map")
    return config


def run_one(n, batch_name, cases, queries, duration, emu_class, config, generate, device, out):
    directory = out / batch_name / f"N{n:04d}"
    directory.mkdir(parents=True, exist_ok=False)
    ticks_per_run = int(round(duration / INTERVAL_S))
    times = np.concatenate([j * INTERVAL_S + np.arange(1, n+1, dtype=np.float64) / n * INTERVAL_S
                            for j in range(ticks_per_run)])
    times[-1] = duration
    all_times = np.concatenate(([0.0], times))
    right = np.searchsorted(all_times, queries, side="left").clip(0, len(times))
    left = np.maximum(right - 1, 0)
    stops = sorted(set([0, len(times)] + right.tolist() + left.tolist()))
    initial = np.full(SHAPE, 0.25, dtype=np.float32)
    frames = np.full((len(times), *SHAPE), 0.25, dtype=np.float32)
    for index, case in enumerate(cases):
        y, x = divmod(index, SHAPE[1])
        initial[y, x] = case["initial"]
        frames[:, y, x] = intensity(case, times)
    frames_t = torch.as_tensor(frames, device=device)
    def fresh():
        emu = emu_class(res=SHAPE, config=config, device=device)
        emu.initialize(initial, t=0.0)
        if not bool((emu.pos_thres == C).all()) or not bool((emu.neg_thres == C).all()):
            raise AssertionError("Actual threshold maps differ from the oracle's fixed threshold")
        return emu
    trace_emu = fresh()
    initial_state = snapshot(trace_emu)
    captured = {0: initial_state["lp_log_frame"]}
    chunks = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    previous = 0
    with torch.no_grad():
        for stop in stops:
            if stop == 0:
                continue
            chunks.extend(generate(trace_emu, frames_t[previous:stop], times[previous:stop].tolist()))
            captured[stop] = trace_emu.lp_log_frame.detach().cpu().numpy().copy()
            previous = stop
    torch.cuda.synchronize()
    trace_wall = time.perf_counter() - start
    events = flatten_chunks(chunks)
    padding_events = int(np.count_nonzero(events[:, 2] * SHAPE[1] + events[:, 1] >= len(cases)))
    final_state = snapshot(trace_emu)
    full_emu = fresh()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        full_events = flatten_chunks(generate(full_emu, frames_t, times.tolist()))
    torch.cuda.synchronize()
    single_call_wall = time.perf_counter() - start
    full_state = snapshot(full_emu)
    equality = {key: raw_equal(final_state[key], full_state[key]) for key in final_state}
    equality["events"] = raw_equal(events, full_events)
    scalar_trace, scalar_full = scalar_state(trace_emu), scalar_state(full_emu)
    # fallback counters can count different call boundaries even when the exact
    # reference fallback preserves physical outputs; retain, don't hide them.
    equality["scalar_physical_state"] = all(scalar_trace[key] == scalar_full[key] for key in
        ("t_previous", "t_start", "frame_counter", "num_events_total", "num_events_on", "num_events_off"))
    fraction = np.divide(queries-all_times[left], all_times[right]-all_times[left],
                         out=np.zeros_like(queries), where=all_times[right] != all_times[left])
    lp_queries = np.stack([(1-a)*captured[int(lo)] + a*captured[int(hi)]
                           for lo, hi, a in zip(left, right, fraction)])
    np.save(directory / "events.npy", events)
    np.save(directory / "single_call_events.npy", full_events)
    np.save(directory / "times.npy", times)
    np.save(directory / "query_times.npy", queries)
    np.savez_compressed(directory / "actual_linear_inputs.npz", initial=initial, frames=frames)
    np.save(directory / "actual_lp_queries.npy", lp_queries)
    np.savez_compressed(directory / "initial_sensor_state.npz", **initial_state)
    np.savez_compressed(directory / "final_sensor_state.npz", **final_state)
    np.savez_compressed(directory / "single_call_final_sensor_state.npz", **full_state)
    stop_array = np.asarray(stops, dtype=np.int64)
    np.savez_compressed(directory / "captured_actual_lp_states.npz", sample_indices=stop_array,
                        times=all_times[stop_array], states=np.stack([captured[int(k)] for k in stop_array]))
    receipt = dict(n_per_10ms=n, actual_input_count=len(times), duration_s=duration,
        dt_min_s=float(np.diff(all_times).min()), dt_max_s=float(np.diff(all_times).max()),
        max_unclamped_epsilon=float(np.max(frames.astype(np.float64).clip(0, 1) *
                                             np.diff(all_times)[:, None, None] / TAU)),
        trace_wall_s=trace_wall, single_call_sensor_wall_s=single_call_wall,
        timing_scope="small noise-off analytic batch; not full simulator throughput",
        cuda_path_qualified=(scalar_trace["fallback_count"] == 0 and scalar_full["fallback_count"] == 0),
        split_vs_single_call_byte_equal=equality, trace_scalar_state=scalar_trace,
        single_call_scalar_state=scalar_full, capture_stop_count=len(stops),
        trace_rng_unchanged=raw_equal(initial_state["rng"], final_state["rng"]),
        constant_padding_false_event_count=padding_events,
        sampler_used=False, sampler_cap_used=False, frame_dtype=str(frames.dtype),
        sensor_lp_dtype=str(final_state["lp_log_frame"].dtype), package_threshold=C,
        rng_call_policy="actual gpu_sensor implementation; zero noise must consume no draws")
    json_write(directory / "receipt.json", receipt)
    if not all(equality.values()):
        raise AssertionError(f"Actual split/full sensor byte gate failed at {batch_name} N={n}: {equality}")
    if not receipt["trace_rng_unchanged"]:
        raise AssertionError("Noise-off diagnostic unexpectedly consumed RNG")
    if padding_events:
        raise AssertionError("Constant padding pixels emitted unexpected events")
    rows = []
    oracle_event_rows = []
    for index, case in enumerate(cases):
        y, x = divmod(index, SHAPE[1])
        expected = oracle_events(case, duration)
        oracle_event_rows.extend([[float(stamp), float(x), float(y), float(polarity)]
                                  for stamp, polarity in expected])
        validation = validate_oracle(case, expected, duration)
        actual_pixel = events[(events[:, 1] == x) & (events[:, 2] == y)][:, [0, 3]]
        truth_state = oracle_states(case, queries)
        errors = (lp_queries[:, y, x] - truth_state) / C
        rms, p99 = float(np.sqrt(np.mean(errors**2))), float(np.percentile(np.abs(errors), 99))
        metrics = event_metrics(actual_pixel, expected, BUDGETS["event_matching_tolerance_s"])
        root_ok = (metrics["matched_time_p95_s"] is not None and
                   metrics["matched_time_p95_s"] <= BUDGETS["matched_time_p95_s"]) if len(expected) else len(actual_pixel) == 0
        count_ok = abs(metrics["count_bias_fraction"]) <= BUDGETS["absolute_count_bias_fraction"] if len(expected) else len(actual_pixel) == 0
        passed = (rms <= BUDGETS["state_rmse_over_C"] and p99 <= BUDGETS["state_p99_over_C"]
                  and metrics["f1"] >= BUDGETS["event_f1"] and root_ok and count_ok)
        row = dict(n=n, batch=batch_name, name=case["name"], group=case["group"],
            protected=case["protected"], y=y, x=x, state_rmse_over_C=rms,
            state_p99_over_C=p99, state_max_over_C=float(np.max(np.abs(errors))),
            metrics=metrics, oracle_validation=validation, passes_registered_pilot_budgets=passed,
            cuda_path_qualified=receipt["cuda_path_qualified"],
            qualified_cuda_passing=bool(passed and receipt["cuda_path_qualified"]),
            analytic_events=expected.tolist(), actual_events=actual_pixel.tolist(),
            nominal_delta_log=case["nominal_delta_log"], illumination=case["illumination"],
            sign=case["sign"], phase_s=case["phase_s"])
        rows.append(row)
    json_write(directory / "metrics.json", rows)
    oracle_event_array = np.asarray(oracle_event_rows, dtype=np.float64).reshape(-1, 4)
    if len(oracle_event_array):
        oracle_event_array = oracle_event_array[np.argsort(oracle_event_array[:, 0], kind="stable")]
    np.save(directory / "analytic_events.npy", oracle_event_array)
    np.save(directory / "analytic_lp_queries.npy",
            np.stack([oracle_states(case, queries) for case in cases], axis=1))
    return rows, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n", type=int, nargs="+", default=DEFAULT_N)
    parser.add_argument("--query-count", type=int, default=301)
    parser.add_argument("--duration", type=float, default=0.030)
    parser.add_argument("--cpu-probe-n", type=int, default=63)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new output directory; prior evidence must not be overwritten")
    args.out.mkdir(parents=True)
    try:
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("Actual CUDA sensor is required; no silent CPU substitution")
        if any(n < 1 or n*int(round(args.duration/INTERVAL_S)) >= 32768 for n in args.n):
            raise ValueError("N is outside the actual kernel's validated temporal range")
        if abs(args.duration / INTERVAL_S - round(args.duration / INTERVAL_S)) > 1e-10 or args.duration < 0.030:
            raise ValueError("Duration must be an integer multiple of 10ms and >=30ms")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.cuda.set_device(torch.device(args.device))
        root = args.package_root.resolve()
        if root.name == "genesis_event_plugin":
            root = root.parent
        package = root / "genesis_event_plugin"
        sys.path.insert(0, str(root))
        emulator_module = importlib.import_module("genesis_event_plugin.dvs_emulator")
        noise_module = importlib.import_module("genesis_event_plugin.noise_model")
        sensor_module = importlib.import_module("genesis_event_plugin.gpu_sensor")
        kernel_module = importlib.import_module("genesis_event_plugin.gpu_kernel")
        for module in (emulator_module, noise_module, sensor_module, kernel_module):
            if not Path(module.__file__).resolve().is_relative_to(package):
                raise AssertionError("Imported module is outside the selected package")
        profile_path = args.profile or root / "stage9_sensor_profile.json"
        profile = json.loads(profile_path.read_text()) if profile_path.exists() else None
        config = make_config(noise_module.DvsNoiseConfig, profile)
        groups = make_cases()
        queries = np.linspace(0.0, args.duration, args.query_count, dtype=np.float64)
        for batch in groups.values():
            if len(batch) > math.prod(SHAPE):
                raise AssertionError("Case batch exceeds fixed pixel layout")
            for case in batch:
                validate_oracle(case, oracle_events(case, args.duration), args.duration)
        inventory = [{"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
                      "sha256": sha(path)} for path in sorted(package.rglob("*"))
                     if path.is_file() and path.suffix in {".py", ".cu", ".cpp", ".h", ".json"}]
        registration = dict(schema="analytic_sensor_convergence_v1", package_root=str(root),
            source_inventory=inventory, script_sha256=sha(__file__),
            profile_path=str(profile_path), profile_sha256=sha(profile_path) if profile_path.exists() else None,
            diagnostic_config=asdict(config), production_refractory_retained_s=REFRACTORY_S,
            diagnostic_changes="noise/mismatch off; fixed .2 spatial thresholds; no production profile edits",
            n_per_10ms=args.n, duration_s=args.duration, query_count=args.query_count,
            budgets=BUDGETS, budgets_status="registered pilot engineering budgets, not community consensus",
            device=args.device, device_name=torch.cuda.get_device_name(), torch_version=torch.__version__,
            cuda_version=torch.version.cuda, deterministic_algorithms=True, cases=groups,
            reference="independent exact scalar piecewise-constant RC + threshold roots, no geometry",
            validity="single-event protected steps; pulses/near-threshold independently labeled sensitivity",
            interpolation="actual LP sample states linearly interpolated to fixed absolute queries",
            input_edge_convention="right-continuous physical signal sampled at unchanged grid right endpoints",
            cpu_probe_n_requested=args.cpu_probe_n)
        json_write(args.out / "registration.json", registration)
        # Load the original extension before any reported measurement.
        kernel_module.load_kernel()
        warm = emulator_module.DvsEmulator(res=SHAPE, config=config, device=args.device)
        warm.initialize(np.full(SHAPE, 0.25, dtype=np.float32), t=0.0)
        sensor_module.generate_device_sequence(warm, torch.full((1, *SHAPE), 0.25, device=args.device), [1e-5])
        all_rows, receipts = [], []
        for n in args.n:
            for batch_name, cases in groups.items():
                print(json.dumps({"phase": "start", "N": n, "batch": batch_name}), flush=True)
                rows, receipt = run_one(n, batch_name, cases, queries, args.duration,
                    emulator_module.DvsEmulator, config, sensor_module.generate_device_sequence,
                    args.device, args.out)
                all_rows.extend(rows)
                receipts.append(receipt)
                print(json.dumps({"phase": "done", "N": n, "batch": batch_name,
                    "passed": sum(row["passes_registered_pilot_budgets"] for row in rows),
                    "cases": len(rows), "fallbacks": receipt["trace_scalar_state"]["fallback_count"]}), flush=True)
                json_write(args.out / "partial_metrics.json", all_rows)
        # Independent device bridge check at one density, using the frozen public
        # CPU emulator without changing its model or event localization code.
        cpu_probe = {"requested_n": args.cpu_probe_n, "run": False}
        if args.cpu_probe_n in args.n:
            n = args.cpu_probe_n
            batch = groups["protected"]
            directory = args.out / "protected" / f"N{n:04d}"
            times = np.load(directory / "times.npy")
            cpu = emulator_module.DvsEmulator(res=SHAPE, config=config, device="cpu")
            initial = np.full(SHAPE, 0.25, dtype=np.float32)
            for i, case in enumerate(batch):
                initial[divmod(i, SHAPE[1])] = case["initial"]
            cpu.initialize(initial, t=0.0)
            cpu_chunks = []
            for stamp in times:
                frame = np.full(SHAPE, 0.25, dtype=np.float32)
                for i, case in enumerate(batch):
                    frame[divmod(i, SHAPE[1])] = intensity(case, np.asarray([stamp]))[0]
                cpu_chunks.append(cpu.generate_events(frame, float(stamp)))
            cpu_events = flatten_chunks(cpu_chunks)
            gpu_events = np.load(directory / "events.npy")
            np.save(args.out / "cpu_probe_events.npy", cpu_events)
            cpu_state = snapshot(cpu)
            np.savez_compressed(args.out / "cpu_probe_state.npz", **cpu_state)
            gpu_state = np.load(directory / "final_sensor_state.npz")
            state_errors = {key: float(np.max(np.abs(cpu_state[key].astype(np.float64)-gpu_state[key].astype(np.float64))))
                            for key in STATE_KEYS}
            # CPU/GPU logarithm implementations can differ in final bits. This
            # bridge tolerance is explicitly separate from analytic error budgets.
            by_pixel = lambda data: data[np.lexsort((data[:, 0], data[:, 3], data[:, 2], data[:, 1]))]
            ce, ge = by_pixel(cpu_events), by_pixel(gpu_events)
            count_match = ce.shape == ge.shape
            address_match = count_match and np.array_equal(ce[:, 1:], ge[:, 1:])
            max_time_error = float(np.max(np.abs(ce[:, 0]-ge[:, 0]))) if count_match and len(ce) else 0.0 if count_match else None
            passed = bool(address_match and max_time_error <= 1e-10 and
                          max(state_errors.values()) <= 1e-10)
            cpu_probe = dict(requested_n=n, run=True, passed=passed,
                event_shape_equal=count_match, addresses_polarities_equal=address_match,
                max_event_time_error_s=max_time_error, state_max_abs_errors=state_errors,
                bridge_tolerance=1e-10, bitwise_required=False,
                reason="independent CPU/GPU log last-bit differences; raw arrays retained")
            json_write(args.out / "cpu_probe.json", cpu_probe)
            if not passed:
                raise AssertionError("CPU public emulator bridge check failed")
        summaries = []
        for n in args.n:
            protected = [row for row in all_rows if row["n"] == n and row["protected"]]
            summaries.append(dict(n=n, protected_count=len(protected),
                protected_pass_count=sum(row["passes_registered_pilot_budgets"] for row in protected),
                all_protected_meet_quality_budgets=all(row["passes_registered_pilot_budgets"] for row in protected),
                cuda_path_qualified=all(row["cuda_path_qualified"] for row in protected),
                all_protected_pass=all(row["qualified_cuda_passing"] for row in protected),
                worst_state_rmse_over_C=max(row["state_rmse_over_C"] for row in protected),
                worst_state_p99_over_C=max(row["state_p99_over_C"] for row in protected),
                complete_reference_success_fraction=float(np.mean([row["metrics"]["complete_reference_success"] for row in protected]))))
        qualified = [row["n"] for row in summaries if row["all_protected_pass"]]
        result = dict(status="complete", experiment="noise_off_analytic_sensor_only",
            summaries=summaries, smallest_tested_protected_passing_n=min(qualified) if qualified else None,
            recommendation_scope="analytic piecewise-constant signal tests only; geometry/noise unfinished",
            universal_or_production_step_recommendation=False, cpu_probe=cpu_probe,
            raw_case_count=len(all_rows), script_sha256=sha(__file__))
        json_write(args.out / "metrics.json", all_rows)
        json_write(args.out / "summary.json", result)
        with (args.out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            columns = ["n", "batch", "name", "group", "protected", "illumination", "sign", "phase_s",
                       "state_rmse_over_C", "state_p99_over_C", "state_max_over_C", "passes_registered_pilot_budgets",
                       "cuda_path_qualified", "qualified_cuda_passing"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: row[key] for key in columns} for row in all_rows)
        files = [{"path": str(path.relative_to(args.out)), "size_bytes": path.stat().st_size,
                  "sha256": sha(path)} for path in sorted(args.out.rglob("*")) if path.is_file()]
        json_write(args.out / "artifact_index.json", files)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        json_write(args.out / "FAILURE.json", dict(status="failed", type=type(exc).__name__,
                    message=str(exc), traceback=traceback.format_exc(), script_sha256=sha(__file__)))
        raise


if __name__ == "__main__":
    main()
