"""Smooth log-ramp diagnostic using the actual frozen sensor and an ODE oracle.

This file reuses the already-run analytic_sensor_sweep.py harness, never changes
that file, and patches only its own imported TEST callbacks. No simulator module
is monkey-patched. The independent continuous reference uses SciPy DOP853,
including intensity-dependent bandwidth, and validates a twice-refined solve.

Usage:
  python -B smooth_sensor_sweep.py --package-root /frozen/candidate_release \
      --out /new/smooth_result --device cuda:0
  python -B smooth_sensor_sweep.py --out /new/reference_check --reference-only

Endpoint radiance levels are rounded once to float32. Between those endpoints the
physical signal is continuous and log-linear over the registered ramp duration.
Actual production samples are float32; the ODE integrates the continuous signal,
not a candidate-specific quantized/staircase signal. Quantization is declared.
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

import analytic_sensor_sweep as runner


DEFAULT_N = [63, 126, 189, 252, 377, 754, 1508]
_DURATION = 0.030
_CACHE = {}
_SOLVER = None
_SCIPY_VERSION = None


def make_cases():
    cases = []
    for light in (0.08, 0.4, 1.6):
        for sign in (1, -1):
            for amplitude in (0.24, 0.30, 0.36):
                for width in (0.002, 0.005, 0.010):
                    a, b = runner.levels(light, sign, amplitude)
                    cases.append(dict(name=f"ramp_I{light:g}_p{sign:+d}_d{amplitude:g}_r{width:.3f}",
                        group="smooth_single_crossing_ramp", protected=True,
                        initial=a, final=b, illumination=light, sign=sign,
                        nominal_delta_log=amplitude, phase_s=0.001137,
                        ramp_duration_s=width,
                        effective_delta_log=math.log(b)-math.log(a)))
        a = float(np.float32(light))
        cases.append(dict(name=f"smooth_constant_I{light:g}", group="zero_event_control",
            protected=True, initial=a, final=a, illumination=light, sign=0,
            nominal_delta_log=0.0, phase_s=0.0, ramp_duration_s=0.0,
            effective_delta_log=0.0))
    return {"protected": cases}


def physical_log(case, stamp):
    if not case["sign"]:
        return math.log(case["initial"])
    weight = min(1.0, max(0.0, (float(stamp)-case["phase_s"])/case["ramp_duration_s"]))
    return math.log(case["initial"]) + case["effective_delta_log"] * weight


def intensity(case, times):
    stamps = np.asarray(times, dtype=np.float64)
    if not case["sign"]:
        return np.full(stamps.shape, case["initial"], dtype=np.float32)
    weight = np.clip((stamps-case["phase_s"])/case["ramp_duration_s"], 0.0, 1.0)
    # Preserve exactly the declared endpoint levels, avoiding an exp(log(I))
    # roundtrip at observations outside the open ramp interval.
    values = np.exp(math.log(case["initial"]) + case["effective_delta_log"]*weight)
    values = np.where(weight == 0, case["initial"], np.where(weight == 1, case["final"], values))
    return values.astype(np.float32)


def _initialize_scipy():
    global _SOLVER, _SCIPY_VERSION
    if _SOLVER is None:
        import scipy
        from scipy.integrate import solve_ivp
        _SOLVER, _SCIPY_VERSION = solve_ivp, scipy.__version__


def _solve(case, *, refined):
    _initialize_scipy()
    initial_log = math.log(case["initial"])
    def rhs(stamp, state):
        log_level = physical_log(case, stamp)
        linear_level = math.exp(log_level)
        rate = min(linear_level, 1.0)/runner.TAU
        return [rate*(log_level-state[0])]
    target = initial_log + case["sign"]*runner.C
    def threshold(stamp, state):
        return state[0]-target
    threshold.direction = case["sign"]
    threshold.terminal = False
    base_step = min(0.0002, case["ramp_duration_s"]/16) if case["sign"] else 0.0002
    options = dict(method="DOP853", rtol=5e-12 if refined else 1e-11,
        atol=5e-14 if refined else 1e-13, max_step=base_step/2 if refined else base_step,
        dense_output=True)
    start = time.perf_counter()
    # For a constant control the threshold is not an event; do not attach a
    # zero-valued event function, which would create roots at every solver step.
    solution = _SOLVER(rhs, (0.0, _DURATION), [initial_log],
                       events=threshold if case["sign"] else None, **options)
    if not solution.success:
        raise RuntimeError(f"ODE reference failed: {case['name']}: {solution.message}")
    roots = solution.t_events[0] if case["sign"] else np.empty(0, dtype=np.float64)
    if len(roots) != int(bool(case["sign"])):
        raise AssertionError(f"Smooth oracle must complete exactly one crossing: {case['name']}, roots={roots}")
    if case["sign"]:
        if not runner.C < abs(case["effective_delta_log"]) < 2*runner.C:
            raise AssertionError("Ramp amplitude violates single-crossing oracle design")
        if float(roots[0])+runner.REFRACTORY_S < runner.REFRACTORY_S:
            raise AssertionError("Oracle root precedes sensor initialization")
        root_state_error = abs(float(solution.sol(roots[0])[0])-target)
        if root_state_error > 1e-10:
            raise AssertionError("DOP853 event root does not satisfy its independently stated threshold")
    else:
        root_state_error = 0.0
    metadata = dict(method="DOP853", rtol=options["rtol"], atol=options["atol"],
        max_step_s=options["max_step"], success=bool(solution.success), nfev=int(solution.nfev),
        accepted_mesh_points=len(solution.t), root_times_s=roots.tolist(),
        threshold_root_state_error=root_state_error, wall_s=time.perf_counter()-start)
    return solution, metadata


def reference(case):
    if case["name"] in _CACHE:
        return _CACHE[case["name"]]
    coarse, coarse_meta = _solve(case, refined=False)
    fine, fine_meta = _solve(case, refined=True)
    # A denser reference-only validation grid and both derivative-kink times;
    # candidate scoring still uses the original registered common query grid.
    queries = np.unique(np.concatenate((np.linspace(0.0, _DURATION, 1201),
                        np.asarray([case["phase_s"], case["phase_s"]+case["ramp_duration_s"]]))))
    queries = queries[(queries >= 0) & (queries <= _DURATION)]
    errors = (fine.sol(queries)[0]-coarse.sol(queries)[0])/runner.C
    rmse = float(np.sqrt(np.mean(errors**2)))
    p99 = float(np.percentile(np.abs(errors), 99))
    root_error = abs(fine_meta["root_times_s"][0]-coarse_meta["root_times_s"][0]) if case["sign"] else 0.0
    limits = dict(state_rmse_over_C=0.2*runner.BUDGETS["state_rmse_over_C"],
                  state_p99_over_C=0.2*runner.BUDGETS["state_p99_over_C"],
                  root_time_error_s=0.2*runner.BUDGETS["matched_time_p95_s"])
    passed = bool(rmse < limits["state_rmse_over_C"] and p99 < limits["state_p99_over_C"]
                  and root_error < limits["root_time_error_s"])
    receipt = dict(case=case["name"], passed=passed, coarse=coarse_meta, refined=fine_meta,
        state_rmse_difference_over_C=rmse, state_p99_difference_over_C=p99,
        state_max_difference_over_C=float(np.max(np.abs(errors))),
        event_root_difference_s=root_error, convergence_limits=limits,
        interpretation="20% of pilot error budgets; convergence of numerical ODE reference, not sensor calibration")
    if not passed:
        raise AssertionError(f"ODE reference unresolved; no candidate ranking allowed: {receipt}")
    _CACHE[case["name"]] = (fine, receipt)
    return _CACHE[case["name"]]


def oracle_states(case, queries):
    solution, _ = reference(case)
    return np.asarray(solution.sol(np.asarray(queries, dtype=np.float64))[0], dtype=np.float64)


def oracle_events(case, duration):
    if abs(duration-_DURATION) > 1e-15:
        raise AssertionError("Oracle duration differs from registered horizon")
    _, receipt = reference(case)
    return np.asarray([[stamp, case["sign"]] for stamp in receipt["refined"]["root_times_s"]],
                       dtype=np.float64).reshape(-1, 2)


def validate_oracle(case, events, duration):
    _, receipt = reference(case)
    if len(events) != int(bool(case["sign"])):
        raise AssertionError("Invalid smooth reference event count")
    if len(events) and float(events[0, 0]) < case["phase_s"]:
        raise AssertionError("Continuous reference emitted before the ramp began")
    return dict(reference_converged=receipt["passed"],
        reference_root_refinement_difference_s=receipt["event_root_difference_s"],
        reference_root_state_error=receipt["refined"]["threshold_root_state_error"],
        single_crossing_guarantee="monotonic RC response, C<abs(delta_log)<2C; first event cannot be refractory-blocked")


def cpu_probe(n, cases, config, emulator_class, out, duration):
    directory = out/"protected"/f"N{n:04d}"
    times = np.load(directory/"times.npy")
    initial = np.full(runner.SHAPE, 0.25, dtype=np.float32)
    for index, case in enumerate(cases):
        initial[divmod(index, runner.SHAPE[1])] = case["initial"]
    emulator = emulator_class(res=runner.SHAPE, config=config, device="cpu")
    emulator.initialize(initial, t=0.0)
    chunks = []
    for stamp in times:
        frame = np.full(runner.SHAPE, 0.25, dtype=np.float32)
        for index, case in enumerate(cases):
            frame[divmod(index, runner.SHAPE[1])] = intensity(case, np.asarray([stamp]))[0]
        chunks.append(emulator.generate_events(frame, float(stamp)))
    actual = runner.flatten_chunks(chunks)
    gpu = np.load(directory/"events.npy")
    state = runner.snapshot(emulator)
    np.save(out/"cpu_probe_events.npy", actual)
    np.savez_compressed(out/"cpu_probe_state.npz", **state)
    with np.load(directory/"final_sensor_state.npz") as other:
        state_errors = {key: float(np.max(np.abs(state[key].astype(np.float64)-other[key].astype(np.float64))))
                        for key in runner.STATE_KEYS}
    ordered = lambda values: values[np.lexsort((values[:, 0], values[:, 3], values[:, 2], values[:, 1]))]
    left, right = ordered(actual), ordered(gpu)
    count_ok = left.shape == right.shape
    address_ok = count_ok and np.array_equal(left[:, 1:], right[:, 1:])
    time_error = float(np.max(np.abs(left[:, 0]-right[:, 0]))) if count_ok and len(left) else 0.0 if count_ok else None
    passed = bool(address_ok and time_error <= 1e-10 and max(state_errors.values()) <= 1e-10)
    receipt = dict(n=n, passed=passed, addresses_polarities_equal=address_ok,
        max_event_time_error_s=time_error, state_max_abs_errors=state_errors,
        bridge_tolerance=1e-10, bitwise_required=False,
        reason="separate CPU/GPU arithmetic bridge; raw arrays saved; analytic budgets unchanged")
    runner.json_write(out/"cpu_probe.json", receipt)
    if not passed:
        raise AssertionError("Frozen CPU/GPU smooth-signal bridge failed")
    return receipt


def main():
    global _DURATION
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n", type=int, nargs="+", default=DEFAULT_N)
    parser.add_argument("--query-count", type=int, default=301)
    parser.add_argument("--duration", type=float, default=0.030)
    parser.add_argument("--cpu-probe-n", type=int, default=63)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a fresh result directory; never overwrite prior evidence")
    args.out.mkdir(parents=True)
    try:
        _DURATION = args.duration
        if _DURATION < 0.030 or abs(_DURATION/runner.INTERVAL_S-round(_DURATION/runner.INTERVAL_S)) > 1e-10:
            raise ValueError("Horizon must be >=30ms and an integer multiple of 10ms")
        groups = make_cases()
        runner.intensity, runner.oracle_states = intensity, oracle_states
        runner.oracle_events, runner.validate_oracle = oracle_events, validate_oracle
        reference_receipts = []
        for case in groups["protected"]:
            _, receipt = reference(case)
            reference_receipts.append(receipt)
        reference_result = dict(status="complete", all_passed=all(row["passed"] for row in reference_receipts),
            scipy_version=_SCIPY_VERSION, cases=reference_receipts,
            equation="ds/dt=clamp(I(t),0,1)/tau*(log(I(t))-s); tau=1/(2*pi*300)",
            root="first continuous ON/OFF crossing of L0+sign*.2; no timestamp rounding",
            signal="continuous log-linear ramp between float32-rounded endpoint levels",
            refinement="rtol and atol halved, max_step halved; differences <20% registered pilot budgets")
        runner.json_write(args.out/"reference_convergence.json", reference_result)
        queries = np.linspace(0.0, _DURATION, args.query_count, dtype=np.float64)
        np.save(args.out/"reference_query_times.npy", queries)
        np.save(args.out/"reference_lp_queries.npy", np.stack([oracle_states(case, queries) for case in groups["protected"]], axis=1))
        if args.reference_only:
            runner.json_write(args.out/"summary.json", dict(status="reference_only_complete",
                reference_all_passed=reference_result["all_passed"], case_count=len(groups["protected"]),
                source_sha256=runner.sha(__file__), cases=groups))
            print(json.dumps({"status": "reference_only_complete", "cases": len(reference_receipts)}), flush=True)
            return
        if args.package_root is None:
            raise ValueError("--package-root required for actual sensor execution")
        if not torch.cuda.is_available() or not args.device.startswith("cuda"):
            raise RuntimeError("Actual CUDA required; no silent CPU substitution")
        if any(n < 1 or n*int(round(_DURATION/runner.INTERVAL_S)) >= 32768 for n in args.n):
            raise ValueError("Requested N exceeds the frozen kernel's temporal range")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.cuda.set_device(torch.device(args.device))
        root = args.package_root.resolve()
        if root.name == "genesis_event_plugin":
            root = root.parent
        package = root/"genesis_event_plugin"
        sys.path.insert(0, str(root))
        emulator_module = importlib.import_module("genesis_event_plugin.dvs_emulator")
        noise_module = importlib.import_module("genesis_event_plugin.noise_model")
        sensor_module = importlib.import_module("genesis_event_plugin.gpu_sensor")
        kernel_module = importlib.import_module("genesis_event_plugin.gpu_kernel")
        for module in (emulator_module, noise_module, sensor_module, kernel_module):
            if not Path(module.__file__).resolve().is_relative_to(package):
                raise AssertionError("Wrong package imported")
        profile_path = args.profile or root/"stage9_sensor_profile.json"
        profile = json.loads(profile_path.read_text()) if profile_path.exists() else None
        config = runner.make_config(noise_module.DvsNoiseConfig, profile)
        inventory = [{"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
                      "sha256": runner.sha(path)} for path in sorted(package.rglob("*"))
                     if path.is_file() and path.suffix in {".py", ".cu", ".cpp", ".h", ".json"}]
        runner.json_write(args.out/"registration.json", dict(schema="smooth_sensor_convergence_v1",
            package_root=str(root), source_inventory=inventory, script_sha256=runner.sha(__file__),
            reused_runner_sha256=runner.sha(runner.__file__), diagnostic_config=asdict(config),
            profile_path=str(profile_path), profile_sha256=runner.sha(profile_path) if profile_path.exists() else None,
            cases=groups, n_per_10ms=args.n, duration_s=_DURATION, query_count=args.query_count,
            budgets=runner.BUDGETS, scipy_version=_SCIPY_VERSION, torch_version=torch.__version__,
            device=args.device, device_name=torch.cuda.get_device_name(), deterministic_algorithms=True,
            reference="independent DOP853 + continuous first-crossing root, twice-refined convergence checked",
            signal_quantization="continuous float64 oracle between rounded float32 endpoints; actual samples float32",
            query_interpolation="actual LP states at unchanged surrounding input ticks, linear numerical trajectory",
            temporal_noise="off diagnostic only; production profile and source untouched",
            production_refractory_retained_s=runner.REFRACTORY_S,
            recommendation_scope="smooth analytic sensor signals only; geometry and full-noise gates remain required"))
        kernel_module.load_kernel()
        warm = emulator_module.DvsEmulator(res=runner.SHAPE, config=config, device=args.device)
        warm.initialize(np.full(runner.SHAPE, 0.25, dtype=np.float32), t=0.0)
        sensor_module.generate_device_sequence(warm, torch.full((1, *runner.SHAPE), 0.25, device=args.device), [1e-5])
        rows = []
        for n in args.n:
            print(json.dumps({"phase": "start", "N": n, "signal": "smooth_log_ramp"}), flush=True)
            batch, receipt = runner.run_one(n, "protected", groups["protected"], queries, _DURATION,
                emulator_module.DvsEmulator, config, sensor_module.generate_device_sequence, args.device, args.out)
            rows.extend(batch)
            runner.json_write(args.out/"partial_metrics.json", rows)
            print(json.dumps({"phase": "done", "N": n,
                "quality_passed": sum(row["passes_registered_pilot_budgets"] for row in batch),
                "cases": len(batch), "cuda_path_qualified": receipt["cuda_path_qualified"]}), flush=True)
        bridge = cpu_probe(args.cpu_probe_n, groups["protected"], config, emulator_module.DvsEmulator,
                           args.out, _DURATION) if args.cpu_probe_n in args.n else {"run": False}
        summaries = []
        for n in args.n:
            selected = [row for row in rows if row["n"] == n]
            summaries.append(dict(n=n, protected_count=len(selected),
                protected_pass_count=sum(row["passes_registered_pilot_budgets"] for row in selected),
                all_protected_meet_quality_budgets=all(row["passes_registered_pilot_budgets"] for row in selected),
                cuda_path_qualified=all(row["cuda_path_qualified"] for row in selected),
                all_protected_pass=all(row["qualified_cuda_passing"] for row in selected),
                worst_state_rmse_over_C=max(row["state_rmse_over_C"] for row in selected),
                worst_state_p99_over_C=max(row["state_p99_over_C"] for row in selected),
                complete_reference_success_fraction=float(np.mean([row["metrics"]["complete_reference_success"] for row in selected]))))
        qualified = [row["n"] for row in summaries if row["all_protected_pass"]]
        summary = dict(status="complete", experiment="noise_off_smooth_log_ramp_sensor_only",
            reference_all_converged=reference_result["all_passed"], summaries=summaries,
            smallest_tested_protected_passing_n=min(qualified) if qualified else None,
            universal_or_production_step_recommendation=False, cpu_probe=bridge,
            recommendation_scope="these analytic ramps only; geometry/noise unfinished",
            raw_case_count=len(rows), script_sha256=runner.sha(__file__))
        runner.json_write(args.out/"metrics.json", rows)
        runner.json_write(args.out/"summary.json", summary)
        columns = ["n", "batch", "name", "group", "protected", "illumination", "sign", "phase_s",
            "state_rmse_over_C", "state_p99_over_C", "state_max_over_C", "passes_registered_pilot_budgets",
            "cuda_path_qualified", "qualified_cuda_passing"]
        with (args.out/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: row[key] for key in columns} for row in rows)
        inventory = [{"path": str(path.relative_to(args.out)), "size_bytes": path.stat().st_size,
                      "sha256": runner.sha(path)} for path in sorted(args.out.rglob("*")) if path.is_file()]
        runner.json_write(args.out/"artifact_index.json", inventory)
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        runner.json_write(args.out/"FAILURE.json", dict(status="failed", type=type(exc).__name__,
            message=str(exc), traceback=traceback.format_exc(), script_sha256=runner.sha(__file__)))
        raise


if __name__ == "__main__":
    main()
