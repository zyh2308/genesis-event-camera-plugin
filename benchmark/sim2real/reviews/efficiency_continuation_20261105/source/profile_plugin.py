"""Separate actual-entry CUDA profile; these observations are never timing results."""
from __future__ import annotations
import argparse
import functools
import importlib
import json
import os
from pathlib import Path
import sys

from measure_plugin import (checked_output, configure_mode, core_sources, load_fixture,
                            measured_run, sha, validate_run, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", default="s2_00")
    parser.add_argument("--intervals", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fixture", type=Path,
                        help="Explicit archived fixture snapshot when copied outside repository")
    args = parser.parse_args()
    if args.intervals < 1 or not args.case.replace("_", "").isalnum():
        parser.error("Positive intervals and a safe case ID are required")
    configure_mode(args.deterministic)
    out = checked_output(args.core, args.out)
    out.mkdir(parents=True, exist_ok=False)
    core = args.core.resolve(strict=True)
    before = core_sources(core)
    fixture = load_fixture(core, out, args.deterministic, args.fixture)
    torch = fixture.torch
    inputs = fixture.Fixture(args.case, args.intervals)
    write_json(out / "physical_inputs.json", inputs.description)
    warm, warm_events = fixture.run_ours(fixture.Fixture(args.case, min(2, args.intervals)), "ours_resident")
    validate_run(warm, fixture.observed[-1], min(2, args.intervals))
    del warm_events
    fixture.gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    targets = [
        ("gpu_geometry", "project_endpoint_to_alpha_stack_torch", "projection"),
        ("gpu_geometry", "_splat_projected_coverage_batch_torch", "coverage_splat"),
        ("gpu_geometry", "depth_aware_bidirectional_warp_projected_batch_torch", "warp"),
        ("gpu_interpolator", "finish_geometry", "finish"),
        ("gpu_sensor", "generate_device_sequence", "sensor"),
    ]
    replaced, calls = [], {}
    for module_name, function_name, short_label in targets:
        module = importlib.import_module("genesis_event_plugin." + module_name)
        original = getattr(module, function_name)
        label = "plugin_phase/" + short_label
        calls[label] = 0
        def wrapped(function, phase):
            @functools.wraps(function)
            def invoke(*positional, **keywords):
                calls[phase] += 1
                with torch.profiler.record_function(phase):
                    return function(*positional, **keywords)
            return invoke
        wrapper = wrapped(original, label)
        # In particular optimized.py holds an imported sensor-function alias.
        # Patch only references to this exact callable within the selected package.
        for name, loaded in list(sys.modules.items()):
            if name == "genesis_event_plugin" or name.startswith("genesis_event_plugin."):
                for attr, value in list(vars(loaded).items()):
                    if value is original:
                        replaced.append((loaded, attr, original))
                        setattr(loaded, attr, wrapper)
    result = {"schema": "separate_actual_plugin_profile_v1", "status": "running",
              "performance_claimed": False, "profile_observation_changes_timing": True}
    write_json(out / "profile.json", result)
    try:
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                profile_memory=True, record_shapes=True) as profiler:
            row, events = measured_run(fixture, inputs)
        validate_run(row, fixture.observed[-1], args.intervals)
        operators = []
        for item in profiler.key_averages():
            operators.append({
                "key": item.key, "count": int(item.count),
                "cpu_total_us": float(item.cpu_time_total),
                "cpu_self_us": float(item.self_cpu_time_total),
                "device_total_us": float(item.device_time_total),
                "device_self_us": float(item.self_device_time_total),
                "cpu_memory_bytes": int(item.cpu_memory_usage),
                "cpu_self_memory_bytes": int(item.self_cpu_memory_usage),
                "device_memory_bytes": int(getattr(item, "device_memory_usage", 0)),
                "device_self_memory_bytes": int(getattr(item, "self_device_memory_usage", 0)),
            })
        if not all(calls.values()):
            raise RuntimeError("At least one required phase was not observed: " + repr(calls))
        if core_sources(core) != before or sha(fixture.fixture_path) != fixture.fixture_sha256:
            raise RuntimeError("Immutable core or archived fixture changed during profiling")
        profiler.export_chrome_trace(str(out / "trace.json"))
        (out / "operators_by_device_self.txt").write_text(
            profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
        result.update(status="complete", case=args.case, intervals=args.intervals,
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                      core=str(core), core_source_sha256=before,
                      index_reuse_v1_anchor=fixture.v1_anchor,
                      fixture_source_path=str(fixture.fixture_path),
                      fixture_source_sha256=fixture.fixture_sha256,
                      fixture_snapshot_sha256=fixture.fixture_snapshot_sha256,
                      profile_path=str(fixture.profile_path), profile_sha256=fixture.profile_sha256,
                      profile=fixture.PROFILE, script_sha256=sha(__file__),
                      build_environment={key: os.environ.get(key) for key in
                                         ("CUDA_HOME", "CC", "CXX", "MAX_JOBS",
                                          "TORCH_CUDA_ARCH_LIST", "EVENTGENESIS_GPU_BUILD_DIR")},
                      phase_calls=calls, operators=operators,
                      input_array_layout={k: {"dtype": str(v.dtype), "shape": list(v.shape), "device": "cpu"}
                                          for k, v in inputs.data.items()},
                      instrumented_run=row, observation=fixture.observed[-1],
                      phase_times_are_inclusive_and_overlap=True,
                      phase_times_must_not_be_summed=True,
                      gpu_name=torch.cuda.get_device_name(),
                      real_data_used=False, training_started=False)
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      phase_calls=calls)
        raise
    finally:
        for module, name, original in reversed(replaced):
            setattr(module, name, original)
        write_json(out / "profile.json", result)
    print("PLUGIN_PROFILE_COMPLETE", json.dumps(calls), flush=True)


if __name__ == "__main__":
    main()
