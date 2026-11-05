"""Fresh-process actual-plugin timing, using the archived immutable fixture.

No GPU work is performed by the coordinator. Each trial runs in a new Python
process, sets its numerical mode before Torch import, warms the private kernel,
then creates a fresh plugin. The fixture's four function/class bodies are
compiled unchanged; its legacy import-path mutations and V2E code are not run.
"""
from __future__ import annotations
import argparse
import ast
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

FIXTURE_NAMES = ("digest", "Fixture", "validate", "run_ours")
SOURCE_SUFFIXES = {".py", ".json", ".cu", ".cpp"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def configure_mode(deterministic):
    if "torch" in sys.modules:
        raise RuntimeError("Numerical mode must be configured before importing Torch")
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
                      MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    sys.dont_write_bytecode = True


def checked_output(core, out):
    core = core.resolve(strict=True)
    out = out.resolve()
    if out == core or core in out.parents or out in core.parents:
        raise ValueError("Output and immutable core must be disjoint")
    return out


def core_sources(core):
    return {str(p.relative_to(core)): sha(p) for p in sorted(core.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in SOURCE_SUFFIXES}


def json_value(value):
    if dataclasses.is_dataclass(value):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return value.item()
    return str(value)


def load_fixture(core, out, deterministic, fixture_path=None):
    """Import one requested static package, then compile unchanged fixture bodies."""
    import cv2
    import h5py
    import numpy as np
    import torch
    core = core.resolve(strict=True)
    build = os.environ.get("EVENTGENESIS_GPU_BUILD_DIR")
    if not build:
        raise RuntimeError("An explicitly isolated EVENTGENESIS_GPU_BUILD_DIR is required")
    checked_output(core, Path(build))
    for key in ("CUDA_HOME", "CC", "CXX"):
        if not os.environ.get(key):
            raise RuntimeError("Use the recorded private toolchain; missing " + key)
    if os.environ.get("MAX_JOBS") != "2" or os.environ.get("TORCH_CUDA_ARCH_LIST") != "7.0":
        raise RuntimeError("This V100S experiment requires MAX_JOBS=2 and TORCH_CUDA_ARCH_LIST=7.0")
    if any(k == "genesis_event_plugin" or k.startswith("genesis_event_plugin.") for k in sys.modules):
        raise RuntimeError("Each package must run in a genuinely fresh process")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("An explicit coordinated CUDA_VISIBLE_DEVICES is required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one explicitly assigned CUDA device is required")
    profile_path = core / "stage9_sensor_profile.json"
    profile = json.loads(profile_path.read_text())
    for name, expected in profile["core_canonical_LF_sha256"].items():
        actual = hashlib.sha256((core / "genesis_event_plugin" / name).read_text().encode()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Unchanged reference source mismatch: {name}")
    sys.path.insert(0, str(core))
    from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
    from genesis_event_plugin.noise_model import DvsNoiseConfig
    from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
    from genesis_event_plugin.optimized import OptimizedGenesisPhysicsEventPlugin
    for name, module in list(sys.modules.items()):
        if name == "genesis_event_plugin" or name.startswith("genesis_event_plugin."):
            location = getattr(module, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(core):
                raise RuntimeError(f"Package import escaped requested core: {name}")
    if fixture_path is None:
        fixture_path = Path(__file__).resolve().parents[2] / "sampler_cost_20261105/source/compare_static_template.py"
    fixture_path = Path(fixture_path).resolve(strict=True)
    manifest_path = fixture_path.parent.parent / "release/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    archived = {entry["path"]: entry["sha256"] for entry in manifest["candidate_files"]}
    allowed = "genesis_event_plugin/gpu_geometry.py"
    for relative, expected in archived.items():
        if relative != allowed and sha(core / relative) != expected:
            raise RuntimeError("Non-geometry source changed from archived index-reuse v1: " + relative)
    v1_anchor = {"manifest_path": str(manifest_path), "manifest_sha256": sha(manifest_path),
                 "archived_geometry_sha256": archived[allowed],
                 "actual_geometry_sha256": sha(core / allowed),
                 "all_other_manifest_files_byte_identical": True}
    source = fixture_path.read_text()
    replacements = {
        "CORE=ROOT/'release_candidate_static'": "CORE=Path(" + repr(str(core)) + ")",
        "ROOT=Path(__file__).resolve().parent": "ROOT=Path(" + repr(str(out)) + ")",
        "sys.path[:0]=[str(CORE),str(OLD),str(SRC)]": "sys.path.insert(0,str(CORE))",
    }
    adapted = source
    for old, new in replacements.items():
        if adapted.count(old) != 1:
            raise RuntimeError("Unexpected archived fixture structure: " + old)
        adapted = adapted.replace(old, new)
    snapshot = out / "fixture_snapshot.py"
    snapshot.write_text(adapted)
    tree = ast.parse(adapted)
    nodes = []
    for name in FIXTURE_NAMES:
        matches = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one unchanged fixture definition: {name}")
        nodes.append(matches[0])
    root = Path("/home/zhaoyuhan/Eventbased_WAM")
    observed = []
    class ObservedPlugin(OptimizedGenesisPhysicsEventPlugin):
        def close(self):
            # run_ours calls close only after the timed CUDA synchronization.
            emu = self.dvs_emulator
            tensors = {k: {"dtype": str(v.dtype), "device": str(v.device), "shape": list(v.shape)}
                       for k, v in vars(emu).items() if torch.is_tensor(v)}
            observed.append({
                "optimization_active": bool(self.optimization_active),
                "optimization_status": json_value(self.optimization_status),
                "plugin_fallbacks": int(getattr(self, "fallbacks", 0)),
                "sensor_fallbacks": int(getattr(emu, "_candidate_fallbacks", 0)),
                "last_fallback_flags": int(getattr(emu, "_last_fallback_flags", 0)),
                "last_sampling": json_value(self.interpolator.last_sampling),
                "plugin_total_steps": int(self.total_steps),
                "plugin_total_events": int(self.total_events),
                "actual_sensor_config": json_value(emu.cfg),
                "sensor_tensor_layout": tensors,
            })
            return super().close()
    namespace = dict(__name__="archived_actual_plugin_fixture", __file__=str(snapshot),
                     Path=Path, ROOT=out, CORE=core, P=root,
                     R=root / "runs/sim2real/20261103-stage9-direct-v1",
                     PROFILE=profile, SimpleNamespace=SimpleNamespace,
                     cv2=cv2, h5py=h5py, np=np, torch=torch, gc=gc, time=time,
                     hashlib=hashlib, Resident=ObservedPlugin,
                     GenesisPhysicsEventPlugin=GenesisPhysicsEventPlugin,
                     DvsNoiseConfig=DvsNoiseConfig, AdaptiveSamplingConfig=AdaptiveSamplingConfig)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(snapshot), "exec"), namespace)
    return SimpleNamespace(**namespace, observed=observed,
                           fixture_path=fixture_path, fixture_sha256=sha(fixture_path),
                           fixture_snapshot_sha256=sha(snapshot),
                           profile_path=profile_path, profile_sha256=sha(profile_path),
                           v1_anchor=v1_anchor)


def validate_run(row, observation, expected_intervals):
    if row["fallbacks"] or observation["plugin_fallbacks"] or observation["sensor_fallbacks"]:
        raise RuntimeError("Reference fallback occurred; this is not a fast-path result")
    if not observation["optimization_active"]:
        raise RuntimeError("Actual optimized path is inactive")
    if len(row["substeps"]) != expected_intervals or set(row["substeps"]) != {377}:
        raise RuntimeError("Expected uncapped native377 sample grid on every interval")
    sampling = observation["last_sampling"]
    if isinstance(sampling, dict) and sampling.get("capped"):
        raise RuntimeError("Native sampler was capped")
    if row["events"]["dtype"] != "float64":
        raise RuntimeError("Public raw events must remain native FP64")


def measured_run(fixture, inputs):
    """Keep the archived setup/generation timers; collect peaks at their boundary."""
    torch = fixture.torch
    boundary = {}
    original_reset = torch.cuda.reset_peak_memory_stats
    def reset(device=None):
        boundary.update(setup_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                        setup_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
        return original_reset(device)
    original_reset()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats = reset
    try:
        row, events = fixture.run_ours(inputs, "ours_resident")
    finally:
        torch.cuda.reset_peak_memory_stats = original_reset
    boundary.update(generation_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                    generation_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    boundary["pipeline_peak_allocated_bytes"] = max(boundary["setup_peak_allocated_bytes"], boundary["generation_peak_allocated_bytes"])
    boundary["pipeline_peak_reserved_bytes"] = max(boundary["setup_peak_reserved_bytes"], boundary["generation_peak_reserved_bytes"])
    row.update(boundary)
    return row, events


def worker(args):
    configure_mode(args.deterministic)
    out = checked_output(args.core, args.out)
    out.mkdir(parents=True, exist_ok=False)
    core = args.core.resolve(strict=True)
    before = core_sources(core)
    fixture = load_fixture(core, out, args.deterministic, args.fixture)
    torch, np = fixture.torch, fixture.np
    inputs = fixture.Fixture(args.case, args.intervals)
    write_json(out / "physical_inputs.json", inputs.description)
    warm_row, warm_events = fixture.run_ours(fixture.Fixture(args.case, min(2, args.intervals)), "ours_resident")
    validate_run(warm_row, fixture.observed[-1], min(2, args.intervals))
    write_json(out / "warmup.json", {"row": warm_row, "observation": fixture.observed[-1]})
    del warm_events
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    row, events = measured_run(fixture, inputs)
    observation = fixture.observed[-1]
    validate_run(row, observation, args.intervals)
    if events.dtype != np.float64:
        raise RuntimeError("Output precision changed")
    if core_sources(core) != before or sha(fixture.fixture_path) != fixture.fixture_sha256:
        raise RuntimeError("Immutable core or archived fixture changed during measurement")
    np.save(out / "events.npy", events, allow_pickle=False)
    row.update(
        case=args.case, trial=args.trial_index, intervals=args.intervals, core=str(core),
        initialized_pipeline_s=row["setup_s"] + row["wall_s"],
        simulated_s=float(inputs.ts[-1]),
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        gpu_uuid_visible=os.environ["CUDA_VISIBLE_DEVICES"], gpu_name=torch.cuda.get_device_name(),
        runtime={"torch": torch.__version__, "cuda": torch.version.cuda,
                 "num_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                 "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32},
        input_array_layout={k: {"dtype": str(v.dtype), "shape": list(v.shape), "device": "cpu"}
                            for k, v in inputs.data.items()},
        event_output_device="cpu", observation=observation,
        profile=fixture.PROFILE, profile_path=str(fixture.profile_path), profile_sha256=fixture.profile_sha256,
        core_source_sha256=before, index_reuse_v1_anchor=fixture.v1_anchor,
        fixture_source_path=str(fixture.fixture_path),
        fixture_source_sha256=fixture.fixture_sha256, fixture_snapshot_sha256=fixture.fixture_snapshot_sha256,
        measurement_script_sha256=sha(__file__),
        build_environment={key: os.environ.get(key) for key in
                           ("CUDA_HOME", "CC", "CXX", "MAX_JOBS",
                            "TORCH_CUDA_ARCH_LIST", "EVENTGENESIS_GPU_BUILD_DIR")},
        fresh_process=True, warmed_private_kernel=True, fresh_measured_plugin=True,
        timing_boundary="preloaded CPU physical input to sorted CPU native raw events; setup plus generation; rendering, input disk I/O, process/JIT startup and output serialization excluded",
        peak_boundary="setup and generation maxima; includes persistent runtime allocations; no unrelated model loaded",
        real_data_used=False, training_started=False, tolerance_acceptance_used=False)
    write_json(out / "result.json", row)
    print("PLUGIN_MEASUREMENT_COMPLETE", args.case, args.trial_index, row["initialized_pipeline_s"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--intervals", type=int, default=20)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fixture", type=Path, help="Explicit archived fixture snapshot when copied outside repository")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trial-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.intervals < 1 or args.trials < 1 or not args.case.replace("_", "").isalnum():
        parser.error("Positive interval/trial counts and one safe case ID are required")
    if args.worker:
        return worker(args)
    out = checked_output(args.core, args.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    report = {"schema": "actual_plugin_fresh_process_timing_v1", "status": "running",
              "deterministic_algorithms": args.deterministic, "rows": rows,
              "paired_order_controlled_by_external_driver": True}
    write_json(out / "timings.json", report)
    try:
        for trial in range(args.trials):
            destination = out / f"trial{trial}"
            argv = [sys.executable, "-u", str(Path(__file__).resolve()),
                    "--core", str(args.core.resolve()), "--out", str(destination),
                    "--case", args.case, "--intervals", str(args.intervals),
                    "--worker", "--trial-index", str(trial)]
            if args.deterministic:
                argv.append("--deterministic")
            if args.fixture:
                argv.extend(["--fixture", str(args.fixture.resolve())])
            with (out / f"trial{trial}.log").open("x") as log:
                subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
            rows.append(json.loads((destination / "result.json").read_text()))
            write_json(out / "timings.json", report)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(out / "timings.json", report)
    print("ALL_PLUGIN_TRIALS_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()
