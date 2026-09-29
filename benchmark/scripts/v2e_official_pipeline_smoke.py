#!/usr/bin/env python3
"""Run the official V2E image-folder path with SuperSloMo enabled.

This is a readiness smoke only. It refuses the direct EventEmulator shortcut,
loads every frozen V2E parameter from ``benchmark/configs/v2e.yaml``, and
records the exact CLI used for the official chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_MODEL_URL = (
    "https://drive.google.com/file/d/1ETID_4xqLpRBrRo1aOT7Yphs3QqWR_fx/view?usp=sharing"
)


def _load_yaml(path: Path):
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _stop(message: str, **fields) -> int:
    result = {"status": "STOP", "formal_metrics_executed": False, "error": message, **fields}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2


def _find_h5_events(path: Path):
    import h5py
    with h5py.File(path, "r") as handle:
        if "events" not in handle:
            raise RuntimeError(f"official V2E output has no events dataset: {path}")
        dataset = handle["events"]
        if dataset.ndim != 2 or dataset.shape[1] != 4:
            raise RuntimeError(f"unexpected official V2E events shape: {dataset.shape}")
        shape = tuple(dataset.shape)
        dtype = dataset.dtype
        sample = dataset[:]
    return "events", shape, dtype, sample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _headless_desktop_shim(output: Path) -> Path:
    """Provide only v2e's optional GUI opener without changing its pipeline."""

    root = output / "_v2e_runtime_compat"
    package = root / "v2ecore"
    package.mkdir(parents=True, exist_ok=True)
    (package / "desktop.py").write_text(
        "def open(path):\n    return None\n", encoding="utf-8"
    )
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--python", dest="python_exe", default=sys.executable)
    parser.add_argument("--slomo-model", type=Path, default=None)
    args = parser.parse_args()

    source = args.source or Path(os.environ.get(
        "V2E_SOURCE", "/home/科研/Eventbased_WAM/code/v2e_source"
    ))
    v2e = source / "v2e.py"
    model = args.slomo_model or Path(os.environ.get(
        "V2E_SLOMO_MODEL", str(source / "input/SuperSloMo39.ckpt")
    ))
    if not v2e.is_file():
        return _stop("V2E source/v2e.py not found", source=str(source))
    if not model.is_file():
        return _stop(
            "official SuperSloMo checkpoint is missing; direct EventEmulator is forbidden",
            expected_model=str(model),
            official_source=OFFICIAL_MODEL_URL,
        )
    frames = sorted(args.frames.glob("*.png"))
    if len(frames) < 3:
        return _stop("same-Genesis replay did not produce at least 3 intermediate frames", frame_count=len(frames))

    config_path = ROOT / "benchmark/configs/v2e.yaml"
    protocol_path = ROOT / "benchmark/protocol_v3.yaml"
    config = _load_yaml(config_path)
    protocol = _load_yaml(protocol_path)
    params = config["parameters"]
    pipeline = protocol["methods"]["v2e"]["pipeline"]
    source_frame_rate = float(pipeline["source_frame_rate_hz"])
    timestamp_resolution = float(pipeline["timestamp_resolution_s"])
    # v3 explicitly freezes the non-auto 100 Hz -> 0.1 ms path.
    auto_timestamp_resolution = False
    seed = int(params["seed"])
    cli_values = {
        "input_frame_rate": source_frame_rate,
        "auto_timestamp_resolution": auto_timestamp_resolution,
        "timestamp_resolution": timestamp_resolution,
        "pos_thres": float(params["pos_thres"]),
        "neg_thres": float(params["neg_thres"]),
        "sigma_thres": float(params["sigma_thres"]),
        "cutoff_hz": float(params["cutoff_hz"]),
        "leak_rate_hz": float(params["leak_rate_hz"]),
        "shot_noise_rate_hz": float(params["shot_noise_rate_hz"]),
        "refractory_period": float(params["refractory_period_s"]),
        "leak_jitter_fraction": float(params["leak_jitter_fraction"]),
        "noise_rate_cov_decades": float(params["noise_rate_cov_decades"]),
        "dvs_emulator_seed": seed,
        "slomo_model": str(model),
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compat_root = _headless_desktop_shim(output)
    event_file = output / "events.h5"
    log_file = output / "official_v2e.log"
    cmd = [
        args.python_exe, str(v2e),
        "--input", str(args.frames),
        "--input_frame_rate", str(cli_values["input_frame_rate"]),
        "--auto_timestamp_resolution", str(cli_values["auto_timestamp_resolution"]),
        "--timestamp_resolution", str(cli_values["timestamp_resolution"]),
        "--pos_thres", str(cli_values["pos_thres"]),
        "--neg_thres", str(cli_values["neg_thres"]),
        "--sigma_thres", str(cli_values["sigma_thres"]),
        "--cutoff_hz", str(cli_values["cutoff_hz"]),
        "--leak_rate_hz", str(cli_values["leak_rate_hz"]),
        "--shot_noise_rate_hz", str(cli_values["shot_noise_rate_hz"]),
        "--refractory_period", str(cli_values["refractory_period"]),
        "--leak_jitter_fraction", str(cli_values["leak_jitter_fraction"]),
        "--noise_rate_cov_decades", str(cli_values["noise_rate_cov_decades"]),
        "--dvs_emulator_seed", str(cli_values["dvs_emulator_seed"]),
        "--slomo_model", str(model),
        "--output_folder", str(output),
        "--dvs_h5", event_file.name,
        "--dvs_text", "None",
        "--skip_video_output",
        "--no_preview",
        "--overwrite",
    ]
    command_record = [str(item) for item in cmd]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(compat_root), str(source), environment.get("PYTHONPATH", "")) if item
    )
    completed = subprocess.run(
        cmd, cwd=source, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    log_file.write_text(completed.stdout, encoding="utf-8")
    (output / "v2e_cli.json").write_text(json.dumps({
        "config": str(config_path),
        "config_hash_scope": "exact bytes of benchmark/configs/v2e.yaml",
        "official_source": "https://github.com/SensorsINI/v2e",
        "official_checkpoint_source": OFFICIAL_MODEL_URL,
        "checkpoint_path": str(model),
        "checkpoint_size_bytes": model.stat().st_size,
        "checkpoint_sha256": _sha256(model),
        "frozen_parameters": cli_values,
        "argv": command_record,
        "headless_desktop_shim": str(compat_root / "v2ecore/desktop.py"),
    }, indent=2), encoding="utf-8")
    if completed.returncode != 0:
        return _stop("official V2E pipeline exited non-zero", returncode=completed.returncode, log=str(log_file))
    if not event_file.is_file():
        return _stop("official V2E pipeline produced no HDF5 event output", log=str(log_file))

    try:
        dataset, shape, dtype, sample = _find_h5_events(event_file)
    except Exception as exc:
        return _stop(f"cannot read official V2E output: {type(exc).__name__}: {exc}", log=str(log_file))
    if sample.shape[0] == 0:
        return _stop("official V2E pipeline produced an empty event stream", log=str(log_file))

    timestamps = np.asarray(sample[:, 0], dtype=np.float64) / 1e6
    duration = float(timestamps.max() - timestamps.min())
    expected_duration = (len(frames) - 1) / source_frame_rate
    duration_error = duration - expected_duration
    forbidden = re.compile(
        r"(undersampl|cutoff.*warning|warning.*300|disable_slomo|rescal.*timestamp)",
        re.I,
    )
    forbidden_lines = [line for line in completed.stdout.splitlines() if forbidden.search(line)]
    if forbidden_lines:
        return _stop(
            "official V2E log contains a forbidden cadence/timestamp warning",
            lines=forbidden_lines[:10], log=str(log_file),
        )
    slowdown_match = re.search(r"slowdown_factor=(\d+)", completed.stdout)
    if slowdown_match is None:
        return _stop("official V2E log did not report a computed slowdown factor", log=str(log_file))
    slowdown_factor = int(slowdown_match.group(1))
    expected_slowdown = int(np.ceil((1.0 / source_frame_rate) / timestamp_resolution))
    if slowdown_factor != expected_slowdown:
        return _stop(
            "official V2E slowdown factor differs from frozen cadence contract",
            observed=slowdown_factor, expected=expected_slowdown, log=str(log_file),
        )
    if abs(duration_error) > max(0.002, 0.02 * expected_duration):
        return _stop(
            "official V2E output duration differs from same replay duration",
            output_duration_s=duration, expected_duration_s=expected_duration,
            duration_error_s=duration_error, log=str(log_file),
        )
    result = {
        "status": "PASS",
        "formal_metrics_executed": False,
        "official_pipeline": True,
        "super_slomo_enabled": True,
        "source_frame_count": len(frames),
        "source_frame_rate_hz": source_frame_rate,
        "effective_timestamp_resolution_s": timestamp_resolution,
        "computed_slowdown_factor": slowdown_factor,
        "expected_slowdown_factor": expected_slowdown,
        "dataset": dataset,
        "event_shape": list(shape),
        "dtype": str(dtype),
        "output_duration_s": duration,
        "expected_duration_s": expected_duration,
        "duration_error_s": duration_error,
        "checkpoint_path": str(model),
        "checkpoint_size_bytes": model.stat().st_size,
        "checkpoint_sha256": _sha256(model),
        "forbidden_warning_lines": [],
        "config": str(config_path),
        "seed": seed,
        "dvs_emulator_seed": seed,
        "argv_record": str(output / "v2e_cli.json"),
    }
    (output / "v2e_pipeline_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
