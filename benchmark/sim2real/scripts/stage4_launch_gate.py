#!/usr/bin/env python3
"""Fingerprint-enforcing launcher for corrected Stage-4 STE-FlowNet runs.

No run directory or trainer process is created until the receipt is complete,
all phase-required checks PASS, and current code/config/data/environment hashes
exactly match the receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


FOUNDATION = [
    "renderer_segmentation_mapping",
    "depth_K_pixel_center",
    "trajectory_initialization",
    "steflow_model_and_loss_warp_cuda",
    "epe_controls",
    "correlation_operator_cuda",
    "bank_manifest_and_geometry",
    "paired_adapters_and_shared_labels",
    "event_window_recount",
    "event_representation_and_edge_cases",
    "fingerprint_completeness",
    "frozen_core_unchanged",
    "synthetic_only_guard",
]
PILOT = FOUNDATION + ["paired_formal_epoch_pilot"]
RESUME = PILOT + ["resume_equivalence_epoch_boundary"]
FULL = RESUME + ["launcher_negative_controls"]
CODE_FILES = [
    "benchmark/sim2real/scripts/generate_stage4_flow_bank.py",
    "benchmark/sim2real/scripts/prepare_steflow_stage0.py",
    "benchmark/sim2real/scripts/prepare_stage4_steflow_bank.py",
    "benchmark/sim2real/scripts/run_stage4_v2e_shard.py",
    "benchmark/sim2real/scripts/finalize_stage4_v2e.py",
    "benchmark/sim2real/scripts/stage4_v2e_artifacts.py",
    "benchmark/sim2real/scripts/run_steflow_stage2_pilot.py",
    "benchmark/sim2real/scripts/train_steflow_stage4.py",
    "benchmark/sim2real/scripts/stage4_launch_gate.py",
    "benchmark/sim2real/scripts/stage4_correctness_preflight.py",
    "benchmark/sim2real/scripts/stage4_resume_equivalence.py",
    "benchmark/sim2real/scripts/stage4_update_preflight_result.py",
    "benchmark/sim2real/scripts/stage4_validate_pilot.py",
    "benchmark/sim2real/scripts/stage4_gate_negative_controls.py",
    "benchmark/sim2real/scripts/run_stage4_corrected_v2_server.sh",
    "benchmark/sim2real/compat/steflow_align_corners.py",
    "genesis_event_plugin/physics_plugin.py",
    "genesis_event_plugin/genesis_adapter.py",
    "genesis_event_plugin/interpolator.py",
    "genesis_event_plugin/physics_warp.py",
]
CORE_FILES = [
    "genesis_event_plugin/__init__.py",
    "genesis_event_plugin/adaptive_sampler.py",
    "genesis_event_plugin/calibration.py",
    "genesis_event_plugin/camera_margin.py",
    "genesis_event_plugin/camera_model.py",
    "genesis_event_plugin/dvs_emulator.py",
    "genesis_event_plugin/evaluation/__init__.py",
    "genesis_event_plugin/evaluation/evaluator.py",
    "genesis_event_plugin/evaluation/event_reader.py",
    "genesis_event_plugin/evaluation/event_schema.py",
    "genesis_event_plugin/evaluation/pecs_metrics.py",
    "genesis_event_plugin/evaluation/plan_a_runner.py",
    "genesis_event_plugin/genesis_adapter.py",
    "genesis_event_plugin/physics_plugin.py",
    "genesis_event_plugin/interpolator.py",
    "genesis_event_plugin/physics_warp.py",
    "genesis_event_plugin/physics_interpolator.py",
    "genesis_event_plugin/noise_model.py",
    "genesis_event_plugin/plugin.py",
    "genesis_event_plugin/radiometry.py",
    "genesis_event_plugin/readout.py",
    "genesis_event_plugin/recorder.py",
    "genesis_event_plugin/torch_physics.py",
    "genesis_event_plugin/utils/__init__.py",
    "genesis_event_plugin/utils/dvs_core.py",
    "genesis_event_plugin/utils/flow_utils.py",
    "genesis_event_plugin/utils/motion_blur.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_fingerprint(repo: Path, config: Path, bank: Path, v2e: Path,
                      data: Path, model: Path, genesis_root: Path) -> dict:
    code = {}
    for relative in CODE_FILES:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"required code input missing: {path}")
        code[relative] = sha256(path)
    core = {relative: sha256(repo / relative) for relative in CORE_FILES}
    config_hash = sha256(config)
    model_hashes = {
        "STEFlow.py": sha256(model / "models/STEFlow.py"),
        "multiscaleloss.py": sha256(model / "multiscaleloss.py"),
        "corr.py": sha256(model / "models/corr.py"),
        "git_commit": subprocess.check_output(["git", "-C", str(model), "rev-parse", "HEAD"], text=True).strip(),
    }
    upstream_expected = {
        "STEFlow.py": "eaf1f02ba5addc319c9736307960172cb2e3693a714d536d2bad478c0964c785",
        "multiscaleloss.py": "ff401317e49ce37ae0f8e9bfd6b82f133549b4a0271be35095f74bdc50a439c0",
        "corr.py": "2274d68f0bf1880c6b9806bc7835ea0243570af1b5a4415e0a968e154a8ec26a",
    }
    if any(model_hashes[key] != expected for key, expected in upstream_expected.items()):
        raise RuntimeError(f"locked STE-FlowNet source changed: {model_hashes}")
    bank_manifest_path = bank / "bank_manifest.json"
    split_path = bank / "registered_split.json"
    h5_path = bank / "shared_trajectory.h5"
    v2e_manifest_path = v2e / "v2e_complete_manifest.json"
    adapter_manifest_path = data / "training_bank_manifest.json"
    for path in (bank_manifest_path, split_path, h5_path, v2e_manifest_path, adapter_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"required data manifest/artifact missing: {path}")
    bank_manifest = json.loads(bank_manifest_path.read_text())
    if bank_manifest.get("schema") != "stage4_shared_flow_bank_v2_geometry_corrected":
        raise RuntimeError(f"unaccepted shared bank schema {bank_manifest.get('schema')}")
    if sha256(h5_path) != bank_manifest.get("bank_sha256"):
        raise RuntimeError("shared bank content hash does not match its manifest")
    adapter_manifest = json.loads(adapter_manifest_path.read_text())
    adapter_hashes = {}
    for row in adapter_manifest.get("entries", []):
        path = Path(row["path"])
        if not path.is_absolute():
            path = data / path
        actual = sha256(path)
        if actual != row.get("sha256"):
            raise RuntimeError(f"adapter file hash mismatch: {path}")
        adapter_hashes[str(path)] = actual
    if len(adapter_hashes) != 64:
        raise RuntimeError(f"expected 64 immutable paired adapter files, got {len(adapter_hashes)}")
    raw_v2e_hashes, v2e_input_hashes = {}, {}
    v2e_manifest = json.loads(v2e_manifest_path.read_text())
    for row in v2e_manifest.get("episodes", []):
        raw_path = Path(row["raw_events_path"])
        if sha256(raw_path) != row.get("raw_events_sha256"):
            raise RuntimeError(f"V2E raw-event artifact hash mismatch: {raw_path}")
        raw_v2e_hashes[str(raw_path)] = row["raw_events_sha256"]
        for frame in row.get("input_frame_manifest", []):
            frame_path = Path(frame["path"])
            if sha256(frame_path) != frame.get("sha256"):
                raise RuntimeError(f"V2E input-frame artifact hash mismatch: {frame_path}")
            v2e_input_hashes[str(frame_path)] = frame["sha256"]
    if len(raw_v2e_hashes) != 32 or len(v2e_input_hashes) != 6432:
        raise RuntimeError(f"V2E raw/input artifact census mismatch: {len(raw_v2e_hashes)}/{len(v2e_input_hashes)}")
    project = repo.parents[1]
    v2e_source = project / "external/sources/v2e-cac6384445/v2e-pinned"
    v2e_entry = v2e_source / "v2e.py"
    v2e_model = project / "external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt"
    v2e_source_commit = subprocess.check_output(["git", "-C", str(v2e_source), "rev-parse", "HEAD"], text=True).strip()
    config_value = json.loads(config.read_text())
    expected_v2e_commit = config_value["generators"]["v2e"]["commit"]
    if v2e_source_commit != expected_v2e_commit:
        raise RuntimeError(f"pinned V2E source commit mismatch: {v2e_source_commit} != {expected_v2e_commit}")
    v2e_code = {"commit": v2e_source_commit, "entry_sha256": sha256(v2e_entry), "slomo_checkpoint_sha256": sha256(v2e_model)}
    genesis_files = {}
    for relative in ("genesis/vis/camera.py", "genesis/vis/visualizer.py",
                     "genesis/vis/rasterizer_context.py", "genesis/ext/pyrender/renderer.py"):
        path = genesis_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"runtime renderer source is missing: {path}")
        genesis_files[relative] = sha256(path)
    try:
        import torch
        torch_version, cuda_version = torch.__version__, torch.version.cuda
    except Exception as exc:
        raise RuntimeError(f"cannot import runtime torch: {exc}") from exc
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch_version, "torch_cuda_build": cuda_version,
        "genesis_world": package_version("genesis-world"),
        "numpy": package_version("numpy"), "h5py": package_version("h5py"),
        "opencv_python": package_version("opencv-python"),
    }
    inputs = {
        "config": {"path": str(config), "sha256": config_hash},
        "bank": {"manifest": sha256(bank_manifest_path), "split": sha256(split_path), "h5": sha256(h5_path)},
        "v2e_manifest": sha256(v2e_manifest_path),
        "v2e_source": v2e_code, "v2e_raw_artifacts": raw_v2e_hashes,
        "v2e_input_frames": v2e_input_hashes,
        "adapter_manifest": sha256(adapter_manifest_path), "adapter_files": adapter_hashes,
        "model": model_hashes, "genesis_renderer_sources": genesis_files,
        "frozen_core": core, "environment": environment,
    }
    return {"code": code, "inputs": inputs, "fingerprint_sha256": canonical_hash({"code": code, "inputs": inputs})}


def validate_report(report: dict, phase: str, fingerprint: dict | None) -> list[str]:
    required = {"pilot": FOUNDATION, "resume-check": PILOT, "full": FULL}[phase]
    errors = []
    if report.get("schema") != "stage4_preflight_result_v1":
        errors.append("preflight report schema missing/unsupported")
    if report.get("overall_status") != "PASS":
        errors.append(f"preflight overall_status is {report.get('overall_status')!r}")
    checks = {row.get("check"): row.get("status") for row in report.get("checks", [])}
    for name in required:
        if name not in checks:
            errors.append(f"required check missing: {name}")
        elif checks[name] != "PASS":
            errors.append(f"required check {name} is {checks[name]!r}")
    if fingerprint is not None and report.get("fingerprint") != fingerprint:
        errors.append("preflight code/data/config/model/environment fingerprint is stale")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "resume-check", "full"), required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--v2e-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--genesis-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--generator", choices=("eventgenesis", "v2e"), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--test-only-sentinel", type=Path,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    repo = args.repo_root.resolve(); config = args.config.resolve()
    bank = args.bank_root.resolve(); v2e = args.v2e_root.resolve(); data = args.data_root.resolve()
    model = args.model_root.resolve(); genesis_root = args.genesis_root.resolve()
    if args.phase == "pilot" and (args.max_updates is None or not 1 <= args.max_updates <= 198):
        parser.error("pilot phase requires --max-updates in [1,198]")
    if args.phase == "pilot" and args.resume is not None:
        parser.error("pilot phase must start from registered seed initialization, not a prior checkpoint")
    if args.phase == "resume-check" and (args.max_updates is None or not 1 <= args.max_updates <= 199):
        parser.error("resume-check phase requires --max-updates in [1,199]")
    if args.phase == "full" and args.max_updates is not None:
        parser.error("full phase does not accept an update cap")
    if args.phase == "full" and args.resume is None:
        parser.error("full phase must continue from the fully validated epoch pilot checkpoint")
    if args.phase == "resume-check" and args.resume is not None and not args.resume.is_file():
        parser.error("resume checkpoint is missing")
    if args.phase == "full" and args.resume is not None and not args.resume.is_file():
        parser.error("full-run resume checkpoint is missing")

    if not args.receipt.is_file():
        raise SystemExit(f"LAUNCH BLOCKED: missing preflight receipt {args.receipt}")
    try:
        report = json.loads(args.receipt.read_text())
    except Exception as exc:
        raise SystemExit(f"LAUNCH BLOCKED: cannot read preflight receipt: {exc}") from exc
    errors = validate_report(report, args.phase, None)
    if errors:
        raise SystemExit("LAUNCH BLOCKED:\n- " + "\n- ".join(errors))
    try:
        fingerprint = build_fingerprint(repo, config, bank, v2e, data, model, genesis_root)
    except Exception as exc:
        raise SystemExit(f"LAUNCH BLOCKED: fingerprint construction failed: {exc}") from exc
    errors = validate_report(report, args.phase, fingerprint)
    if errors:
        raise SystemExit("LAUNCH BLOCKED:\n- " + "\n- ".join(errors))
    print(json.dumps({"status": "GATE_PASS", "phase": args.phase,
                      "fingerprint_sha256": fingerprint["fingerprint_sha256"],
                      "receipt": str(args.receipt.resolve()), "run_root": str(args.run_root.resolve())}, indent=2), flush=True)
    if args.verify_only:
        return
    if args.test_only_sentinel is not None:
        if os.environ.get("STAGE4_GATE_TEST_MODE") != "1":
            raise SystemExit("--test-only-sentinel is restricted to gate negative-control tests")
        sentinel = args.test_only_sentinel.resolve()
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("trainer branch reached\n")
        return
    trainer = repo / "benchmark/sim2real/scripts/train_steflow_stage4.py"
    command = [sys.executable, str(trainer), "--model-root", str(model), "--data-root", str(data),
               "--config", str(config), "--run-root", str(args.run_root.resolve()),
               "--generator", args.generator, "--seed", "0"]
    if args.resume:
        command += ["--resume", str(args.resume.resolve())]
    if args.max_updates is not None:
        command += ["--max-updates", str(args.max_updates)]
    env = os.environ.copy(); env["STAGE4_LAUNCH_GATE_VERIFIED"] = fingerprint["fingerprint_sha256"]
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
