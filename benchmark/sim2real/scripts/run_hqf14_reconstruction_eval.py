#!/usr/bin/env python3
"""Run both frozen Stage-5 E2VID candidates over all 14 HQF sequences.

No outputs are created until the frozen protocol, full dataset hashes, and both
synthetic-best checkpoint hashes pass preflight. Actual HQF inference/scoring
requires the explicit ``--enable-real-scoring`` flag. ``--preflight-only`` is
read-only and does not load models or evaluate scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SIM2REAL = ROOT / "benchmark/sim2real"
SCRIPTS = SIM2REAL / "scripts"
DEFAULT_PROTOCOL = SIM2REAL / "protocol/hqf_14_reconstruction_test_frozen_20261024_v1.json"
DEFAULT_DATA_MANIFEST = SIM2REAL / "evidence/S5/20261021-recon-flow-localization/20261021-hqf14-manifest-validation-v3/hqf_14_file_manifest_v3.json"
DEFAULT_TRAINING_MANIFEST = SIM2REAL / "evidence/S5/20261021-recon-flow-localization/reconstruction_training_artifact_manifest.json"
INFER_SCRIPT = SCRIPTS / "infer_hqf_e2vid_candidate.py"
SCORE_SCRIPT = SCRIPTS / "score_hqf14_reconstruction.py"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
from benchmark.sim2real.scripts.score_hqf14_reconstruction import (  # noqa: E402
    EXPECTED_ARMS,
    sha256_file,
    validate_data_manifest,
    verify_frozen_protocol,
)
from verify_pinned_source import verify_pinned_source  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def checkpoint_path_for(protocol: dict[str, Any], arm: str) -> Path:
    return Path(protocol["protocol_freeze"]["checkpoint_bindings"][arm]["checkpoint_path"]).resolve()


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Path]]:
    protocol_path = args.protocol.resolve()
    protocol, protocol_sha = verify_frozen_protocol(protocol_path)
    config_record = protocol["protocol_freeze"]["model_config"]
    model_config = ROOT / config_record["path"]
    if sha256_file(model_config) != config_record["sha256"]:
        raise RuntimeError("registered reconstruction model config hash mismatch")
    data_manifest_path = args.data_manifest.resolve()
    data_manifest, data_rows = validate_data_manifest(protocol, data_manifest_path, args.hqf_root.resolve())
    checkpoints = {}
    for arm in EXPECTED_ARMS:
        record = protocol["protocol_freeze"]["checkpoint_bindings"][arm]
        path = checkpoint_path_for(protocol, arm)
        if not path.is_file():
            raise FileNotFoundError(f"{arm} registered synthetic-best checkpoint missing: {path}")
        if path.stat().st_size != record["checkpoint_size_bytes"] or sha256_file(path) != record["checkpoint_sha256"]:
            raise RuntimeError(f"{arm} checkpoint size/SHA does not match registered synthetic-best")
        checkpoints[arm] = path
    if not args.evreal_root.is_dir() or not args.model_root.is_dir():
        raise FileNotFoundError("pinned EVREAL and event_cnn_minimal source roots are required")
    manifest_path = ROOT / protocol["protocol_freeze"]["source_bundle"]["path"]
    source_bundle = load_json(manifest_path)
    source_roots = {}
    for row in source_bundle.get("repositories", []):
        source_roots[row.get("repo")] = row
    for repo_name, source_root in (("EVREAL", args.evreal_root.resolve()),
                                   ("event_cnn_minimal", args.model_root.resolve())):
        record = source_roots.get(repo_name)
        if not record:
            raise RuntimeError(f"source bundle does not pin {repo_name}")
        verification = verify_pinned_source(source_root, repo_name, record["commit"], manifest_path)
        if verification.get("verification") != "clean_git_checkout":
            raise RuntimeError(f"{repo_name} source checkout is not clean and pinned: {verification}")
    return protocol, {"manifest": data_manifest, "rows": data_rows}, protocol_sha, checkpoints


def write_run_manifest(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--hqf-root", type=Path, required=True)
    parser.add_argument("--evreal-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--enable-real-scoring", action="store_true",
                        help="explicitly run the frozen one-time HQF inference and score protocol")
    args = parser.parse_args()
    if not args.preflight_only and not args.enable_real_scoring:
        parser.error("actual HQF inference/scoring requires --enable-real-scoring")

    started = utc_now()
    protocol, data, protocol_sha, checkpoints = preflight(args)
    print(json.dumps({"status": "preflight_pass", "protocol_sha256": protocol_sha,
                      "sequences": len(protocol["sequences"]), "arms": list(EXPECTED_ARMS),
                      "checkpoint_sha256": {arm: protocol["protocol_freeze"]["checkpoint_bindings"][arm]["checkpoint_sha256"] for arm in EXPECTED_ARMS},
                      "data_manifest_sha256": sha256_file(args.data_manifest.resolve()),
                      "real_scoring_enabled": not args.preflight_only}, indent=2))
    if args.preflight_only:
        return 0

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse/overwrite an existing HQF run directory: {output_root}")
    output_root.mkdir(parents=True)
    inference_root = output_root / "inference"
    log_root = output_root / "logs"
    log_root.mkdir()
    run_manifest_path = output_root / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema": "hqf14_paired_inference_run_v1",
        "status": "inference_running",
        "created_utc": started,
        "finished_utc": None,
        "real_scoring_enabled": True,
        "real_scores_computed": False,
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": protocol_sha,
        "data_manifest_path": str(args.data_manifest.resolve()),
        "data_manifest_sha256": sha256_file(args.data_manifest.resolve()),
        "hqf_root": str(args.hqf_root.resolve()),
        "evreal_root": str(args.evreal_root.resolve()),
        "model_root": str(args.model_root.resolve()),
        "model_config_path": str(ROOT / protocol["protocol_freeze"]["model_config"]["path"]),
        "model_config_sha256": protocol["protocol_freeze"]["model_config"]["sha256"],
        "inference_root": str(inference_root),
        "inference_script": str(INFER_SCRIPT),
        "inference_script_sha256": sha256_file(INFER_SCRIPT),
        "score_script": str(SCORE_SCRIPT),
        "score_script_sha256": sha256_file(SCORE_SCRIPT),
        "device": args.device,
        "arms": {},
    }
    write_run_manifest(run_manifest_path, manifest)
    for arm in EXPECTED_ARMS:
        checkpoint = checkpoints[arm]
        arm_record = {"checkpoint_path": str(checkpoint),
                      "checkpoint_sha256": protocol["protocol_freeze"]["checkpoint_bindings"][arm]["checkpoint_sha256"],
                      "inference_manifest_sha256": {}, "sequences": {}}
        manifest["arms"][arm] = arm_record
        write_run_manifest(run_manifest_path, manifest)
        for name in protocol["sequences"]:
            sequence_output = inference_root / arm / name
            sequence_output.mkdir(parents=True)
            command = [sys.executable, "-u", str(INFER_SCRIPT),
                       "--evreal-root", str(args.evreal_root.resolve()),
                       "--model-root", str(args.model_root.resolve()),
                       "--hqf-root", str(args.hqf_root.resolve()),
                       "--checkpoint", str(checkpoint), "--sequence", name,
                       "--model-config", manifest["model_config_path"],
                       "--output-root", str(sequence_output), "--device", args.device,
                       "--save-every", "1", "--save-float-arrays"]
            safe = f"{arm}__{name}"
            stdout_path = log_root / f"{safe}.stdout.log"
            stderr_path = log_root / f"{safe}.stderr.log"
            seq_started = utc_now()
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                proc = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
            seq_finished = utc_now()
            seq_result = {"command": command, "started_utc": seq_started,
                          "finished_utc": seq_finished, "exit_code": proc.returncode,
                          "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}
            if proc.returncode != 0:
                arm_record["sequences"][name] = seq_result
                manifest["status"] = "inference_failed"
                manifest["finished_utc"] = utc_now()
                write_run_manifest(run_manifest_path, manifest)
                raise RuntimeError(f"HQF inference failed for {arm}/{name}; see {stderr_path}")
            inf_path = sequence_output / "inference_manifest.json"
            if not inf_path.is_file():
                raise RuntimeError(f"inference returned 0 but did not write {inf_path}")
            inf_sha = sha256_file(inf_path)
            arm_record["inference_manifest_sha256"][name] = inf_sha
            seq_result["inference_manifest_path"] = str(inf_path)
            seq_result["inference_manifest_sha256"] = inf_sha
            arm_record["sequences"][name] = seq_result
            write_run_manifest(run_manifest_path, manifest)
    manifest["status"] = "all_28_inferences_complete_unscored"
    manifest["finished_utc"] = utc_now()
    write_run_manifest(run_manifest_path, manifest)

    score_path = output_root / "paired_score.json"
    score_command = [sys.executable, "-u", str(SCORE_SCRIPT), "--protocol", str(args.protocol.resolve()),
                     "--data-manifest", str(args.data_manifest.resolve()), "--hqf-root", str(args.hqf_root.resolve()),
                     "--inference-root", str(inference_root), "--run-manifest", str(run_manifest_path),
                     "--model-root", str(args.model_root.resolve()), "--output", str(score_path),
                     "--device", args.device, "--enable-real-scoring"]
    score_started = utc_now()
    score_stdout, score_stderr = log_root / "score.stdout.log", log_root / "score.stderr.log"
    with score_stdout.open("wb") as stdout, score_stderr.open("wb") as stderr:
        score_proc = subprocess.run(score_command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    manifest["score_command"] = score_command
    manifest["score_started_utc"] = score_started
    manifest["score_finished_utc"] = utc_now()
    manifest["score_exit_code"] = score_proc.returncode
    manifest["score_stdout_path"] = str(score_stdout)
    manifest["score_stderr_path"] = str(score_stderr)
    if score_proc.returncode != 0:
        manifest["status"] = "scoring_failed"
        manifest["finished_utc"] = utc_now()
        write_run_manifest(run_manifest_path, manifest)
        raise RuntimeError(f"HQF scorer failed; see {score_stderr}")
    manifest["status"] = "paired_real_hqf14_scoring_complete"
    manifest["real_scores_computed"] = True
    manifest["score_path"] = str(score_path)
    manifest["score_sha256"] = sha256_file(score_path)
    manifest["finished_utc"] = utc_now()
    write_run_manifest(run_manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "run_manifest": str(run_manifest_path),
                      "score_path": str(score_path), "score_sha256": manifest["score_sha256"]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
