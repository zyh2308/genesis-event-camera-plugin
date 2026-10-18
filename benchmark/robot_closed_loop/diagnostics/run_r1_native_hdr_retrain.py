#!/usr/bin/env python3
"""Run one isolated native-HDR Ours vs unchanged V2E R1 development retrain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.run_paired_generator_smoke import (  # noqa: E402
    _make_ours_events,
    _sha256,
    _write_hash_manifest,
    _write_source_episode,
)


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _manifest(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        result[name] = digest
    return result


def _write_source_contract(path: Path, contract: Dict[str, Any]) -> None:
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")


def _timestamp_hash(cache_path: Path) -> str:
    with h5py.File(cache_path, "r") as handle:
        timestamps = np.asarray(handle["timestamps_s"][:])
    return hashlib.sha256(timestamps.tobytes()).hexdigest()


def _run_fit(
    *, runner_python: Path, scene_config: Path, source_root: Path, out: Path,
    train_ids: List[int], validation_ids: List[int], training: Dict[str, Any],
    source_label: str, log_path: Path,
) -> Dict[str, Any]:
    command = [
        str(runner_python), "-m", "benchmark.robot_closed_loop.diagnostics.train_event_state_development",
        "--config", str(scene_config), "--source-results", str(source_root), "--out", str(out),
        "--train-episodes", ",".join(map(str, train_ids)),
        "--validation-episodes", ",".join(map(str, validation_ids)),
        "--epochs", str(int(training["max_epochs"])), "--patience", str(int(training["patience"])),
        "--batch-size", str(int(training["batch_size"])), "--seed", str(int(training["seed"])),
        "--threads", str(int(training["threads"])), "--architecture", str(training["network"]),
        "--source-label", source_label,
    ]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(training["threads"]),
        "MKL_NUM_THREADS": str(training["threads"]),
        "OPENBLAS_NUM_THREADS": str(training["threads"]),
        "PYTHONPATH": str(ROOT),
    }
    process = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.write_text(process.stdout, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"fit failed ({source_label}): {process.stdout[-3000:]}")
    summary_path = out / "development_summary.json"
    checkpoint_path = out / "spatial_event_state.pt"
    return {
        "checkpoint_sha256": _sha256(checkpoint_path),
        "summary_sha256": _sha256(summary_path),
        "source_manifest_sha256": _sha256(source_root / "SHA256SUMS"),
        "validation_metrics": json.loads(summary_path.read_text(encoding="utf-8"))["validation_metrics"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--v2e-training-source", type=Path, required=True,
                        help="immutable V2E training source from the audited paired small9 batch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path,
                        default=ROOT / "benchmark/robot_closed_loop/configs/r1_native_hdr_small9_development_v1.json")
    parser.add_argument("--scene-config", type=Path,
                        default=ROOT / "benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json")
    parser.add_argument("--control-config", type=Path,
                        default=ROOT / "benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--episode-ids", default="0,1,2,3,4,5,6,7,8")
    args = parser.parse_args()

    experiment_path = args.experiment_config.resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    training = experiment["training"]
    train_ids = [int(value) for value in training["train_episode_ids"]]
    validation_ids = [int(value) for value in training["validation_episode_ids"]]
    episode_ids = [int(value) for value in args.episode_ids.split(",") if value.strip()]
    if set(train_ids) & set(validation_ids) or set(episode_ids) != set(train_ids) | set(validation_ids):
        raise ValueError("episode IDs must match the non-overlapping episode-level train/validation split")

    cache_root = args.cache_root.resolve()
    old_v2e_root = args.v2e_training_source.resolve()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite prior native-HDR run: {out}")
    old_manifest_path = old_v2e_root / "SHA256SUMS"
    if not old_manifest_path.is_file():
        raise FileNotFoundError(old_manifest_path)
    old_manifest = _manifest(old_manifest_path)
    expected_v2e_manifest = experiment["v2e_input_endpoint"]["source_manifest_sha256"]
    if _sha256(old_manifest_path) != expected_v2e_manifest:
        raise ValueError("V2E source manifest does not match the native-HDR experiment config")

    old_contract = json.loads((old_v2e_root / "episode_split_contract.json").read_text(encoding="utf-8"))
    if (old_contract["train_episode_ids"] != train_ids
            or old_contract["validation_episode_ids"] != validation_ids
            or old_contract["training_seed"] != int(training["seed"])
            or old_contract["architecture"] != training["network"]
            or old_contract["fixed_budget"]["max_epochs"] != int(training["max_epochs"])
            or old_contract["fixed_budget"]["patience"] != int(training["patience"])
            or old_contract["fixed_budget"]["batch_size"] != int(training["batch_size"])
            or old_contract["fixed_budget"]["threads"] != int(training["threads"])):
        raise ValueError("archived V2E split/budget differs from the locked paired retrain contract")

    out.mkdir(parents=True)
    replay_root = out / "ours_native_hdr_generation"
    replay_root.mkdir()
    ours_root = out / "ours_native_hdr_training"
    v2e_root = out / "v2e_training_reused"
    ours_root.mkdir()
    split_contract = {
        "train_episode_ids": train_ids,
        "validation_episode_ids": validation_ids,
        "split_unit": "episode",
        "training_seed": int(training["seed"]),
        "architecture": training["network"],
        "fixed_budget": {
            "max_epochs": int(training["max_epochs"]),
            "patience": int(training["patience"]),
            "batch_size": int(training["batch_size"]),
            "threads": int(training["threads"]),
        },
        "purpose": "native scene-linear HDR vs official V2E small-cache development fit; not formal test",
    }
    split_sha = hashlib.sha256(json.dumps(split_contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _write_source_contract(ours_root / "episode_split_contract.json", {**split_contract, "sha256": split_sha})

    # Preserve the V2E archive byte-for-byte, including its existing manifest.
    shutil.copytree(old_v2e_root, v2e_root)
    if _sha256(v2e_root / "SHA256SUMS") != expected_v2e_manifest:
        raise RuntimeError("copied V2E event source changed during isolated archival")

    cache_inputs: Dict[str, str] = {}
    episode_records = []
    generation_started = time.perf_counter()
    for episode_id in episode_ids:
        episode_name = f"episode_{episode_id:03d}"
        cache_dir = cache_root / episode_name
        cache_path = cache_dir / "trajectory_cache.h5"
        result_path = cache_dir / "episode_result.json"
        trace_path = cache_dir / "oracle_trace.jsonl"
        frame_manifest = cache_dir / "frame_manifest.json"
        for required in (cache_path, result_path, trace_path, frame_manifest):
            if not required.is_file():
                raise FileNotFoundError(required)
        cache_inputs[str(cache_path)] = _sha256(cache_path)
        v2e_episode = v2e_root / "episodes" / episode_name
        v2e_pair_path = v2e_episode / "pair_provenance.json"
        v2e_events_path = v2e_episode / "direct_hdr_events_1khz_2khz.npz"
        for required in (v2e_pair_path, v2e_events_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        pair = json.loads(v2e_pair_path.read_text(encoding="utf-8"))
        expected = {
            "shared_cache_sha256": _sha256(cache_path),
            "frame_manifest_sha256": _sha256(frame_manifest),
            "episode_result_sha256": _sha256(result_path),
            "trace_sha256": _sha256(trace_path),
        }
        mismatches = {key: {"pair": pair.get(key), "observed": value}
                      for key, value in expected.items() if pair.get(key) != value}
        if mismatches:
            raise ValueError(f"cached physical rollout provenance mismatch in {episode_name}: {mismatches}")
        event_relpath = f"episodes/{episode_name}/direct_hdr_events_1khz_2khz.npz"
        if old_manifest.get(event_relpath) != _sha256(v2e_events_path):
            raise ValueError(f"V2E source event hash differs from the preserved manifest for {episode_name}")
        expected_split = "train" if episode_id in train_ids else "validation"
        if pair.get("split") != expected_split:
            raise ValueError(f"V2E episode split label mismatch for {episode_name}")

        result = json.loads(result_path.read_text(encoding="utf-8"))
        if int(pair["episode_id"]) != episode_id or int(pair["seed"]) != int(result["seed"]):
            raise ValueError(f"episode identity mismatch for {episode_name}")
        trace = _jsonl(trace_path)
        replay_episode = replay_root / episode_name
        replay_episode.mkdir()
        replay_started = time.perf_counter()
        events = _make_ours_events(cache_path, replay_episode, frames_dir=None, response_config=None)
        generation_wall_s = time.perf_counter() - replay_started
        _write_source_episode(ours_root, episode_id, result, trace, events)
        ours_episode = ours_root / "episodes" / episode_name
        ours_pair = {
            "episode_id": episode_id,
            "seed": int(result["seed"]),
            "speed_m_s": float(result["launch_speed_m_s"]),
            "launch_y_m": float(result["start_y_m"]),
            "shared_cache_sha256": _sha256(cache_path),
            "frame_manifest_sha256": _sha256(frame_manifest),
            "episode_result_sha256": _sha256(result_path),
            "trace_sha256": _sha256(trace_path),
            "v2e_event_archive_sha256": _sha256(v2e_events_path),
            "split_contract_sha256": split_sha,
            "split": expected_split,
            "ours_generator": "GenesisPhysicsEventPlugin from cached scene-linear HDR radiance",
            "ours_event_count": int(len(events)),
            "same_physical_rollout_timestamps_labels_and_split_as_v2e": True,
            "v2e_source_manifest_sha256": expected_v2e_manifest,
        }
        (ours_episode / "pair_provenance.json").write_text(json.dumps(ours_pair, indent=2), encoding="utf-8")
        episode_records.append({
            "episode_id": episode_id,
            "seed": int(result["seed"]),
            "speed_m_s": float(result["launch_speed_m_s"]),
            "launch_y_m": float(result["start_y_m"]),
            "cache_sha256": _sha256(cache_path),
            "timestamps_sha256": _timestamp_hash(cache_path),
            "trace_sha256": _sha256(trace_path),
            "ours_event_count": int(len(events)),
            "ours_event_archive_sha256": _sha256(ours_episode / "direct_hdr_events_1khz_2khz.npz"),
            "v2e_event_archive_sha256": _sha256(v2e_events_path),
            "ours_generation_wall_s": generation_wall_s,
        })
        print(json.dumps({"episode_id": episode_id, "ours_event_count": int(len(events)),
                          "ours_generation_wall_s": generation_wall_s}), flush=True)

    generation_wall = time.perf_counter() - generation_started
    _write_hash_manifest(ours_root)
    fit_started = time.perf_counter()
    runner_python = Path(args.python).resolve()
    fit_records = {
        "ours_native_hdr": _run_fit(
            runner_python=runner_python, scene_config=args.scene_config.resolve(), source_root=ours_root,
            out=out / "ours_fit", train_ids=train_ids, validation_ids=validation_ids, training=training,
            source_label="native scene-linear HDR Ours events", log_path=out / "ours_fit.log",
        ),
        "v2e_official": _run_fit(
            runner_python=runner_python, scene_config=args.scene_config.resolve(), source_root=v2e_root,
            out=out / "v2e_fit", train_ids=train_ids, validation_ids=validation_ids, training=training,
            source_label="unchanged official V2E events from audited paired batch", log_path=out / "v2e_fit.log",
        ),
    }
    fit_wall = time.perf_counter() - fit_started

    reference = args.reference_root.resolve()
    deceleration = float(json.loads(args.scene_config.read_text(encoding="utf-8"))["controller"]["initial_deceleration_assumption_m_s2"])
    evaluation_out = out / "common_reference_eval"
    eval_command = [
        str(runner_python), "-m", "benchmark.robot_closed_loop.diagnostics.evaluate_r1_common_direct_hdr",
        "--scene-config", str(args.scene_config.resolve()), "--control-config", str(args.control_config.resolve()),
        "--reference-root", str(reference),
        "--ours-checkpoint", str(out / "ours_fit/spatial_event_state.pt"),
        "--v2e-checkpoint", str(out / "v2e_fit/spatial_event_state.pt"),
        "--output", str(evaluation_out), "--deceleration", str(deceleration),
    ]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(training["threads"]),
                   "MKL_NUM_THREADS": str(training["threads"]), "OPENBLAS_NUM_THREADS": str(training["threads"]),
                   "PYTHONPATH": str(ROOT)}
    eval_started = time.perf_counter()
    process = subprocess.run(eval_command, cwd=ROOT, env=environment, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (out / "common_reference_eval.log").write_text(process.stdout, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"common Direct-HDR evaluation failed: {process.stdout[-3000:]}")
    eval_wall = time.perf_counter() - eval_started

    summary = {
        "status": "PASS",
        "experiment_id": experiment["experiment_id"],
        "purpose": "small native HDR development retrain and common Direct-HDR evaluation; not ranking or formal test",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "experiment_config_sha256": _sha256(experiment_path),
        "scene_config_sha256": _sha256(args.scene_config.resolve()),
        "control_config_sha256": _sha256(args.control_config.resolve()),
        "cache_root": str(cache_root),
        "cache_input_sha256": cache_inputs,
        "v2e_input_source": str(old_v2e_root),
        "v2e_source_manifest_sha256": _sha256(old_manifest_path),
        "ours_source_manifest_sha256": _sha256(ours_root / "SHA256SUMS"),
        "episode_split": {**split_contract, "sha256": split_sha},
        "episodes": episode_records,
        "fits": fit_records,
        "common_reference": {
            "root": str(reference),
            "manifest_sha256": _sha256(reference / "SHA256SUMS"),
            "summary_sha256": _sha256(reference / "r0_5_reference_summary.json"),
            "both_models_read_same_files_and_poll_times": True,
            "evaluation_summary_sha256": _sha256(evaluation_out / "r1_common_direct_hdr_summary.json"),
            "evaluation_poll_csv_sha256": _sha256(evaluation_out / "r1_common_direct_hdr_polls.csv"),
        },
        "frozen_core_modified": False,
        "boundaries": {
            "ours_uses_native_scene_linear_hdr": True,
            "v2e_events_reused_byte_for_byte_from_existing_official_pipeline": True,
            "models_receive_only_causal_event_tensor": True,
            "GT_used_only_for_training_labels_and_post_inference_scoring": True,
            "existing_nine_episode_reference_is_development_only": True,
            "no_physics_rerun_or_new_rollout_capture": True,
        },
        "cost_s": {
            "cached_native_hdr_event_generation_all_episodes": generation_wall,
            "two_equal_budget_cpu_model_fits": fit_wall,
            "common_reference_evaluation": eval_wall,
            "end_to_end_this_process": generation_wall + fit_wall + eval_wall,
            "excludes": "original physical rollouts, V2E generation, and original Direct-HDR reference capture",
        },
        "interpretation": "This run isolates the downstream value of training on native-HDR Ours events while retaining the already-generated V2E baseline. It is not a response-matched sensor comparison and is not a held-out estimate.",
    }
    (out / "native_hdr_retrain_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_hash_manifest(out)
    print(json.dumps({
        "status": "PASS", "output": str(out), "cost_s": summary["cost_s"],
        "checkpoint_sha256": {key: value["checkpoint_sha256"] for key, value in fit_records.items()},
        "evaluation_summary": json.loads((evaluation_out / "r1_common_direct_hdr_summary.json").read_text(encoding="utf-8"))["overall"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
