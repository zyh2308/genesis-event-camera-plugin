#!/usr/bin/env python3
"""Cross-check one E2VID synthetic fixture from shared bank through metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--generator", choices=("eventgenesis", "v2e"), required=True)
    parser.add_argument("--metrics-result", type=Path, required=True,
                        help="use a completed, hash-bound synthetic metric result")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite verifier output: {output}")
    fixture_root = run / "fixture" / args.generator
    sequence_root = fixture_root / f"s5q_02_train_{args.generator}"
    inference_root = run / "inference" / args.generator
    fixture = read_json(fixture_root / "synthetic_fixture_manifest.json")
    inference = read_json(inference_root / "inference_manifest.json")
    metrics_path = args.metrics_result.resolve()
    metrics = read_json(metrics_path)
    fixture_manifest_path = fixture_root / "synthetic_fixture_manifest.json"
    inference_manifest_path = inference_root / "inference_manifest.json"
    if fixture.get("schema") != "stage5_synthetic_e2vid_memmap_fixture_v1":
        raise RuntimeError("unexpected synthetic fixture schema")
    if (fixture.get("real_data_used") is not False or fixture.get("real_scoring_enabled") is not False
            or fixture.get("checkpoint_selection_allowed") is not False):
        raise RuntimeError("fixture is not registered as isolated synthetic-only plumbing")
    if fixture.get("rollout_id") != "s5q_02_train" or fixture.get("split") != "train":
        raise RuntimeError("fixture rollout differs from the registered shared smoke")
    if fixture.get("initial_anchor_excluded_from_model_targets") is not True or fixture.get("target_frame_count") != 200:
        raise RuntimeError("fixture target endpoints do not exclude exactly the initial recurrent boundary")
    if (inference.get("status") != "synthetic_fixture_event_only_inference_completed_no_GT_scoring"
            or inference.get("model", {}).get("reference_frame_pixels_used_as_input") is not False
            or inference.get("model", {}).get("real_GT_scoring_performed") is not False
            or inference.get("sequence", {}).get("num_samples_processed") != 200
            or inference.get("model", {}).get("recurrent_reset_calls") != 1):
        raise RuntimeError("event-only recurrent inference contract failed")
    if inference.get("model", {}).get("checkpoint_sha256") != metrics.get("checkpoint_sha256"):
        raise RuntimeError("inference and metric artifacts use different candidate checkpoint hashes")
    if (metrics.get("schema") != "stage5_synthetic_reconstruction_metric_result_v1"
            or metrics.get("status") != "synthetic_plumbing_pass_not_model_quality_or_generator_result"
            or metrics.get("real_data_used") is not False
            or metrics.get("real_scoring_enabled") is not False
            or metrics.get("checkpoint_selection_allowed") is not False
            or metrics.get("target_frame_count") != 200):
        raise RuntimeError("metric output is not the registered synthetic-only 200-endpoint run")
    if (metrics.get("fixture_manifest", {}).get("sha256") != sha256(fixture_manifest_path)
            or metrics.get("inference_manifest", {}).get("sha256") != sha256(inference_manifest_path)
            or metrics.get("checkpoint_sha256") != inference["model"].get("checkpoint_sha256")):
        raise RuntimeError("metric result is not bound to these exact fixture, inference and checkpoint artifacts")
    metric_rows = metrics.get("per_frame", [])
    if [row.get("sample_index") for row in metric_rows] != list(range(1, 201)):
        raise RuntimeError("metric rows do not cover all 200 inferable endpoints in order")
    if Path(inference["sequence"]["path"]).resolve() != sequence_root:
        raise RuntimeError("inference path differs from this fixture")
    bank = Path(fixture["physical_bank_path"]).resolve()
    if not bank.is_file() or sha256(bank) != fixture.get("physical_bank_sha256"):
        raise RuntimeError("fixture bank path/hash does not verify")
    for name, item in fixture.get("files", {}).items():
        path = Path(item["path"]).resolve()
        if path.parent != sequence_root or not path.is_file() or sha256(path) != item.get("sha256"):
            raise RuntimeError(f"fixture input file fails manifest hash check: {name}")
    for row in inference.get("registered_inference", {}).get("saved_images", []):
        image_path = Path(row["path"]).resolve()
        if not image_path.is_file() or sha256(image_path) != row.get("sha256"):
            raise RuntimeError(f"saved uint8 reconstruction does not verify: {image_path}")
        path = Path(row["float_array_path"]).resolve()
        if not path.is_file() or sha256(path) != row.get("float_array_sha256"):
            raise RuntimeError(f"saved float32 reconstruction does not verify: {path}")
    if len(inference["registered_inference"].get("saved_images", [])) != 200:
        raise RuntimeError("inference artifact is missing one or more registered endpoints")

    first_pass_exit_path = run / "exit_code.txt"
    first_pass_exit = first_pass_exit_path.read_text().strip() if first_pass_exit_path.is_file() else None
    result = {
        "schema": "stage5_synthetic_reconstruction_fixture_verification_v1",
        "status": "synthetic_fixture_end_to_end_pass_no_real_scoring",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": args.generator,
        "real_data_used": False,
        "real_scoring_enabled": False,
        "checkpoint_selection_allowed": False,
        "candidate_checkpoint": {
            "path": inference["model"]["checkpoint_path"],
            "sha256": inference["model"]["checkpoint_sha256"],
        },
        "physical_bank": {
            "path": str(bank), "size_bytes": bank.stat().st_size,
            "sha256": fixture["physical_bank_sha256"],
        },
        "fixture_manifest": {"path": str(fixture_root / "synthetic_fixture_manifest.json"),
                              "sha256": sha256(fixture_root / "synthetic_fixture_manifest.json")},
        "inference_manifest": {"path": str(inference_root / "inference_manifest.json"),
                               "sha256": sha256(inference_root / "inference_manifest.json")},
        "metric_result": {"path": str(metrics_path), "sha256": sha256(metrics_path),
                           "metric_mean": metrics["metric_mean"],
                           "lpips_source": metrics.get("lpips_source")},
        "verifier_script": {"path": str(Path(__file__).resolve()),
                            "sha256": sha256(Path(__file__).resolve())},
        "event_windows": {
            "image_endpoints_including_anchor": fixture["image_endpoint_count_including_anchor"],
            "inferable_target_endpoints": fixture["target_frame_count"],
            "model_calls": inference["sequence"]["num_samples_processed"],
            "recurrent_state_reset_calls": inference["model"]["recurrent_reset_calls"],
            "anchor_skipped": inference["sequence"]["initial_anchor_endpoint_excluded_from_model_and_metrics"],
            "event_only_input": inference["model"]["input_contract"],
        },
        "initial_wrapper_exit_code": first_pass_exit,
        "initial_wrapper_failure_reason": (
            "The first combined invocation hit an old PerceptualSimilarity import of skimage.measure.compare_ssim; "
            "the compatibility alias was added, and the same frozen inference artifact was then evaluated by the "
            "registered metric script into the hash-bound metrics_result path."
            if first_pass_exit not in (None, "0") else None
        ),
        "interpretation": "pipeline wiring and numeric contracts only; metrics are not a model-quality or generator-ranking result",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "generator": args.generator,
                      "targets": result["event_windows"]["inferable_target_endpoints"],
                      "metric_sha256": result["metric_result"]["sha256"],
                      "output": str(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
