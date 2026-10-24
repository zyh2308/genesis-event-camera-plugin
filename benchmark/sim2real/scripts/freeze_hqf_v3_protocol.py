#!/usr/bin/env python3
"""Create an immutable execution copy of the reviewed HQF v3 candidate.

The candidate remains unchanged. This script binds the freeze to the exact
candidate, 14-sequence data manifest, corrected-Stage5 training artifact
manifest, model config, pinned source bundle, and evaluation entry code.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SIM2REAL = ROOT / "benchmark/sim2real"
CANDIDATE = SIM2REAL / "protocol/hqf_14_reconstruction_test_candidate_20261021_v3.json"
DATA_MANIFEST = SIM2REAL / "evidence/S5/20261021-recon-flow-localization/20261021-hqf14-manifest-validation-v3/hqf_14_file_manifest_v3.json"
TRAINING_MANIFEST = SIM2REAL / "evidence/S5/20261021-recon-flow-localization/reconstruction_training_artifact_manifest.json"
MODEL_CONFIG = SIM2REAL / "configs/stage4_corrected_v2_flowdt1_seed0.json"
TRAINING_CONFIG = SIM2REAL / "configs/stage4_training_launch_v1.json"
SOURCE_BUNDLE = SIM2REAL / "manifests/downstream_source_bundle.json"
INFER_SCRIPT = SIM2REAL / "scripts/infer_hqf_e2vid_candidate.py"
SCORE_SCRIPT = SIM2REAL / "scripts/score_hqf14_reconstruction.py"
RUN_SCRIPT = SIM2REAL / "scripts/run_hqf14_reconstruction_eval.py"
SELFTEST_SCRIPT = SIM2REAL / "scripts/test_hqf14_reconstruction_eval.py"
FREEZE_SCRIPT = SIM2REAL / "scripts/freeze_hqf_v3_protocol.py"
DEFAULT_OUTPUT = SIM2REAL / "protocol/hqf_14_reconstruction_test_frozen_20261024_v1.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def preview_indices(frame_count: int) -> list[int]:
    """Fixed endpoints at 0/25/50/75/100% over inferable indices 1..N-1."""
    if frame_count < 2:
        raise ValueError(f"HQF sequence needs at least two frames, got {frame_count}")
    n = frame_count - 1
    offsets = [int((k * (n - 1)) // 4) for k in range(5)]
    return sorted({1 + value for value in offsets})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {output}")

    for path in (CANDIDATE, DATA_MANIFEST, TRAINING_MANIFEST, MODEL_CONFIG, TRAINING_CONFIG,
                 SOURCE_BUNDLE, INFER_SCRIPT, SCORE_SCRIPT, RUN_SCRIPT, SELFTEST_SCRIPT, FREEZE_SCRIPT):
        if not path.is_file():
            raise FileNotFoundError(path)

    candidate = read_json(CANDIDATE)
    data = read_json(DATA_MANIFEST)
    training = read_json(TRAINING_MANIFEST)
    model_config = read_json(MODEL_CONFIG)
    training_config = read_json(TRAINING_CONFIG)
    if candidate.get("schema") != "hqf_14_reconstruction_test_candidate_v3":
        raise ValueError("source protocol is not HQF candidate v3")
    if candidate.get("sequences") != [row.get("sequence") for row in data.get("sequences", [])]:
        raise ValueError("HQF candidate and data manifest sequence lists differ or are reordered")
    if (data.get("status") != "ready_unscored" or data.get("expected_sequence_count") != 14
            or data.get("ready_unscored_sequence_count") != 14
            or data.get("real_scoring_enabled") is not False
            or data.get("real_gt_metrics_computed") is not False):
        raise ValueError("only a complete, unscored 14-sequence data manifest can be frozen")
    if (training.get("schema") != "stage5_reconstruction_training_artifact_manifest_v1"
            or training.get("scope") != "completed corrected-TC synthetic-only paired reconstruction seed-0 training"
            or training.get("real_scoring_enabled") is not False
            or training.get("frozen_simulator_core") != "e79e656ac00f7847b98db9e12bf43b093f3afff1"):
        raise ValueError("training artifact manifest is not the expected corrected Stage-5 pair")
    if not isinstance(model_config.get("reconstruction", {}).get("model_args"), dict):
        raise ValueError("registered reconstruction model args are missing")
    if training.get("shared", {}).get("config_sha256") != sha256(TRAINING_CONFIG):
        raise ValueError("Stage-5 training artifact config hash differs from the registered training config")

    arms = {}
    for name in ("eventgenesis", "v2e"):
        record = training.get("arms", {}).get(name, {})
        best = record.get("best_checkpoint", {})
        if (record.get("status") != "complete" or record.get("exit_code") != 0
                or record.get("best_epoch") is None or record.get("best_synthetic_validation_lpips_vgg") is None
                or not best.get("path") or best.get("size_bytes", 0) <= 0 or len(best.get("sha256", "")) != 64):
            raise ValueError(f"{name} is missing a completed synthetic-validation-best checkpoint record")
        arms[name] = {
            "checkpoint_path": best["path"],
            "checkpoint_size_bytes": best["size_bytes"],
            "checkpoint_sha256": best["sha256"],
            "selection": "corrected Stage-5 synthetic LPIPS-VGG best; real HQF is not used for selection",
            "best_epoch": record["best_epoch"],
            "best_synthetic_validation_lpips_vgg": record["best_synthetic_validation_lpips_vgg"],
        }

    previews = {}
    for row in data["sequences"]:
        name = row["sequence"]
        if row.get("status") != "ready_unscored" or row.get("evreal_memmap", {}).get("exists") is not True:
            raise ValueError(f"sequence {name} is not ready_unscored")
        files = row["evreal_memmap"].get("files", {})
        missing = row["evreal_memmap"].get("missing_files", [])
        if missing:
            raise ValueError(f"sequence {name} has missing converted files: {missing}")
        images = files.get("images.npy")
        if not images:
            raise ValueError(f"sequence {name} does not bind images.npy")
        shapes = row["evreal_memmap"].get("schema_checks", {}).get("shapes", {})
        image_shape = shapes.get("images")
        if not image_shape or len(image_shape) != 4 or image_shape[-1] != 1:
            raise ValueError(f"sequence {name} manifest does not register a valid images.npy shape: {image_shape}")
        previews[name] = {
            "num_frames_including_anchor": int(image_shape[0]),
            "sample_indices": preview_indices(int(image_shape[0])),
            "rule": "pre-inference fixed 0/25/50/75/100 percent positions over inferable sample indices 1..N-1; deduplicated; never output-selected",
        }

    frozen = copy.deepcopy(candidate)
    frozen["schema"] = "hqf_14_reconstruction_test_frozen_v1"
    frozen["status"] = "frozen_execution_protocol_unscored"
    frozen["real_scores_authorized"] = True
    frozen["real_scoring_authorized"] = True
    frozen["real_scoring_enabled"] = False
    frozen["protocol_freeze"] = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_candidate_path": str(CANDIDATE.relative_to(ROOT)),
        "source_candidate_sha256": sha256(CANDIDATE),
        "source_candidate": {"path": str(CANDIDATE.relative_to(ROOT)), "sha256": sha256(CANDIDATE)},
        "execution_policy": "The user authorized the first fixed HQF v3 two-arm evaluation. This file freezes that candidate for the run; do not edit or use its scores to tune/select anything. The evaluation has not been run by this freeze step.",
        "required_real_run_flag": "--enable-real-scoring",
        "data_manifest": {
            "path": str(DATA_MANIFEST.relative_to(ROOT)),
            "sha256": sha256(DATA_MANIFEST),
            "status": "ready_unscored",
            "expected_sequence_count": 14,
        },
        "training_artifact_manifest": {
            "path": str(TRAINING_MANIFEST.relative_to(ROOT)),
            "sha256": sha256(TRAINING_MANIFEST),
        },
        "model_config": {
            "path": str(MODEL_CONFIG.relative_to(ROOT)),
            "sha256": sha256(MODEL_CONFIG),
        },
        "training_config": {
            "path": str(TRAINING_CONFIG.relative_to(ROOT)),
            "sha256": sha256(TRAINING_CONFIG),
        },
        "source_bundle": {
            "path": str(SOURCE_BUNDLE.relative_to(ROOT)),
            "sha256": sha256(SOURCE_BUNDLE),
        },
        "execution_code": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (INFER_SCRIPT, SCORE_SCRIPT, RUN_SCRIPT, SELFTEST_SCRIPT, FREEZE_SCRIPT)
        },
        "checkpoint_bindings": arms,
        "pre_registered_preview_indices": previews,
        "metric_contract": {
            "per_frame": ["LPIPS-AlexNet", "SSIM", "PSNR", "MSE"],
            "sequence_aggregation": "arithmetic mean of all inferable frames in the sequence; PSNR exact-match frames have +infinity",
            "dataset_aggregation": "equal arithmetic mean of the 14 sequence means; not weighted by sequence length",
            "prediction_storage": "float32 [H,W], clipped once to [0,1] by the registered inference adapter",
            "real_data_used_for_training_or_selection": False,
            "checkpoint_selection_uses_real_data": False,
        },
    }
    frozen["data_integrity"]["score_gate"] = (
        "The 2026-10-24 user execution instruction freezes the already-reviewed v3 source-order/time-integrity rule for one first HQF evaluation. No score-based tuning, subset selection, or checkpoint selection is permitted."
    )
    frozen["data_integrity"]["score_gate_research_review_source"] = "reviews/recon_flow_localization_20261019/REVIEW.md"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(frozen, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    digest = sha256(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {output.name}\n")
    print(json.dumps({"status": frozen["status"], "protocol": str(output),
                      "sha256": digest, "sequences": len(previews),
                      "preview_indices": previews}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
