#!/usr/bin/env python3
"""Build a compact, verifiable 45-epoch archive from the retained flow logs.

The original first epoch lives under the paired pilot directory. Epochs 2–45
live under the resumed full-training directory. This script joins them only
after checking contiguous optimizer updates and epoch validation coverage.
Large logs/checkpoints remain at their original g1 paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ARMS = ("eventgenesis", "v2e")
EXPECTED_UPDATES = 8910
UPDATES_PER_EPOCH = 198


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finite(record: dict, keys: tuple[str, ...], source: str) -> None:
    for key in keys:
        if key in record and record[key] is not None and not math.isfinite(float(record[key])):
            raise ValueError(f"non-finite {key} in {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "flow_seed0_45_epoch_curve.csv").exists() or (output / "artifact_index.json").exists():
        raise FileExistsError(f"refusing to overwrite an existing flow archive in {output}")

    arm_curves: dict[str, list[dict]] = {}
    artifacts: list[dict] = []
    exit_receipts: dict[str, dict] = {}
    for arm in ARMS:
        pilot_log = campaign / "pilot" / arm / "training_metrics.jsonl"
        full_log = campaign / "full_training_retry1" / arm / "training_metrics.jsonl"
        pilot = read_jsonl(pilot_log)
        full = read_jsonl(full_log)
        updates = [r for r in pilot + full if "optimizer_update" in r]
        update_ids = [int(r["optimizer_update"]) for r in updates]
        if update_ids != list(range(1, EXPECTED_UPDATES + 1)):
            raise ValueError(f"{arm}: expected contiguous updates 1..{EXPECTED_UPDATES}, got {len(update_ids)} records")
        for row in updates:
            finite(row, ("loss", "photometric_loss", "smoothness_loss", "gradient_l2_norm"), arm)
        validation = [r for r in pilot + full if r.get("event") == "synthetic_validation"]
        per_epoch = {int(r["epoch"]): r for r in validation if r.get("epoch") is not None}
        if set(per_epoch) != set(range(1, 46)):
            raise ValueError(f"{arm}: synthetic validation must cover epochs 1..45, got {sorted(per_epoch)}")
        initial = [r for r in pilot if r.get("event") == "initial_synthetic_validation"]
        if len(initial) != 1 or int(initial[0].get("optimizer_updates", -1)) != 0:
            raise ValueError(f"{arm}: missing unique initial zero-update validation")
        per_update_epoch_counts = defaultdict(int)
        for row in updates:
            epoch = int(row["epoch"])
            per_update_epoch_counts[epoch] += 1
        if any(per_update_epoch_counts[e] != UPDATES_PER_EPOCH for e in range(1, 46)):
            raise ValueError(f"{arm}: every epoch must contain {UPDATES_PER_EPOCH} optimizer updates")

        epochs = []
        for epoch in range(1, 46):
            train = [r for r in updates if int(r["epoch"]) == epoch]
            val = per_epoch[epoch]
            finite(val, ("model_mean_epe_px_per_30ms", "zero_flow_mean_epe_px_per_30ms"), arm)
            epochs.append({
                "generator": arm,
                "epoch": epoch,
                "first_optimizer_update": min(int(r["optimizer_update"]) for r in train),
                "last_optimizer_update": max(int(r["optimizer_update"]) for r in train),
                "train_loss_mean": sum(float(r["loss"]) for r in train) / len(train),
                "photometric_loss_mean": sum(float(r["photometric_loss"]) for r in train) / len(train),
                "smoothness_loss_mean": sum(float(r["smoothness_loss"]) for r in train) / len(train),
                "gradient_l2_mean": sum(float(r["gradient_l2_norm"]) for r in train) / len(train),
                "synthetic_validation_epe_px_per_30ms": float(val["model_mean_epe_px_per_30ms"]),
                "zero_flow_epe_px_per_30ms": float(val["zero_flow_mean_epe_px_per_30ms"]),
                "best_epoch_at_validation": int(val["best_epoch"]),
                "best_synthetic_validation_epe_px_per_30ms": float(val["best_epe"]),
                "validation_rollout_count": len(val.get("model_epe_by_rollout", {})),
                "real_data_used": bool(val.get("real_data_used", False)),
            })
        if any(row["real_data_used"] for row in epochs):
            raise ValueError(f"{arm}: real data was marked in a synthetic validation record")
        arm_curves[arm] = epochs

        paths = [
            pilot_log, full_log,
            campaign / "pilot" / arm / "live_status.json",
            campaign / "full_training_retry1" / arm / "live_status.json",
            campaign / "pilot" / arm / "run_metadata.json",
            campaign / "full_training_retry1" / arm / "run_metadata.json",
            campaign / "pilot" / arm / "synthetic_best.pt",
            campaign / "pilot" / arm / "last.pt",
            campaign / "full_training_retry1" / arm / "synthetic_best.pt",
            campaign / "full_training_retry1" / arm / "last.pt",
            campaign / "resume_retry1" / f"full_resume_{arm}.command.txt",
            campaign / "resume_retry1" / f"full_resume_{arm}.exit_code.txt",
        ]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            artifacts.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
        pilot_status = json.loads((campaign / "pilot" / arm / "live_status.json").read_text())
        full_status = json.loads((campaign / "full_training_retry1" / arm / "live_status.json").read_text())
        if (full_status.get("status"), full_status.get("completed_updates"), full_status.get("total_updates")) != (
                "complete", EXPECTED_UPDATES, EXPECTED_UPDATES):
            raise ValueError(f"{arm}: final status does not prove complete registered budget: {full_status}")
        if full_status.get("best_epoch") != int(per_epoch[45]["best_epoch"]):
            raise ValueError(f"{arm}: final live status and final curve best epoch differ")
        process_exit = int((campaign / "resume_retry1" / f"full_resume_{arm}.exit_code.txt").read_text().strip())
        if process_exit != 0:
            raise ValueError(f"{arm}: resumed full-training process exit code is {process_exit}")
        exit_receipts[arm] = {
            "pilot_status": pilot_status.get("status"),
            "pilot_completed_updates": pilot_status.get("completed_updates"),
            "full_status": full_status.get("status"),
            "completed_updates": full_status.get("completed_updates"),
            "total_updates": full_status.get("total_updates"),
            "training_process_exit_code": process_exit,
            "best_epoch": full_status.get("best_epoch"),
            "best_synthetic_validation_epe_px_per_30ms": full_status.get("best_synthetic_validation_epe_px_per_30ms"),
            "initial_state_sha256": full_status.get("initial_state_sha256"),
            "terminal_state_sha256": full_status.get("current_state_sha256"),
            "last_checkpoint": full_status.get("last_checkpoint"),
            "last_checkpoint_sha256": sha256(Path(full_status["last_checkpoint"])),
            "updated_utc": full_status.get("updated_utc"),
        }

    if exit_receipts["eventgenesis"]["initial_state_sha256"] != exit_receipts["v2e"]["initial_state_sha256"]:
        raise ValueError("paired runs do not share the registered initial model state")
    fields = list(arm_curves["eventgenesis"][0])
    curve_csv = output / "flow_seed0_45_epoch_curve.csv"
    with curve_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for arm in ARMS:
            writer.writerows(arm_curves[arm])

    campaign_exit = campaign / "exit_code.txt"
    continuation_exit = campaign / "continue_after_data.exit_code.txt"
    root_exit = int(campaign_exit.read_text().strip()) if campaign_exit.is_file() else None
    continuation_exit_code = int(continuation_exit.read_text().strip()) if continuation_exit.is_file() else None
    retry_exit_file = campaign / "resume_retry1" / "exit_code.txt"
    retry_exit = int(retry_exit_file.read_text().strip()) if retry_exit_file.is_file() else None
    if retry_exit != 0:
        raise ValueError(f"successful retry supervisor exit code must be 0, got {retry_exit}")
    for arm in ARMS:
        for name in ("pilot_eventgenesis.exit_code.txt", "pilot_v2e.exit_code.txt"):
            exit_path = campaign / name
            if not exit_path.is_file() or exit_path.read_text().strip() != "0":
                raise ValueError(f"paired epoch-1 pilot process did not exit 0: {exit_path}")
    artifacts.extend({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                      for path in (campaign_exit, continuation_exit, retry_exit_file,
                                   campaign / "resume_retry1" / "full_resume_eventgenesis.exit_code.txt",
                                   campaign / "resume_retry1" / "full_resume_v2e.exit_code.txt",
                                   campaign / "resume_retry1" / "campaign_complete.txt")
                      if path.is_file())

    index = {
        "schema": "stage4_flow_seed0_full_curve_archive_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": "45-epoch corrected-v2 dt1 seed-0 synthetic training; epoch 1 is the completed paired pilot, epochs 2-45 are the continuation",
        "epochs": 45,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "updates_per_generator": EXPECTED_UPDATES,
        "curve_csv": str(curve_csv.relative_to(output.parent)),
        "curve_csv_size_bytes": curve_csv.stat().st_size,
        "curve_csv_sha256": sha256(curve_csv),
        "real_data_used": False,
        "real_scores_authorized": False,
        "campaign_exit_code": root_exit,
        "continuation_exit_code": continuation_exit_code,
        "successful_retry_supervisor_exit_code": retry_exit,
        "exit_code_note": "The original campaign wrapper exits 2 and its first post-data continuation exits 1 after the strict replay/equivalence gate failed. The preserved retry1 resume check then completed; each completed per-arm 45-epoch trainer and retry1 supervisor exited 0. These earlier gate failures are retained and are not training-process exit codes.",
        "arms": exit_receipts,
        "artifacts": artifacts,
    }
    index_path = output / "artifact_index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "curve_csv": str(curve_csv), "curve_csv_sha256": index["curve_csv_sha256"],
                      "campaign_exit_code": root_exit, "continuation_exit_code": continuation_exit_code,
                      "arms": exit_receipts}, indent=2))


if __name__ == "__main__":
    main()
