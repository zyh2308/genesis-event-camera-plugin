#!/usr/bin/env python3
"""GPU-side audit of retry02 logs and checkpoints after final JSON serialization failed."""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--executed-script", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args()
    out = a.output_root.resolve()
    if out.exists():
        raise FileExistsError(out)
    repo = a.repo_root.resolve()
    run = a.run_root.resolve()
    sys.path[:0] = [str(repo / "benchmark/sim2real/scripts"),
                    str(repo / "benchmark/sim2real/compat"),
                    str(a.executed_script.resolve().parent)]
    import run_stage7_replicad_gtfit_retry02 as prior
    from stage5_flow_supervised_pilot import (
        SOURCE_COMMIT, model_output_to_flow_hw, model_source, state_sha256,
    )
    from verify_pinned_source import verify_pinned_source

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint audit")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = json.loads(a.config.read_text())
    manifest_path = a.data_root.resolve() / "training_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("real_data_used") is not False:
        raise RuntimeError("real data must not enter the checkpoint audit")
    rows = prior.load_rows(manifest)
    if len(rows["validation"]) != 2:
        raise RuntimeError("expected two fixed independent validation rollouts")
    masks = {}
    for split_rows in rows.values():
        for row in split_rows:
            x = np.asarray(row["arms"]["eventgenesis"]["events"])
            y = np.asarray(row["arms"]["v2e"]["events"])
            if x.shape != y.shape:
                raise RuntimeError("paired event tensors differ")
            valid = np.asarray(row["arms"]["eventgenesis"]["flow_valid"], dtype=bool)
            masks[row["rollout_id"]] = (np.any(x != 0, axis=(0, 3)) |
                                        np.any(y != 0, axis=(0, 3))) & valid
    expected_updates = int(config["diagnostic_budget"]["updates_per_arm"])
    arms = config["pairing"]["arms"]
    if arms != ["eventgenesis", "v2e"]:
        raise RuntimeError(f"unexpected paired arms {arms}")
    common = {
        "schema": "stage7_gtfit_retry02_postrun_checkpoint_audit_v1",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_process_exit_code": 1,
        "training_process_failure": "JSON summary rejected torch CUDA UUID object after training/checkpoint/restore loop",
        "synthetic_only": True,
        "formal_checkpoint_selection": False,
        "gpu": torch.cuda.get_device_name(device),
        "source_commit": SOURCE_COMMIT,
        "source_verification": verify_pinned_source(
            a.model_root.resolve(), "STE-FlowNet", SOURCE_COMMIT,
            repo / "benchmark/sim2real/manifests/downstream_source_bundle.json"),
        "executed_training_script": str(a.executed_script.resolve()),
        "executed_training_script_sha256": digest(a.executed_script),
        "data_manifest_sha256": digest(manifest_path),
        "config_sha256": digest(a.config),
        "arms": {},
    }
    initial_hashes = set()
    for arm in arms:
        arm_dir = run / "flow_gtfit_retry02" / arm
        log_path = arm_dir / "updates.jsonl"
        ckpt_path = arm_dir / "checkpoint_final.pt"
        records = [json.loads(s) for s in log_path.read_text().splitlines() if s.strip()]
        updates = [x for x in records if x.get("record_type") == "optimizer_update"]
        evals = [x for x in records if x.get("record_type") == "evaluation"
                 and x.get("split") == "validation" and x.get("update") == expected_updates]
        if len(updates) != expected_updates or len(evals) != 2:
            raise RuntimeError(f"{arm}: update/evaluation counts {len(updates)}/{len(evals)}")
        for i, rec in enumerate(updates, 1):
            if rec.get("update") != i or rec.get("optimizer_step_values") != [i]:
                raise RuntimeError(f"{arm}: optimizer update trace is not continuous at {i}")
            if (not math.isfinite(float(rec["loss_before_update"])) or
                    not math.isfinite(float(rec["gradient_l2_norm"])) or
                    float(rec["gradient_l2_norm"]) <= 0):
                raise RuntimeError(f"{arm}: invalid loss/gradient at update {i}")
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        if saved.get("generator") != arm or int(saved.get("updates", -1)) != expected_updates:
            raise RuntimeError(f"{arm}: checkpoint identity or update count mismatch")
        model, _ = model_source(a.model_root.resolve(), device)
        model.load_state_dict(saved["state_dict"], strict=True)
        model.eval()
        state_hash = state_sha256(model)
        if state_hash != saved.get("state_sha256"):
            raise RuntimeError(f"{arm}: loaded checkpoint state hash mismatch")
        initial_hashes.add(saved.get("initial_state_sha256"))
        optimizer = torch.optim.Adam([
            {"params": model.bias_parameters(), "weight_decay": float(config["pairing"]["weight_decay_biases"])},
            {"params": model.weight_parameters(), "weight_decay": float(config["pairing"]["weight_decay_weights"])},
        ], lr=float(config["pairing"]["learning_rate"]), betas=tuple(config["pairing"]["betas"]))
        optimizer.load_state_dict(saved["optimizer"])
        opt_steps = prior.optimizer_steps(optimizer)
        if opt_steps != [expected_updates]:
            raise RuntimeError(f"{arm}: restored optimizer steps are {opt_steps}")
        repeat, _ = model_source(a.model_root.resolve(), device)
        repeat.load_state_dict(saved["state_dict"], strict=True)
        repeat.eval()
        validation = []
        for row in rows["validation"]:
            sample = row["arms"][arm]
            helper = {"model_output_to_flow_hw": model_output_to_flow_hw}
            p1 = prior.evaluate(model, sample, masks[row["rollout_id"]], device, helper)
            p2 = prior.evaluate(repeat, sample, masks[row["rollout_id"]], device, helper)
            max_abs = float(np.max(np.abs(p1["prediction"] - p2["prediction"])))
            if max_abs > 1e-6:
                raise RuntimeError(f"{arm}/{row['rollout_id']}: repeat prediction mismatch {max_abs}")
            log_row = next(x for x in evals if x["rollout_id"] == row["rollout_id"])
            keys = ("full_valid_epe_px_per_30ms", "common_event_active_epe_px_per_30ms",
                    "zero_flow_full_valid_epe_px_per_30ms", "zero_flow_common_event_active_epe_px_per_30ms")
            compare = {}
            for key in keys:
                observed, expected = p1[key], log_row[key]
                ok = ((observed is None and expected is None) or
                      (observed is not None and expected is not None and
                       math.isfinite(float(observed)) and abs(float(observed)-float(expected)) <= 2e-5))
                compare[key] = {"logged": expected, "recomputed": observed, "within_2e-5": ok}
                if not ok:
                    raise RuntimeError(f"{arm}/{row['rollout_id']}: metric mismatch {key}")
            validation.append({
                "rollout_id": row["rollout_id"],
                "prediction_hash_recomputed": p1["prediction_flow_sha256"],
                "prediction_hash_matches_log": p1["prediction_flow_sha256"] == log_row["prediction_flow_sha256"],
                "repeat_checkpoint_max_abs_prediction_delta": max_abs,
                "metrics": compare,
            })
        common["arms"][arm] = {
            "updates": len(updates),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_bytes": ckpt_path.stat().st_size,
            "checkpoint_sha256": digest(ckpt_path),
            "state_sha256_saved": saved["state_sha256"],
            "state_sha256_loaded": state_hash,
            "restored_optimizer_steps": opt_steps,
            "initial_state_sha256": saved["initial_state_sha256"],
            "validation": validation,
        }
        del model, repeat, optimizer, saved
        torch.cuda.empty_cache()
    if len(initial_hashes) != 1:
        raise RuntimeError(f"paired starts differ: {initial_hashes}")
    common["paired_initial_state_equal"] = True
    common["status"] = "postrun_checkpoint_and_validation_audit_passed"
    common["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out.mkdir(parents=True)
    path = out / "restore_and_validation_audit.json"
    path.write_text(json.dumps(common, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": common["status"], "output": str(path),
                      "arms": {k: {"updates": v["updates"], "checkpoint_sha256": v["checkpoint_sha256"],
                                  "validation": [{"rollout": r["rollout_id"],
                                                  "active_epe": r["metrics"]["common_event_active_epe_px_per_30ms"]["recomputed"],
                                                  "zero_active_epe": r["metrics"]["zero_flow_common_event_active_epe_px_per_30ms"]["recomputed"],
                                                  "prediction_hash_matches_log": r["prediction_hash_matches_log"]}
                                                 for r in v["validation"]]}
                               for k,v in common["arms"].items()}}, indent=2))


if __name__ == "__main__":
    main()
