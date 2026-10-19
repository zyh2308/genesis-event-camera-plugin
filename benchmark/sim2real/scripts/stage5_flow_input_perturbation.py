#!/usr/bin/env python3
"""Fixed synthetic-best test of STE-FlowNet event-input dependence.

The three variants perturb the stored event representation before model input:
normal, all-zero, and reverse-order across the five time bins within each
polarity. The reversal preserves total counts and per-pixel spatial support.
Masks are frozen once from normal EventGenesis/V2E inputs and reused for every
variant. This is a small sensitivity diagnostic, not a correctness or
real-domain performance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark/sim2real/scripts"))
sys.path.insert(0, str(ROOT / "benchmark/sim2real/compat"))
from run_steflow_stage2_pilot import model_output_to_flow_hw, pure_torch_spatial_correlation
from steflow_align_corners import patch_ste
from verify_pinned_source import verify_pinned_source

SOURCE_COMMIT = "eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6"
FLOW_UNIT = "px_per_30ms_interval"
STRATA = {"s0": "camera_motion_static_scene", "s1": "independent_rigid_motion_static_camera",
          "s2": "camera_and_independent_rigid_motion", "s3": "combined_motion_with_occlusion"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_npz_selected(path: Path, index: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {}
        for key in ("events", "flow_gt", "flow_valid", "sample_ids", "event_start_s", "event_end_s", "label_sha256"):
            value = np.asarray(archive[key])
            # Rollout-level label hashes are stored as zero-dimensional NPZ
            # metadata; the sample arrays carry a leading window dimension.
            result[key] = value.copy() if value.ndim == 0 else np.asarray(value[index]).copy()
        return result


def load_checkpoint_model(path: Path, arm: str, model_root: Path, data_manifest_sha: str,
                          config_sha: str, device: torch.device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    checks = {"generator": arm, "seed": 0, "flow_unit": FLOW_UNIT,
              "data_manifest_sha256": data_manifest_sha, "config_sha256": config_sha,
              "resume_phase": "inference", "resume_allowed": False}
    mismatch = {k: (saved.get(k), v) for k, v in checks.items() if saved.get(k) != v}
    if mismatch:
        raise RuntimeError(f"{arm} synthetic-best metadata mismatch: {mismatch}")
    if not isinstance(saved.get("state_dict"), dict):
        raise RuntimeError(f"{arm} checkpoint lacks STE-FlowNet state_dict")
    sys.path.insert(0, str(model_root))
    import models
    import models.corr as corr
    import multiscaleloss
    corr.spatial_correlation_sample = pure_torch_spatial_correlation
    upstream = patch_ste(models, multiscaleloss, model_root)
    model = models.steflow().to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    current_hash = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        current_hash.update(name.encode())
        current_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    if saved.get("current_state_sha256") != current_hash.hexdigest():
        raise RuntimeError(f"{arm} checkpoint model-state hash mismatch")
    model.eval()
    metadata = {k: saved.get(k) for k in ("generator", "seed", "epoch", "optimizer_update", "next_epoch",
                                          "best_epoch", "best_synthetic_validation_epe", "initial_state_sha256",
                                          "current_state_sha256")}
    del saved
    return model, upstream, metadata


def predict(model, events: np.ndarray, device: torch.device) -> np.ndarray:
    if events.shape != (2, 260, 346, 5) or not np.isfinite(events).all():
        raise ValueError(f"expected finite events [2,260,346,5], got {events.shape}")
    value = torch.from_numpy(np.ascontiguousarray(events, dtype=np.float32)).unsqueeze(0)
    value = F.pad(value, (0, 0, 3, 3, 46, 46)).to(device)
    with torch.inference_mode():
        output = model(value, 352, use_dt4=False, flow_seq=False)
        if tuple(output.shape) != (1, 2, 352, 352):
            raise RuntimeError(f"unexpected STE output shape {tuple(output.shape)}")
        flow = model_output_to_flow_hw(output)[:, 46:306, 3:349]
    if flow.shape != (2, 260, 346) or not np.isfinite(flow).all():
        raise RuntimeError("decoded event-only flow must be finite [u,v,260,346]")
    return flow


def epe(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float | None:
    count = int(mask.sum())
    if not count:
        return None
    return float(np.hypot(pred[0, mask] - gt[0, mask], pred[1, mask] - gt[1, mask]).mean(dtype=np.float64))


def summarize_field(flow: np.ndarray, mask: np.ndarray) -> dict:
    count = int(mask.sum())
    if not count:
        return {"pixels": 0, "mean_magnitude": None, "median_magnitude": None, "mean_u": None, "mean_v": None}
    mag = np.hypot(flow[0], flow[1])[mask]
    return {"pixels": count, "mean_magnitude": float(mag.mean()), "median_magnitude": float(np.median(mag)),
            "mean_u": float(flow[0][mask].mean()), "mean_v": float(flow[1][mask].mean())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--eventgenesis-checkpoint", type=Path, required=True)
    p.add_argument("--v2e-checkpoint", type=Path, required=True)
    p.add_argument("--eventgenesis-sha256", required=True)
    p.add_argument("--v2e-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("pinned STE-FlowNet path requires CUDA")
    torch.cuda.set_device(device)
    model_root, data_root = args.model_root.resolve(), args.data_root.resolve()
    for arm, path, expected in (("eventgenesis", args.eventgenesis_checkpoint, args.eventgenesis_sha256),
                                ("v2e", args.v2e_checkpoint, args.v2e_sha256)):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"registered {arm} synthetic-best checkpoint is missing or changed")
    source = verify_pinned_source(model_root, "STE-FlowNet", SOURCE_COMMIT,
                                  ROOT / "benchmark/sim2real/manifests/downstream_source_bundle.json")
    bank_manifest_path = data_root / "training_bank_manifest.json"
    manifest_path = data_root / "training_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha, config_sha = sha256(manifest_path), sha256(args.config.resolve())
    if manifest.get("flow_unit") != "pixels per 30ms endpoint interval":
        raise RuntimeError("diagnostic data manifest flow unit differs from px/30ms")
    entry_index = {(row["generator"], row["rollout_id"]): row for row in manifest["entries"]}
    validation_ids = sorted({rid for arm, rid in entry_index if arm == "eventgenesis"
                             and next(row for row in manifest["entries"] if row["generator"] == arm and row["rollout_id"] == rid)["split"] == "validation"})
    if len(validation_ids) != 8 or any(not any(rid.startswith(prefix + "_") for prefix in STRATA) for rid in validation_ids):
        raise RuntimeError(f"expected fixed validation rollouts across four strata, got {validation_ids}")
    # One fixed dt1_000 window from each of the eight registered validation rollouts.
    rows = []
    for rollout_id in validation_ids:
        pair = {}
        for arm in ("eventgenesis", "v2e"):
            registered = entry_index[(arm, rollout_id)]
            path = Path(registered["path"])
            if not path.is_file() or sha256(path) != registered["sha256"]:
                raise RuntimeError(f"registered synthetic adapter changed: {path}")
            pair[arm] = load_npz_selected(path, 0)
        eg, v2 = pair["eventgenesis"], pair["v2e"]
        for arm, sample in pair.items():
            if sample["flow_gt"].shape != (2, 260, 346) or not np.isfinite(sample["flow_gt"]).all():
                raise RuntimeError(f"{arm} flow GT must be finite [u,v,260,346]")
            if sample["flow_valid"].shape != (260, 346) or not np.isin(sample["flow_valid"], (0, 1)).all():
                raise RuntimeError(f"{arm} flow validity must be binary [260,346]")
        if not (np.array_equal(eg["flow_gt"], v2["flow_gt"]) and np.array_equal(eg["flow_valid"], v2["flow_valid"])
                and np.array_equal(eg["sample_ids"], v2["sample_ids"]) and np.array_equal(eg["label_sha256"], v2["label_sha256"])):
            raise RuntimeError(f"paired GT, validity, ID, or label hash mismatch for {rollout_id}")
        if str(eg["sample_ids"]) != f"{rollout_id}/dt1_000":
            raise RuntimeError(f"fixed first window ID changed for {rollout_id}: {eg['sample_ids']}")
        normal_active = {arm: np.any(pair[arm]["events"] != 0, axis=(0, 3)) for arm in pair}
        union = normal_active["eventgenesis"] | normal_active["v2e"]
        valid = eg["flow_valid"].astype(bool)
        masks = {"full_valid": valid, "frozen_normal_event_active_union": valid & union}
        variant_inputs = {}
        for arm in ("eventgenesis", "v2e"):
            raw = eg["events"] if arm == "eventgenesis" else v2["events"]
            if raw.shape != (2, 260, 346, 5):
                raise RuntimeError(f"unexpected raw time-bin representation: {raw.shape}")
            reversed_bins = raw[..., ::-1].copy()
            variant_inputs[arm] = {"normal": raw, "all_zero_raw_events": np.zeros_like(raw),
                                   "reversed_five_time_bins": reversed_bins}
            if not np.array_equal(raw.sum(axis=-1), reversed_bins.sum(axis=-1)):
                raise RuntimeError("time reversal did not preserve per-polarity pixel event counts")
            if not np.array_equal(np.any(raw != 0, axis=-1), np.any(reversed_bins != 0, axis=-1)):
                raise RuntimeError("time reversal changed spatial event support")
        row = {"sample_id": str(eg["sample_ids"]), "rollout_id": rollout_id,
               "stratum": STRATA[rollout_id[:2]], "event_start_s": float(eg["event_start_s"]),
               "event_end_s": float(eg["event_end_s"]), "flow_unit": FLOW_UNIT,
               "valid_pixels": int(valid.sum()), "frozen_active_union_pixels": int((valid & union).sum()),
               "event_counts": {arm: int(pair[arm]["events"].sum(dtype=np.float64)) for arm in pair},
               "variants": {}}
        zero_epe = epe(np.zeros_like(eg["flow_gt"]), eg["flow_gt"], valid)
        row["zero_flow_epe_full_valid"] = zero_epe
        row["zero_flow_epe_active_union"] = epe(np.zeros_like(eg["flow_gt"]), eg["flow_gt"], valid & union)
        for arm in ("eventgenesis", "v2e"):
            row["variants"][arm] = {}
            for variant, raw in variant_inputs[arm].items():
                row["variants"][arm][variant] = {
                    "raw_event_tensor_sha256": hashlib.sha256(np.ascontiguousarray(raw).view(np.uint8)).hexdigest(),
                    "total_event_count": int(raw.sum(dtype=np.float64)),
                    "model_input_changed_from_normal": bool(not np.array_equal(raw, variant_inputs[arm]["normal"])),
                }
        row["_pair"] = pair
        row["_variant_inputs"] = variant_inputs
        row["_masks"] = masks
        rows.append(row)

    result_models = {}
    initial_hashes = set()
    for arm, checkpoint in (("eventgenesis", args.eventgenesis_checkpoint.resolve()), ("v2e", args.v2e_checkpoint.resolve())):
        model, source_patch, checkpoint_meta = load_checkpoint_model(checkpoint, arm, model_root, manifest_sha, config_sha, device)
        initial_hashes.add(checkpoint_meta.get("initial_state_sha256"))
        result_models[arm] = {"checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint), **checkpoint_meta},
                              "source_patch": source_patch, "samples": []}
        for row in rows:
            pair, variants, masks = row["_pair"], row["_variant_inputs"], row["_masks"]
            gt = pair[arm]["flow_gt"].astype(np.float32, copy=False)
            baseline = predict(model, variants[arm]["normal"], device)
            sample_result = {"sample_id": row["sample_id"], "regions": {}}
            for region, mask in masks.items():
                sample_result["regions"][region] = {
                    "normal_epe_px_per_30ms": epe(baseline, gt, mask),
                    "normal_prediction": summarize_field(baseline, mask),
                    "zero_flow_epe_px_per_30ms": epe(np.zeros_like(gt), gt, mask),
                }
            for variant in ("all_zero_raw_events", "reversed_five_time_bins"):
                changed = predict(model, variants[arm][variant], device)
                for region, mask in masks.items():
                    sample_result["regions"][region][variant] = {
                        "epe_px_per_30ms": epe(changed, gt, mask),
                        "prediction": summarize_field(changed, mask),
                        "mean_absolute_output_change_from_normal": float(np.abs(changed[:, mask] - baseline[:, mask]).mean()) if mask.any() else None,
                        "output_vector_sha256": hashlib.sha256(np.ascontiguousarray(changed).view(np.uint8)).hexdigest(),
                    }
            result_models[arm]["samples"].append(sample_result)
        del model
        torch.cuda.empty_cache()
    if len(initial_hashes) != 1:
        raise RuntimeError(f"synthetic-best checkpoint initialization hashes differ: {sorted(initial_hashes)}")

    for row in rows:
        del row["_pair"], row["_variant_inputs"], row["_masks"]
    out = {"schema": "stage5_flow_input_perturbation_v1", "status": "completed",
           "started_utc": started_utc, "device": str(device),
           "model_source": source, "data_root": str(data_root), "data_manifest_sha256": manifest_sha,
           "training_config_sha256": config_sha,
           "fixed_window_rule": "dt1_000 from all eight registered validation rollouts; two per each of four fixed strata",
           "input_variants": {"normal": "stored 5-bin event tensor",
                              "all_zero_raw_events": "raw stored event volume set to zero before padding/model encoding",
                              "reversed_five_time_bins": "reverse time-bin order for each polarity before padding; preserve per-pixel counts and support"},
           "mask_rule": "same valid mask and same frozen union of normal EventGenesis/V2E active pixels for every variant",
           "interpretation_limit": "input sensitivity is not evidence that the model decoded correct motion; fixed 8-window synthetic diagnosis only",
           "frozen_masks_and_inputs": rows, "models": result_models,
           "real_data_used": False, "real_scores_authorized": False}
    out["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": out["status"], "samples": len(rows), "output": str(args.output),
                      "real_data_used": False, "real_scores_authorized": False}, indent=2))


if __name__ == "__main__":
    main()
