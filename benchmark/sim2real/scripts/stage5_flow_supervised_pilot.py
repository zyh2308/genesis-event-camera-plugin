#!/usr/bin/env python3
"""Small paired supervised STE-FlowNet diagnostic on shared synthetic GT.

This is not a replacement for the registered photometric/smoothness protocol.
It uses four fixed training rollouts and eight held-out rollouts, identical
seed/sampling/update budget, and the pinned model's four output scales. At each
scale, GT is area-consistent downsampled, vector magnitudes are scaled by the
x/y resolution ratios, and the validity support is conservatively downsampled.
The objective is the standard robust endpoint error at the official four
STE-FlowNet scales with the source recipe's equal scale weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
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
ORIGINAL_SCALE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
FLOW_UNIT = "px_per_30ms_interval"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_sample(path: Path, index: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        result = {}
        for key in ("events", "flow_gt", "flow_valid", "sample_ids", "label_sha256"):
            value = np.asarray(z[key])
            # Rollout-level label hashes are scalar metadata, while the data
            # and sample identifiers carry a leading window dimension.
            result[key] = value.copy() if value.ndim == 0 else np.asarray(value[index]).copy()
        return result


def model_source(model_root: Path, device: torch.device):
    sys.path.insert(0, str(model_root))
    import models
    import models.corr as corr
    import multiscaleloss
    corr.spatial_correlation_sample = pure_torch_spatial_correlation
    patch = patch_ste(models, multiscaleloss, model_root)
    return models.steflow().to(device), patch


def pad_sample(sample: dict[str, np.ndarray], device: torch.device):
    events = torch.from_numpy(sample["events"].astype(np.float32, copy=False)).unsqueeze(0)
    events = F.pad(events, (0, 0, 3, 3, 46, 46)).to(device)
    flow = torch.from_numpy(sample["flow_gt"].astype(np.float32, copy=False)).unsqueeze(0)
    flow = F.pad(flow, (3, 3, 46, 46)).to(device)
    valid = torch.from_numpy(sample["flow_valid"].astype(np.float32, copy=False)).view(1, 1, 260, 346)
    valid = F.pad(valid, (3, 3, 46, 46)).to(device)
    return events, flow, valid


def multiscale_supervised_epe(outputs, gt_full, valid_full) -> tuple[torch.Tensor, list[float]]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
        raise RuntimeError("training-mode STE-FlowNet must return exactly four flow scales")
    weighted, scalars, weight_sum = None, [], 0.0
    target_h, target_w = gt_full.shape[-2:]
    finite_pixels = torch.isfinite(gt_full).all(dim=1, keepdim=True)
    valid_full = ((valid_full > 0.5) & finite_pixels).to(dtype=gt_full.dtype)
    gt_full = torch.where(finite_pixels.expand_as(gt_full), gt_full, torch.zeros_like(gt_full))
    for scale_index, (prediction, weight) in enumerate(zip(outputs, ORIGINAL_SCALE_WEIGHTS)):
        if prediction.ndim != 4 or prediction.shape[1] != 2:
            raise ValueError(f"scale {scale_index} flow must be [B,2,H,W], got {tuple(prediction.shape)}")
        h, w = prediction.shape[-2:]
        coverage = F.interpolate(valid_full, size=(h, w), mode="area")
        numerator = F.interpolate(gt_full * valid_full, size=(h, w), mode="bilinear", align_corners=False)
        target = numerator / coverage.clamp_min(1e-6)
        target = target.clone()
        target[:, 0] *= w / target_w
        target[:, 1] *= h / target_h
        mask = coverage[:, 0] >= 0.999
        if not bool(mask.any()):
            raise RuntimeError(f"scale {scale_index} has no fully valid GT support")
        if prediction.shape != target.shape or prediction.shape[0] != valid_full.shape[0]:
            raise ValueError(f"scale {scale_index} target shape {tuple(target.shape)} != prediction {tuple(prediction.shape)}")
        delta = prediction - target
        endpoint = torch.sqrt(delta.square().sum(dim=1) + 1e-6)
        loss = endpoint[mask].mean()
        weighted = loss * weight if weighted is None else weighted + loss * weight
        scalars.append(float(loss.detach().cpu()))
        weight_sum += weight
    return weighted / weight_sum, scalars


def predict_eval(model, events, device) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        full = model(events, 352, use_dt4=False, flow_seq=False)
    out = model_output_to_flow_hw(full)[:, 46:306, 3:349]
    if out.shape != (2, 260, 346) or not np.isfinite(out).all():
        raise RuntimeError("STE evaluation flow must be finite [2,260,346]")
    return out


def verify_train_eval_output_parity(model, sample: dict[str, np.ndarray], device: torch.device) -> dict:
    """Check that mode-dependent return structure does not change the full flow."""
    events, _, _ = pad_sample(sample, device)
    model.train()
    with torch.inference_mode():
        train_outputs = model(events, 352, use_dt4=False, flow_seq=False)
    if not isinstance(train_outputs, (tuple, list)) or len(train_outputs) != 4:
        raise RuntimeError("training-mode STE-FlowNet must return four scales for the mode-parity check")
    train_flow = model_output_to_flow_hw(train_outputs[0])[:, 46:306, 3:349]
    eval_flow = predict_eval(model, events, device)
    max_abs = float(np.max(np.abs(train_flow - eval_flow)))
    if not np.allclose(train_flow, eval_flow, atol=1e-6, rtol=0):
        raise RuntimeError(f"STE train/eval full-flow output differs for a fixed input: max_abs={max_abs}")
    return {"sample_id": str(sample["sample_ids"]), "max_abs_difference": max_abs,
            "flow_sha256": hashlib.sha256(np.ascontiguousarray(eval_flow).view(np.uint8)).hexdigest(),
            "status": "pass"}


def epe_pair(pred, gt, mask):
    count = int(mask.sum())
    if not count:
        return None
    return float(np.hypot(pred[0, mask] - gt[0, mask], pred[1, mask] - gt[1, mask]).mean(dtype=np.float64))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--updates", type=int, default=80)
    p.add_argument("--train-windows-per-rollout", type=int, default=16,
                   help="fixed evenly spaced samples per training rollout; default reproduces the 64-window pilot")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.updates < 1:
        raise ValueError("--updates must be positive")
    if not 1 <= args.train_windows_per_rollout <= 64:
        raise ValueError("--train-windows-per-rollout must be within [1,64]")
    out_root = args.output_root.resolve()
    if out_root.exists():
        raise FileExistsError(f"refusing to overwrite pilot output: {out_root}")
    out_root.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("pinned STE-FlowNet pilot requires CUDA")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model_root, data_root = args.model_root.resolve(), args.data_root.resolve()
    source = verify_pinned_source(model_root, "STE-FlowNet", SOURCE_COMMIT,
                                  ROOT / "benchmark/sim2real/manifests/downstream_source_bundle.json")
    manifest_path = data_root / "training_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha, config_sha = sha256(manifest_path), sha256(args.config.resolve())
    if manifest.get("flow_unit") != "pixels per 30ms endpoint interval":
        raise RuntimeError("the GT pilot expects shared physical pixels-per-30ms flow labels")
    rows = manifest["entries"]
    by_arm = {(row["generator"], row["rollout_id"]): row for row in rows}
    train_rollouts = [f"s{i}_00" for i in range(4)]
    validation_rollouts = sorted(rid for arm, rid in by_arm if arm == "eventgenesis" and by_arm[(arm, rid)]["split"] == "validation")
    if len(validation_rollouts) != 8:
        raise RuntimeError(f"expected eight independent validation rollouts, got {validation_rollouts}")
    for rid in train_rollouts:
        if any(by_arm[(arm, rid)]["split"] != "train" for arm in ("eventgenesis", "v2e")):
            raise RuntimeError(f"registered training rollout split mismatch: {rid}")

    train_rows = []
    for rid in train_rollouts:
        eg_row, v2_row = by_arm[("eventgenesis", rid)], by_arm[("v2e", rid)]
        for row in (eg_row, v2_row):
            path = Path(row["path"])
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise RuntimeError(f"registered data artifact changed: {path}")
        # Place each selected window at the center of an equal-sized stratum.
        # In particular, a two-window overfit check uses indices 16 and 48,
        # avoiding rollout endpoints while spanning the fixed interval.
        sample_indices = np.floor(
            (np.arange(args.train_windows_per_rollout, dtype=np.float64) + 0.5)
            * 64 / args.train_windows_per_rollout
        ).astype(int).tolist()
        for sample_index in sample_indices:
            eg = load_sample(Path(eg_row["path"]), sample_index)
            v2 = load_sample(Path(v2_row["path"]), sample_index)
            if not (np.array_equal(eg["flow_gt"], v2["flow_gt"]) and np.array_equal(eg["flow_valid"], v2["flow_valid"])
                    and np.array_equal(eg["sample_ids"], v2["sample_ids"]) and np.array_equal(eg["label_sha256"], v2["label_sha256"])):
                raise RuntimeError(f"paired shared labels differ for {rid} sample {sample_index}")
            train_rows.append({"rollout_id": rid, "sample_index": sample_index,
                               "sample_id": str(eg["sample_ids"]),
                               "arms": {"eventgenesis": eg, "v2e": v2}})
    expected_train_samples = len(train_rollouts) * args.train_windows_per_rollout
    if len(train_rows) != expected_train_samples:
        raise RuntimeError(f"the fixed GT pilot expected {expected_train_samples} training samples, got {len(train_rows)}")

    val_rows = []
    for rid in validation_rollouts:
        samples = {}
        for arm in ("eventgenesis", "v2e"):
            row = by_arm[(arm, rid)]
            path = Path(row["path"])
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise RuntimeError(f"registered validation artifact changed: {path}")
            samples[arm] = load_sample(path, 0)
        if not (np.array_equal(samples["eventgenesis"]["flow_gt"], samples["v2e"]["flow_gt"])
                and np.array_equal(samples["eventgenesis"]["flow_valid"], samples["v2e"]["flow_valid"])):
            raise RuntimeError(f"paired validation labels differ for {rid}")
        val_rows.append({"rollout_id": rid, "sample_id": str(samples["eventgenesis"]["sample_ids"]), "arms": samples})

    result = {"schema": "stage5_steflow_synthetic_gt_supervised_pilot_v2", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "source": source,
              "data_manifest_sha256": manifest_sha, "config_sha256": config_sha,
              "objective": "equal-weight four-scale robust endpoint error from the pinned STE-FlowNet Charbonnier family; target vector resized per scale; validity coverage >=0.999",
              "training_rollouts": train_rollouts, "training_samples": len(train_rows),
              "train_windows_per_rollout": args.train_windows_per_rollout,
              "fixed_training_sample_indices_per_rollout": sample_indices,
              "validation_rollouts": validation_rollouts, "validation_windows_per_rollout": 1,
              "updates_per_arm": args.updates, "batch_size": 1, "real_data_used": False,
              "flow_gt_used_for_optimizer": True, "protocol_status": "separate diagnostic; no formal protocol or checkpoint selection change", "arms": {}}
    sampler = np.random.default_rng(20261019).permutation(len(train_rows)).tolist()
    for arm in ("eventgenesis", "v2e"):
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        model, source_patch = model_source(model_root, device)
        initial_hash = state_sha256(model)
        if "paired_initial_hash" in result and initial_hash != result["paired_initial_hash"]:
            raise RuntimeError("paired GT pilot initial states differ")
        result["paired_initial_hash"] = initial_hash
        optimizer = torch.optim.Adam([
            {"params": model.bias_parameters(), "weight_decay": 0.0},
            {"params": model.weight_parameters(), "weight_decay": .0004},
        ], lr=.0004, betas=(.9, .999))
        model.train()
        metrics_path = out_root / f"{arm}_updates.jsonl"
        started = time.monotonic()
        with metrics_path.open("w") as log:
            for update in range(args.updates):
                sample_row = train_rows[sampler[update % len(sampler)]]
                sample = sample_row["arms"][arm]
                events, gt, valid = pad_sample(sample, device)
                outputs = model(events, 352, use_dt4=False, flow_seq=False)
                loss, scale_losses = multiscale_supervised_epe(outputs, gt, valid)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{arm} update {update+1}: non-finite supervised loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                    raise RuntimeError(f"{arm} update {update+1}: invalid gradients")
                grad_norm = float(torch.sqrt(sum(g.detach().float().square().sum() for g in gradients)).cpu())
                if not np.isfinite(grad_norm) or grad_norm <= 0:
                    raise RuntimeError(f"{arm} update {update+1}: zero gradient")
                optimizer.step()
                record = {"update": update+1, "sample_id": sample_row["sample_id"],
                          "rollout_id": sample_row["rollout_id"], "loss": float(loss.detach().cpu()),
                          "scale_endpoint_losses": scale_losses, "gradient_l2_norm": grad_norm,
                          "finite_gradient_tensors": len(gradients), "time_utc": datetime.now(timezone.utc).isoformat()}
                log.write(json.dumps(record, sort_keys=True)+"\n"); log.flush()
        model.eval()
        parity_train = verify_train_eval_output_parity(model, train_rows[0]["arms"][arm], device)
        parity_val = verify_train_eval_output_parity(model, val_rows[0]["arms"][arm], device)
        def eval_rows(source_rows):
            details=[]
            with torch.inference_mode():
                for row in source_rows:
                    sample=row["arms"][arm]
                    events,gt,valid=pad_sample(sample,device)
                    pred=model_output_to_flow_hw(model(events,352,use_dt4=False,flow_seq=False))[:,46:306,3:349]
                    gt_np=sample["flow_gt"].astype(np.float32,copy=False)
                    mask=sample["flow_valid"].astype(bool,copy=False)
                    details.append({"rollout_id":row["rollout_id"],"sample_id":row["sample_id"],
                                    "epe_px_per_30ms":epe_pair(pred,gt_np,mask),"zero_flow_epe_px_per_30ms":epe_pair(np.zeros_like(gt_np),gt_np,mask),
                                    "valid_pixels":int(mask.sum()),"prediction_mean_magnitude":float(np.hypot(pred[0],pred[1]).mean())})
            return details
        train_eval=eval_rows(train_rows)
        validation=eval_rows(val_rows)
        checkpoint=out_root/f"{arm}_last.pt"
        state={"state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"generator":arm,"seed":0,
               "updates":args.updates,"initial_state_sha256":initial_hash,"state_sha256":state_sha256(model),
               "data_manifest_sha256":manifest_sha,"config_sha256":config_sha,"flow_unit":FLOW_UNIT,
               "real_data_used":False,"validation_used_for_selection":False}
        torch.save(state,checkpoint)
        restored,_=model_source(model_root,device)
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        restored.load_state_dict(saved["state_dict"],strict=True); restored.eval()
        restored_hash=state_sha256(restored)
        if restored_hash!=state["state_sha256"]:
            raise RuntimeError(f"{arm}: restore weight hash mismatch")
        restored_eval=[]
        with torch.inference_mode():
            for row in val_rows:
                sample=row["arms"][arm]; events,_,_=pad_sample(sample,device)
                pred=model_output_to_flow_hw(restored(events,352,use_dt4=False,flow_seq=False))[:,46:306,3:349]
                restored_eval.append(hashlib.sha256(np.ascontiguousarray(pred).view(np.uint8)).hexdigest())
        model_eval_hashes=[]
        with torch.inference_mode():
            for row in val_rows:
                sample=row["arms"][arm]; events,_,_=pad_sample(sample,device)
                pred=model_output_to_flow_hw(model(events,352,use_dt4=False,flow_seq=False))[:,46:306,3:349]
                model_eval_hashes.append(hashlib.sha256(np.ascontiguousarray(pred).view(np.uint8)).hexdigest())
        if restored_eval!=model_eval_hashes:
            raise RuntimeError(f"{arm}: ordinary eval outputs changed after checkpoint restore")
        result["arms"][arm]={"initial_state_sha256":initial_hash,"final_state_sha256":state["state_sha256"],
            "updates":args.updates,"checkpoint":{"path":str(checkpoint),"size_bytes":checkpoint.stat().st_size,"sha256":sha256(checkpoint),"restore_hash_equal":True,"ordinary_eval_outputs_equal":True},
            "source_patch":source_patch,"train_subset_mean_epe_px_per_30ms":float(np.mean([r["epe_px_per_30ms"] for r in train_eval])),
            "independent_validation_mean_epe_px_per_30ms":float(np.mean([r["epe_px_per_30ms"] for r in validation])),
            "independent_validation_zero_flow_mean_epe_px_per_30ms":float(np.mean([r["zero_flow_epe_px_per_30ms"] for r in validation])),
            "train_eval_output_parity":{"train":parity_train,"validation":parity_val},
            "train_subset":train_eval,"independent_validation":validation,
            "wall_seconds":time.monotonic()-started}
        del model,restored,optimizer
        torch.cuda.empty_cache()
    result["status"]="complete"
    result["finished_utc"]=datetime.now(timezone.utc).isoformat()
    output_path=out_root/"summary.json"
    output_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":result["status"],"output":str(output_path),"updates_per_arm":args.updates,
                      "train_subset_epe":{a:result["arms"][a]["train_subset_mean_epe_px_per_30ms"] for a in result["arms"]},
                      "independent_val_epe":{a:result["arms"][a]["independent_validation_mean_epe_px_per_30ms"] for a in result["arms"]}},indent=2))

if __name__=="__main__":
    main()
