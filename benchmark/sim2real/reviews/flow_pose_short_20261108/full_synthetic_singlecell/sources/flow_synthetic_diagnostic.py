"""Read-only synthetic flow diagnostics: bank CPU audit or bounded GPU probe.

No optimizer is constructed. No real file, real prediction or real score is
read. Frozen source/data/checkpoints are only opened for reading. All outputs
must go to a new directory; existing training/evaluation roots are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_module(path, name):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def validate_contract(c):
    R = Path(c["campaign"])
    require(c["schema"] == "flow_priority_synthetic_diagnostic_v1", "Unknown diagnostic contract")
    require(c["no_optimizer_updates"] is True and c["no_real_data"] is True
            and c["real_scores_used"] is False, "Synthetic-only read-only scope required")
    require(c["methods"] == ["eventgenesis", "v2e"] and c["seed"] == 0, "Fixed paired methods/seed required")
    require(sha(R / "sources/training_recipe.json") == c["recipe_sha256"], "Recipe hash differs")
    recipe = json.loads((R / "sources/training_recipe.json").read_text())
    require(recipe["frozen_simulator_core"] == c["frozen_simulator_core"], "Frozen core changed")
    require(recipe["flow_dt1"]["expected_optimizer_updates"] == 8910, "Formal training budget differs")
    for method in c["methods"]:
        b = c["methods_binding"][method]
        require(b["method"] == method and b["seed"] == 0 and b["completed_updates"] == 8910,
                "Fixed completed synthetic checkpoint required")
        require(sha(b["data_manifest"]) == b["data_manifest_sha256"], "Data manifest changed")
    return R


def entries(c, split):
    lookup = {}
    for method in c["methods"]:
        b = c["methods_binding"][method]
        manifest = json.loads(Path(b["data_manifest"]).read_text())
        rows = sorted((r for r in manifest["entries"] if r["split"] == split), key=lambda r: r["rollout_id"])
        require(len(rows) == (24 if split == "train" else 8), "Wrong whole-rollout split")
        lookup[method] = {r["rollout_id"]: r for r in rows}
    require(set(lookup["eventgenesis"]) == set(lookup["v2e"]), "Paired rollout IDs differ")
    return lookup


def load_episode(row, verify_hash=True):
    path = Path(row["path"])
    require(path.name == row["rollout_id"] + ".npz", "Unexpected rollout path")
    if verify_hash:
        require(sha(path) == row["sha256"], "Synthetic episode file changed: " + str(path))
    with np.load(path, allow_pickle=False) as f:
        data = {k: np.asarray(f[k]) for k in f.files}
    require(data["events"].shape == (66, 2, 260, 346, 5), "Wrong event raster/interval count")
    require(data["flow_gt"].shape == (66, 2, 260, 346) and data["flow_valid"].shape == (66, 260, 346), "Wrong label layout")
    require(np.isfinite(data["flow_gt"]).all() and np.isin(data["flow_valid"], [0, 1]).all(), "Invalid synthetic labels")
    require(np.allclose(data["event_end_s"] - data["event_start_s"], .03, atol=1e-8, rtol=0), "Not fixed30ms endpoints")
    return data


def pair_equal(a, b):
    for key in ("flow_gt", "flow_valid", "gray_prev", "gray_next", "sample_ids", "event_start_s", "event_end_s"):
        require(np.array_equal(a[key], b[key]), "Generator-specific physical labels/gray/time: " + key)


def active(events):
    return np.any(events > 0, axis=(0, 3))


def regions(gt, valid, common_active):
    magnitude = np.sqrt(np.square(gt.astype(np.float64)).sum(axis=0))
    valid = valid.astype(bool) & np.isfinite(gt).all(axis=0)
    result = {"all_physical_valid": valid, "common_event_active_valid": valid & common_active,
              "event_inactive_valid": valid & ~common_active,
              "invalid_geometry_or_visibility": ~valid}
    edges = [0, .1, .5, 1, 2, 5, np.inf]
    for lo, hi in zip(edges, edges[1:]):
        result[f"motion_{lo:g}_{hi:g}px"] = valid & (magnitude >= lo) & (magnitude < hi)
        result[f"active_motion_{lo:g}_{hi:g}px"] = result[f"motion_{lo:g}_{hi:g}px"] & common_active
    return magnitude, result


def region_summary(prediction, gt, valid, common_active):
    magnitude, masks = regions(gt, valid, common_active)
    error = None if prediction is None else np.sqrt(np.square(prediction.astype(np.float64) - gt).sum(axis=0))
    predmag = None if prediction is None else np.sqrt(np.square(prediction.astype(np.float64)).sum(axis=0))
    result = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        if name == "invalid_geometry_or_visibility":
            result[name] = dict(pixels=n, note="Not identified as pure occlusion; invalid GT is zeroed by original adapter")
            continue
        row = dict(pixels=n, fraction_of_sensor=n / valid.size,
                   gt_mean_motion_px=float(magnitude[mask].mean()) if n else None)
        if prediction is not None:
            row.update(epe_px=float(error[mask].mean()) if n else None,
                       predicted_mean_motion_px=float(predmag[mask].mean()) if n else None,
                       prediction_below0p1_fraction=float((predmag[mask] < .1).mean()) if n else None)
        result[name] = row
    return result


def bank_audit(c, out, deadline):
    rows = []
    for split in ("train", "validation"):
        lookup = entries(c, split)
        for rid in sorted(lookup["eventgenesis"]):
            require(time.monotonic() < deadline, "Bounded CPU audit deadline exceeded")
            a, b = (load_episode(lookup[m][rid]) for m in c["methods"])
            pair_equal(a, b)
            windows = []
            for i in range(66):
                aa, bb = active(a["events"][i]), active(b["events"][i])
                windows.append(dict(index=i, sample_id=str(a["sample_ids"][i]),
                    eventgenesis_active_fraction=float(aa.mean()), v2e_active_fraction=float(bb.mean()),
                    common_active_fraction=float((aa & bb).mean()), union_active_fraction=float((aa | bb).mean()),
                    metrics=region_summary(None, a["flow_gt"][i], a["flow_valid"][i], aa & bb)))
            rows.append(dict(split=split, rollout=rid, stratum=lookup["eventgenesis"][rid]["stratum"], windows=windows))
            print("SYNTHETIC_BANK_AUDITED", split, rid, flush=True)
    write(out / "bank_motion_coverage.json", dict(status="complete", rows=rows,
        train_windows=1584, validation_windows=528, GT_role="diagnostics only; no optimizer", real_data_used=False,
        note="Pooled/window distributions are diagnostic only; unchanged formal validation remains rollout equal"))


def norm(tensors):
    return float(sum(t.detach().double().square().sum() for t in tensors).sqrt())


def cosine(a, b):
    na, nb = norm(a), norm(b)
    return float(sum((x.detach().double() * y.detach().double()).sum() for x, y in zip(a, b)) / (na * nb)) if na and nb else None


def gradient_probe(trainer, loss_module, outputs, gray, event, flow, valid, common):
    import torch
    import torch.nn.functional as F
    fields = tuple(value.detach().requires_grad_(True) for value in outputs)
    require(all(torch.isfinite(value).all() for value in fields), "Nonfinite model prediction")
    total, photo, smooth = trainer.objective_dt1(fields, gray[:, 0:1], gray[:, 1:2], event, loss_module)
    require(all(torch.isfinite(value) for value in (total, photo, smooth)), "Nonfinite frozen objective")
    gp = torch.autograd.grad(photo, fields, retain_graph=True)
    gs = torch.autograd.grad(10 * smooth, fields, retain_graph=True)
    require(all(torch.isfinite(value).all() for value in (*gp, *gs)), "Nonfinite diagnostic gradient")
    result = dict(photo_sum_multiscale=float(photo.detach()), smooth_mean_multiscale=float(smooth.detach()),
        objective=float(total.detach()), photo_gradient_l2=norm(gp), weighted_smooth_gradient_l2=norm(gs),
        smooth_to_photo_gradient_ratio=norm(gs) / max(norm(gp), 1e-30),
        photo_smooth_gradient_cosine=cosine(gp, gs), gradient_space="detached flow outputs, not full network parameters",
        GT_used_in_optimizer=False, per_scale=[])
    for index, (field, pgrad, sgrad) in enumerate(zip(fields, gp, gs)):
        h, w = field.shape[-2:]
        target = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        target = target * field.new_tensor([w / 256, h / 256]).view(1, 2, 1, 1)
        mask = F.interpolate(valid.float(), size=(h, w), mode="nearest")[:, 0] > .5
        support = F.adaptive_max_pool2d(common.float(), (h, w))[:, 0] > 0
        difference = (field - target).square().sum(dim=1).add(1e-12).sqrt()
        oracle = difference[mask].mean()
        ge = torch.autograd.grad(oracle, field, retain_graph=True)[0]
        energy = pgrad.square().sum(dim=1)
        denom = float(energy.sum())
        result["per_scale"].append(dict(index=index, shape=list(field.shape),
            photo_l2=norm([pgrad]), weighted_smooth_l2=norm([sgrad]),
            photo_vs_GT_EPE_gradient_cosine=cosine([pgrad], [ge]),
            smooth_vs_GT_EPE_gradient_cosine=cosine([sgrad], [ge]),
            photo_gradient_energy_common_active_valid=float(energy[mask & support].sum()) / denom if denom else None,
            photo_gradient_energy_invalid_geometry_or_visibility=float(energy[~mask].sum()) / denom if denom else None))
    # Exact finest-scale fixed-coordinate warp, only a diagnostic oracle.
    cases = {"zero": torch.zeros_like(flow), "prediction": fields[0].detach(), "GT_projection_diagnostic_only": flow}
    result["finest_photometric_oracle"] = {}
    for name, field in cases.items():
        warped = loss_module.warp(gray[:, 1:2], field)
        residual = (warped - gray[:, 0:1]).square().add(1e-6).pow(.45)[:, 0]
        mm = valid[:, 0] > .5
        aa = mm & (common[:, 0] > 0)
        result["finest_photometric_oracle"][name] = dict(
            physical_valid_mean=float(residual[mm].mean()) if mm.any() else None,
            common_active_valid_mean=float(residual[aa].mean()) if aa.any() else None)
    return result


def gpu_probe(c, out, deadline):
    import torch
    import torch.nn as nn
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == c["GPU_UUID"] and torch.cuda.device_count() == 1,
            "GPU3 fixed UUID must be exclusively exposed by existing queue; never useGPU2/5")
    require(sha(c["training_source"]) == c["training_source_sha256"], "Frozen training source differs")
    require(sha(c["compat_source"]) == c["compat_sha256"], "Frozen warp source differs")
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True;torch.backends.cudnn.benchmark = False
    trainer = load_module(c["training_source"], "flow_diagnostic_frozen_trainer")
    sys.path.insert(0, c["model_root"])
    import models, models.corr, multiscaleloss
    models.corr.spatial_correlation_sample = trainer.pure_torch_spatial_correlation
    trainer.patch_ste(models, multiscaleloss, Path(c["model_root"]))
    device = torch.device("cuda:0")
    val = entries(c, "validation"); train = entries(c, "train")
    results = []
    for method in c["methods"]:
        require(time.monotonic() < deadline, "Bounded GPU diagnostic deadline exceeded")
        binding = c["methods_binding"][method]
        require(sha(binding["selected_checkpoint"]) == binding["selected_checkpoint_sha256"], "Fixed synthetic checkpoint changed")
        checkpoint = torch.load(binding["selected_checkpoint"], map_location="cpu", weights_only=False)
        require(checkpoint["generator"] == method and checkpoint["seed"] == 0
                and checkpoint["config_sha256"] == c["recipe_sha256"]
                and checkpoint["data_manifest_sha256"] == binding["data_manifest_sha256"], "CP identity/config/data mismatch")
        model = models.steflow().to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint
        require(not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()),
                "Unexpected BN state; do not mutate it in a read-only probe")
        before = trainer.state_sha256(model)
        validation = []
        model.eval()
        for rid in sorted(val[method]):
            require(time.monotonic() < deadline, "Bounded GPU diagnostic deadline exceeded")
            a, b = (load_episode(val[m][rid]) for m in c["methods"])
            pair_equal(a, b)
            data = a if method == "eventgenesis" else b
            for i in c["validation_indices"]:
                require(time.monotonic() < deadline, "Bounded GPU diagnostic deadline exceeded")
                with torch.inference_mode():
                    event, _, _ = trainer.val_input(data, i, device)
                    predicted = trainer.model_output_to_flow_hw(model(event, 352, use_dt4=False, flow_seq=False))[:, 46:306, 3:349]
                validation.append(dict(rollout=rid, index=i, sample_id=str(data["sample_ids"][i]),
                    metrics=region_summary(predicted, data["flow_gt"][i], data["flow_valid"][i],
                                           active(a["events"][i]) & active(b["events"][i]))))
            print("SYNTHETIC_PREDICTIONS_AUDITED", method, rid, flush=True)
        # One shared8-example augmented TRAIN batch: two fixed windows per stratum.
        events=[];grays=[];flows=[];valids=[];commons=[];sample_ids=[];sample_labels=[]
        for stratum in range(4):
            rid=sorted(rid for rid in train[method] if rid.startswith(f"s{stratum}_"))[0]
            a,b=(load_episode(train[m][rid]) for m in c["methods"]);pair_equal(a,b)
            for i in c["training_sample_indices"]:
                transformed=[]
                for data in (a,b):
                    sample={"events":data["events"][i:i+1],"gray_prev":data["gray_prev"][i],"gray_next":data["gray_next"][i],
                            "flow_gt":data["flow_gt"][i],"flow_valid":data["flow_valid"][i]}
                    rng=np.random.default_rng(c["augmentation_seed"]+stratum*1000+i)
                    transformed.append(trainer.apply_joint_transform(sample,rng,device))
                require(torch.equal(transformed[0][1],transformed[1][1]) and torch.equal(transformed[0][2],transformed[1][2])
                        and torch.equal(transformed[0][3],transformed[1][3]),"Augmented physical labels/gray differ")
                item=transformed[c["methods"].index(method)]
                events.append(item[0]);grays.append(item[1]);flows.append(item[2]);valids.append(item[3])
                commons.append((transformed[0][0].sum(dim=(1,4))>0)[:,None] & (transformed[1][0].sum(dim=(1,4))>0)[:,None])
                sample_ids.append(str(a["sample_ids"][i]));sample_labels.append(array_sha(item[2].cpu().numpy()))
        model.train()
        require(time.monotonic() < deadline, "Bounded GPU diagnostic deadline exceeded")
        event=torch.cat(events);gray=torch.cat(grays);flow=torch.cat(flows);valid=torch.cat(valids);common=torch.cat(commons)
        with torch.no_grad():
            output=model(event,256,use_dt4=False,flow_seq=False)
        require(isinstance(output,(tuple,list)) and len(output)==4,"Expected four formal training scales")
        gradient=gradient_probe(trainer,multiscaleloss,output,gray,event,flow,valid,common)
        require(trainer.state_sha256(model)==before,"Read-only diagnostic changed model state")
        require(sha(binding["selected_checkpoint"])==binding["selected_checkpoint_sha256"],"Source checkpoint changed during diagnostic")
        require(time.monotonic() < deadline, "Bounded GPU diagnostic deadline exceeded")
        results.append(dict(method=method,seed=0,checkpoint_sha256=binding["selected_checkpoint_sha256"],
            validation=validation,gradient_diagnostic=gradient,training_sample_ids=sample_ids,training_labels_sha256=sample_labels,
            model_state_unchanged=True,optimizer_updates=0))
        write(out/(method+"_probe.json"),results[-1])
        del model,event,gray,flow,valid,common,output
        torch.cuda.empty_cache()
    require(results[0]["training_sample_ids"]==results[1]["training_sample_ids"]
            and results[0]["training_labels_sha256"]==results[1]["training_labels_sha256"],"Paired probe labels differ")
    write(out/"completion.json",dict(status="complete",results=results,paired_labels_verified=True,
        synthetic_validation_windows_per_method=32,optimizer_updates=0,GT_flow_optimizer_supervision=False,
        real_data_used=False,real_scores_used=False,formal_weights_modified=False,
        phase="diagnostic only; no automatic paired shorttrain",GPU_UUID=c["GPU_UUID"]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("bank","probe"));p.add_argument("--contract",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();c=json.loads(a.contract.read_text());validate_contract(c)
    require(not a.out.exists(),"Refusing existing output directory")
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/"contract.json",c)
    deadline=time.monotonic()+c["deadline_seconds"]
    try:
        if a.mode=="bank":
            bank_audit(c,a.out,deadline)
            write(a.out/"completion.json",dict(status="complete",real_data_used=False,optimizer_updates=0,mode="CPU bank audit"))
        else:gpu_probe(c,a.out,deadline)
    except BaseException:
        write(a.out/"failure.json",dict(status="failed",error=traceback.format_exc(),real_data_used=False,optimizer_updates=0))
        raise


if __name__=="__main__":main()
