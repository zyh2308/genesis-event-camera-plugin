"""Zero-update synthetic module/gradient diagnostic; hybrids are not candidates."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

GPU4 = 'GPU-0602549d-5c0b-133e-63ab-a01d1f6a49d3'
COMPLETION_SHA = '6540c8e62b3606376641590fa267bf90111e889e279213ab4f6a8537c2cc2ac0'
ARMS = ('clean_control', 'physical_supervised', 'paired_consistency')
PREFIXES = ('features.', 'center.', 'size.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def swap_state(candidate, control, prefixes):
    assert candidate.keys() == control.keys() and all(p in PREFIXES for p in prefixes)
    result = {}
    for key, value in candidate.items():
        assert value.shape == control[key].shape and value.dtype == control[key].dtype
        if not key.startswith(PREFIXES): assert torch.equal(value, control[key])
        result[key] = (control[key] if key.startswith(tuple(prefixes)) else value).detach().clone()
    return result


def cosine(first, second):
    a, b = first.double().flatten(), second.double().flatten()
    aa, bb = a.norm(), b.norm()
    return float((a @ b) / (aa*bb)) if aa > 0 and bb > 0 else None


def drift(candidate, control):
    result = {}
    for prefix in PREFIXES:
        keys = sorted(k for k in candidate if k.startswith(prefix))
        a = torch.cat([candidate[k].double().flatten() for k in keys])
        b = torch.cat([control[k].double().flatten() for k in keys])
        delta = a-b
        result[prefix] = dict(numel=len(a), delta_l2=float(delta.norm()), max_abs=float(delta.abs().max()),
                              relative_l2=float(delta.norm()/b.norm()) if b.norm() > 0 else None,
                              parameter_cosine=cosine(a, b))
    return result


def gradient_vectors(loss, model):
    named = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
    gradients = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
    return {prefix: torch.cat([(torch.zeros_like(p) if g is None else g).detach().cpu().double().flatten()
                               for (key, p), g in zip(named, gradients) if key.startswith(prefix)])
            for prefix in PREFIXES if any(key.startswith(prefix) for key, _ in named)}


def summarize_gradients(vectors):
    result = {}
    for prefix in next(iter(vectors.values())):
        v = {name: groups[prefix] for name, groups in vectors.items()}
        legacy = .5*v['normal']+.25*v['half']+.25*v['eighth']
        priority = .5*v['normal']+.375*v['half']+.125*v['eighth']
        result[prefix] = dict(norms={k:float(x.norm()) for k,x in v.items()},
            normal_half_cosine=cosine(v['normal'],v['half']), normal_eighth_cosine=cosine(v['normal'],v['eighth']),
            half_eighth_cosine=cosine(v['half'],v['eighth']),
            legacy_mixture_half_cosine=cosine(legacy,v['half']), priority_mixture_half_cosine=cosine(priority,v['half']),
            teacher_half_vs_half_GT_cosine=cosine(v['half_teacher'],v['half']),
            teacher_normal_vs_half_GT_cosine=cosine(v['normal_teacher'],v['half']))
    return result


def fixed_subset(rows, per_rollout):
    groups = {}
    for row in rows: groups.setdefault(row['rid'], []).append(row)
    result = []
    for rid in sorted(groups):
        group = groups[rid]
        result += [group[i] for i in np.linspace(0,len(group)-1,min(per_rollout,len(group)),dtype=int)]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','study','native-source-dir','radiance-train','radiance-validation','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--native-manifest-sha256',required=True)
    a = p.parse_args(); start=time.monotonic(); deadline=start+540
    def bound(): assert time.monotonic()<deadline,'Bounded read-only diagnostic exceeded540s'
    assert not a.out.exists() and a.out.parent.is_dir()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in (GPU4,'4') and torch.cuda.device_count()==1
    actual=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).strip()
    assert actual==GPU4
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.dont_write_bytecode=True
    own=Path(__file__).resolve().parent
    for name,digest in json.loads((own/'diagnostic_source_manifest.json').read_text()).items():assert sha(own/name)==digest
    assert sha(a.study/'completion.json')==COMPLETION_SHA
    done=json.loads((a.study/'completion.json').read_text());assert done['status']=='complete' and done['real_data_used'] is False
    execution=json.loads((a.study/'priority_execution_receipt.json').read_text());assert len(execution['records'])==36
    assert sha(a.native_source_dir/'source_manifest.json')==a.native_manifest_sha256
    for name,digest in json.loads((a.native_source_dir/'source_manifest.json').read_text()).items():assert sha(a.native_source_dir/name)==digest
    for name,digest in json.loads((a.study.parent/'sources/source_manifest.json').read_text()).items():assert sha(a.study.parent/'sources'/name)==digest
    sys.path.insert(0,str(a.native_source_dir));sys.path.insert(0,str(a.study.parent/'sources'))
    native=load(a.native_source_dir/'pose_two_branch_coverage.py','bound_prior_pose_diagnostic')
    shared=native.shared;R=a.campaign;A=R/'recovery_20261105_mainline'
    scale=shared.load('diagnostic_scale',A/'pose_scale_pilot_20261106.py')
    fresh=shared.load('diagnostic_fresh',A/'pose_fresh_confirm_20261106_v3.py')
    from pose_priority_policy import center_supervision,center_retention,depth_retention
    sys.path.insert(0,str(R/'sources/benchmark/sim2real/scripts'))
    from run_pose_stage2_pilot import encode_event_window
    asset=json.loads((A/'pose_prepared_full_v5/protocol.json').read_text())['assets']['vertices']
    assert sha(asset['path'])==asset['sha256'];vertices=np.load(asset['path'],allow_pickle=False)
    a.out.mkdir(exist_ok=False)
    outputs=dict(schema='pose_half_forgetting_readonly_diagnostic_v1',status='running',optimizer_updates=0,
        real_data_used=False,hybrids_are_diagnostic_not_deployable=True,forward_and_gradients_seeds=[0],
        teacher_gradients_unweighted=True,actual_teacher_coefficients=dict(normal=2,half_expected=1.5),
        parameter_drift_seeds=[0,1,2],source_completion_sha256=COMPLETION_SHA,parameter_drift=[],forwards=[],gradients=[],inputs=[])
    states={};teachers={};weights=[]
    for method in ('eventgenesis','v2e'):
        for seed in range(3):
            bound();tag=f'{method}_seed{seed}'
            row=next(r for r in done['results'] if (r['method'],r['seed'])==(method,seed))
            teacher_path=Path(row['localization_path']);assert sha(teacher_path)==row['localization_sha256']
            teacher=torch.load(teacher_path,map_location='cpu',weights_only=True);teachers[method,seed]=teacher
            weights.append(dict(path=str(teacher_path),sha256=sha(teacher_path),state_sha256=state_sha(teacher)))
            for branch in ('depth','center'):
                for arm in ARMS:
                    path=a.study/(tag+'_'+arm+('_localizer' if branch=='center' else '')+'.pt')
                    state=torch.load(path,map_location='cpu',weights_only=True)
                    evidence=next(r for r in execution['records'] if (r['method'],r['seed'],r['arm'],r['branch'])==(method,seed,arm,branch))
                    assert state_sha(state)==evidence['final_state_sha256'] and state_sha(teacher)==evidence['source_state_sha256']
                    states[method,seed,branch,arm]=state
                    weights.append(dict(path=str(path),sha256=sha(path),state_sha256=state_sha(state)))
                control=states[method,seed,branch,'clean_control']
                for arm in ARMS:
                    candidate=states[method,seed,branch,arm]
                    outputs['parameter_drift'].append(dict(method=method,seed=seed,branch=branch,arm=arm,
                        versus_control=drift(candidate,control),versus_source_teacher=drift(candidate,teacher)))
    selected=sorted(json.loads((a.radiance_train/'completion.json').read_text())['selected'])[:2]
    valfiles=sorted((a.radiance_validation/'half/pose_bank/physical').glob('*.h5'));assert len(valfiles)==4
    assert not (set(selected)&{p.stem for p in valfiles})
    normal_validation=A/'jobs/119_pose_fresh_confirm_paired_20261106/results/pose_bank'
    geometry_binding=None
    for method in ('eventgenesis','v2e'):
        bound();splits={}
        for split,ids,normal_bank,relit_root,k in [
            ('train',selected,R/'pose_bank',a.radiance_train,4),
            ('validation',[p.stem for p in valfiles],normal_validation,a.radiance_validation,2)]:
            conditions={}
            for condition in ('normal','half','eighth'):
                bank=normal_bank if condition=='normal' else relit_root/condition/'pose_bank'
                rows=fresh.adapter(bank,[bank/'physical'/(rid+'.h5') for rid in ids],method,encode_event_window)
                conditions[condition]=fixed_subset(rows,k)
            identity=native.identity(conditions['normal'])
            assert identity==native.identity(conditions['half'])==native.identity(conditions['eighth'])
            assert len(identity)==8
            if split=='validation':
                if geometry_binding is None:geometry_binding=identity
                else:assert geometry_binding==identity
            images={c:shared.tensors(rows,torch.device('cuda')) for c,rows in conditions.items()}
            labels={c:shared.labels(scale,rows,vertices,torch.device('cuda')) for c,rows in conditions.items()}
            assert torch.equal(labels['normal'],labels['half']) and torch.equal(labels['normal'],labels['eighth'])
            splits[split]=(images,labels)
            outputs['inputs'].append(dict(method=method,split=split,ids=[r['id'] for r in conditions['normal']],
                labels_sha256=hashlib.sha256(labels['normal'].cpu().numpy().tobytes()).hexdigest(),
                input_sha256={c:hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for c,t in images.items()}))
        for branch in ('depth','center'):
            bound();model=scale.CenterScale().cuda().eval()
            teacher=scale.CenterScale().cuda().eval();teacher.load_state_dict(teachers[method,0],strict=True);teacher.requires_grad_(False)
            control=states[method,0,branch,'clean_control']
            variants=[('clean_control',control)]
            for arm in ARMS[1:]:
                candidate=states[method,0,branch,arm]
                variants += [(arm,candidate),(arm+'__restore_features',swap_state(candidate,control,['features.'])),
                             (arm+'__restore_heads',swap_state(candidate,control,['center.','size.']))]
                if branch=='depth':variants += [(arm+'__restore_size',swap_state(candidate,control,['size.'])),
                                                (arm+'__restore_center',swap_state(candidate,control,['center.']))]
            for variant,state in variants:
                bound();model.load_state_dict(state,strict=True)
                with torch.inference_mode():
                    for split,(images,labels) in splits.items():
                        for condition in ('normal','half','eighth'):
                            logits,uv,size,shape=model(images[condition]);target=labels[condition]
                            z=(shape+target[:,3]-size).exp();truth=target[:,4].exp()
                            measured=dict(method=method,seed=0,branch=branch,variant=variant,split=split,
                                condition=condition,windows=8,center_px=float(((uv-target[:,:2])*uv.new_tensor([346,260])).norm(dim=1).mean()))
                            if branch=='depth': measured.update(depth_mae_m=float((z-truth).abs().mean()),depth_bias_m=float((z-truth).mean()),
                                log_size_mae=float((size-target[:,2]).abs().mean()),log_shape_bias=float((shape-(target[:,4]-target[:,3]+target[:,2])).mean()))
                            outputs['forwards'].append(measured)
            images,labels=splits['train']
            for arm in ARMS:
                bound();model.load_state_dict(states[method,0,branch,arm],strict=True)
                model.requires_grad_(True)
                if branch=='center':model.size.requires_grad_(False)
                predictions={c:model(x) for c,x in images.items()}
                supervision={c:(shared.supervised(model,pred,labels[c]) if branch=='depth' else center_supervision(model,pred,labels[c]))
                             for c,pred in predictions.items()}
                with torch.no_grad():refs={c:teacher(images[c]) for c in ('normal','half')}
                supervision['normal_teacher']=(depth_retention(predictions['normal'],refs['normal'],include_center=True).mean()
                    if branch=='depth' else center_retention(predictions['normal'],refs['normal']).mean())
                supervision['half_teacher']=(depth_retention(predictions['half'],refs['half']).mean()
                    if branch=='depth' else center_retention(predictions['half'],refs['half']).mean())
                vectors={c:gradient_vectors(loss,model) for c,loss in supervision.items()}
                outputs['gradients'].append(dict(method=method,seed=0,branch=branch,arm=arm,
                    losses={c:float(loss.detach()) for c,loss in supervision.items()},groups=summarize_gradients(vectors)))
            del model,teacher
            torch.cuda.empty_cache()
    for row in weights:assert sha(row['path'])==row['sha256'],'Read-only source weights changed'
    assert sha(a.study/'completion.json')==COMPLETION_SHA
    outputs.update(status='complete',elapsed_seconds=time.monotonic()-start,source_weights=weights,
                   limitation='Eight fixed synthetic windows per split; seed0 functional probes; swaps not trained candidates; no gate or adoption decision')
    write(a.out/'diagnostic.json',outputs)
    print('POSE_READONLY_DIAGNOSTIC_COMPLETE',len(outputs['forwards']),len(outputs['gradients']),flush=True)


if __name__=='__main__':main()
