"""Zero-update fixed TRAIN component-gradient diagnostic for pose branches.

All32 matching physical windows and three illuminations are fixed before model
outputs. No affine augmentation is applied in this probe. These are local loss
gradients at frozen endpoints, not the historical minibatch or AdamW step.
"""
from __future__ import annotations
import argparse,gc,hashlib,importlib.util,json,math,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.dont_write_bytecode=True

CORE='62da4f35ce3c5c96f04268bd1687a27657af9238'
GPU4='GPU-0602549d-5c0b-133e-63ab-a01d1f6a49d3'
CONDITIONS=('normal','half','eighth')
PREFIXES=('features','center','size')
WEIGHTS={'normal':.5,'half':.375,'eighth':.125,'normal_teacher':2.}

def need(x,message):
    if not x:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def state_sha(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def tensor_sha(x):return array_sha(x.detach().cpu().numpy())

def depth_components(model,pred,target):
    logits,uv,size,shape=pred
    heat=(-.5*(model.grid[None]-target[:,None,:2]).square().sum(-1)*16**2).softmax(1)
    center=-(heat*logits.log_softmax(1)).sum(1).mean()+4*F.smooth_l1_loss(uv,target[:,:2],beta=.1)
    return dict(center=center,size=2*F.smooth_l1_loss(size,target[:,2],beta=.1),
        shape=F.smooth_l1_loss(shape,target[:,4]-target[:,3]+target[:,2],beta=.1),
        logdepth=2*F.smooth_l1_loss(shape+target[:,3]-size,target[:,4],beta=.1))

def select_cohort(rows,rids):
    groups={rid:[] for rid in rids}
    for row in rows:
        if row['rid'] in groups:groups[row['rid']].append(row)
    selected=[]
    for rid in rids:
        values=sorted(groups[rid],key=lambda row:row['id']);need(len(values)>=4,'Fewer than4 visible windows in fixedTRAIN rollout: '+rid)
        selected.extend(values[i] for i in np.linspace(0,len(values)-1,4,dtype=int))
    need(len(selected)==32 and len({r['id'] for r in selected})==32,'Exactly32 unique fixedTRAIN windows required')
    return selected

def collect(loss,model):
    named=list(model.named_parameters());trainable=[(n,p) for n,p in named if p.requires_grad]
    grads=torch.autograd.grad(loss,[p for _,p in trainable],retain_graph=True,allow_unused=True)
    by_name={n:g for (n,_),g in zip(trainable,grads)}
    need(all(g is None or torch.isfinite(g).all() for g in grads),'Nonfinite component gradient')
    pieces={k:[] for k in ('full',*PREFIXES)}
    for name,p in named:
        g=by_name.get(name);v=torch.zeros(p.numel(),dtype=torch.float64) if g is None else g.detach().cpu().double().flatten()
        pieces['full'].append(v)
        for prefix in PREFIXES:
            if name.startswith(prefix+'.'):pieces[prefix].append(v)
    vectors={k:torch.cat(v) if v else torch.zeros(0,dtype=torch.float64) for k,v in pieces.items()}
    return vectors,dict(named_parameters=len(named),trainable_parameters=len(trainable),nonNone_gradient_tensors=sum(g is not None for g in grads),
        unused_trainable_names=[n for (n,_),g in zip(trainable,grads) if g is None],frozen_names=[n for n,p in named if not p.requires_grad],finite=True)

def combine(items):return {k:sum(weight*v[k] for weight,v in items) for k in items[0][1]}
def compare(a,b):
    out={}
    for k in a:
        x,y=a[k],b[k];xx=float(x.norm());yy=float(y.norm());dot=float((x*y).sum())
        out[k]=dict(left_norm=xx,right_norm=yy,dot=dot,cosine=dot/(xx*yy) if xx and yy else None,
                    target_directional_derivative_for_negative_left_gradient=-dot)
    return out
def vector_record(v):return {k:dict(norm=float(t.norm()),sha256=tensor_sha(t),elements=t.numel()) for k,t in v.items()}

def mixture_diagnostics(vectors):
    physical=combine([(WEIGHTS[c],vectors[c+'/supervised']) for c in CONDITIONS])
    teacher=combine([(2.,vectors['normal_teacher'])]);mixture=combine([(1.,physical),(1.,teacher)])
    P,T=physical['full'],teacher['full'];denom=float(P.square().sum())
    coefficient=min(float((P*T).sum())/denom,0.) if denom else 0.
    # One full-space projection coefficient, not a separate favorable prefix fit.
    projected_teacher=combine([(1.,teacher),(-coefficient,physical)])
    projected_mixture=combine([(1.,physical),(1.,projected_teacher)])
    ratios={}
    for k in physical:
        a=float(physical[k].norm());b=float(teacher[k].norm());ratios[k]=float(mixture[k].norm())/(a+b) if a+b else None
    pairs=dict(physical_vs_teacher=compare(physical,teacher))
    for c in CONDITIONS:
        pairs['physical_vs_'+c]=compare(physical,vectors[c+'/supervised'])
        pairs['teacher_vs_'+c]=compare(teacher,vectors[c+'/supervised'])
        pairs['actual_expected_mixture_vs_'+c]=compare(mixture,vectors[c+'/supervised'])
        pairs['projected_teacher_mixture_vs_'+c]=compare(projected_mixture,vectors[c+'/supervised'])
    for a,b in (('normal','half'),('normal','eighth'),('half','eighth')):pairs[a+'_vs_'+b]=compare(vectors[a+'/supervised'],vectors[b+'/supervised'])
    return dict(weights=WEIGHTS,physical_expected=vector_record(physical),normal_teacher_weighted=vector_record(teacher),
        actual_expected_mixture=vector_record(mixture),cancellation_ratio=ratios,comparisons=pairs,
        projected_teacher_diagnostic=dict(full_space_removed_negative_coefficient=coefficient,
            projection='T_projected=T-min(dot(T,P)/dot(P,P),0)*P; P=.5*N+.375*H+.125*D; T=2*normal_teacher',
            same_coefficient_all_prefixes=True,projected_teacher=vector_record(projected_teacher),projected_mixture=vector_record(projected_mixture),
            optimizer_updates=0,training_treatment_registered=False))

def load_policy(ctx):
    path=Path(ctx.study).parent/'sources/pose_priority_policy.py'
    sys.path.insert(0,str(path.parent));spec=importlib.util.spec_from_file_location('trainprobe_bound_priority_policy',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module,path

def prepare_train(ctx,c):
    manifest=read(ctx.R/'pose_bank/bank_manifest.json')
    rids=sorted(r.get('rollout_id',r.get('rollout')) for r in manifest['rollouts'] if r['split']=='train')
    prepared=read(Path(c['radiance_train'])/'completion.json');need(len(rids)==96 and set(rids)==set(prepared['selected']) and prepared['status']=='complete','Original complete96 TRAIN coverage required')
    rids=rids[:8];banks={'normal':ctx.R/'pose_bank',**{l:Path(c['radiance_train'])/l/'pose_bank' for l in ('half','eighth')}}
    result={};cohort_metadata={};paired_labels=None
    for method in c['methods']:
        data={};reference=None
        for condition,bank in banks.items():
            rows=ctx.fresh.adapter(bank,[bank/'physical'/(rid+'.h5') for rid in rids],method,ctx.encode)
            rows=select_cohort(rows,rids);identity=ctx.native.identity(rows)
            if reference is None:reference=identity
            else:need(identity==reference,'TRAIN relighting changed physical identity/GT')
            x=ctx.shared.tensors(rows,ctx.device);y=ctx.shared.labels(ctx.scale,rows,ctx.vertices,ctx.device)
            need(x.shape==(32,3,128,128) and torch.isfinite(x).all() and torch.isfinite(y).all(),'Wrong native TRAIN tensor/labels')
            data[condition]=(x,y)
        need(torch.equal(data['normal'][1],data['half'][1]) and torch.equal(data['normal'][1],data['eighth'][1]),'TRAIN GT differs under relighting')
        labels=tensor_sha(data['normal'][1])
        if paired_labels is None:paired_labels=labels
        else:need(labels==paired_labels,'Cross-generator TRAIN labels differ')
        result[method]=data;cohort_metadata[method]=dict(rollouts=rids,sample_ids=[v['id'] for v in rows],
            labels_sha256=labels,input_sha256={k:tensor_sha(v[0]) for k,v in data.items()},windows=32,
            source='first8 sorted originalTRAIN rid;4 linspace visible windows each; matched raw event encode and native area128',
            affine_applied=False,validation_samples_used=False)
    return result,cohort_metadata

def all_records(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:yield value
        for child in value.values():yield from all_records(child)
    elif isinstance(value,(list,tuple)):
        for child in value:yield from all_records(child)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    started=time.monotonic();c=read(args.contract);deadline=started+c['timeout_seconds']
    need(c['frozen_core']==CORE and c['gpu_uuid']==GPU4 and c['methods']==['eventgenesis','v2e'] and c['seeds']==[0,1,2] and c['arm_mapping']=={'clean_control':'C0','physical_supervised':'C1','paired_consistency':'slow'} and c['timeout_seconds']==1200,'Wrong registered TRAIN diagnostic scope')
    need(args.out.is_absolute() and not args.out.exists() and not args.out.is_symlink() and args.out.parent.is_dir(),'Unique absolute new output required')
    args.out.mkdir(exist_ok=False)
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        import export_pose_factors as E
        ctx=E.load_context(c,check_gpu=True);policy,policy_path=load_policy(ctx)
        data,cohort=prepare_train(ctx,c);write(args.out/'cohort.json',dict(methods=cohort,optimizer_updates=0,GT_role='TRAIN physical supervision diagnostic only; no optimizer/calibration'))
        source_before={r['path']:r['sha256'] for r in all_records(ctx.source_records)}
        source_before[str(policy_path)]=sha(policy_path);outputs=[]
        rows={(r['method'],r['seed']):r for r in ctx.completion['results']}
        for method in c['methods']:
            for seed in c['seeds']:
                run=rows[method,seed];teacher_path=Path(run['localization_path']);need(sha(teacher_path)==run['localization_sha256'],'Initial normal teacher checkpoint bytes differ')
                teacher=ctx.scale.CenterScale().to(ctx.device).eval();teacher.load_state_dict(torch.load(teacher_path,map_location='cpu',weights_only=True),strict=True);teacher.requires_grad_(False)
                teacher_state=state_sha(teacher.state_dict());source_before[str(teacher_path)]=sha(teacher_path)
                with torch.no_grad():reference=teacher(data[method]['normal'][0])
                for legacy_arm,canonical in c['arm_mapping'].items():
                    for branch in ('depth','center'):
                        need(time.monotonic()<deadline,'1200s TRAIN diagnostic budget reached')
                        model,training_record,path=E.load_branch(ctx,method,seed,legacy_arm,branch)
                        model.requires_grad_(True)
                        if branch=='center':model.size.requires_grad_(False)
                        need(training_record['source_state_sha256']==teacher_state,'Teacher must be actual common frozen source of this branch')
                        before=state_sha(model.state_dict());need(before==training_record['final_state_sha256'],'Endpoint state differs from actual training receipt')
                        source_before[str(path)]=sha(path);model.train()
                        inventory=[dict(name=n,shape=list(t.shape),requires_grad=t.requires_grad) for n,t in model.named_parameters()]
                        predictions={condition:model(value[0]) for condition,value in data[method].items()}
                        vectors={};terms={}
                        for condition,pred in predictions.items():
                            y=data[method][condition][1]
                            components=depth_components(model,pred,y) if branch=='depth' else dict(center=policy.center_supervision(model,pred,y))
                            total=sum(components.values())
                            native=ctx.shared.supervised(model,pred,y) if branch=='depth' else policy.center_supervision(model,pred,y)
                            need(torch.equal(total,native),'Native supervision component formula parity failed')
                            for name,loss in {**components,'supervised':native}.items():
                                need(time.monotonic()<deadline,'1200s gradient diagnostic budget reached')
                                v,meta=collect(loss,model);key=condition+'/'+name;vectors[key]=v;terms[key]=dict(loss=float(loss.detach()),gradients=vector_record(v),gradient_metadata=meta)
                        reference_loss=(policy.depth_retention(predictions['normal'],reference,include_center=True).mean() if branch=='depth' else policy.center_retention(predictions['normal'],reference).mean())
                        v,meta=collect(reference_loss,model);vectors['normal_teacher']=v;terms['normal_teacher']=dict(loss=float(reference_loss.detach()),actual_weight=2.,gradients=vector_record(v),gradient_metadata=meta)
                        mixture=mixture_diagnostics(vectors);component_comparisons={}
                        if branch=='depth':
                            for condition in CONDITIONS:
                                for component in ('center','size','shape'):
                                    component_comparisons[condition+'/'+component+'_vs_logdepth']=compare(vectors[condition+'/'+component],vectors[condition+'/logdepth'])
                                component_comparisons[condition+'/teacher_vs_logdepth']=compare(vectors['normal_teacher'],vectors[condition+'/logdepth'])
                        need(all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in teacher.parameters()),'autograd diagnostic populated mutable .grad unexpectedly')
                        need(before==state_sha(model.state_dict()) and teacher_state==state_sha(teacher.state_dict()),'Read-only gradient diagnostic changed model/teacher state')
                        result=dict(method=method,seed=seed,arm=canonical,legacy_arm=legacy_arm,branch=branch,training_windows=32,conditions=list(CONDITIONS),
                            model_path=str(path),model_file_sha256=sha(path),model_state_before_sha256=before,model_state_after_sha256=state_sha(model.state_dict()),
                            teacher_path=str(teacher_path),teacher_file_sha256=sha(teacher_path),teacher_state_sha256=teacher_state,source_training_record=training_record,
                            parameter_inventory=inventory,parameter_inventory_sha256=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                            losses_and_gradients=terms,mixture=mixture,component_comparisons=component_comparisons,
                            optimizer_updates=0,Adam_direction_computed=False,affine_applied=False,cohort_labels_sha256=cohort[method]['labels_sha256'],
                            expected_mixture_recipe='C1/slow expected illumination mixture evaluated at each endpoint; at C0 this is a counterfactual, not its normal-only historical objective',
                            interpretation='Positive gradient cosine implies local negative-left-gradient lowers the right task to first order, not an actual AdamW step; conflict at trained endpoints may be a regularized equilibrium. Projection is a numerical TRAIN-only diagnostic, not a registered treatment.')
                        target=args.out/f'{method}_seed{seed}_{canonical}_{branch}.json';write(target,result)
                        outputs.append(dict(method=method,seed=seed,arm=canonical,branch=branch,path=str(target),sha256=sha(target)))
                        print('POSE_TRAIN_GRADIENT',method,seed,canonical,branch,flush=True)
                        del model,predictions,vectors,terms,reference_loss,components,result,pred,loss,total,native,v;gc.collect();torch.cuda.empty_cache()
                del teacher,reference;gc.collect();torch.cuda.empty_cache()
        need(len(outputs)==36 and time.monotonic()<deadline,'Expected36 branch endpoints inside1200s')
        for path,digest in source_before.items():need(sha(path)==digest,'Original source/model file changed: '+path)
        write(args.out/'completion.json',dict(schema='pose_train_factor_gradient_diagnostic_v1',status='complete',optimizer_updates=0,real_scores_read=False,
            results=outputs,models=36,methods=c['methods'],seeds=c['seeds'],branches=['depth','center'],arms=c['arm_mapping'],cohort=cohort,
            source_records=ctx.source_records,verified_original_file_sha256=source_before,contract_sha256=sha(args.contract),worker_sha256=sha(Path(__file__)),
            physical_task_weights=WEIGHTS,feature_projection_is_diagnostic_only=True,no_training_treatment_selected=True,
            validation_gradients_used=False,real_gradients_used=False,affine_applied=False,Adam_direction_computed=False,elapsed_seconds=time.monotonic()-started))
    except BaseException:
        write(args.out/'failure.json',dict(status='failed',error=traceback.format_exc(),optimizer_updates=0,real_scores_read=False));raise

if __name__=='__main__':main()
