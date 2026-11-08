"""Zero-update RAW C1 confirmation on the fixed new32 synthetic trajectories.

Only the already frozen encoder, model forward, crop/rotation and ADD-S scorer
are reused. No test GT enters model inputs or readout. Resource ownership and
timeout are supplied by the existing GPU parent, not created by this worker.
"""
from __future__ import annotations
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
import prepare_confirmation as P

def assess_scores(scores,draws=10000,seed=P.BOOTSTRAP_SEED):
    """Two predeclared primary conditions; pair seeds inside each trajectory."""
    expected={(m,s,k) for m in P.METHODS for s in (0,1,2) for k in ('normal','half','eighth')}
    keys=[(r['method'],r['seed'],r['condition']) for r in scores]
    P.need(len(keys)==len(set(keys))==18 and set(keys)==expected,'Complete18 condition cells required')
    by={k:r for k,r in zip(keys,scores,strict=True)};rids=sorted(by[('eventgenesis',0,'normal')]['score']['by_rollout'])
    P.need(len(rids)==32,'Exactly32 physical trajectory groups required')
    for r in scores:
        P.need(sorted(r['score']['by_rollout'])==rids,'Trajectory inventory differs')
        for v in r['score']['by_rollout'].values():
            P.need(all(np.isfinite(x) for x in v.values()) and 0<=v['success']<=1,'Nonfinite/invalid score')
    rng=np.random.default_rng(seed);indices=rng.integers(len(rids),size=(draws,len(rids)));checks=[]
    for condition in ('normal','half'):
        delta=np.asarray([[by[('eventgenesis',s,condition)]['score']['by_rollout'][rid]['success']-by[('v2e',s,condition)]['score']['by_rollout'][rid]['success'] for rid in rids] for s in (0,1,2)])
        per_seed=delta.mean(1);paired=delta.mean(0);lo,hi=np.quantile(paired[indices].mean(1),[.0125,.9875])
        checks.append(dict(condition=condition,metric='ADD-S_success',difference='eventgenesis_minus_v2e',seed_mean_delta=per_seed.tolist(),
            mean_delta=float(paired.mean()),ci_level=.975,paired_trajectory_ci=[float(lo),float(hi)],passed=bool((per_seed>0).all() and lo>0)))
    return dict(schema='pose_confirm32_conditional_advantage_v1',passed=all(r['passed'] for r in checks),checks=checks,
        draws=draws,bootstrap_seed=seed,resampling_unit='trajectory_after_paired_seed_mean',familywise_ci_level=.95,
        formal30_status='unchanged_not_retested',formal8_status='unchanged_not_retested',real_scoring_authorized=False,
        scope='independent synthetic conditional confirmation; not real generalization',mandatory_negative_condition='eighth')

def baseline_references(path):
    values=P.read(path);rows=[r for r in values if r.get('arm')=='control']
    P.need(len(rows)==18 and all(r['readout']=='native_global' and r['windows']==64 for r in rows),'Actual18 RAW C1 baseline scores required')
    result={(r['method'],r['seed'],r['condition']):r['score'] for r in rows}
    P.need(len(result)==18,'Duplicate native baseline score');return result

def compare_score(current,reference):
    P.need(current.keys()==reference.keys() and current['by_rollout'].keys()==reference['by_rollout'].keys(),'Baseline score fields/inventory differ')
    worst=0.
    for a,b in [(current['summary'],reference['summary']),*((current['by_rollout'][r],reference['by_rollout'][r]) for r in current['by_rollout'])]:
        P.need(a.keys()==b.keys(),'Baseline metric fields differ')
        for k in a:
            P.need(np.isfinite(a[k]) and np.isclose(a[k],b[k],rtol=1e-6,atol=1e-8),'Native complete-pose baseline parity failed: '+k)
            worst=max(worst,abs(a[k]-b[k]))
    return worst

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('registration','out'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--registration-sha256',required=True);ap.add_argument('--event-receipt-sha256',required=True)
    a=ap.parse_args();P.need(P.sha(a.registration)==a.registration_sha256,'Registered contract differs');c=P.read(a.registration);c['_registration_sha256']=a.registration_sha256
    P.validate_registration(c);P.need(a.out.is_absolute() and a.out.parent.is_dir() and not a.out.exists(),'Fresh absolute evaluation output required');a.out.mkdir()
    start=time.monotonic()
    try:
        F=P.common(c);root=Path(c['root']);event=root/'event_completion.json';P.need(P.sha(event)==a.event_receipt_sha256,'Actual replay completion SHA differs');receipt=P.read(event)
        P.need(receipt['status']=='complete' and receipt['optimizer_updates']==0 and receipt['real_data_used'] is False and receipt['registration_sha256']==a.registration_sha256,'Replay scope/registration differs')
        P.verify_records(receipt['artifacts']);P.verify_records([c['export_source'],c['base_contract'],c['baseline_scores']])
        E=P.load('confirmation_bound_export',c['export_source']['path']);base=P.read(c['base_contract']['path'])
        import torch,cv2
        from scipy.spatial import cKDTree
        torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        ctx=E.load_context(base,check_gpu=True);P.need(P.sha(ctx.fresh.__file__)==P.FRESH_SHA,'Actual adapter/fresh implementation differs')
        references=baseline_references(c['baseline_scores']['path']);tree=cKDTree(ctx.vertices);results=[];models=[];parities=[];eligibility=[];case_count=0;paired_identity=None
        with (a.out/'predictions.jsonl').open('x') as stream,torch.inference_mode():
            for method in P.METHODS:
                rows={};views={}
                for condition in c['conditions']:
                    bank=Path(receipt['banks'][condition]);files=sorted((bank/'physical').glob('*.h5'));P.need(len(files)==32,'Actual32 physical files required')
                    rows[condition]=F.adapter(bank,files,method,ctx.encode);ids=[r['id'] for r in rows[condition]];P.need(len(ids)==len(set(ids)) and len({r['rid'] for r in rows[condition]})==32,'Missing trajectory or duplicate window')
                    identity=ctx.native.identity(rows[condition]);paired_identity=paired_identity or identity;P.need(identity==paired_identity,'Cross-method/light ordered GT/time identity differs')
                    expected=[f'{p.stem}/{k:03d}' for p in files for k in range(4,20)];P.need(set(ids)<=set(expected),'Unexpected window selection')
                    eligibility.append(dict(method=method,condition=condition,eligible_ids=ids,excluded_by_original_geometry_rule=[x for x in expected if x not in set(ids)],rule='original fresh.adapter anchor in-frame; no event/prediction-dependent selection'))
                    views[condition]=ctx.shared.tensors(rows[condition],ctx.device)
                old_rows,_=E.aligned_rows(ctx,base,method,with_gains=False);old_views={k:ctx.shared.tensors(v,ctx.device) for k,v in old_rows.items()}
                for seed in (0,1,2):
                    E.check_gpu(base);P.need(time.monotonic()-start<3600,'Fixed3600s evaluation limit exceeded')
                    depth,_,_=E.load_branch(ctx,method,seed,'physical_supervised','depth');center,_,_=E.load_branch(ctx,method,seed,'physical_supervised','center')
                    parent=next(r for r in ctx.completion['results'] if (r['method'],r['seed'])==(method,seed));rp=ctx.A/f'jobs/117_pose_rotation_paired_20261106/results/{method}_seed{seed}_rotation.pt'
                    E.remember(ctx.source_records,rp,parent['parents']['rotation_sha256']);head=ctx.rotation.RotationHead().to(ctx.device).eval();head.load_state_dict(torch.load(rp,map_location='cpu',weights_only=True)['model'],strict=True)
                    for model in (depth,center,head):model.requires_grad_(False)
                    states=[E.state_sha(m.state_dict()) for m in (depth,center,head)]
                    for condition in c['conditions']:
                        old=old_rows[condition];images=old_views[condition];K=[r['K'] for r in old]
                        d=E.predict_raw(depth,images,ctx.scale,K);uv=E.predict_raw(center,images,ctx.scale,K)['uv'];R=ctx.shared.frozen_rotation(ctx.rotation,head,images,uv,d['size'],K)
                        gt=ctx.shared.labels(ctx.scale,old,ctx.vertices,ctx.device).cpu().numpy();score=ctx.shared.score(ctx.rotation,old,uv,d['z'],d['size'],gt[:,2],R,ctx.vertices,tree,ctx.threshold)
                        error=compare_score(score,references[(method,seed,condition)]);parities.append(dict(method=method,seed=seed,condition=condition,passed=True,max_abs=error))
                    for condition in c['conditions']:
                        values=rows[condition];images=views[condition];K=[r['K'] for r in values]
                        d=E.predict_raw(depth,images,ctx.scale,K);uv=E.predict_raw(center,images,ctx.scale,K)['uv'];R=ctx.shared.frozen_rotation(ctx.rotation,head,images,uv,d['size'],K)
                        P.need(np.isfinite(uv).all() and np.isfinite(R).all() and np.isfinite(d['z']).all() and np.isfinite(d['size']).all(),'Nonfinite frozen prediction')
                        gt=ctx.shared.labels(ctx.scale,values,ctx.vertices,ctx.device).cpu().numpy();score=ctx.shared.score(ctx.rotation,values,uv,d['z'],d['size'],gt[:,2],R,ctx.vertices,tree,ctx.threshold)
                        results.append(dict(method=method,seed=seed,condition=condition,arm='C1_RAW',score=score,window_ids=[r['id'] for r in values],windows=len(values)))
                        for i,row in enumerate(values):
                            record=dict(method=method,seed=seed,condition=condition,rid=row['rid'],sample_id=row['id'],pred_uv=uv[i],z_m=d['z'][i],size=d['size'][i],R_pred=R[i],K=row['K'],T_gt=row['T'],anchor=row['keys'][0],gt_uv=row['uv'][0],gt_log_size=gt[i,2])
                            stream.write(json.dumps(record,allow_nan=False,default=E.json_default)+'\n');case_count+=1
                    P.need(all(E.state_sha(m.state_dict())==state for m,state in zip((depth,center,head),states,strict=True)),'Frozen state changed')
                    models.append(dict(method=method,seed=seed,states_sha256=states,unchanged=True));del depth,center,head;torch.cuda.empty_cache()
                del rows,views,old_rows,old_views
        P.need(len(parities)==18 and len(results)==18 and len(models)==6 and case_count==3*sum(len(x['eligible_ids']) for x in eligibility),'Incomplete confirmation matrix')
        P.verify_records(receipt['artifacts']);P.verify_records(ctx.source_records);P.verify_records(c['source_records']);P.verify_records([c['export_source'],c['base_contract'],c['baseline_scores']]);P.check_science_and_core(c)
        decision=assess_scores(results,c['bootstrap_draws'],c['bootstrap_seed'])
        P.write(a.out/'scores.json',results);P.write(a.out/'eligibility.json',eligibility);P.write(a.out/'baseline_parity.json',parities);P.write(a.out/'conditional_advantage.json',decision)
        P.write(a.out/'completion.json',dict(schema='pose_confirm32_frozen_evaluation_v1',status='complete',optimizer_updates=0,real_data_used=False,GT_forward_inputs=False,
            registration_sha256=a.registration_sha256,event_completion=P.record(event),cases=case_count,score_rows=18,models=models,all_baseline_parities_passed=True,
            conditional_advantage=decision,source_records=ctx.source_records+c['source_records'],artifacts=[P.record(p) for p in a.out.iterdir() if p.is_file()],elapsed_seconds=time.monotonic()-start))
    except BaseException as e:
        P.write(a.out/'failure.json',dict(status='failed',optimizer_updates=0,real_data_used=False,error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
