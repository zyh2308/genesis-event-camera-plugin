"""Stage9 MVSEC preparation and fail-closed evaluation bridge.

Reuses the existing pinned loader, GT propagation and metrics; never opens a
real dataset or loads a model in prepare/self-test. Research must freeze the
prepared protocol/manifest and issue the existing hash-bound authorization
before evaluate can run. Missing outdoor data remains explicit in every result.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,os,sys,tempfile,time
from pathlib import Path

METHODS=('eventgenesis','v2e','vid2e','film_esim','rife_esim')
SEEDS=(0,1,2)
REQUIRED=('indoor_flying1','indoor_flying2','indoor_flying3','outdoor_day1')
EXPECTED_INDOOR={'indoor_flying1':2165,'indoor_flying2':2624,'indoor_flying3':2911}
REGISTERED_INDOOR_SHA='46e0141bde84b8c3c3cc96615cec247f44bb8488ae33bdaf626d4f23d8dca02f'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)

def module(path,name):
    path=Path(path).resolve(strict=True);sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def checkpoint_binding(campaign,method,seed,recipe_hash,*,verify_last=True):
    run=(Path(campaign)/'training/flow'/f'seed{seed}'/method).resolve()
    receipt=run/'completion.json'
    if not receipt.is_file():return None
    r=read(receipt)
    if (r.get('task'),r.get('method'),r.get('seed'))!=('flow',method,seed):
        raise ValueError('Completion identity mismatch: '+str(receipt))
    if r.get('status')!='complete' or r.get('completed_updates')!=8910:
        raise ValueError('Formal45epoch/8910 budget not completed: '+str(receipt))
    if r.get('recipe_sha256')!=recipe_hash or r.get('real_data_used_in_training') is not False:
        raise ValueError('Recipe/isolation mismatch: '+str(receipt))
    selected=Path(r['selected_checkpoint']).resolve(strict=True)
    if selected.parent!=run or selected.name!='synthetic_best.pt':
        raise ValueError('Only this Stage9 run synthetic_best is accepted, never last/preflight/old weights')
    actual=sha(selected)
    if actual!=r['selected_checkpoint_sha256']:raise ValueError('Selected checkpoint SHA256 mismatch')
    initial=r.get('initial_state_sha256','')
    if len(initial)!=64:raise ValueError('Missing initialization fingerprint')
    last=run/'last.pt'
    if verify_last and (not last.is_file() or sha(last)!=r.get('last_checkpoint_sha256')):
        raise ValueError('Completed last checkpoint absent or SHA256 mismatch')
    data=Path(campaign)/'adapters'/method/'flow/training_bank_manifest.json'
    if not data.is_file():raise FileNotFoundError('Stage9 actual flow bank manifest missing: '+str(data))
    return dict(method=method,seed=seed,run=str(run),completed_updates=8910,
                completion=str(receipt),completion_sha256=sha(receipt),
                selected_checkpoint=str(selected),selected_checkpoint_sha256=actual,
                selected_checkpoint_size_bytes=selected.stat().st_size,
                last_checkpoint=str(last),last_checkpoint_sha256=r.get('last_checkpoint_sha256'),
                initial_state_sha256=initial,recipe_sha256=recipe_hash,
                data_manifest=str(data.resolve()),data_manifest_sha256=sha(data),
                correlation_backend=r.get('correlation_backend','original_pytorch'))

def validate_binding(binding):
    selected=Path(binding['selected_checkpoint']).resolve(strict=True)
    if selected.parent!=Path(binding['run']).resolve() or selected.name!='synthetic_best.pt':
        raise ValueError('Invalid frozen Stage9 checkpoint path')
    for key,hkey in [('selected_checkpoint','selected_checkpoint_sha256'),('completion','completion_sha256'),('data_manifest','data_manifest_sha256')]:
        if sha(binding[key])!=binding[hkey]:raise ValueError('Frozen artifact changed: '+key)
    if selected.stat().st_size!=binding['selected_checkpoint_size_bytes']:raise ValueError('Frozen weight size changed')

def prepare(a):
    root=a.campaign.resolve(strict=True);recipe=root/'sources/training_recipe.json';rh=sha(recipe)
    recipe_data=read(recipe)
    if recipe_data.get('schema')!='stage9_newcore_training_recipe_v1' or recipe_data.get('frozen_simulator_core')!='62da4f35ce3c5c96f04268bd1687a27657af9238':
        raise ValueError('Only the registered new-core Stage9 recipe is accepted')
    if recipe_data['flow_dt1']['expected_optimizer_updates']!=8910 or recipe_data['flow_dt1']['effective_batch']!=8:
        raise ValueError('Frozen Stage9 flow budget/batch contract differs')
    if sha(a.registration)!=a.registration_sha256:
        raise ValueError('Registration SHA256 differs from explicitly pinned window manifest')
    registration=read(a.registration);windows=registration['windows']
    observed={s:len([r for r in windows if r['sequence']==s]) for s in REQUIRED}
    for seq,n in EXPECTED_INDOOR.items():
        if observed[seq]!=n:raise ValueError('Registered indoor window count changed: '+seq)
    if len({(r['sequence'],r['aps_start_index']) for r in windows})!=len(windows):
        raise ValueError('Duplicate registered windows')
    available=[s for s in REQUIRED if observed[s] and s in registration['files']]
    missing=[s for s in REQUIRED if s not in available]
    if missing not in ([],['outdoor_day1']):raise ValueError('Unexpected required-data gap: '+str(missing))
    bindings={};incomplete=[]
    for seed in SEEDS:
        for method in METHODS:
            key=f'{method}/seed{seed}'
            b=checkpoint_binding(root,method,seed,rh)
            if b is None:incomplete.append(key)
            else:bindings[key]=b
            print('PREPARE',key,'ready' if b else 'training_incomplete',flush=True)
        same=[b['initial_state_sha256'] for b in bindings.values() if b['seed']==seed]
        if len(set(same))>1:raise ValueError('Generator-specific initialization differs at seed '+str(seed))
    source={label:dict(path=str(path.resolve(strict=True)),sha256=sha(path)) for label,path in
            [('adapter',a.adapter_script),('metrics',a.metric_script),('model_compat',a.compat_script)]}
    # No hashes or paths from the historical two-arm checkpoint_binding are
    # reused. Only immutable real-data window registration is consumed.
    protocol=dict(schema='stage9_mvsec_dt1_protocol_v2',status='candidate_pending_research_review',
        real_scores_authorized=False,required_sequences=list(REQUIRED),evaluation_sequences=available,
        missing_required_sequences=missing,complete_required_scope=not missing,
        result_scope='required_four_sequence_final' if not missing else 'indoor_only_preliminary',
        methods=list(METHODS),seeds=list(SEEDS),training_budget_updates=8910,dt=1,
        model_commit='eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6',
        selected_weights='completed Stage9 synthetic_best only; no real feedback',
        flow_unit='px_per_interval',event_window='(APS timestamp[i],APS timestamp[i+1]]',
        input='same native260x346 real events,5Gaussian equal-count slices,center-pad352; event-only model input',
        primary_roi_native_yxyx=[0,260,0,346],
        legacy_rois_native_yxyx={s:([2,192,45,301] if s=='outdoor_day1' else [2,258,45,301]) for s in REQUIRED},
        primary_mask='finite nonzero GT AND identical common real-event support',
        secondary='fixed legacy256 ROI, same mask rule, always reported; not selected by scores',
        metrics=['aee_px_per_interval','outlier_3px_and_5pct_percent','zero_flow_aee_px_per_interval'],
        empty_support='keep index/denominator metadata; undefined score, never fabricated zero',
        aggregation='per-window then equal-sequence; seed mean/sampleSD and paired sequence-cluster CI; no four-sequence aggregate while outdoor missing',
        registration_sha256=sha(a.registration),source_scripts=source,
        no_real_data_model_selection=True)
    manifest=dict(schema='stage9_mvsec_matrix_manifest_v2',status='candidate_pending_research_review',
        real_scores_authorized=False,campaign=str(root),recipe=str(recipe),recipe_sha256=rh,
        registration=str(a.registration.resolve()),registration_sha256=sha(a.registration),
        required_sequences=list(REQUIRED),evaluation_sequences=available,missing_required_sequences=missing,
        registered_window_counts=observed,total_registered_windows=len(windows),
        complete_checkpoint_matrix=not incomplete,missing_training_runs=incomplete,
        checkpoint_binding=bindings,source_scripts=source,
        result_scope=protocol['result_scope'])
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'protocol_candidate.json',protocol);write(a.out/'manifest_candidate.json',manifest)
    write(a.out/'PREPARATION_STATUS.json',dict(real_scores_computed=False,weights_ready=len(bindings),
        required_weights=15,missing_training_runs=incomplete,required_sequences=list(REQUIRED),
        available_sequences=available,missing_sequences=missing,
        protocol_freeze_and_hash_bound_authorization_still_required=True))
    print('PREPARED_NO_REAL_SCORING',a.out,flush=True)

def authorize(a):
    p=read(a.protocol);m=read(a.manifest)
    # Reuse the original explicit protocol/manifest SHA-bound authorization.
    ms=Path(m['source_scripts']['metrics']['path'])
    if sha(ms)!=m['source_scripts']['metrics']['sha256']:raise ValueError('Frozen metrics source changed')
    metrics=module(ms,'stage9_pinned_mvsec_metrics')
    metrics.authorize_real_evaluation(a.protocol,a.manifest,a.authorization)
    if p['schema']!='stage9_mvsec_dt1_protocol_v2' or m['schema']!='stage9_mvsec_matrix_manifest_v2':raise ValueError('Old Stage4/unpatched protocol cannot be used')
    if set(m['source_scripts'])!={'adapter','metrics','model_compat'}:raise ValueError('Frozen training model compatibility patch must be bound')
    if p['methods']!=list(METHODS) or p['seeds']!=list(SEEDS) or p['dt']!=1:raise ValueError('Frozen matrix changed')
    if p['required_sequences']!=list(REQUIRED) or m['required_sequences']!=list(REQUIRED):raise ValueError('Required outdoor sequence cannot be silently removed')
    if not m['complete_checkpoint_matrix'] or m['missing_training_runs']:raise ValueError('Full15checkpoint matrix not frozen')
    expected={f'{method}/seed{seed}' for method in METHODS for seed in SEEDS}
    if set(m['checkpoint_binding'])!=expected:raise ValueError('Frozen matrix must contain exactly15 completed runs')
    if p['registration_sha256']!=m['registration_sha256'] or sha(m['registration'])!=m['registration_sha256']:raise ValueError('Window registration changed')
    if sha(m['recipe'])!=m['recipe_sha256']:raise ValueError('Formal training recipe changed')
    if p['evaluation_sequences']!=m['evaluation_sequences'] or p['missing_required_sequences']!=m['missing_required_sequences']:raise ValueError('Protocol/manifest sequence scope differs')
    if set(p['evaluation_sequences'])|set(p['missing_required_sequences'])!=set(REQUIRED):raise ValueError('Declared sequence coverage incomplete')
    if p['missing_required_sequences'] and p['result_scope']!='indoor_only_preliminary':raise ValueError('Missing outdoor requires explicit preliminary scope')
    if read(a.authorization).get('evaluation_scope')!=p['result_scope']:raise ValueError('Authorization must explicitly name this full/partial evaluation scope')
    for label,item in m['source_scripts'].items():
        if p['source_scripts'][label]!=item or sha(item['path'])!=item['sha256']:raise ValueError('Source binding differs: '+label)
    for b in m['checkpoint_binding'].values():validate_binding(b)
    return p,m,metrics

def evaluate(a):
    p,m,metrics=authorize(a)
    key=f'{a.method}/seed{a.seed}';binding=m['checkpoint_binding'][key]
    registration=read(m['registration'])
    # Imports and all real-data/model reads happen only after authorization.
    import numpy as np
    import torch,h5py
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not a.device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('GPU inference required; lease must be provided by caller')
    if torch.cuda.device_count()!=1 or not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-'):raise RuntimeError('Require one explicitly leased physical GPU UUID')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    device=torch.device(a.device);torch.cuda.set_device(device)
    adapter=module(m['source_scripts']['adapter']['path'],'stage9_pinned_mvsec_adapter')
    model,ckpt,commit,verification=adapter.load_pinned_model(a.model_root,Path(binding['selected_checkpoint']),1,device)
    # The existing inference adapter predates Stage9: apply the exact frozen
    # TRAINING compatibility layer, including the internal model warp, before
    # any inference. This is parity with training, not a new model change.
    compat=module(m['source_scripts']['model_compat']['path'],'stage9_frozen_model_compat')
    import importlib
    compatibility=compat.patch_ste(importlib.import_module('models'),importlib.import_module('multiscaleloss'),a.model_root)
    for field,expected in [('generator',a.method),('seed',a.seed),('config_sha256',m['recipe_sha256']),('data_manifest_sha256',binding['data_manifest_sha256']),('initial_state_sha256',binding['initial_state_sha256'])]:
        if ckpt.get(field)!=expected:raise ValueError('Selected checkpoint metadata mismatch: '+field)
    a.out.mkdir(parents=True,exist_ok=False);rows=[];per_sequence=[]
    with (a.out/'window_metrics.jsonl').open('x') as stream:
        for seq in p['evaluation_sequences']:
            file=registration['files'][seq]
            data_path=Path(file['data_path_on_g1']);gt_path=Path(file['gt_path_on_g1'])
            for path,label in [(data_path,'data'),(gt_path,'gt')]:
                if path.stat().st_size!=file[label+'_size_bytes'] or sha(path)!=file[label+'_sha256']:raise ValueError('Real dataset fingerprint mismatch: '+str(path))
            data=adapter.load_mvsec_events_and_endpoints(data_path)
            registered=[x for x in registration['windows'] if x['sequence']==seq]
            with h5py.File(gt_path,'r') as h:
                gt=h['davis/left/flow_dist'];gt_times=np.asarray(h['davis/left/flow_dist_ts'],dtype=np.float64).reshape(-1)
                for expected in registered:
                    row,_,_,prediction,truth,_,events=adapter.run_one_window(model,data,gt,gt_times,expected['aps_start_index'],1,352,device)
                    if row['event_input_sha256']!=expected['event_input_sha256'] or metrics.sha256_array(truth)!=expected['gt_flow_sha256']:raise ValueError('Registered input/GT changed: '+expected['sample_id'])
                    if row['start_time_s']!=expected['start_time_s'] or row['end_time_s']!=expected['end_time_s']:raise ValueError('Absolute APS endpoints differ')
                    scored={}
                    for name,roi in [('full_native',p['primary_roi_native_yxyx']),('legacy_roi',p['legacy_rois_native_yxyx'][seq])]:
                        mask,counts=metrics.common_valid_mask(truth,events,events,evaluation_roi_native_yxyx=roi)
                        if metrics.sha256_array(mask)!=expected['gt_support']['common_mask_sha256_'+name]:raise ValueError('Registered common mask changed: '+expected['sample_id'])
                        value=metrics.compute_window_metrics(prediction,truth,mask)
                        zero=metrics.compute_window_metrics(np.zeros_like(truth),truth,mask)
                        scored[name]=dict(**value,zero_flow_aee_px_per_interval=zero['aee_px_per_interval'],mask_sha256=metrics.sha256_array(mask),mask_counts=counts)
                    item=dict(sequence=seq,aps_start_index=expected['aps_start_index'],sample_id=expected['sample_id'],
                        method=a.method,seed=a.seed,event_input_sha256=row['event_input_sha256'],
                        gt_sha256=expected['gt_flow_sha256'],event_count=row['event_count'],metrics=scored,
                        checkpoint_sha256=binding['selected_checkpoint_sha256'],inference_inputs=['events only'])
                    stream.write(json.dumps(item,allow_nan=False)+'\n');rows.append(item)
                    if len(rows)%100==0:stream.flush();print('MVSEC_FIXED_EVALUATION',key,seq,len(rows),flush=True)
            for roi_name in ['full_native','legacy_roi']:
                selected=[x['metrics'][roi_name] for x in rows if x['sequence']==seq and x['metrics'][roi_name]['status']=='scored']
                per_sequence.append(dict(method=a.method,seed=a.seed,sequence=seq,roi=roi_name,
                    registered_windows=len(registered),scored_windows=len(selected),empty_windows=len(registered)-len(selected),
                    aee_px_per_interval=float(np.mean([x['aee_px_per_interval'] for x in selected])) if selected else None,
                    outlier_3px_and_5pct_percent=float(np.mean([x['outlier_3px_and_5pct_percent'] for x in selected])) if selected else None,
                    zero_flow_aee_px_per_interval=float(np.mean([x['zero_flow_aee_px_per_interval'] for x in selected])) if selected else None))
            del data
    write(a.out/'result.json',dict(status='authorized_real_score_full_scope' if not p['missing_required_sequences'] else 'authorized_indoor_partial_score',
        method=a.method,seed=a.seed,required_sequences=list(REQUIRED),evaluation_sequences=p['evaluation_sequences'],
        missing_required_sequences=p['missing_required_sequences'],complete_required_scope=not p['missing_required_sequences'],
        four_sequence_aggregate_published=False,per_sequence=per_sequence,window_count=len(rows),
        protocol_sha256=sha(a.protocol),manifest_sha256=sha(a.manifest),authorization_sha256=sha(a.authorization),
        checkpoint_sha256=binding['selected_checkpoint_sha256'],source_verification=verification,
        model_commit=commit,training_model_compatibility=compatibility,
        model_compat_script_sha256=m['source_scripts']['model_compat']['sha256'],no_real_score_feedback=True))
    with (a.out/'sequence_metrics.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per_sequence[0]));w.writeheader();w.writerows(per_sequence)
    print('MVSEC_STAGE9_EVALUATION_COMPLETE',key,'partial' if p['missing_required_sequences'] else 'complete_required_scope')

def selftest(a):
    metrics=module(a.metric_script,'stage9_metrics_cpu_selftest')
    import numpy as np
    passed=[]
    with tempfile.TemporaryDirectory(prefix='stage9_mvsec_bridge_') as tmp:
        root=Path(tmp);(root/'sources').mkdir();recipe=root/'sources/training_recipe.json'
        recipe.write_text(json.dumps(dict(schema='stage9_newcore_training_recipe_v1',frozen_simulator_core='62da4f35ce3c5c96f04268bd1687a27657af9238',flow_dt1=dict(expected_optimizer_updates=8910,effective_batch=8))))
        for seed in SEEDS:
            for method in METHODS:
                run=root/'training/flow'/f'seed{seed}'/method;run.mkdir(parents=True)
                best=run/'synthetic_best.pt';best.write_bytes(b'synthetic_fixture_no_model')
                last=run/'last.pt';last.write_bytes(b'synthetic_fixture_completed_last')
                data=root/'adapters'/method/'flow/training_bank_manifest.json';data.parent.mkdir(parents=True,exist_ok=True);data.write_text('{}')
                r=dict(task='flow',method=method,seed=seed,status='complete',completed_updates=8910,recipe_sha256=sha(recipe),real_data_used_in_training=False,
                       selected_checkpoint=str(best),selected_checkpoint_sha256=sha(best),last_checkpoint_sha256=sha(last),initial_state_sha256=str(seed)*64)
                (run/'completion.json').write_text(json.dumps(r))
                assert checkpoint_binding(root,method,seed,sha(recipe)) is not None
        passed.append('all15_identity_budget_hash_bindings')
        run=root/'training/flow/seed0/eventgenesis';rp=run/'completion.json';original=read(rp)
        for label,changes in [('invalid_checkpoint_hash',{'selected_checkpoint_sha256':'f'*64}),('wrong_seed',{'seed':1}),('incomplete_budget',{'completed_updates':20}),('old_or_last_weight',{'selected_checkpoint':str(run/'last.pt')}),('real_training_leak',{'real_data_used_in_training':True}),('wrong_recipe',{'recipe_sha256':'b'*64})]:
            rp.write_text(json.dumps({**original,**changes}))
            try:checkpoint_binding(root,'eventgenesis',0,sha(recipe))
            except (ValueError,FileNotFoundError):passed.append(label)
            else:raise AssertionError('Invalid fixture accepted: '+label)
        rp.write_text(json.dumps(original))
        reg=root/'registration.json';reg.write_text('{}')
        try:prepare(argparse.Namespace(campaign=root,registration=reg,registration_sha256='0'*64))
        except ValueError as e:
            if 'Registration SHA256' not in str(e):raise
            passed.append('wrong_registration_hash_rejected_before_dataset_access')
        else:raise AssertionError('Unpinned registration accepted')
        gt=np.zeros((2,260,346),np.float64);events=np.zeros((2,260,346,5),np.uint8)
        gt[0,10,50]=4;events[0,10,50,0]=1
        mask,_=metrics.common_valid_mask(gt,events,events,evaluation_roi_native_yxyx=[0,260,0,346])
        z=metrics.compute_window_metrics(np.zeros_like(gt),gt,mask)
        assert z['aee_px_per_interval']==4 and z['outlier_3px_and_5pct_percent']==100
        passed.append('mature_metric_vector_zero_flow_oracle')
        protocol=root/'protocol.json';manifest=root/'manifest.json';auth=root/'auth.json'
        protocol.write_text(json.dumps({'status':'candidate_pending_research_review','real_scores_authorized':False}))
        manifest.write_text('{}');auth.write_text('{}')
        try:metrics.authorize_real_evaluation(protocol,manifest,auth)
        except PermissionError:passed.append('existing_real_score_hash_gate_rejects_closed_protocol')
        else:raise AssertionError('Scoring gate bypassed')
        # Validate the actual frozen compatibility function on a tractable CPU
        # zero-flow identity. Negative control must expose the old convention.
        import torch
        import torch.nn.functional as F
        compat=module(a.compat_script,'stage9_compat_cpu_selftest')
        class Dummy:
            def mesh_grid(self,b,h,w):
                yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
                return torch.stack((xx,yy),dim=0)[None].repeat(b,1,1,1)
            def norm_grid(self,g):
                g=g.clone();g[:,0]=2*g[:,0]/(g.shape[-1]-1)-1;g[:,1]=2*g[:,1]/(g.shape[-2]-1)-1
                return g.permute(0,2,3,1)
        dummy=Dummy();x=torch.arange(35,dtype=torch.float64).reshape(1,1,5,7);flow=torch.zeros(1,2,5,7,dtype=torch.float64)
        correct=compat.model_flow_warp(dummy,x,flow)
        wrong=F.grid_sample(x,dummy.norm_grid(dummy.mesh_grid(1,5,7).double()),padding_mode='border',align_corners=False)
        if float((correct-x).abs().max())>1e-12 or float((wrong-x).abs().max())<.01:raise AssertionError('Frozen model warp identity/negative control failed')
        passed.append('frozen_training_model_warp_identity_and_old_convention_negative_control')
    result=dict(passed=True,checks=passed,real_data_used=False,model_loaded=False,GPU_used=False)
    if a.output:write(a.output,result)
    print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    s=sub.add_parser('prepare');s.add_argument('--campaign',type=Path,required=True);s.add_argument('--registration',type=Path,required=True)
    s.add_argument('--registration-sha256',default=REGISTERED_INDOOR_SHA,help='Explicit approved registration hash; an extended outdoor registration needs its own reviewed hash')
    s.add_argument('--adapter-script',type=Path,required=True);s.add_argument('--metric-script',type=Path,required=True);s.add_argument('--out',type=Path,required=True);s.set_defaults(func=prepare)
    s.add_argument('--compat-script',type=Path,required=True)
    s=sub.add_parser('evaluate');s.add_argument('--protocol',type=Path,required=True);s.add_argument('--manifest',type=Path,required=True);s.add_argument('--authorization',type=Path,required=True)
    s.add_argument('--method',choices=METHODS,required=True);s.add_argument('--seed',type=int,choices=SEEDS,required=True);s.add_argument('--model-root',type=Path,required=True);s.add_argument('--device',default='cuda:0');s.add_argument('--out',type=Path,required=True);s.set_defaults(func=evaluate)
    s=sub.add_parser('self-test');s.add_argument('--metric-script',type=Path,required=True);s.add_argument('--compat-script',type=Path,required=True);s.add_argument('--output',type=Path);s.set_defaults(func=selftest)
    a=p.parse_args();a.func(a)

if __name__=='__main__':main()
