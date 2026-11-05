"""Stage9 HQF15 bridge. Preparation is stdlib-only and never runs inference.

Existing Stage5 files remain unchanged. Actual metrics use an isolated imported
copy of the pinned scorer: only the list of run identities and protocol path are
supplied by this bridge; window checks and metric arithmetic are unchanged.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

METHODS = ('eventgenesis','v2e','vid2e','film_esim','rife_esim')
SEEDS = (0,1,2)
CORE = '62da4f35ce3c5c96f04268bd1687a27657af9238'
LEGACY_INFER_CORE = 'e79e656ac00f7847b98db9e12bf43b093f3afff1'
INFERENCE_STATUS = 'fixed_sample_event_only_inference_completed_no_GT_scoring'
REL = Path('benchmark/sim2real')
FILES = ('events_ts.npy','events_xy.npy','events_p.npy','images.npy','images_ts.npy',
         'image_event_indices.npy','metadata.json','conversion_manifest.json')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def require(ok,message):
    if not ok:raise RuntimeError(message)
def utc():return datetime.now(timezone.utc).isoformat()
def record(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'size_bytes':p.stat().st_size,'sha256':sha(p)}
def check_record(r):
    p=Path(r['path']).resolve(strict=True)
    require(p.is_file() and p.stat().st_size==r['size_bytes'] and sha(p)==r['sha256'],
            'Bound artifact changed: '+str(p))
    return p
def dump_new(path,value):
    p=Path(path)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def collect_training(campaign,recipe_path):
    """All15 formal receipts and selected weights; no partial fallback."""
    campaign=Path(campaign).resolve(strict=True)
    recipe=read(recipe_path);recipe_sha=sha(recipe_path)
    require(recipe['frozen_simulator_core']==CORE,'Wrong simulator core for Stage9')
    require(recipe['reconstruction']['expected_optimizer_updates']==12000,'Wrong reconstruction budget')
    require(recipe['seeds']==list(SEEDS),'Registered seeds differ')
    rows=[];missing=[]
    for seed in SEEDS:
        for method in METHODS:
            run=campaign/'training/reconstruction'/f'seed{seed}'/method
            cp=run/'completion.json'
            if not cp.is_file():missing.append(f'{method}/seed{seed}');continue
            value=read(cp)
            require((value.get('task'),value.get('method'),value.get('seed'))==('reconstruction',method,seed),'Receipt identity mismatch: '+str(cp))
            require(value.get('status')=='complete' and value.get('completed_updates')==12000,'Incomplete formal budget: '+str(cp))
            require(value.get('recipe_sha256')==recipe_sha,'Recipe fingerprint mismatch: '+str(cp))
            require(value.get('real_data_used_in_training') is False and value.get('real_scoring_performed') is False,'Training receipt real-data guard differs: '+str(cp))
            selected=Path(value['selected_checkpoint']).resolve(strict=True)
            require(selected.parent==run.resolve() and selected.name=='synthetic_best.pt','Only this run synthetic_best.pt is eligible')
            require(sha(selected)==value.get('selected_checkpoint_sha256'),'Selected checkpoint SHA mismatch: '+str(selected))
            init=value.get('initial_state_sha256','')
            require(len(init)==64 and all(c in '0123456789abcdef' for c in init),'Missing initialization fingerprint')
            rows.append(dict(run_id=f'{method}_seed{seed}',method=method,seed=seed,
                             completion=record(cp),checkpoint=record(selected),
                             initial_state_sha256=init,completed_updates=12000))
    require(not missing,'Missing formal training receipts: '+', '.join(missing))
    require(len(rows)==15,'Formal run matrix must contain15 entries')
    for seed in SEEDS:
        require(len({r['initial_state_sha256'] for r in rows if r['seed']==seed})==1,
                f'Paired initialization differs for seed{seed}')
    return rows

def inspect_data(manifest_path,hqf_root,sequences):
    manifest=read(manifest_path);root=Path(hqf_root).resolve(strict=True)
    require(manifest.get('status')=='ready_unscored','HQF data manifest status differs')
    require(manifest.get('real_scoring_enabled') is False and manifest.get('real_gt_metrics_computed') is False,'HQF manifest must remain a pre-score artifact')
    require(manifest.get('expected_sequence_count')==14 and manifest.get('ready_unscored_sequence_count')==14,'HQF14 data not complete')
    require([x.get('sequence') for x in manifest['sequences']]==sequences,'HQF sequence ordering changed')
    for row in manifest['sequences']:
        name=row['sequence'];m=row['evreal_memmap']
        require(row['status']=='ready_unscored' and m.get('exists') is True and not m.get('missing_files'),'Unready HQF sequence: '+name)
        require(Path(m['path']).resolve()==(root/name).resolve(),'Registered HQF root changed: '+name)
        require(set(m['files'])==set(FILES),'Incomplete HQF file inventory: '+name)
        for filename in FILES:
            f=root/name/filename;expected=m['files'][filename]
            require(f.is_file() and f.stat().st_size==expected['size_bytes'] and sha(f)==expected['sha256'],
                    'HQF file mismatch: '+str(f))
    return manifest

def verify_downstream_sources(p):
    """Re-run the pinned tree verifier at every verification/launch boundary."""
    verifier_path=check_record(p['evaluation_sources']['verify_pinned_source.py'])
    bundle=check_record(p['downstream_source_bundle'])
    spec=importlib.util.spec_from_file_location('stage9_launch_source_verifier',verifier_path)
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    rows={x['repo']:x for x in read(bundle)['repositories']}
    checks={}
    for name,key in [('EVREAL','evreal_root'),('event_cnn_minimal','model_root')]:
        require(name in rows,'Missing downstream source pin: '+name)
        checks[name]=verifier.verify_pinned_source(Path(p[key]),name,rows[name]['commit'],bundle)
    return checks

def build_protocol(repo,campaign,hqf_root,model_root,evreal_root):
    repo=Path(repo).resolve(strict=True);campaign=Path(campaign).resolve(strict=True)
    recipe_path=campaign/'sources/training_recipe.json'
    # Refuse incomplete training before reading any large real dataset files.
    runs=collect_training(campaign,recipe_path)
    inherited=repo/REL/'protocol/hqf_14_reconstruction_test_frozen_20261024_v1.json'
    old=read(inherited)
    require(len(old['sequences'])==14 and len(set(old['sequences']))==14,'Inherited HQF14 sequence inventory differs')
    lock=old['protocol_freeze'];data_path=(repo/lock['data_manifest']['path']).resolve(strict=True)
    require(sha(data_path)==lock['data_manifest']['sha256'],'Inherited data manifest SHA differs')
    recipe=read(recipe_path);old_model=read(repo/lock['model_config']['path'])['reconstruction']['model_args']
    require(recipe['reconstruction']['model_args']==old_model,'Stage9 architecture differs from inherited evaluator')
    require(old['dataset_role']['primary']=='HQF all 14 sequences','Unexpected primary reconstruction dataset')
    inspect_data(data_path,hqf_root,old['sequences'])
    scripts=repo/REL/'scripts'
    bindings={name:record(scripts/name) for name in ('infer_hqf_e2vid_candidate.py','score_hqf14_reconstruction.py','verify_pinned_source.py','evaluate_synthetic_reconstruction_fixture.py')}
    bundle=(repo/lock['source_bundle']['path']).resolve(strict=True)
    require(sha(bundle)==lock['source_bundle']['sha256'],'Pinned downstream bundle SHA differs')
    # Full source verification supports either clean git or exact pinned copies.
    spec=importlib.util.spec_from_file_location('stage9_source_verifier',scripts/'verify_pinned_source.py')
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    source_rows={x['repo']:x for x in read(bundle)['repositories']}
    source_checks={}
    for name,path in [('EVREAL',evreal_root),('event_cnn_minimal',model_root)]:
        source_checks[name]=verifier.verify_pinned_source(Path(path),name,source_rows[name]['commit'],bundle)
    return dict(schema='stage9_hqf15_frozen_protocol_v1',status='prepared_unscored',created_utc=utc(),
        task='reconstruction',methods=list(METHODS),seeds=list(SEEDS),expected_runs=15,
        expected_sequence_runs=210,real_scores_computed=False,real_scoring_enabled=False,
        authorization_required=True,real_test_exposure='HQF14 previously seen benchmark re-evaluation; not a blind test',
        dataset_role={'primary':'HQF all14 registered sequences','extension':'HS-ERGB separate protocol, not implemented here'},
        hqf_root=str(Path(hqf_root).resolve()),model_root=str(Path(model_root).resolve()),evreal_root=str(Path(evreal_root).resolve()),
        repository=str(repo),campaign=str(campaign),sequences=old['sequences'],runs=runs,
        inherited_protocol=record(inherited),data_manifest=record(data_path),training_recipe=record(recipe_path),
        downstream_source_bundle=record(bundle),source_verification=source_checks,evaluation_sources=bindings,
        bridge_source=record(__file__),inherited_contract={key:copy.deepcopy(old[key]) for key in
          ('time_alignment','photometry_and_metrics','raw_bag_contract','data_integrity','model')},
        pre_registered_preview_indices=copy.deepcopy(lock['pre_registered_preview_indices']),
        selection='per-run synthetic_best.pt only; no real selection or fallback to last/preflight',
        score_identity_policy='actual method_seed identifiers; no pretending other methods are V2E',
        aggregation='frames within sequence; equal14 sequence means; then3seed mean/sampleSD; paired sequence bootstrap',
        no_real_feedback_to_training_or_generator=True)

def verify_protocol(path,full_data=True):
    path=Path(path).resolve(strict=True);p=read(path)
    require(p.get('schema')=='stage9_hqf15_frozen_protocol_v1','Not a Stage9 HQF15 protocol')
    require(p.get('status')=='prepared_unscored' and p.get('real_scores_computed') is False,'Protocol is not unscored')
    require(p.get('methods')==list(METHODS) and p.get('seeds')==list(SEEDS),'Run matrix changed')
    require(p.get('expected_runs')==15 and p.get('expected_sequence_runs')==210,'Incomplete evaluation matrix')
    require(len(p['runs'])==15 and {(x['method'],x['seed']) for x in p['runs']}=={(m,s) for m in METHODS for s in SEEDS},'Duplicate or missing run identity')
    side=path.with_suffix(path.suffix+'.sha256')
    require(side.is_file() and side.read_text().split()[0]==sha(path),'Protocol sidecar SHA mismatch')
    for key in ('inherited_protocol','data_manifest','training_recipe','downstream_source_bundle','bridge_source'):check_record(p[key])
    for rec in p['evaluation_sources'].values():check_record(rec)
    verify_downstream_sources(p)
    current=collect_training(Path(p['campaign']),check_record(p['training_recipe']))
    require(current==p['runs'],'Formal receipt/checkpoint bindings changed')
    if full_data:inspect_data(p['data_manifest']['path'],p['hqf_root'],p['sequences'])
    return p

def authorize(protocol_path,auth_path):
    a=read(auth_path)
    require(a.get('schema')=='stage9_final_real_evaluation_authorization_v1','Explicit evaluation authorization record required')
    require(a.get('authorized') is True and a.get('task')=='reconstruction','Real evaluation not authorized')
    require(a.get('protocol_sha256')==sha(protocol_path),'Authorization binds another protocol')
    require(bool(a.get('source_user_instruction')) and bool(a.get('research_verification_note')),'Authorization provenance missing')
    require(a.get('no_feedback_to_training_or_selection') is True,'Real-test isolation guard missing')
    return verify_protocol(protocol_path)

def load_scorer(p):
    source=check_record(p['evaluation_sources']['score_hqf14_reconstruction.py'])
    sys.path.insert(0,str(source.parent));sys.path.insert(0,p['repository'])
    spec=importlib.util.spec_from_file_location('stage9_isolated_hqf_metric_library',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    helper=check_record(p['evaluation_sources']['evaluate_synthetic_reconstruction_fixture.py'])
    for function in (module.classical_metrics,module.load_lpips_alex,module.lpips_input,module.validate_pair):
        require(Path(inspect.getsourcefile(function)).resolve()==helper,'Metric helper imported outside the bound source tree')
    module.EXPECTED_ARMS=tuple(x['run_id'] for x in p['runs'])
    return module

def validate_inference_contract(p,row,seq,path):
    """Stage9 identities plus every non-metric guard of the legacy validator."""
    item=read(path);model=item.get('model',{});sequence=item.get('sequence',{})
    cp=check_record(row['checkpoint']);registered=item.get('registered_inference',{})
    label=row['run_id']+'/'+seq
    require(item.get('status')==INFERENCE_STATUS,label+' incomplete inference status')
    require(item.get('self_test',{}).get('is_synthetic_fixture') is False,label+' is a fixture')
    require(model.get('checkpoint_path')==str(cp) and model.get('checkpoint_sha256')==row['checkpoint']['sha256'],label+' wrong checkpoint')
    require(model.get('checkpoint_generator_metadata')==row['method'],label+' wrong generator identity')
    require(model.get('checkpoint_model_args')==read(p['training_recipe']['path'])['reconstruction']['model_args'],label+' wrong model architecture')
    require(model.get('checkpoint_state_dict_key')=='model',label+' wrong model state key')
    require(model.get('public_EVREAL_weights_loaded') is False,label+' substituted public weights')
    require(model.get('reference_frame_pixels_used_as_input') is False and model.get('real_GT_scoring_performed') is False,label+' event-only isolation failed')
    require(model.get('input_contract')=='event-only; [1,5,H,W]',label+' wrong input contract')
    require(model.get('recurrent_reset_calls')==1,label+' wrong recurrent reset count')
    require(model.get('reset_policy')=='one reset at the initial anchor; first model call uses events from (t0,t1]; no reset inside sequence',label+' wrong recurrent reset policy')
    require(model.get('output_contract_after_crop')==[1,1,180,240],label+' wrong native crop contract')
    for name,key in [('EVREAL','evreal_root'),('event_cnn_minimal','model_root')]:
        require(model.get('source_verification',{}).get(name,{}).get('commit')==p['source_verification'][name]['commit'],label+' wrong '+name+' source commit')
    require(model.get('model_source_path')==str(Path(p['model_root']).resolve()),label+' model root changed')
    require(sequence.get('name')==seq and sequence.get('path')==str((Path(p['hqf_root'])/seq).resolve()),label+' wrong sequence path or identity')
    n=sequence.get('num_frames_in_memmap',0)
    require(isinstance(n,int) and n>=2 and sequence.get('num_samples_processed')==n-1 and sequence.get('num_inferable_event_endpoints')==n-1,label+' incomplete endpoint count')
    require(sequence.get('initial_anchor_endpoint_excluded_from_model_and_metrics') is True and sequence.get('timestamps_strictly_increasing') is True,label+' wrong anchor/timing contract')
    require(registered.get('save_every_n_samples')==1 and registered.get('save_float_arrays') is True,label+' float32 outputs missing')
    require(registered.get('saved_sample_indices')==list(range(1,n)),label+' incomplete saved inventory')
    require([x.get('sample_index') for x in sequence.get('samples',[])]==list(range(1,n)),label+' incomplete sample inventory')
    require([x.get('sample_index') for x in registered.get('saved_images',[])]==list(range(1,n)),label+' incomplete prediction inventory')
    data_row=next(x for x in read(p['data_manifest']['path'])['sequences'] if x['sequence']==seq)
    provenance=item.get('provenance',{});data_files=provenance.get('data_files',{})
    for filename in FILES[:-1]:
        require(data_files.get(filename,{}).get('sha256')==data_row['evreal_memmap']['files'][filename]['sha256'],label+' inference data hash differs: '+filename)
    reported_core=provenance.get('frozen_simulator_core_commit')
    require(reported_core in (LEGACY_INFER_CORE,CORE),label+' unexpected runner core metadata')
    return item

def preflight_metric_assets(p):
    """Check metric runtime and local assets before210 costly inferences."""
    import torch
    import cv2
    import skimage.metrics
    import skimage.measure
    weights=Path(p['model_root'])/'PerceptualSimilarity/models/weights/v0.1/alex.pth'
    backbone=Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'
    require(weights.is_file(),'Missing local LPIPS weights; automatic download disabled')
    require(backbone.is_file(),'Missing local AlexNet backbone; automatic download disabled')
    before={'metric_weights':record(weights),'backbone':record(backbone)}
    scorer=load_scorer(p)
    metric,source,actual_weights,actual_backbone=scorer.load_lpips_alex(Path(p['model_root']),torch.device('cpu'))
    require(Path(source).resolve().is_relative_to(Path(p['model_root']).resolve()),'LPIPS source imported outside pinned model tree')
    require(Path(actual_weights).resolve()==weights.resolve() and Path(actual_backbone).resolve()==backbone.resolve(),'Metric asset locations changed')
    for value in before.values():check_record(value)
    del metric
    return dict(status='cpu_metric_initialization_passed_no_real_images',metric_source=record(source),**before)

def run_evaluation(protocol_path,auth_path,out,device):
    p=authorize(protocol_path,auth_path);out=Path(out).resolve()
    require(not out.exists(),'Refusing existing evaluation output directory')
    if str(device).startswith('cuda'):
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8',':16:8'),'CUDA deterministic inference requires CUBLAS_WORKSPACE_CONFIG before launch')
    metric_preflight=preflight_metric_assets(p)
    out.mkdir(parents=True);(out/'logs').mkdir()
    manifest=dict(schema='stage9_hqf15_run_v1',status='running',created_utc=utc(),
                  protocol_sha256=sha(protocol_path),authorization=record(auth_path),
                  real_scores_computed=False,arms={},frozen_simulator_core=CORE,metric_preflight=metric_preflight,
                  legacy_infer_provenance_note='Unchanged legacy infer hardcodes its historical core e79 in raw provenance. Stage9 training core is bound by this protocol, recipe and checkpoint receipts; raw metadata is retained without rewriting.',
                  run_identity={x['run_id']:{'method':x['method'],'seed':x['seed']} for x in p['runs']})
    run_path=out/'run_manifest.json'
    def save():run_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    save()
    try:
        for row in p['runs']:
            rid=row['run_id'];cp=check_record(row['checkpoint'])
            arm=dict(checkpoint_path=str(cp),checkpoint_sha256=row['checkpoint']['sha256'],sequences={},inference_manifest_sha256={})
            manifest['arms'][rid]=arm;save()
            for seq in p['sequences']:
                target=out/'inference'/rid/seq
                argv=[sys.executable,'-u',p['evaluation_sources']['infer_hqf_e2vid_candidate.py']['path'],
                      '--evreal-root',p['evreal_root'],'--model-root',p['model_root'],
                      '--hqf-root',p['hqf_root'],'--checkpoint',str(cp),'--sequence',seq,
                      '--model-config',p['training_recipe']['path'],'--output-root',str(target),
                      '--device',device,'--save-every','1','--save-float-arrays']
                log=out/'logs'/f'{rid}__{seq}.log'
                with log.open('x') as f:result=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,cwd=p['repository'])
                require(result.returncode==0,f'Inference failed {rid}/{seq}; retained log {log}')
                inf=target/'inference_manifest.json';item=validate_inference_contract(p,row,seq,inf)
                arm['sequences'][seq]=dict(exit_code=0,command=argv,manifest=record(inf),
                    effective_training_core=CORE,legacy_runner_reported_core=item['provenance']['frozen_simulator_core_commit'])
                arm['inference_manifest_sha256'][seq]=sha(inf);save()
        manifest['status']='all_210_inferences_complete_unscored';save()
        dump_new(out/'inference_manifest_frozen.json',manifest)
        score_existing(p,protocol_path,out,device)
        manifest.update(status='complete',real_scores_computed=True,finished_utc=utc(),score=record(out/'score.json'));save()
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc),finished_utc=utc());save();raise

def score_existing(p,protocol_path,out,device):
    require(verify_protocol(protocol_path,full_data=False)==p,'Protocol bindings changed before scoring')
    scorer=load_scorer(p)
    scorer.DEFAULT_PROTOCOL=Path(protocol_path).resolve()
    import torch
    protocol=dict(schema=p['schema'],sequences=p['sequences'],photometry_and_metrics=p['inherited_contract']['photometry_and_metrics'],
        protocol_freeze={'data_manifest':{'path':p['data_manifest']['path'],'sha256':p['data_manifest']['sha256']},
                         'pre_registered_preview_indices':p['pre_registered_preview_indices']})
    data,by=scorer.validate_data_manifest(protocol,Path(p['data_manifest']['path']),Path(p['hqf_root']))
    data['__path__']=p['data_manifest']['path'];data['__sha256__']=p['data_manifest']['sha256']
    frozen_run=out/'inference_manifest_frozen.json'
    run=read(frozen_run);inferences={}
    require(run['status']=='all_210_inferences_complete_unscored' and run['protocol_sha256']==sha(protocol_path),'Immutable inference manifest binding differs')
    for key in ('metric_source','metric_weights','backbone'):check_record(run['metric_preflight'][key])
    for row in p['runs']:
        rid=row['run_id'];inferences[rid]={}
        for seq in p['sequences']:
            rec=run['arms'][rid]['sequences'][seq]['manifest'];path=check_record(rec)
            inf=validate_inference_contract(p,row,seq,path)
            inferences[rid][seq]=inf
    metric,source,weights,backbone=scorer.load_lpips_alex(Path(p['model_root']),torch.device(device))
    require(record(source)==run['metric_preflight']['metric_source'] and record(weights)==run['metric_preflight']['metric_weights'] and record(backbone)==run['metric_preflight']['backbone'],'Metric assets changed after preflight')
    provenance={'model':'unchanged pinned PerceptualSimilarity v0.1 AlexNet',
                'metric_source':record(source),'metric_weights':record(weights),'backbone':record(backbone),
                'metric_loop':'unchanged score_pair, isolated module run identities supplied by Stage9 bridge'}
    result=scorer.score_pair(protocol=protocol,protocol_sha256=sha(protocol_path),data_manifest=data,data_rows=by,
        hqf_root=Path(p['hqf_root']),inference_manifests=inferences,run_manifest_path=frozen_run,
        output_path=out/'score.json',metric_model=metric,device=torch.device(device),metric_provenance=provenance)
    with (out/'per_sequence_scores.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['task','metric','method','seed','sequence','value'])
        for row in p['runs']:
            for seq,s in result['per_arm'][row['run_id']]['per_sequence'].items():
                m=s['metrics'];values={'lpips_alex':m['lpips_alex_mean'],'ssim':m['ssim_mean'],'mse':m['mse_mean']}
                values['psnr_db']=float('inf') if m['psnr']['mean_is_positive_infinity'] else m['psnr']['mean_db']
                for name,value in values.items():w.writerow(['reconstruction',name,row['method'],row['seed'],seq,value])

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for name in ('repo','campaign','hqf-root','model-root','evreal-root','out'):p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('verify');p.add_argument('--protocol',type=Path,required=True)
    p=sub.add_parser('run')
    for name in ('protocol','authorization','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--enable-real-scoring',action='store_true',required=True)
    a=parser.parse_args()
    if a.command=='prepare':
        require(not a.out.exists(),'Use a fresh protocol output directory')
        protocol=build_protocol(a.repo,a.campaign,a.hqf_root,a.model_root,a.evreal_root)
        a.out.mkdir(parents=True);target=a.out/'hqf15_protocol.json';dump_new(target,protocol)
        target.with_suffix('.json.sha256').write_text(sha(target)+'  '+target.name+'\n')
        template=dict(schema='stage9_final_real_evaluation_authorization_v1',authorized=False,task='reconstruction',
          protocol_sha256=sha(target),source_user_instruction='',research_verification_note='',no_feedback_to_training_or_selection=True)
        dump_new(a.out/'authorization_template_DO_NOT_RUN.json',template)
        print(json.dumps({'status':'prepared_unscored','runs':15,'sequence_runs':210,'protocol':str(target),'sha256':sha(target)},indent=2))
    elif a.command=='verify':
        p=verify_protocol(a.protocol);print(json.dumps({'status':'verified_unscored','runs':len(p['runs']),'real_scores_computed':False}))
    else:run_evaluation(a.protocol,a.authorization,a.out,a.device)

if __name__=='__main__':main()
