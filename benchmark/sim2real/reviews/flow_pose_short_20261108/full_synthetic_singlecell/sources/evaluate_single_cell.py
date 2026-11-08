"""Full frozen synthetic evaluation only; no optimizer and no real files."""
from __future__ import annotations
import argparse,gc,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
import numpy as np
import flow_synthetic_diagnostic as D

METHODS=('eventgenesis','v2e','vid2e','film_esim','rife_esim')
STRATA=('camera_motion_static_scene','independent_rigid_motion_static_camera','camera_and_independent_rigid_motion','combined_motion_with_occlusion')
ROOT=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-flow-full-synthetic-readonly-rife-seed1-v1')
def need(ok,message):
    if not ok:raise ValueError(message)

def validate(c,seeds,methods,out,visible):
    need(c['schema']=='flow_full_synthetic_readonly_cell_v1' and c['root']==str(ROOT),'Wrong frozen scope/root')
    need(c['methods']==list(METHODS) and c['seeds']==[0,1,2] and c['windows_per_checkpoint']==528,'Full15x528 scope required')
    need(c['optimizer_updates']==0 and c['real_data_used'] is False and c['real_scores_used'] is False and c['checkpoint_selection_changed'] is False,'Read-only synthetic scope required')
    shard='rife_seed1' if seeds==[1] and methods==['rife_esim'] else None
    need(shard is not None and out==ROOT/shard,'Only registered shards and new output roots accepted')
    need(visible==c['shards'][shard]['gpu_uuid'],'Only registered shard GPU may be exposed')
    need(set(c['checkpoint_binding'])=={f'{m}/seed{s}' for m in METHODS for s in (0,1,2)},'Exactly15 fixed checkpoints required')
    for key,b in c['checkpoint_binding'].items():
        need(key==f"{b['method']}/seed{b['seed']}" and b['completed_updates']==8910,'Wrong checkpoint identity/budget')
        need(Path(b['selected_checkpoint'])==Path(b['run'])/'synthetic_best.pt','Only original synthetic_best accepted')
    for seed in (0,1,2):need(len({c['checkpoint_binding'][f'{m}/seed{seed}']['initial_state_sha256'] for m in METHODS})==1,'Methods must share the same original initialization within seed')
    return shard

def aggregate(rows):
    """Equal nonempty windows, then equal rollouts; empty support stays None."""
    result={}
    for name in rows[0]:
        if name=='invalid_geometry_or_visibility':continue
        keys=('epe_px','zero_epe_px','gt_mean_motion_px','predicted_mean_motion_px','prediction_below0p1_fraction','fraction_of_sensor')
        entry={}
        for key in keys:
            values=[r[name][key] for r in rows if r[name].get(key) is not None]
            entry[key]=float(np.mean(values)) if values else None
        entry['supported_members']=sum(r[name].get('epe_px') is not None for r in rows)
        entry['total_members']=len(rows)
        result[name]=entry
    return result

def source_tree(modelroot,bundle,expected):
    pin=next(r for r in json.loads(bundle.read_text())['repositories'] if r['repo']=='STE-FlowNet')
    need(pin['commit']==expected['commit'],'Model commit differs')
    rows=[]
    for f in pin['files']:
        p=Path(f['path']);need(not p.is_absolute() and '..' not in p.parts,'Unsafe model source path')
        actual=modelroot/p;need(actual.is_file() and actual.stat().st_size==f['size_bytes'] and D.sha(actual)==f['sha256'],'Model source changed')
        rows.append([f['path'],f['size_bytes'],f['sha256']])
    tree=hashlib.sha256(json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    need(len(rows)==14 and sum(r[1] for r in rows)==expected['bytes_verified'] and tree==expected['tree_sha256'],'Model source tree changed')
    return dict(commit=pin['commit'],files_verified=len(rows),tree_sha256=tree,bytes_verified=sum(r[1] for r in rows))

def entries(c):
    lookup={}
    for m in METHODS:
        binding=c['checkpoint_binding'][m+'/seed0'];p=Path(binding['data_manifest'])
        need(D.sha(p)==binding['data_manifest_sha256'],'Frozen synthetic manifest changed')
        records=json.loads(p.read_text())['entries'];rows={r['rollout_id']:r for r in records if r['split']=='validation'}
        need(len(rows)==8 and len(records)==32,'Original24train/8val split differs')
        need({st:sum(r['stratum']==st for r in rows.values()) for st in STRATA}=={st:2 for st in STRATA},'Four equal validation strata required')
        lookup[m]=rows
        for seed in (0,1,2):
            b=c['checkpoint_binding'][f'{m}/seed{seed}']
            need(b['data_manifest']==str(p) and b['data_manifest_sha256']==binding['data_manifest_sha256'],'Seed-specific data differs')
    ids=set(lookup['eventgenesis'])
    for m in METHODS:need(set(lookup[m])==ids,'Generator-specific validation rollout IDs')
    for rid in ids:need(len({lookup[m][rid]['stratum'] for m in METHODS})==1,'Generator-specific stratum')
    return lookup

def best_history(binding,update):
    p=Path(binding['run'])/'training_metrics.jsonl';matches=[]
    with p.open() as f:
        for line in f:
            if '"synthetic_validation"' not in line:continue
            row=json.loads(line)
            if row.get('optimizer_updates')==update:matches.append(row)
    need(len(matches)==1,'Selected checkpoint must have exactly one original full validation row')
    expected=float(matches[0]['model_mean_epe_px_per_30ms']);need(math.isfinite(expected),'Nonfinite original validation')
    return dict(path=str(p),sha256=D.sha(p),optimizer_update=update,epe_px=expected)

def available_ram():
    rows=Path('/proc/meminfo').read_text().splitlines()
    return int(next(r.split()[1] for r in rows if r.startswith('MemAvailable:')))*1024

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--methods',nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    c=json.loads(a.contract.read_text());shard=validate(c,a.seeds,a.methods,a.out.resolve(),os.environ.get('CUDA_VISIBLE_DEVICES'))
    need(not a.out.exists(),'Refusing to overwrite evaluation')
    from singlecell_contract import fallback_admission
    fallback_proof=fallback_admission(c)
    need(a.contract.resolve()==ROOT/'sources/contract.json','Only root-bound contract accepted')
    manifest=json.loads((ROOT/'sources/source_manifest.json').read_text())
    for name,h in manifest.items():need(Path(name).name==name and D.sha(ROOT/'sources'/name)==h,'Changed evaluation source: '+name)
    for name,record in c['sources'].items():need(D.sha(record['path'])==record['sha256'],'Frozen training source changed: '+name)
    need(D.sha(c['campaign']+'/sources/training_recipe.json')==c['recipe_sha256'],'Frozen recipe changed')
    recipe=json.loads(Path(c['campaign']+'/sources/training_recipe.json').read_text())
    need(recipe['frozen_simulator_core']==c['frozen_simulator_core'] and recipe['flow_dt1']['expected_optimizer_updates']==8910,'Core/budget changed')
    lookup=entries(c);ram=available_ram();need(ram>=8*1024**3,'Require at least8GiB available RAM per worker')
    import torch
    need(torch.cuda.is_available() and torch.cuda.device_count()==1,'Only one leased CUDA device required')
    need('RTX 3090' in torch.cuda.get_device_name(0),'Registered GPU hardware must be3090')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    trainer=D.load_module(c['sources']['trainer']['path'],'frozen_synthetic_eval_trainer')
    tree=source_tree(Path(c['model_root']),Path(c['campaign'])/'sources/downstream_source_bundle.json',c['model_source'])
    sys.path.insert(0,c['model_root']);import models,models.corr,multiscaleloss
    models.corr.spatial_correlation_sample=trainer.pure_torch_spatial_correlation
    trainer.patch_ste(models,multiscaleloss,Path(c['model_root']))
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+1200;results=[]
    D.write(a.out/'admission.json',dict(shard=shard,seeds=a.seeds,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],gpu_name=torch.cuda.get_device_name(0),ram_available_bytes=ram,source_manifest_sha256=D.sha(ROOT/'sources/source_manifest.json'),model_source=tree,optimizer_updates=0,real_data_used=False))
    try:
        for seed in a.seeds:
            for method in a.methods:
                need(time.monotonic()<deadline,'Twenty-minute singlecell budget exceeded')
                b=c['checkpoint_binding'][f'{method}/seed{seed}'];cp_path=Path(b['selected_checkpoint'])
                need(D.sha(cp_path)==b['selected_checkpoint_sha256'] and cp_path.stat().st_size==b['selected_checkpoint_size_bytes'],'Selected checkpoint changed')
                need(D.sha(b['completion'])==b['completion_sha256'],'Completed formal run receipt changed')
                cp=torch.load(cp_path,map_location='cpu',weights_only=False)
                for key,value in [('generator',method),('seed',seed),('config_sha256',c['recipe_sha256']),('data_manifest_sha256',b['data_manifest_sha256']),('initial_state_sha256',b['initial_state_sha256'])]:need(cp.get(key)==value,'Selected checkpoint metadata differs: '+key)
                need(cp['real_data_used'] is False and 1<=cp['optimizer_update']<=8910,'Checkpoint must be synthetic only')
                history=best_history(b,cp['optimizer_update']);model=models.steflow().cuda();model.load_state_dict(cp['state_dict'],strict=True)
                need(trainer.state_sha256(model)==cp['current_state_sha256'],'Loaded model state differs');del cp;model.eval();torch.cuda.reset_peak_memory_stats()
                rolls=[];episode_bindings=[]
                for rid in sorted(lookup[method]):
                    need(time.monotonic()<deadline and available_ram()>=2*1024**3,'Resource/deadline limit reached')
                    eg=D.load_episode(lookup['eventgenesis'][rid]);v2=D.load_episode(lookup['v2e'][rid]);D.pair_equal(eg,v2)
                    data=eg if method=='eventgenesis' else v2 if method=='v2e' else D.load_episode(lookup[method][rid])
                    D.pair_equal(eg,data);need(np.isfinite(data['events']).all() and np.min(data['events'])>=0,'Invalid nonnegative event count input')
                    episode_bindings.append(dict(rollout=rid,method_npz_sha256=lookup[method][rid]['sha256'],common_event_sources={m:lookup[m][rid]['sha256'] for m in ('eventgenesis','v2e')},physical_labels_sha256={k:D.array_sha(eg[k]) for k in ('flow_gt','flow_valid','sample_ids','event_start_s','event_end_s')}))
                    windows=[]
                    for i in range(66):
                        need(time.monotonic()<deadline,'Twenty-minute singlecell budget exceeded')
                        with torch.inference_mode():
                            event,_,_=trainer.val_input(data,i,torch.device('cuda:0'))
                            pred=trainer.model_output_to_flow_hw(model(event,352,use_dt4=False,flow_seq=False))[:,46:306,3:349]
                        need(pred.shape==(2,260,346) and np.isfinite(pred).all(),'Nonfinite/wrong native prediction')
                        gt=data['flow_gt'][i];valid=data['flow_valid'][i];common=D.active(eg['events'][i])&D.active(v2['events'][i])
                        regions=D.region_summary(pred,gt,valid,common);zero=D.region_summary(np.zeros_like(pred),gt,valid,common)
                        for name,row in regions.items():
                            if name!='invalid_geometry_or_visibility':row['zero_epe_px']=zero[name]['epe_px']
                        epe,_=trainer.compute_epe(pred,gt,valid,prediction_unit='px_per_interval',gt_unit='px_per_interval')
                        need(abs(epe-regions['all_physical_valid']['epe_px'])<=1e-6,'Native metric consistency failed');regions['all_physical_valid']['epe_px']=epe
                        windows.append(dict(index=i,sample_id=str(data['sample_ids'][i]),common_active_fraction=float(common.mean()),regions=regions))
                    rolls.append(dict(rollout=rid,stratum=lookup[method][rid]['stratum'],windows=windows,regions=aggregate([w['regions'] for w in windows])))
                    print('SYNTHETIC_READONLY',method,seed,rid,flush=True);del eg,v2,data;gc.collect()
                regions=aggregate([r['regions'] for r in rolls]);actual=regions['all_physical_valid']['epe_px']
                need(abs(actual-history['epe_px'])<=1e-5,'Full528 synthetic_best parity failed')
                need(D.sha(cp_path)==b['selected_checkpoint_sha256'],'Original selected checkpoint changed during evaluation')
                result=dict(method=method,seed=seed,checkpoint_sha256=b['selected_checkpoint_sha256'],checkpoint_path=str(cp_path),window_count=528,rollout_count=8,regions=regions,strata={st:aggregate([r['regions'] for r in rolls if r['stratum']==st]) for st in STRATA},rollouts=rolls,episode_bindings=episode_bindings,original_history=history,parity=dict(passed=True,current_epe_px=actual,original_epe_px=history['epe_px'],absolute_tolerance=1e-5),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),optimizer_updates=0,real_data_used=False)
                D.write(a.out/f'{method}_seed{seed}.json',result);results.append(dict(method=method,seed=seed,sha256=D.sha(a.out/f'{method}_seed{seed}.json'),regions=regions))
                del model;gc.collect();torch.cuda.empty_cache()
        D.write(a.out/'completion.json',dict(schema='flow_full_synthetic_readonly_cell_result_v1',fallback_admission=fallback_proof,status='complete',shard=shard,seeds=a.seeds,cells=len(results),windows_per_checkpoint=528,results=results,elapsed_seconds=time.monotonic()-started,source_manifest_sha256=D.sha(ROOT/'sources/source_manifest.json'),contract_sha256=D.sha(a.contract),optimizer_updates=0,real_data_used=False,real_scores_used=False,checkpoint_selection_changed=False))
    except BaseException:
        if not (a.out/'failure.json').exists():D.write(a.out/'failure.json',dict(status='failed',error=traceback.format_exc(),optimizer_updates=0,real_data_used=False))
        raise

if __name__=='__main__':main()
