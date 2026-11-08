"""Strict runtime-only admission for the existing owned GPU3 queue."""
from __future__ import annotations
import hashlib,json,os,subprocess,time
from pathlib import Path
GPU_UUID='GPU-253ab9b8-a0e5-4f12-0579-261818cf31dd'
GPU_NAME='Tesla V100S-PCIE-32GB'
PARENT_PID=335161
SCIENCE_SHA='d2860889e582223bf248d7b8e8e6fcc8f80aa71041a371243fdae06010e312ff'


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024**2),b''):h.update(data)
    return h.hexdigest()


def record(path):
    p=Path(path);return dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha(p))


def validate_contract(c,scientific_contract_sha256):
    need(c['schema']=='pose_aux_mask_runtime_resource_v1' and c['scientific_contract_sha256']==scientific_contract_sha256==SCIENCE_SHA,'Unregistered science/resource contract')
    need(c['gpu_index']==3 and c['gpu_uuid']==GPU_UUID and c['gpu_name']==GPU_NAME and c['parent_pid']==PARENT_PID,'Unregistered GPU/parent')
    ticks=c['parent_start_ticks'];need((isinstance(ticks,int) and not isinstance(ticks,bool) and ticks>0) or (isinstance(ticks,str) and ticks.isdigit() and str(int(ticks))==ticks and int(ticks)>0),'Exact canonical parent start ticks required')
    argv=c['parent_cmdline'];script=c['parent_script_path']
    need(isinstance(argv,list) and len(argv)>=6 and all(isinstance(v,str) and v for v in argv),'Exact parent argv required')
    need(Path(script).is_absolute() and Path(script).name=='server_queue.py' and argv.count(script)==1,'Original server_queue script path required')
    need(argv[-4:]==[c['campaign'],'3',GPU_UUID,'5'],'Parent queue campaign/index/UUID/poll arguments differ')
    need(len(c['parent_script_sha256'])==64 and all(x in '0123456789abcdef' for x in c['parent_script_sha256']),'Exact parent script SHA required')


def parse_proc_stat(value):
    # comm may contain whitespace or parentheses; fields after its final ')' start at state(field3).
    end=value.rfind(')');parts=value[end+1:].split();need(end>0 and len(parts)>19,'Invalid Linux parent stat')
    return dict(state=parts[0],start_ticks=int(parts[19]))


def verify_snapshot(c,s,current_pid):
    """Pure CPU validator. No mutation, GPU calls, or permissive missing fields."""
    need(s['cuda_visible_devices']==GPU_UUID,'CUDA_VISIBLE_DEVICES must bind the exact GPU3 UUID')
    need(s['gpu_index']==3 and s['gpu_uuid']==c['gpu_uuid']==GPU_UUID and s['gpu_name']==c['gpu_name']==GPU_NAME,'Physical GPU identity differs')
    need(s['cuda_device_count']==1 and s['torch_gpu_name']==GPU_NAME,'Exactly one expected CUDA device required')
    need(s['parent_pid']==PARENT_PID and s['parent_start_ticks']==int(c['parent_start_ticks']),'Queue parent PID was recycled')
    need(s['parent_state'] not in ('Z','X') and s['parent_cmdline']==c['parent_cmdline'],'Queue parent process identity differs')
    need(s['parent_script_sha256']==c['parent_script_sha256'],'Original queue source changed')
    need(set(s['active_gpu_pids'])<={PARENT_PID,current_pid},'Another process occupies the assigned GPU')
    return dict(passed=True,gpu_index=3,gpu_uuid=GPU_UUID,gpu_name=GPU_NAME,queue_parent_pid=PARENT_PID,queue_parent_start_ticks=int(c['parent_start_ticks']),
        queue_parent_script_sha256=c['parent_script_sha256'],active_gpu_pids=sorted(s['active_gpu_pids']),worker_pid=current_pid,cuda_device_count=1,cuda_visible_devices=GPU_UUID)


def check_resource(path,digest,scientific_contract_sha256):
    need(sha(path)==digest,'Root-pinned resource contract bytes differ');c=json.loads(Path(path).read_text());validate_contract(c,scientific_contract_sha256)
    import torch
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID,'Wrong CUDA UUID before initialization')
    physical=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=uuid,name','--format=csv,noheader'],text=True,timeout=15).strip().split(',')
    need(len(physical)==2,'Ambiguous physical GPU query')
    need([x.strip() for x in physical]==[GPU_UUID,GPU_NAME],'Wrong physical GPU before CUDA initialization')
    parent=Path('/proc')/str(PARENT_PID);stat=parse_proc_stat((parent/'stat').read_text());argv=[x.decode() for x in (parent/'cmdline').read_bytes().split(b'\0') if x]
    need(stat['start_ticks']==int(c['parent_start_ticks']) and stat['state'] not in ('Z','X') and argv==c['parent_cmdline'] and sha(c['parent_script_path'])==c['parent_script_sha256'],'Wrong queue parent before CUDA initialization')
    count=torch.cuda.device_count();need(count==1,'CUDA visibility is not exclusive')
    torch_name=torch.cuda.get_device_name(0)
    active=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    pids=[]
    for line in active.splitlines():
        line=line.strip();need(line.isdigit(),'Unrecognized active GPU PID record');pids.append(int(line))
    snapshot=dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_index=3,gpu_uuid=physical[0].strip(),gpu_name=physical[1].strip(),cuda_device_count=count,torch_gpu_name=torch_name,
        parent_pid=PARENT_PID,parent_start_ticks=stat['start_ticks'],parent_state=stat['state'],parent_cmdline=argv,parent_script_sha256=sha(c['parent_script_path']),active_gpu_pids=pids)
    result=verify_snapshot(c,snapshot,os.getpid());result.update(resource_contract=record(path),parent_script=record(c['parent_script_path']),observed_unix_seconds=time.time(),original_frozen_source_gpu_uuid='GPU-0602549d-5c0b-133e-63ab-a01d1f6a49d3',scope='runtime_device_substitution_only_science_unchanged')
    return result


def verify_reference_baseline(ctx,E,caches,reference,forward_completion):
    """Zero-update original C1 parity on this device before the first smoke update."""
    import gc,numpy as np,torch
    from scipy.spatial import cKDTree
    rec=next(r for r in forward_completion['artifacts'] if Path(r['path']).name=='native_scores.json')
    E.remember(ctx.source_records,rec['path'],rec['sha256']);scores={(r['method'],r['seed'],r['condition']):r['score'] for r in json.loads(Path(rec['path']).read_text()) if r['arm']=='C1'}
    need(len(scores)==18,'All18 original C1 native scores required');tree=cKDTree(ctx.vertices);checks=[]
    for method in ('eventgenesis','v2e'):
        item=next(r for r in caches if Path(r['path']).name==f'{method}_RAW_inputs_masks128.pt');need(sha(item['path'])==item['sha256'],'Prepared RAW cache changed before baseline')
        data=torch.load(item['path'],map_location='cpu',weights_only=False)['data']
        for seed in range(3):
            center,_,_=E.load_branch(ctx,method,seed,'physical_supervised','center');depth,_,_=E.load_branch(ctx,method,seed,'physical_supervised','depth')
            parent=next(r for r in ctx.completion['results'] if (r['method'],r['seed'])==(method,seed));rp=ctx.A/f'jobs/117_pose_rotation_paired_20261106/results/{method}_seed{seed}_rotation.pt'
            E.remember(ctx.source_records,rp,parent['parents']['rotation_sha256']);head=ctx.rotation.RotationHead().to(ctx.device).eval();head.load_state_dict(torch.load(rp,map_location='cpu',weights_only=True)['model'],strict=True)
            models=[(m,E.state_sha(m.state_dict())) for m in (center,depth,head)]
            for view,condition in (('matched_normal','normal'),('half','half'),('eighth','eighth')):
                v=data['views'][view];rows=v['rows'];images=v['raw'].to(ctx.device);K=[r['K'] for r in rows];need(len(rows)==64,'Baseline fixed64 windows missing')
                with torch.inference_mode():uv=ctx.native.estimate_centers(center,images);d=E.predict_raw(depth,images,ctx.scale,K)
                maxima={name:0. for name in ('center_uv','depth_uv','log_size','log_shape','z','size')}
                for i,row in enumerate(rows):
                    old=reference[(method,seed,condition,row['id'])]
                    for name,value,expected in [('center_uv',uv[i],old['pred_uv']),('depth_uv',d['uv'][i],old['depth_model_uv']),('log_size',d['log_size'][i],old['log_size']),('log_shape',d['log_shape'][i],old['log_shape']),('z',d['z'][i],old['z_m']),('size',d['size'][i],old['size'])]:
                        error=float(np.max(abs(np.asarray(value)-expected)));need(np.isfinite(error) and error<=2e-6,'GPU3 original C1 reference parity failed: '+name);maxima[name]=max(maxima[name],error)
                R=ctx.shared.frozen_rotation(ctx.rotation,head,images,uv,d['size'],K);rotation_error=float(np.max(abs(R-np.stack([reference[(method,seed,condition,r['id'])]['R_pred'] for r in rows]))))
                need(np.isfinite(rotation_error) and rotation_error<=2e-6,'GPU3 original C1 rotation parity failed')
                score=ctx.shared.score(ctx.rotation,rows,uv,d['z'],d['size'],v['y'][:,2].numpy(),R,ctx.vertices,tree,ctx.threshold);oldscore=scores[(method,seed,condition)]
                E.compare_summary(score['summary'],oldscore['summary']);need(score['by_rollout'].keys()==oldscore['by_rollout'].keys(),'Baseline native rollout inventory differs')
                for rid in score['by_rollout']:E.compare_summary(score['by_rollout'][rid],oldscore['by_rollout'][rid])
                checks.append(dict(method=method,seed=seed,condition=condition,windows=64,max_abs=maxima,rotation_max_abs=rotation_error,native_summary_and_by_rollout_parity=True,passed=True));del images
            need(all(E.state_sha(m.state_dict())==s for m,s in models),'Frozen baseline model state changed');del models,center,depth,head;gc.collect();torch.cuda.empty_cache()
        del data;gc.collect()
    need(len(checks)==18,'Incomplete original C1 numerical baseline')
    return dict(status='passed',optimizer_updates=0,real_data_used=False,execution_gpu_uuid=GPU_UUID,original_source_gpu_uuid='GPU-0602549d-5c0b-133e-63ab-a01d1f6a49d3',checks=checks,prediction_windows=1152,all_models_unchanged=True)
