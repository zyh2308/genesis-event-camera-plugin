"""Run bounded CPU rendering workers and freeze a complete physical bank."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--kind',choices=['rich','pose'],required=True)
args=p.parse_args();campaign=args.campaign.resolve();P=Path('/home/zhaoyuhan/Eventbased_WAM')
R=campaign/(args.kind+'_bank');R.mkdir(exist_ok=False)
(R/'sources').symlink_to(campaign/'sources',target_is_directory=True)
(R/'workers').mkdir();(R/'cache').mkdir()
assert os.cpu_count()>=32
assert os.getloadavg()[0] < .7*os.cpu_count()
available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
assert available>80*1024**3
workers=8
private_lib=P/'runs/sim2real/20261101-epose-diverse-bank-v8/runtime/osmesa/usr/lib/x86_64-linux-gnu'
base=os.environ.copy();base.update(CUDA_VISIBLE_DEVICES='',PYOPENGL_PLATFORM='osmesa',GALLIUM_DRIVER='llvmpipe',LIBGL_ALWAYS_SOFTWARE='1',MESA_GL_VERSION_OVERRIDE='4.5',
    LD_LIBRARY_PATH=str(private_lib),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',LP_NUM_THREADS='1',QD_NUM_THREADS='2',QD_OFFLINE_CACHE='1',PYTHONDONTWRITEBYTECODE='1')
script=campaign/'sources'/('rich_physical_worker.py' if args.kind=='rich' else 'pose_physical_worker.py')
processes=[];receipt={'kind':args.kind,'phase':'native_physical_generation','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'cpu_workers':workers,'threads_per_worker':2,'source_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'worker_processes':[],'real_data_used':False}
for slot in range(workers):
    env=base.copy();cache=R/'cache'/str(slot);cache.mkdir()
    env.update(QD_OFFLINE_CACHE_FILE_PATH=str(cache/'quad'),GS_CACHE_FILE_PATH=str(cache/'genesis'))
    log=(R/'workers'/f'worker{slot}.log').open('ab',buffering=0)
    child=subprocess.Popen([sys.executable,'-u',str(script),'--campaign',str(R),'--slot',str(slot)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,close_fds=True)
    log.close();processes.append(child)
    receipt['worker_processes'].append({'slot':slot,'pid':child.pid,'log':str(R/'workers'/f'worker{slot}.log')})
(R/'processes.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
while any(c.poll() is None for c in processes):
    status=[{'slot':i,'pid':c.pid,'exit_code':c.poll()} for i,c in enumerate(processes)]
    (R/'live.json').write_text(json.dumps({'phase':'native_physical_generation','workers':status,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
    time.sleep(20)
codes=[c.returncode for c in processes]
(R/'worker_exits.json').write_text(json.dumps(codes)+'\n')
assert all(c==0 for c in codes), 'Rendering worker failures: '+repr(codes)
import h5py
import numpy as np
files=sorted((R/'physical').glob('*.h5'))
expected=32 if args.kind=='rich' else 128
assert len(files)==expected,(len(files),expected)
rows=[];train=[];validation=[]
with h5py.File(R/'shared_trajectory.h5.partial','w') as merged:
    episodes=merged.create_group('episodes')
    for file in files:
        with h5py.File(file,'r') as h:
            assert len(h['episodes'])==1
            rid=next(iter(h['episodes']));g=h['episodes'][rid]
            times=np.asarray(g['frame_timestamps_s']);assert np.allclose(np.diff(times),.01,rtol=0,atol=1e-7)
            expected_frames=201 if args.kind=='rich' else 21
            assert len(times)==expected_frames
            assert g['radiance_linear_rgb'].dtype==np.float32 and g['radiance_linear_rgb'].shape==(expected_frames,260,346,3)
            assert g['depth_m'].shape==g['segmentation_link_idx'].shape==(expected_frames,260,346)
            split=str(g.attrs['split']);assert split in ['train','validation']
            (train if split=='train' else validation).append(rid)
            episodes[rid]=h5py.ExternalLink(str(file),'/episodes/'+rid)
            info={'rollout_id':rid,'split':split,'path':str(file),'bytes':file.stat().st_size,'frames':expected_frames}
            if args.kind=='rich':info['scene_id']=str(g.attrs['scene_id'])
        digest=hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024**2),b''):digest.update(chunk)
        info['sha256']=digest.hexdigest();rows.append(info)
    merged.attrs.update(schema='stage9_shared_physical_cache_v1',endpoint_hz=100,real_data_used=False,source_geometry_core_independent=True)
(R/'shared_trajectory.h5.partial').rename(R/'shared_trajectory.h5')
assert len(train)==(24 if args.kind=='rich' else 96) and len(validation)==(8 if args.kind=='rich' else 32)
assert not set(train)&set(validation)
if args.kind=='rich':
    assert not {r['scene_id'] for r in rows if r['split']=='train'} & {r['scene_id'] for r in rows if r['split']=='validation'}
split={'unit':'complete furnished layout' if args.kind=='rich' else 'complete independent synthetic camera/object rollout','train':train,'validation':validation}
(R/'registered_split.json').write_text(json.dumps(split,indent=2)+'\n')
manifest={'schema':'stage9_native_physical_bank_v1','kind':args.kind,'rollouts':rows,'shared_trajectory':str(R/'shared_trajectory.h5'),
    'shared_trajectory_sha256':hashlib.sha256((R/'shared_trajectory.h5').read_bytes()).hexdigest(),
    'registered_split_sha256':hashlib.sha256((R/'registered_split.json').read_bytes()).hexdigest(),
    'real_data_used':False,'generator_event_count_screening':False,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(R/'bank_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'complete','kind':args.kind,'rollouts':len(rows),'train':len(train),'validation':len(validation),'manifest':str(R/'bank_manifest.json')}),flush=True)
