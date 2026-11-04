"""Additional baseline controls; not mixed into the primary matched suite."""
import argparse,gc,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent/'release_candidate_static'))
from genesis_event_plugin.optimized import OptimizedGenesisPhysicsEventPlugin
import compare as c
c.Resident=OptimizedGenesisPhysicsEventPlugin
ROOT=Path(__file__).resolve().parent
out=ROOT/'controls';out.mkdir(exist_ok=False)
f=c.Fixture('s2_00',20)
# More precise original 25us setting, retaining native official interpolation.
slomo=c.new_slomo(2);c.run_v2e(c.Fixture('s2_00',16),slomo)
slomo.auto_upsample=True
original_interpolate=c.interpolate
def auto_interpolate(model,fixture,memory=True,folder=None):
    cache={};folder=folder or str(ROOT/'virtual_auto_png')
    with c.memory_images(model,fixture,cache):relative,factor=model.interpolate(str(fixture.input_dir),str(folder),(fixture.W,fixture.H))
    intervals=np.diff(np.r_[relative,len(fixture.ts)-1])
    timestamps=(relative+.5*intervals)*.01
    images=[cache[k] for k in sorted(cache)]
    assert len(images)==len(timestamps) and np.all(np.diff(timestamps)>0)
    return images,timestamps,float(factor)
c.interpolate=auto_interpolate;auto_rows=[]
try:
    for trial in range(3):
        row,events=c.run_v2e(f,slomo);row.update(trial=trial,substeps=None,matched_temporal_sampling=False);auto_rows.append(row)
finally:c.interpolate=original_interpolate
(out/'auto_sampling.json').write_text(json.dumps(dict(rows=auto_rows,scope='native automatic flow-based upsampling; different temporal density; not included in matched speedup',timestamp_assignment='midpoints reconstructed from native per-batch returned intervals'),indent=2)+'\n')
slomo.auto_upsample=False
slomo.upsampling_factor=400;slomo.batch_size=8
rows=[]
for trial in range(3):
    row,events=c.run_v2e(f,slomo);row.update(trial=trial,factor=400);rows.append(row)
(out/'factor400.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
del slomo;gc.collect();torch.cuda.empty_cache()
# Common numeric photometric input, native FP64 / FP32 sensor implementations.
from genesis_event_plugin.dvs_emulator import DvsEmulator
from genesis_event_plugin.noise_model import DvsNoiseConfig
from resident2 import sensor_device
base=np.linspace(32,128,f.H*f.W,dtype=np.float32).reshape(f.H,f.W)
frames=[np.clip(base*(1+.6*np.sin((j+1)/377*3)),32,224).astype(np.float32) for j in range(377)]
times=np.arange(1,378,dtype=np.float64)*.01/377
sensor_rows=[]
for trial in range(-1,3):
    variants=['ours','v2e'] if trial%2==0 else ['v2e','ours']
    for variant in variants:
        if variant=='ours':
            e=DvsEmulator(res=(f.H,f.W),config=DvsNoiseConfig(**c.PROFILE['noise_config_kwargs']),device='cuda:0');e.initialize(base/255,0.)
        else:
            e=c.emulator(f);e.generate_events(base,0.)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        if variant=='ours':chunks=sensor_device(e,torch.as_tensor(np.stack(frames)/255,device='cuda:0'),list(times))
        else:
            with torch.no_grad():chunks=[e.generate_events(image,float(t)) for image,t in zip(frames,times)]
        chunks=[x for x in chunks if x is not None and len(x)]
        events=np.concatenate(chunks) if chunks else np.empty((0,4));events=events[np.argsort(events[:,0],kind='stable')]
        torch.cuda.synchronize();wall=time.perf_counter()-begin
        if trial>=0:sensor_rows.append(dict(variant=variant,trial=trial,wall_s=wall,events=len(events),dtype=str(events.dtype),peak_bytes=torch.cuda.max_memory_allocated()))
        if variant=='v2e':e.cleanup()
        del e;gc.collect();torch.cuda.empty_cache()
(out/'sensor_only.json').write_text(json.dumps(dict(rows=sensor_rows,source='same deterministic DN32..224 brightness fields; EventGenesis linear input=DN/255',
    full_noise_retained=True,native_precision_retained=True,subframes=377,simulated_s=.01),indent=2)+'\n')
# Longer clip permits the documented batch16 baseline, avoiding a batch8-only claim.
long_fixture=c.Fixture('s2_00',40);slomo=c.new_slomo(2,16)
c.run_v2e(c.Fixture('s2_00',32),slomo);slomo.upsampling_factor=377;slomo.batch_size=16
long_rows=[]
for trial in range(3):
    variants=['ours_resident','v2e_ssm_memory'] if trial%2==0 else ['v2e_ssm_memory','ours_resident']
    for variant in variants:
        row,events=c.run_ours(long_fixture,variant) if variant=='ours_resident' else c.run_v2e(long_fixture,slomo)
        row.update(trial=trial,batch_size=16 if variant=='v2e_ssm_memory' else None,simulated_s=float(long_fixture.ts[-1]));long_rows.append(row)
        (out/'batch16_long.json').write_text(json.dumps(dict(rows=long_rows,completed=False),indent=2)+'\n')
(out/'batch16_long.json').write_text(json.dumps(dict(rows=long_rows,completed=True),indent=2)+'\n')
del slomo;gc.collect();torch.cuda.empty_cache()
# Native stock file-based CLI, including startup/model init; no preview/video.
folder=out/'input_png';folder.mkdir()
for j,image in enumerate(f.images.values()):Image.fromarray(image).save(folder/(str(j).zfill(6)+'.png'))
cmd=[sys.executable,str(c.SRC/'v2e.py'),'--input',str(folder),'--input_frame_rate','100',
    '--auto_timestamp_resolution','False','--timestamp_resolution',str(.01/377),'--slomo_model',str(c.WEIGHT),
    '--batch_size','8','--output_folder',str(out/'stock_cli'),'--unique_output_folder','False',
    '--skip_video_output','--no_preview','--dvs_h5',str(out/'stock_cli_events.h5'),
    '--dvs_aedat2','None','--dvs_aedat4','None','--dvs_text','None',
    '--pos_thres','.2','--neg_thres','.2','--sigma_thres','.03','--cutoff_hz','300',
    '--leak_rate_hz','.01','--leak_jitter_fraction','.1','--noise_rate_cov_decades','.1',
    '--shot_noise_rate_hz','.001','--refractory_period','.0005','--dvs_emulator_seed','42']
begin=time.perf_counter()
with (out/'stock_cli.log').open('x') as log:r=subprocess.run(cmd,cwd=c.SRC,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,timeout=1200)
result=dict(command=cmd,exit_code=r.returncode,wall_s=time.perf_counter()-begin,nominal_physical_duration_s=.2,used_in_primary_comparison=False)
if r.returncode==0:
    import h5py
    with h5py.File(out/'stock_cli_events.h5','r') as h:
        result['datasets']={}
        def visit(name,obj):
            if isinstance(obj,h5py.Dataset):
                d=dict(shape=list(obj.shape),dtype=str(obj.dtype))
                if obj.ndim==2 and obj.shape[1]>=4 and len(obj):d.update(first=list(map(float,obj[0])),last=list(map(float,obj[-1])))
                result['datasets'][name]=d
        h.visititems(visit)
(out/'stock_cli_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
assert r.returncode==0
print('CONTROLS_COMPLETE',flush=True)
