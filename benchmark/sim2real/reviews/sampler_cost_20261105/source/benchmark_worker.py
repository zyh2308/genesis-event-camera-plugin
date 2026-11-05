"""Uninstrumented actual-entry benchmark; one immutable package per process."""
import argparse, atexit, gc, hashlib, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--core',required=True);p.add_argument('--out',required=True);p.add_argument('--method',choices=['frozen_gpu','candidate_gpu','v2e_auto','v2e_auto_min63'],required=True);p.add_argument('--case',required=True);p.add_argument('--intervals',type=int,default=20);p.add_argument('--trials',type=int,default=3)
a=p.parse_args();E=Path(__file__).resolve().parent;V=E.parent/'simulator_efficiency_comparison_20261105c'
out=Path(a.out);out.mkdir(parents=True,exist_ok=False);core=Path(a.core).resolve()
source=(V/'compare_static.py').read_text()
assert source.count("CORE=ROOT/'release_candidate_static'")==1
source=source.replace("CORE=ROOT/'release_candidate_static'",'CORE=Path('+repr(str(core))+')')
helper=out/'static_benchmark_fixture.py';helper.write_text(source)
spec=importlib.util.spec_from_file_location('actual_entry_benchmark',helper)
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
assert not torch.are_deterministic_algorithms_enabled(), 'Normal-mode timing only; exact gates are separate'
c.ROOT=out

def auto_interpolate(model,fixture,memory=True,folder=None):
    cache={}
    with c.memory_images(model,fixture,cache):
        relative,factor=model.interpolate(str(fixture.input_dir),str(out/'virtual_png'),(fixture.W,fixture.H))
    relative=np.asarray(relative,dtype=np.float64)
    edges=np.r_[relative,len(fixture.ts)-1];mids=relative+.5*np.diff(edges)
    times=np.interp(mids,np.arange(len(fixture.ts)),fixture.ts)
    images=[cache[k] for k in sorted(cache)]
    assert len(images)==len(times) and np.all(np.diff(times)>0)
    return images,times,float(factor)

c.interpolate=auto_interpolate
fixture=c.Fixture(a.case,a.intervals)
(out/'physical_inputs.json').write_text(json.dumps(fixture.description,indent=2)+'\n')
slomo=None
if a.method.startswith('v2e'):
    slomo=c.new_slomo(2 if a.method=='v2e_auto' else 63,8);slomo.auto_upsample=True
    warm,_=c.run_v2e(c.Fixture(a.case,16),slomo)
    assert slomo.upsampling_factor==(2 if a.method=='v2e_auto' else 63)
else:
    warm,_=c.run_ours(c.Fixture(a.case,2),'ours_resident')
(out/'warmup.json').write_text(json.dumps(warm,indent=2)+'\n')
rows=[]
for trial in range(a.trials):
    gc.collect();torch.cuda.empty_cache()
    row,events=c.run_v2e(fixture,slomo) if slomo is not None else c.run_ours(fixture,'ours_resident')
    if slomo is None:assert row['fallbacks']==0 and set(row['substeps'])=={377}
    row.update(method=a.method,trial=trial,case=a.case,core=str(core),simulated_s=float(fixture.ts[-1]),
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_name=torch.cuda.get_device_name(),
        deterministic_algorithms=False,full_stage9_noise=True,same_temporal_density_claimed=False,
        timing_boundary='cached CPU physical/display input to sorted CPU raw events; setup, input disk load and serialization excluded')
    row['initialized_pipeline_s']=row['setup_s']+row['wall_s']
    row['primary_timing_boundary']='per-clip initialized pipeline: setup plus generation, including each sensor first frame; preloaded SSM network/private compiled kernel, physical rendering, input disk load and serialization excluded'
    if slomo is not None:
        row['minimum_factor']=int(slomo.upsampling_factor);row['substeps']=None
        row['sampling_note']='official automatic SuperSloMo; inherited 300Hz profile; no equal-quality assertion'
    np.save(out/('trial'+str(trial)+'_events.npy'),events,allow_pickle=False)
    support=(events[:,0]>=.02)&(events[:,0]<=.18)
    row['shared_interior_support']={'start_s':.02,'end_s':.18,'events':int(support.sum()),'sha256':hashlib.sha256(np.ascontiguousarray(events[support]).tobytes()).hexdigest(),'not_a_quality_score':True}
    rows.append(row)
    (out/'timings.json').write_text(json.dumps(dict(rows=rows,status='running'),indent=2)+'\n')
    print('TIMING',json.dumps(row),flush=True)
if slomo is not None:
    slomo.cleanup();atexit.unregister(slomo.cleanup)
(out/'timings.json').write_text(json.dumps(dict(rows=rows,status='complete',core_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (core/'genesis_event_plugin').glob('gpu_*.py')}),indent=2)+'\n')
print('BENCHMARK_COMPLETE',flush=True)
