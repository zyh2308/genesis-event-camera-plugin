"""Both native adaptive samplers, unchanged Stage9 noise and accuracy settings."""
import atexit, gc, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

W=Path(__file__).resolve().parent
V=W.parent/'simulator_efficiency_comparison_20261105c'
spec=importlib.util.spec_from_file_location('accepted_static_comparison',V/'compare_static.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
c.ROOT=W/'native_auto';c.ROOT.mkdir(exist_ok=False)
OUT=c.ROOT
old_interpolate=c.interpolate

def auto_interpolate(model,fixture,memory=True,folder=None):
    cache={}
    with c.memory_images(model,fixture,cache):
        relative,factor=model.interpolate(str(fixture.input_dir),str(OUT/'virtual_png'),(fixture.W,fixture.H))
    relative=np.asarray(relative,dtype=np.float64)
    # Preserve official midpoint samples. Native auto can change factor per batch;
    # get_interpolated_timestamps is not used because it assumes a fixed factor.
    edges=np.r_[relative,len(fixture.ts)-1]
    mids=relative+.5*np.diff(edges)
    timestamps=np.interp(mids,np.arange(len(fixture.ts)),fixture.ts)
    images=[cache[k] for k in sorted(cache)]
    assert len(images)==len(timestamps) and np.all(np.diff(timestamps)>0)
    assert all(x.shape==(fixture.H,fixture.W) for x in images)
    return images,timestamps,float(factor)
c.interpolate=auto_interpolate

rows=[]
for ci,case in enumerate(['s0_00','s1_00','s2_00','s3_00','train_000']):
    f=c.Fixture(case,20)
    (OUT/(case+'_input.json')).write_text(json.dumps(f.description,indent=2)+'\n')
    slomo=c.new_slomo(2,8);slomo.auto_upsample=True
    c.run_ours(c.Fixture(case,2),'ours_resident')
    c.run_v2e(c.Fixture(case,16),slomo)
    for trial in range(3):
        variants=['ours_resident','v2e_native_auto']
        if (ci+trial)%2:variants.reverse()
        for variant in variants:
            row,events=c.run_ours(f,variant) if variant=='ours_resident' else c.run_v2e(f,slomo)
            row.update(case=case,trial=trial,simulated_s=float(f.ts[-1]),sampling='native_adaptive',
                       method=variant,full_stage9_noise_unchanged=True)
            if variant=='v2e_native_auto':
                row['substeps']=None
                row['sampling_note']='official auto enabled; min factor2; 300Hz is our frozen task profile, not an assertion of suitable temporal integration'
            np.save(OUT/(case+'_'+variant+'_'+str(trial)+'.npy'),events,allow_pickle=False)
            rows.append(row)
            with (OUT/'timings.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print('AUTO_TIMING',json.dumps(row),flush=True)
    # A numerical-warning safeguard control; explicit minimum, still auto enabled.
    if case=='s2_00':
        slomo.upsampling_factor=63
        safeguarded=[]
        for trial in range(3):
            row,events=c.run_v2e(f,slomo);row.update(trial=trial,minimum_factor=63,not_matched_quality=True)
            safeguarded.append(row)
        (OUT/'v2e_auto_min63_control.json').write_text(json.dumps({'rows':safeguarded,'scope':'additional documented lowpass-step guard; not default or quality-equivalence proof'},indent=2)+'\n')
    slomo.cleanup();atexit.unregister(slomo.cleanup);del slomo;gc.collect();torch.cuda.empty_cache()
    # Generate one independent old reference on the same fixture and exact profile.
    ref,events=c.run_ours(f,'original_reference')
    np.save(OUT/(case+'_original_reference.npy'),events,allow_pickle=False)
    (OUT/(case+'_reference_receipt.json')).write_text(json.dumps(ref,indent=2)+'\n')
    print('REFERENCE_COMPLETE',case,flush=True)

summary={'status':'complete','rows':rows,'sensor_profile_sha256':hashlib.sha256((c.R/'sources/stage9_sensor_profile.json').read_bytes()).hexdigest(),
         'same_physical_inputs':True,'same_temporal_sampling_density':False,'same_quality_claimed':False,
         'generation_only_not_training':True,'real_data_used':False,
         'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_name':torch.cuda.get_device_name()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('NATIVE_AUTO_SUITE_COMPLETE',flush=True)
