"""Same-input component audit; instrumentation is confined to this test process."""
import gc, hashlib, importlib.util, json, sys, time
from pathlib import Path
import torch

E=Path(__file__).resolve().parent;V=E.parent/'simulator_efficiency_comparison_20261105c'
spec=importlib.util.spec_from_file_location('component_fixture',V/'compare_static.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
c.ROOT=E/'index_component_inputs';c.ROOT.mkdir(exist_ok=False)
torch.use_deterministic_algorithms(True)
from genesis_event_plugin import gpu_geometry as reference
original=reference.depth_aware_bidirectional_warp_projected_batch_torch
candidates={}
for mode in ['indexreuse']:
    path=E/'modules'/('geometry_'+mode.replace('+','_')+'.py')
    sp=importlib.util.spec_from_file_location('candidate_'+mode.replace('+','_').replace('-','_'),path)
    module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module);candidates[mode]=module

def exact(a,b,path=''):
    if torch.is_tensor(a):
        same=a.shape==b.shape and a.dtype==b.dtype
        same=same and bool(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))
        return [] if same else [dict(path=path,shape_a=list(a.shape),shape_b=list(b.shape),dtype=str(a.dtype))]
    if isinstance(a,dict):
        if a.keys()!=b.keys():return [dict(path=path,keys_differ=True)]
        return [x for k in a for x in exact(a[k],b[k],path+'/'+k)]
    if isinstance(a,(list,tuple)):
        if len(a)!=len(b):return [dict(path=path,lengths_differ=True)]
        return [x for i,(u,v) in enumerate(zip(a,b)) for x in exact(u,v,path+'/'+str(i))]
    return [] if a==b else [dict(path=path,value_a=a,value_b=b)]

report={'scope':'same-input component only','deterministic_algorithms':True,'rows':[],'noise_or_sampling_changed':False}
calls=0
def probe(*args,**kwargs):
    global calls
    calls+=1
    expected=original(*args,**kwargs)
    for name,module in candidates.items():
        fn=module.depth_aware_bidirectional_warp_projected_batch_torch
        value=fn(*args,**kwargs);differences=exact(expected,value)
        del value;gc.collect();torch.cuda.empty_cache()
        timing=[]
        if not differences:
            for trial in range(3):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated();start=time.perf_counter()
                value=fn(*args,**kwargs)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                timing.append({'trial':trial,'wall_s':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'incremental_peak_bytes':torch.cuda.max_memory_allocated()-baseline})
                del value
        row={'call':calls,'candidate':name,'bitwise_equal':not differences,'differences':differences,'timings':timing}
        report['rows'].append(row)
        (E/'index_component_results.json').write_text(json.dumps(report,indent=2)+'\n')
        print('COMPONENT',json.dumps(row),flush=True)
    timing=[]
    for trial in range(3):
        torch.cuda.synchronize();start=time.perf_counter();value=original(*args,**kwargs);torch.cuda.synchronize()
        timing.append({'trial':trial,'wall_s':time.perf_counter()-start});del value
    report['rows'].append({'call':calls,'candidate':'reference','bitwise_equal':True,'timings':timing})
    return expected

# Test-process-only interception to capture the real exact projection inputs.
# No candidate or production simulator uses this interception.
reference.depth_aware_bidirectional_warp_projected_batch_torch=probe
try:
    row,events=c.run_ours(c.Fixture('s2_00',1),'ours_resident')
    report['fixture']=row
finally:
    reference.depth_aware_bidirectional_warp_projected_batch_torch=original
from stress_geometry import run_stress
report['stress']=run_stress(reference,candidates)
report.update(status='complete',all_component_cases_exact=all(x['bitwise_equal'] for x in report['rows']))
(E/'index_component_results.json').write_text(json.dumps(report,indent=2)+'\n')
print('COMPONENT_COMPLETE',flush=True)
