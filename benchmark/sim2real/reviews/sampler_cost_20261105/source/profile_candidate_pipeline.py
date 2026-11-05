"""Read-only instrumentation of the unchanged frozen GPU implementation."""
import functools, importlib.util, json, sys
from pathlib import Path
import torch
E=Path(__file__).resolve().parent; V=E.parent/'simulator_efficiency_comparison_20261105c'
spec=importlib.util.spec_from_file_location('prof_fixture', E/'candidate_compare_fixture.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
c.ROOT=E/'candidate_profile_inputs';c.ROOT.mkdir(exist_ok=False)
from genesis_event_plugin import gpu_geometry, gpu_interpolator, gpu_sensor
for module,names in [(gpu_geometry,['project_endpoint_to_alpha_stack_torch','_splat_projected_coverage_batch_torch','depth_aware_bidirectional_warp_projected_batch_torch']), (gpu_interpolator,['finish_geometry']), (gpu_sensor,['generate_device_sequence'])]:
    for name in names:
        original=getattr(module,name)
        def make(fn,label):
            @functools.wraps(fn)
            def wrapper(*args,**kwargs):
                with torch.profiler.record_function(label):return fn(*args,**kwargs)
            return wrapper
        setattr(module,name,make(original,module.__name__+'.'+name))
c.run_ours(c.Fixture('s2_00',1),'ours_resident')
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
    row,events=c.run_ours(c.Fixture('s2_00',1),'ours_resident')
data=[]
for a in p.key_averages():
    data.append(dict(key=a.key,count=a.count,cpu_total_us=a.cpu_time_total,cpu_self_us=a.self_cpu_time_total,cuda_total_us=a.device_time_total,cuda_self_us=a.self_device_time_total))
(E/'candidate_profile_operators.json').write_text(json.dumps(dict(status='complete',row=row,operators=data),indent=2)+'\n')
print(p.key_averages().table(sort_by='self_cpu_time_total',row_limit=25),flush=True)
print(p.key_averages().table(sort_by='self_cuda_time_total',row_limit=25),flush=True)
