"""CUDA event spans around unchanged geometry functions, without extra syncs."""
import json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
source=(root/'bench_pipeline.py').read_text()
insertion='''
import genesis_event_plugin.torch_physics as geometry
phase_events=[]
def wrap_function(name):
    original=getattr(geometry,name)
    def wrapped(*args,**kwargs):
        start=torch.cuda.Event(enable_timing=True);stop=torch.cuda.Event(enable_timing=True)
        start.record();result=original(*args,**kwargs);stop.record()
        phase_events.append((name,start,stop));return result
    setattr(geometry,name,wrapped)
for function in ['project_endpoint_to_alpha_stack_torch','depth_aware_bidirectional_warp_projected_batch_torch']:
    wrap_function(function)
'''
source=source.replace('for trial in range(-1,a.trials):',insertion+'\nfor trial in range(-1,a.trials):')
source+='\ntorch.cuda.synchronize()\n(out/"cuda_phase_spans.json").write_text(json.dumps([{\'name\':n,\'elapsed_ms\':s.elapsed_time(e)} for n,s,e in phase_events],indent=2)+"\\n")\n'
sys.argv=[str(root/'bench_pipeline.py'),'--variant','baseline','--case','s2_00','--trials','1','--out','geometry_profile']
exec(compile(source,str(root/'bench_pipeline.py'),'exec'),{'__name__':'__main__','__file__':str(root/'bench_pipeline.py')})
