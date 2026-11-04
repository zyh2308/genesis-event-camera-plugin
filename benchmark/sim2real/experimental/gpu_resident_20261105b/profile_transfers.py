"""CUPTI copy accounting, excluded from timing tables and regression outputs."""
import json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
condition=sys.argv[1];variant='resident_exact' if condition.startswith('resident') else condition
diagnostics='full' if condition=='resident_full' else 'events'
source=(root/'bench_pipeline_v2.py').read_text()
source=source.replace('    torch.cuda.reset_peak_memory_stats();',"    profiler=None\n    if trial>=0:\n        profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False)\n        profiler.__enter__()\n    torch.cuda.reset_peak_memory_stats();")
source=source.replace('    frame_stack=',"    if profiler is not None:\n        profiler.__exit__(None,None,None);profiler.export_chrome_trace(str(out/'trace.json'))\n    frame_stack=")
sys.argv=[str(root/'bench_pipeline_v2.py'),'--variant',variant,'--diagnostics',diagnostics,'--case','s2_00','--trials','1','--out','profile_'+condition]
exec(compile(source,str(root/'bench_pipeline_v2.py'),'exec'),{'__file__':str(root/'bench_pipeline_v2.py'),'__name__':'__main__'})
trace=json.loads((root/('profile_'+condition)/'trace.json').read_text());copies={}
for event in trace['traceEvents']:
    if event.get('cat')!='gpu_memcpy':continue
    name=event['name'];row=copies.setdefault(name,dict(calls=0,bytes=0,device_duration_us=0.))
    row['calls']+=1;row['bytes']+=int(event.get('args',{}).get('bytes',0));row['device_duration_us']+=float(event.get('dur',0))
(root/('profile_'+condition)/'copy_summary.json').write_text(json.dumps(dict(condition=condition,copies=copies,scope='CUPTI GPU memcpy activities; diagnostic profile, not performance benchmark'),indent=2)+'\n')
print('COPY_SUMMARY',json.dumps(copies),flush=True)
