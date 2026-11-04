"""Alternating-order full capture benchmarks; raw repeats are retained."""
import json,sys,time
from pathlib import Path
import torch
root=Path(__file__).resolve().parent
torch.set_num_interop_threads(1)
source=(root/'bench_pipeline.py').read_text().replace('torch.set_num_interop_threads(1)','pass')
variants=sys.argv[1].split(',') if len(sys.argv)>1 else ['baseline','fused_early']
name=sys.argv[2] if len(sys.argv)>2 else 'paired_suite'
repeats=int(sys.argv[3]) if len(sys.argv)>3 else 5
out=root/name;out.mkdir(exist_ok=False)
results=[];started=time.monotonic()
for case in ['s0_00','s1_00','s2_00','s3_00','train_000']:
    for rep in range(repeats):
        order=variants if rep%2==0 else variants[::-1]
        for variant in order:
            tag=f'{name}/{case}_r{rep}_{variant}'
            sys.argv=[str(root/'bench_pipeline.py'),'--variant',variant,'--case',case,'--trials','1','--intervals','1','--out',tag]
            exec(compile(source,str(root/'bench_pipeline.py'),'exec'),{'__file__':str(root/'bench_pipeline.py'),'__name__':'__main__'})
            summary=json.loads((root/tag/'summary.json').read_text())
            results.append(dict(case=case,repeat=rep,variant=variant,order=order,path=tag,row=summary['rows'][0]))
            (out/'progress.json').write_text(json.dumps(dict(results=results,elapsed_s=time.monotonic()-started),indent=2)+'\n')
(out/'COMPLETE').write_text('complete\n')
print('PAIRED_SUITE_COMPLETE',len(results),flush=True)
