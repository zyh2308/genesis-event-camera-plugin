"""Alternate baseline, prior fusion, resident events, resident full export."""
import json,sys,time
from pathlib import Path
import torch
root=Path(__file__).resolve().parent;torch.set_num_interop_threads(1)
source=(root/'bench_pipeline.py').read_text().replace('torch.set_num_interop_threads(1)','pass')
out=root/'paired_suite';out.mkdir(exist_ok=False);results=[];started=time.monotonic()
conditions=[('baseline','baseline','events'),('prior_fused','prior_fused','events'),('resident_events','resident_exact','events'),('resident_full','resident_exact','full')]
for case in ['s0_00','s1_00','s2_00','s3_00','train_000']:
    for rep in range(3):
        order=conditions if rep%2==0 else conditions[::-1]
        for label,variant,diagnostics in order:
            tag=f'paired_suite/{case}_r{rep}_{label}'
            sys.argv=[str(root/'bench_pipeline.py'),'--variant',variant,'--diagnostics',diagnostics,'--case',case,'--trials','1','--out',tag]
            exec(compile(source,str(root/'bench_pipeline.py'),'exec'),{'__file__':str(root/'bench_pipeline.py'),'__name__':'__main__'})
            summary=json.loads((root/tag/'summary.json').read_text())
            results.append(dict(case=case,repeat=rep,condition=label,variant=variant,diagnostics=diagnostics,path=tag,row=summary['rows'][0]))
            (out/'progress.json').write_text(json.dumps(dict(results=results,elapsed_s=time.monotonic()-started),indent=2)+'\n')
(out/'COMPLETE').write_text('complete\n');print('PAIRED_SUITE_COMPLETE',len(results),flush=True)
