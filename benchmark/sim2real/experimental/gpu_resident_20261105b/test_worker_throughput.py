"""Barrier-synchronized independent workers on the assigned GPU only."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
source=(root/'bench_pipeline_v2.py').read_text()
if len(sys.argv)>1:
    group=Path(sys.argv[1]);worker=int(sys.argv[2]);variant=sys.argv[3]
    barrier='''
    if trial>=0:
        (barrier_group/("ready_"+str(trial)+"_"+str(worker_index))).touch()
        deadline=time.monotonic()+180
        while not (barrier_group/("go_"+str(trial))).exists():
            if time.monotonic()>deadline:raise TimeoutError("worker barrier")
            time.sleep(.02)
'''
    source=source.replace('    torch.cuda.reset_peak_memory_stats();',barrier+'    torch.cuda.reset_peak_memory_stats();')
    source=source.replace('row=dict(trial=trial,wall_s=wall,','row=dict(trial=trial,wall_s=wall,capture_begin_monotonic=begin,capture_end_monotonic=begin+wall,')
    tag=str(group.relative_to(root)/('worker_'+str(worker)))
    sys.argv=[str(root/'bench_pipeline_v2.py'),'--variant',variant,'--case','s2_00','--trials','3','--out',tag]
    exec(compile(source,str(root/'bench_pipeline_v2.py'),'exec'),{'__name__':'__main__','__file__':str(root/'bench_pipeline_v2.py'),'barrier_group':group,'worker_index':worker})
else:
    out=root/'worker_throughput';out.mkdir(exist_ok=False);report=dict(groups=[],real_data_used=False)
    for variant,count in [('baseline',2),('prior_fused',2),('resident_exact',1),('resident_exact',2)]:
        group=out/(variant+'_'+str(count));group.mkdir();children=[];logs=[]
        try:
            for i in range(count):
                log=(group/('worker_'+str(i)+'.log')).open('x');logs.append(log)
                children.append(subprocess.Popen([sys.executable,'-u',__file__,str(group),str(i),variant],stdout=log,stderr=subprocess.STDOUT))
            for trial in range(3):
                deadline=time.monotonic()+180
                while not all((group/('ready_'+str(trial)+'_'+str(i))).exists() for i in range(count)):
                    assert all(p.poll() is None for p in children),'own worker failed before barrier'
                    if time.monotonic()>deadline:raise TimeoutError('group barrier')
                    time.sleep(.1)
                (group/('go_'+str(trial))).touch()
            codes=[p.wait(timeout=180) for p in children];assert all(c==0 for c in codes),codes
            summaries=[json.loads((group/('worker_'+str(i))/'summary.json').read_text()) for i in range(count)]
            measurements=[]
            for trial in range(3):
                rows=[s['rows'][trial] for s in summaries]
                wall=max(r['capture_end_monotonic'] for r in rows)-min(r['capture_begin_monotonic'] for r in rows)
                measurements.append(dict(group_wall_s=wall,intervals_per_s=count/wall,total_events=sum(r['event_count'] for r in rows)))
            row=dict(variant=variant,workers=count,passed=True,measurements=measurements,pids=[p.pid for p in children])
        except Exception as error:
            row=dict(variant=variant,workers=count,passed=False,error=repr(error),pids=[p.pid for p in children])
            # These handles belong exclusively to children created above.
            for process in children:
                if process.poll() is None:process.terminate()
            for process in children:process.wait(timeout=30)
        finally:
            for log in logs:log.close()
        report['groups'].append(row);(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
    print('WORKER_THROUGHPUT_COMPLETE',flush=True)
