"""Seed-balanced g3 worker. Parent owns lease/PG; all15 synthetic gates precede full training."""
import argparse,fcntl,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from selected_common import *

def call(F,argv,log,seconds):
    need(not log.exists(),'Refuse existing log '+str(log));log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as stream:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,cwd=str(F/'sources'))
        until=time.monotonic()+seconds
        while child.poll() is None:
            if (F/'GLOBAL_FAILURE.json').exists() or time.monotonic()>until:
                child.terminate()
                try:child.wait(timeout=45)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=15)
                raise RuntimeError('Own task interrupted by global failure or fixed timeout')
            time.sleep(5)
    save(log.with_suffix('.exit.json'),dict(argv=argv,exit_code=child.returncode))
    need(child.returncode==0,'Own worker failed: '+str(log))

def gates(F,plan,seed):
    for method in METHODS[2:]:
        out=F/'gate_runs'/(method+'_seed'+str(seed))
        cmd=[sys.executable,'-B','-u',str(F/'sources/recon_short_diagnostic.py'),'--campaign',plan['campaign'],'--out',str(out),
             '--phase','mixed','--methods',method,'--seeds',str(seed),'--arms','sigmoid','--updates','300','--validation-length','200']
        call(F,cmd,F/'logs'/('gate_'+method+'_seed'+str(seed)+'.log'),3600)
    own=[]
    for method in METHODS[2:]:
        r=read(F/'gate_runs'/(method+'_seed'+str(seed))/'completion.json');need(r['status']=='complete' and len(r['results'])==1,'Gate worker incomplete')
        own.extend(r['results'])
    save(F/'gate_seed_receipts'/('seed'+str(seed)+'.json'),dict(seed=seed,results=own,passed=True))
    until=time.monotonic()+1800
    while not (F/'ALL15_SYNTHETIC_GATE.json').exists():
        need(not (F/'GLOBAL_FAILURE.json').exists(),'Other lane failed')
        with (F/'.gate.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            paths=[F/'gate_seed_receipts'/('seed'+str(s)+'.json') for s in SEEDS]
            if all(p.exists() for p in paths) and not (F/'ALL15_SYNTHETIC_GATE.json').exists():
                from recon_selected_gate import decide
                allrows=read(F/'sources/inherited_gate_results.json')
                for p in paths:allrows+=read(p)['results']
                for row in allrows:need(sha(Path(row['checkpoint']))==row['checkpoint_sha256'],'Gate CP changed')
                result=decide(allrows);result.update(source_manifest_sha256=sources(F),results_sha256=__import__('hashlib').sha256(json.dumps(allrows,sort_keys=True).encode()).hexdigest())
                save(F/'ALL15_SYNTHETIC_GATE.json',result)
        need(time.monotonic()<until,'Paired gate staging timeout30min');time.sleep(3)
    need(read(F/'ALL15_SYNTHETIC_GATE.json')['passed'],'All15 synthetic gate failed; do not start full runs')

def train(F,plan,method,seed):
    sources(F);run=F/'training'/('seed'+str(seed))/method;run.mkdir(parents=True,exist_ok=False)
    key=method+'/seed'+str(seed);b=plan['resume_sources'].get(key)
    base=[sys.executable,'-B','-u',str(F/'sources/train_e2vid_repaired.py'),'--model-root',plan['hqf']['model_root'],
          '--data-root',str(Path(plan['campaign'])/'adapters'/method/'reconstruction'),'--config',str(F/'sources/training_recipe.json'),
          '--output-root',str(run),'--generator',method,'--seed',str(seed),'--device','cuda:0']
    if b:
        for name in ('last','best'):need(sha(Path(b[name]['checkpoint']))==b[name]['checkpoint_sha256'],'Old resume bytes changed')
        shutil.copy2(b['best']['checkpoint'],run/'synthetic_best.pt')
        selection=read(Path(b['best']['checkpoint']).with_suffix('.json'));selection['checkpoint']=str(run/'synthetic_best.pt')
        need(selection['sha256']==b['best']['checkpoint_sha256'],'Old synthetic selection differs');save(run/'synthetic_best.json',selection)
        oldlog=Path(b['last']['checkpoint']).parent/'training_metrics.jsonl'
        bound=next(r['log'] for r in read(F/'cpu_admission.json')['resume_history_logs'] if r['key']==key)
        need(check(bound)==oldlog,'Historical log lineage differs');shutil.copy2(oldlog,run/'training_metrics.jsonl')
        save(run/'continuation_lineage.json',dict(last=b['last'],best=b['best'],copied_training_log=record(oldlog),
            numerical_note='Original seed0 prefix on g1 RTX3090; continuation on fixed-seed g3 RTX4090, same FP32 recipe; no bitwise whole-trajectory claim'))
        target=b['last']['updates']+1
        call(F,base+['--resume',b['last']['checkpoint'],'--max-updates',str(target)],run/'restore_next.log',3600)
    else:
        call(F,base+['--max-updates','20'],run/'update20.log',3600)
        need(read(run/'run_summary.json')['optimizer_update']==20,'Initial20 incomplete')
        call(F,base+['--resume',str(run/'last.pt'),'--max-updates','21'],run/'resume21.log',3600);target=21
    oracle=read(run/'resume_oracle.json')
    need(oracle['status']=='pass' and oracle['observed_next_update']==target and set(oracle['optimizer_step_counts_after_next_update'])=={target},'Restore oracle failed')
    save(run/'boundary_preflight.json',dict(passed=True,oracle=oracle,checkpoint=record(run/'last.pt')))
    call(F,base+['--resume',str(run/'last.pt')],run/'full_training.log',8*3600)
    summary=read(run/'run_summary.json');need(summary['status']=='complete' and summary['optimizer_update']==12000,'Full200 epochs incomplete')
    selected=read(run/'synthetic_best.json');need(sha(run/'synthetic_best.pt')==selected['sha256'] and selected['selection_source']=='synthetic validation only','Wrong checkpoint selection')
    save(run/'completion.json',dict(status='complete',method=method,seed=seed,optimizer_update=12000,source_manifest_sha256=sources(F),
        initial_state_sha256=summary['initial_model_sha256'],selected_checkpoint=record(run/'synthetic_best.pt'),
        selection_record=record(run/'synthetic_best.json'),summary=summary,real_data_used=False,real_scoring_enabled=False))

def freeze_if_last(F,plan):
    with (F/'.freeze.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        paths=[F/'training'/('seed'+str(s))/m/'completion.json' for s in SEEDS for m in METHODS]
        if not all(p.exists() for p in paths):return False
        if (F/'frozen_checkpoint_matrix.json').exists():return False
        rows=[]
        for p in paths:
            c=read(p);need(c['status']=='complete' and c['optimizer_update']==12000,'Full matrix incomplete')
            check(c['selected_checkpoint']);check(c['selection_record'])
            rows.append(dict(run_id=c['method']+'_seed'+str(c['seed']),method=c['method'],seed=c['seed'],
                checkpoint=c['selected_checkpoint'],selection_record=c['selection_record'],completed_updates=12000,initial_state_sha256=c['initial_state_sha256']))
        for s in SEEDS:need(len({r['initial_state_sha256'] for r in rows if r['seed']==s})==1,'Unpaired initial state')
        save(F/'frozen_checkpoint_matrix.json',dict(schema='recon_selected_frozen_checkpoints_v1',status='frozen_all15_before_real_scoring',
             methods=list(METHODS),seeds=list(SEEDS),real_sequences=list(SEQUENCES),result_scope='posthoc_selected_hqf3_development',
             source_manifest_sha256=sources(F),recipe=record(F/'sources/training_recipe.json'),runs=rows))
        return True

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,choices=[0,1,2],required=True);a=p.parse_args()
    F=a.root.resolve();plan=plan_check(read(F/'sources/registered_plan.json'));need(str(F)==plan['root'],'Wrong root');sources(F)
    try:
        gates(F,plan,a.seed)
        for method in METHODS:train(F,plan,method,a.seed)
        last=freeze_if_last(F,plan)
        if last:
            save(F/'completion.json',dict(status='full15_training_complete_selected_eval_pending',training_runs=15,evaluated_sequence_runs=0,
                real_scoring_enabled=False,real_result_scope=plan['real_result_scope'],full14_historical_results_unchanged=True))
        save(F/'lanes'/('seed'+str(a.seed))/'lane_completion.json',dict(status='complete',seed=a.seed,last_lane_froze_matrix=last,real_scoring_enabled=False,
              note='Every lane releases after training; final completing lane freezes all15; real evaluation needs independent later authorization'))
    except BaseException:
        with (F/'.failure.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not (F/'GLOBAL_FAILURE.json').exists():save(F/'GLOBAL_FAILURE.json',dict(seed=a.seed,error=traceback.format_exc()))
        raise

if __name__=='__main__':main()
