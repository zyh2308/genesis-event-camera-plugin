"""Advance the authorized study only after complete actual development audits."""
from pathlib import Path
import datetime, hashlib, json, os, shutil, subprocess, time, traceback

ROOT=Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/fair_frontend_blind_20261108b')
PY='/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python'
TOOLS=ROOT/'analysis_v1'


def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    state=ROOT/'ADVANCE_STATUS.json';started=time.monotonic();cpus=sorted(os.sched_getaffinity(0))
    def update(stage,**extra):
        state.write_text(json.dumps(dict(stage=stage,updated_utc=stamp(),elapsed_s=time.monotonic()-started,**extra),indent=2))
    def run(label,args,timeout=45000):
        update(label)
        with (ROOT/(label+'.log')).open('x') as log:
            subprocess.run(['nice','-n','10',PY,'-B','-u']+args,env=os.environ,cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    try:
        os.sched_setaffinity(0,cpus[8:10])
        validation=ROOT/'frontend_validation'
        run('audit_development',[str(TOOLS/'collect_fair_stage.py'),'--root',str(validation),
            '--stage','validation','--wait','--recovery-plan',str(validation/'INFRASTRUCTURE_RECOVERY.json')])
        run('lock_frontends',[str(TOOLS/'select_frontend.py'),'--root',str(validation)],timeout=180)
        selection=validation/'FRONTEND_SELECTION.json'
        lock=json.loads(selection.read_text());assert lock['status']=='LOCKED' and lock['locked_utc']
        collision=json.loads((ROOT/'blind_design/SEED_COLLISION_SERVER.json').read_text())
        assert collision['status']=='PASS'
        blind=ROOT/'blind32'
        run('prepare_blind_protocol',[str(ROOT/'tools/make_blind_protocol.py'),
            '--base-runtime',str(ROOT/'BASE_RUNTIME.json'),'--selection',str(selection),
            '--bank',str(ROOT/'blind_design/BLIND32_BANK.json'),'--root',str(blind),'--output',str(blind)],timeout=180)
        shutil.copyfile(validation/'workqueue/fair_workqueue_worker.py',blind/'workqueue/fair_workqueue_worker.py')
        protocol=blind/'EVALUATION_PROTOCOL.json';spec=json.loads(protocol.read_text())
        assert spec['frontend_selection_sha256']==sha(selection)
        (blind/'BLIND_PREREQUISITES.json').write_text(json.dumps(dict(status='PASS',
            locked_utc=lock['locked_utc'],verified_utc=stamp(),selection_sha256=sha(selection),
            protocol_sha256=sha(protocol),bank_sha256=spec['bank_sha256'],
            development_complete_sha256=sha(validation/'FAIR_COLLECTION_COMPLETE.json'),
            camera_original_policy_frozen_core_unchanged=True),indent=2))
        update('launch_blind',frontend_by_method=lock['frontend_by_method'])
        template=json.loads((validation/'catchit_fair_frontend_validation_g1_gpu0_dev_20261108b.job.json').read_text())
        launcher='/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/visual_fast_paired_20261108a/run_gpu_stage_split_cpu.py'
        os.sched_setaffinity(0,cpus)
        jobs=[]
        for gpu in [0,1]:
            worker='g1_gpu'+str(gpu)+'_blind'
            name='catchit_fair_blind32_'+worker+'_20261108b'
            job=dict(name=name,gpu_candidates=[gpu],
                command=['/usr/bin/python3','-u',str(blind/'workqueue/fair_workqueue_worker.py'),
                    '--protocol',str(protocol),'--worker-id',worker],
                env=template['env'],cwd=str(blind),timeout_s=86400)
            jobfile=blind/(name+'.job.json');assert not jobfile.exists();jobfile.write_text(json.dumps(job,indent=2))
            with (blind/(name+'.supervisor.log')).open('x') as log:
                p=subprocess.Popen(['/usr/bin/python3','-u',launcher,'--job',str(jobfile)],cwd=blind,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append(dict(worker_id=worker,supervisor_pid=p.pid,gpu_candidate=gpu))
        (blind/'JOBS_LAUNCHED.json').write_text(json.dumps(dict(jobs=jobs,utc=stamp()),indent=2))
        os.sched_setaffinity(0,cpus[8:10])
        run('audit_blind',[str(TOOLS/'collect_fair_stage.py'),'--root',str(blind),'--stage','blind','--wait'],timeout=46000)
        update('BLIND_COMPLETE',frontend_by_method=lock['frontend_by_method'],
            note='All registered blind outcomes are audited. Cost, figures, media and independent final review remain separate.')
    except BaseException as exc:
        update('STOPPED',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
