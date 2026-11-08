"""Retain OOM attempts, then resume identical registered cases one per GPU."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess

ROOT=Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/fair_frontend_blind_20261108b')
STAGE=ROOT/'frontend_validation'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def relative(p):return str(Path(p).relative_to(STAGE))


def move(src,dst):
    assert src.resolve().is_relative_to(STAGE.resolve())
    dst.parent.mkdir(parents=True,exist_ok=True)
    assert dst.parent.resolve().is_relative_to(STAGE.resolve())
    assert not dst.exists()
    src.rename(dst)


def main():
    manifest=STAGE/'INFRASTRUCTURE_RECOVERY.json'
    assert not manifest.exists(),'Recovery is already registered; do not replay this mutation'
    jobs=list(STAGE.glob('*.job.json'))
    assert len(jobs)==4
    receipts={}
    for jobfile in jobs:
        job=json.loads(jobfile.read_text());resultfile=jobfile.with_name(job['name']+'.result.json')
        assert resultfile.exists(),'Owned validation workers have not all drained'
        receipts[job['command'][-1]]=(jobfile,job,resultfile,json.loads(resultfile.read_text()))
    spec=json.loads((STAGE/'EVALUATION_PROTOCOL.json').read_text())
    historical=[];archived=[]
    for worker,(jobfile,job,resultfile,result) in receipts.items():
        if result['status']=='completed' and result.get('exit_code')==0:continue
        progress=STAGE/'workqueue'/(worker+'.progress.json')
        assert progress.exists()
        entry=dict(worker_id=worker,job_file=relative(jobfile),job_sha256=sha(jobfile),
            result_file=relative(resultfile),result_sha256=sha(resultfile),
            progress_file=relative(progress),progress_sha256=sha(progress),archived_failed_claims=[])
        claims=[p for p in (STAGE/'workqueue/claims').glob('*.json')
                if json.loads(p.read_text()).get('worker_id')==worker]
        for claim in claims:
            record=json.loads(claim.read_text())
            if record['status']=='completed' and record.get('exit_code')==0:continue
            assert record['status']=='infrastructure_failure'
            shard=record['shard'];attempt=STAGE/'infrastructure_failures/attempt0'/('shard'+str(shard).zfill(3))
            attempt.mkdir(parents=True,exist_ok=False)
            target=attempt/'claims'/claim.name
            claim_sha=sha(claim);move(claim,target)
            cases=[c for c in spec['cases'] if c['shard']==shard]
            assert len(cases)==1
            for case in cases:
                source=STAGE/'cases'/case['case_id']
                if source.exists():move(source,attempt/'cases'/case['case_id'])
                for log in STAGE.glob(case['case_id']+'_*.log'):move(log,attempt/log.name)
            for suffix in ['progress','preflight','running','complete']:
                p=STAGE/('shard'+str(shard)+'_'+suffix+'.json')
                if p.exists():move(p,attempt/p.name)
            assert list(attempt.glob('*.log')) and (attempt/('shard'+str(shard)+'_progress.json')).exists()
            hashes={str(p.relative_to(attempt)):sha(p) for p in attempt.rglob('*') if p.is_file()}
            (attempt/'ARCHIVED_SHA256.json').write_text(json.dumps(hashes,indent=2))
            entry['archived_failed_claims'].append(dict(shard=shard,claim_file=relative(target),
                claim_sha256=claim_sha,archived_attempt_root=relative(attempt)))
            archived.append(shard)
        assert entry['archived_failed_claims']
        historical.append(entry)
    assert sorted(archived)==[4,8],('Unexpected failed-shard set: inspect before resuming',archived)
    held=[]
    for p in list((STAGE/'workqueue/claims').glob('*.json')):
        v=json.loads(p.read_text())
        if v['status']=='deferred_capacity_drain':
            move(p,STAGE/'infrastructure_failures/capacity_drain_claims'/p.name);held.append(v['shard'])
    registered=dict(schema='fair_workqueue_infra_recovery_v1',
        registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=sha(STAGE/'EVALUATION_PROTOCOL.json'),reason='cuda_out_of_memory',
        historical_failed_workers=historical,
        replacement_worker_ids=['g1_gpu0_dev_recovery1','g1_gpu1_dev_recovery1'],
        held_unexecuted_shards=held,
        no_algorithm_scene_seed_configuration_changed=True,
        recovery_rule='Repeat the entire interrupted paired shard, preserving every old output. Resume all unexecuted registered shards. One actual worker per GPU; no outcome-based retries.')
    manifest.write_text(json.dumps(registered,indent=2));manifest_sha=sha(manifest)
    template=json.loads((STAGE/'catchit_fair_frontend_validation_g1_gpu0_dev_20261108b.job.json').read_text())
    launcher=Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/visual_fast_paired_20261108a/run_gpu_stage_split_cpu.py')
    launched=[]
    for gpu in [0,1]:
        worker=registered['replacement_worker_ids'][gpu]
        name='catchit_fair_frontend_validation_'+worker+'_20261108b'
        command=list(template['command']);command[-1]=worker
        job={**template,'name':name,'gpu_candidates':[gpu],'command':command,
            'infrastructure_recovery_sha256':manifest_sha}
        if gpu==0:
            job['wait_for_file']=str(ROOT/'cost_smoke_ours25_v1/catchit_fair_cost_smoke_ours25_g1_20261108b.result.json')
        jobfile=STAGE/(name+'.job.json');assert not jobfile.exists();jobfile.write_text(json.dumps(job,indent=2))
        with (STAGE/(name+'.supervisor.log')).open('x') as log:
            proc=subprocess.Popen(['/usr/bin/python3','-u',str(launcher),'--job',str(jobfile)],cwd=STAGE,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launched.append(dict(worker_id=worker,supervisor_pid=proc.pid,gpu_candidate=gpu))
    print(json.dumps(dict(status='RECOVERY_STARTED',archived_failed_shards=archived,
        pending_unexecuted_shards=held,jobs=launched,manifest_sha256=manifest_sha)))


if __name__=='__main__':main()
