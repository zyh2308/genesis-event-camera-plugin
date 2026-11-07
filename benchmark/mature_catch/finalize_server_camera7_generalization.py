"""Finite task-local closeout after both preregistered evaluation shards finish."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,tarfile,time,traceback

ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    os.nice(10);start=time.monotonic();jobs=['catchit_camera7_generalization_g3_shard'+str(i)+'_20261108' for i in range(2)]
    while True:
        results=[ROOT/(j+'.result.json') for j in jobs]
        if all(f.exists() for f in results):
            records=[json.loads(f.read_text()) for f in results]
            assert all(r['status']=='completed' and r['exit_code']==0 for r in records),records
            break
        if time.monotonic()-start>14400:raise TimeoutError('Evaluation closeout exceeded bounded wait')
        time.sleep(20)
    env=dict(os.environ,PYTHONPATH='/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/catchit_20261107a/deps_genesis',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (ROOT/'final_analysis.log').open('w') as out:
        subprocess.run(['/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python',str(ROOT/'analyze_generalization_final.py'),'--root',str(ROOT)],env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
    portable=json.loads((ROOT/'GENERALIZATION_PORTABLE.json').read_text());assert portable['complete'] and not portable['violations'],portable['violations']
    files=set(f for f in (ROOT/'cases').rglob('*') if f.is_file())
    for j in jobs:
        for suffix in ['.result.json','.log','.allocation.json']:
            f=ROOT/(j+suffix)
            if f.exists():files.add(f)
    for i in range(2):
        for suffix in ['_preflight.json','_complete.json','_progress.json','_job.json']:
            f=ROOT/('shard'+str(i)+suffix)
            if f.exists():files.add(f)
    for name in ['PREREGISTERED_PROTOCOL.json','LAUNCH_RECEIPT.json','campaign_worker.py','analyze_generalization_final.py','GENERALIZATION_PORTABLE.json','final_analysis.log','finalize_server_generalization.py']:
        files.add(ROOT/name)
    # Supplementary replays are separately identified and do not alter task outcomes.
    for f in (ROOT/'demo_replay_correct_env').rglob('*'):
        if f.is_file() and (f.parent/'manifest.json').exists():files.add(f)
    manifest=[dict(path=f.relative_to(ROOT).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(files)]
    (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(dict(root=str(ROOT),schema='closed_completed_case_evidence_v1',files=manifest),indent=2))
    text=''.join(row['sha256']+'  '+row['path']+'\n' for row in manifest);(ROOT/'SERVER_SHA256SUMS').write_text(text)
    mismatches=[row['path'] for row in manifest if sha(ROOT/row['path'])!=row['sha256']]
    assert not mismatches,mismatches
    verification=dict(checked=len(manifest),mismatches=mismatches,total_bytes=sum(r['bytes'] for r in manifest),manifest_sha256=sha(ROOT/'SERVER_SHA256SUMS'),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (ROOT/'SERVER_VERIFICATION.json').write_text(json.dumps(verification,indent=2))
    media=[]
    for record in portable['records']:
        base=Path(record['output']);video=base/'episode_000/genesis.mp4'
        if video.exists():media.append(video)
        if record['mode']=='event':media.append(base/'migration_audit/sensor_episode_000/initial_rgb.png')
    for f in (ROOT/'demo_replay_correct_env').rglob('*'):
        if f.is_file() and (f.parent/'manifest.json').exists() and f.suffix in ['.mp4','.jpg','.json']:media.append(f)
    bundle=ROOT/'GENERALIZATION_MEDIA.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=1) as tar:
        for f in sorted(set(media)):tar.add(f,arcname=f.relative_to(ROOT).as_posix(),recursive=False)
    receipt=dict(media_files=len(set(media)),media_bytes=bundle.stat().st_size,media_sha256=sha(bundle),portable_bytes=(ROOT/'GENERALIZATION_PORTABLE.json').stat().st_size,portable_sha256=sha(ROOT/'GENERALIZATION_PORTABLE.json'),verification=verification)
    (ROOT/'CLOSEOUT_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        (ROOT/'CLOSEOUT_ERROR.txt').write_text(traceback.format_exc());raise
