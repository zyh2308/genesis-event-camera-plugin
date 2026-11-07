"""Combine complete disjoint workers into one explicit paired adoption record, with provenance."""
import argparse,datetime,hashlib,json,sys,time,traceback
from pathlib import Path
from recon_repair_gate import decide

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
    temp=path.with_name(path.name+'.partial');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);args=p.parse_args()
    root=args.root;R=args.campaign;assert root==R.parent/'20261107-recon-sigmoid-mixed-v2'
    workers={method:R.parent/('20261107-recon-sigmoid-epoch5-'+method+'-v1') for method in ['eventgenesis','v2e']}
    save(root/'waiting.json',dict(status='waiting_for_both_complete_workers',workers={k:str(v) for k,v in workers.items()},GPU_reserved=False,utc=now()))
    deadline=time.monotonic()+4*3600
    while not all((path/'resource_release.json').exists() for path in workers.values()):
        assert not any((path/'failure.json').exists() for path in workers.values()),'Upstream worker failed'
        if time.monotonic()>deadline:raise TimeoutError('Paired workers did not finish')
        time.sleep(5)
    rows=[];receipts={};manifest=json.loads((root/'sources/source_manifest.json').read_text())
    for file,digest in manifest.items():assert sha(root/'sources'/file)==digest
    for method,worker in workers.items():
        assert json.loads((worker/'completion.json').read_text())['status']=='complete'
        worker_manifest=json.loads((worker/'sources/source_manifest.json').read_text())
        for name,digest in worker_manifest.items():assert sha(worker/'sources'/name)==digest
        assert worker_manifest['recon_short_diagnostic.py']==manifest['recon_short_diagnostic.py']
        payload=json.loads((worker/'mixed/completion.json').read_text());assert payload['status']=='complete' and not payload['real_data_used']
        assert len(payload['results'])==3 and {r['seed'] for r in payload['results']}=={0,1,2}
        for row in payload['results']:
            assert row['method']==method and row['arm']=='sigmoid' and row['updates']==300 and not row['fixed_train_batch']
            assert sha(row['checkpoint'])==row['checkpoint_sha256']
            result=worker/'mixed'/(method+'_seed'+str(row['seed'])+'_sigmoid')/'completion.json'
            row=dict(row,source_result_path=str(result),source_result_sha256=sha(result));rows.append(row)
        receipts[method]=dict(worker=str(worker),source_manifest=worker_manifest,completion_sha256=sha(worker/'mixed/completion.json'),resource_release=json.loads((worker/'resource_release.json').read_text()))
    out=root/'mixed';out.mkdir(exist_ok=False)
    save(out/'protocol.json',dict(real_data_used=False,updates_per_case=300,from_original_initialization=True,
        physical_workers=receipts,reason='120-update study retained: only V2E seed1 contrast ratio0.095764 < predeclared0.1; extend all six to first five epochs without changing model/loss/threshold',
        original_120_update_study=str(R.parent/'20261107-recon-sigmoid-mixed-v1'),thresholds_unchanged=True,
        note='This directory combines actual completed workers; checkpoints remain at the original worker paths.'))
    save(out/'completion.json',dict(status='complete',real_data_used=False,results=rows,source_worker_receipts=receipts))
    decision=decide(rows);decision.update(completion_sha256=sha(out/'completion.json'),gate_source_sha256=manifest['recon_repair_gate.py'])
    save(root/'synthetic_gate.json',decision)
    save(root/'completion.json',dict(status='complete',synthetic_gate_passed=decision['passed'],real_data_used=False,utc=now()))
    save(root/'resource_release.json',dict(status='released',GPU_never_reserved=True,all_upstream_workers_released=True,utc=now()))
    print('EPOCH5_ADOPTION',decision['passed'],len(decision['checks']),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1]);save(root/'failure.json',dict(error=traceback.format_exc(),utc=now()))
        raise
