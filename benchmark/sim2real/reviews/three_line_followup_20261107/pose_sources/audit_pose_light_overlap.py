"""Read-only equality check before treating the new bank as coverage-only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['campaign','full','pilot','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();assert not args.out.exists()
    pilot=json.loads((args.pilot/'completion.json').read_text());full=json.loads((args.full/'completion.json').read_text())
    assert pilot['status']==full['status']=='complete'
    selected=pilot['selected'];all_ids=full['selected']
    assert len(selected)==8 and len(all_ids)==96 and set(selected)<=set(all_ids)
    records=[]
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
        return h.hexdigest()
    for level in ['half','eighth']:
        for method in ['eventgenesis','v2e']:
            old=args.pilot/level/'pose_bank/events'/method;new=args.full/level/'pose_bank/events'/method
            old_receipt={r['rollout_id']:r for r in json.loads((old/'manifest.json').read_text())['rollouts']}
            new_receipt={r['rollout_id']:r for r in json.loads((new/'manifest.json').read_text())['rollouts']}
            assert set(old_receipt)==set(selected) and set(new_receipt)==set(all_ids)
            for rid in all_ids:assert (new/(rid+'.npy')).is_file()
            for rid in selected:
                path=old/(rid+'.npy');counterpart=new/(rid+'.npy')
                a=np.load(path,allow_pickle=False,mmap_mode='r');b=np.load(counterpart,allow_pickle=False,mmap_mode='r')
                assert a.ndim==2 and a.shape[1]==4 and a.shape==b.shape and a.dtype==b.dtype
                assert sha(path)==old_receipt[rid]['events_sha256'] and sha(counterpart)==new_receipt[rid]['events_sha256']
                h=hashlib.sha256()
                for i in range(0,len(a),65536):
                    x,y=np.asarray(a[i:i+65536]),np.asarray(b[i:i+65536])
                    assert np.array_equal(x,y),'Overlap event realization differs: '+str(path)
                    h.update(x.tobytes())
                records.append(dict(level=level,method=method,rollout_id=rid,shape=a.shape,dtype=str(a.dtype),
                    array_sha256=h.hexdigest(),pilot_file_sha256=old_receipt[rid]['events_sha256'],full_file_sha256=new_receipt[rid]['events_sha256']))
    args.out.write_text(json.dumps(dict(status='passed',real_data_used=False,records=records,
        rule='All NPY event arrays consumed by the pinned training adapter in the eight original trajectories remain identical; coverage alone changes'),indent=2)+'\n')
    print('POSE_OVERLAP_IDENTITY_PASSED',len(records),'canonical NPY event files',flush=True)

if __name__=='__main__':main()
