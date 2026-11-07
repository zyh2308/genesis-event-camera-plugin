import hashlib,json,sys,tempfile
from pathlib import Path
import numpy as np
import audit_pose_light_overlap as audit

with tempfile.TemporaryDirectory() as name:
    root=Path(name);pilot=root/'pilot';full=root/'full';pilot.mkdir();full.mkdir()
    ids=[f'train_{i:03}' for i in range(96)];selected=[ids[i] for i in np.linspace(0,95,8,dtype=int)]
    for folder,rids in [(pilot,selected),(full,ids)]:
        (folder/'completion.json').write_text(json.dumps(dict(status='complete',selected=rids)))
        for level in ['half','eighth']:
            for method in ['eventgenesis','v2e']:
                d=folder/level/'pose_bank/events'/method;d.mkdir(parents=True);rows=[]
                for rid in rids:
                    a=np.array([[.1,1,2,1],[.2,2,3,0]],dtype=np.float64);p=d/(rid+'.npy');np.save(p,a)
                    rows.append(dict(rollout_id=rid,events_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
                (d/'manifest.json').write_text(json.dumps(dict(rollouts=rows)))
    sys.argv=['audit','--campaign',str(root),'--full',str(full),'--pilot',str(pilot),'--out',str(root/'good.json')]
    audit.main();assert json.loads((root/'good.json').read_text())['status']=='passed'
    d=full/'half/pose_bank/events/eventgenesis';p=d/'train_000.npy';a=np.load(p);a[0,0]+=.000001;np.save(p,a)
    m=json.loads((d/'manifest.json').read_text());m['rollouts'][0]['events_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();(d/'manifest.json').write_text(json.dumps(m))
    sys.argv[-1]=str(root/'bad.json')
    try:audit.main()
    except AssertionError:pass
    else:raise AssertionError('Changed overlapping event timestamp accepted')
    assert not (root/'bad.json').exists()
out=dict(status='passed',procedural_only=True,real_data_used=False,checks=['all 96 canonical NPY files required in each arm/illumination','32 overlapping files match','valid manifest with changed event array rejected','failure does not publish acceptance'])
Path(__file__).with_name('pose_overlap_preflight.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
