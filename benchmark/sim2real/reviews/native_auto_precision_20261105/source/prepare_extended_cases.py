"""Index actual paired fixtures; never replace production raw events."""
from pathlib import Path
import json
import numpy as np
W=Path(__file__).resolve().parent
R=W.parent/'sim2real/20261103-stage9-direct-v1'
assert json.loads((W/'native_auto/summary.json').read_text())['status']=='complete'
out=W/'conformance_events';out.mkdir(exist_ok=False)
cases=[]
for case in ['s0_00','s1_00','s2_00','s3_00','train_000']:
    kind='pose' if case.startswith('train') else 'rich'
    origin=0. if kind=='pose' else 1.
    paths={}
    for role,filename in [('reference',case+'_original_reference.npy'),('candidate',case+'_ours_resident_0.npy')]:
        raw=np.load(W/'native_auto'/filename)
        assert raw.dtype==np.float64
        raw=raw.copy();raw[:,0]+=origin
        p=out/(case+'_'+role+'.npy');np.save(p,raw,allow_pickle=False);paths[role]=str(p)
    spec=dict(id=case+'_native_adaptive',episode_id=case,physical_h5=str(R/(kind+'_bank/physical')/(case+'.h5')),
              start_s=origin+(0.04 if kind=='pose' else 0),end_s=origin+0.2,**paths)
    if kind=='rich':
        spec.update(flow_template_npz=str(R/'adapters/eventgenesis/flow/eventgenesis/train'/(case+'.npz')),
                    reconstruction_template_h5=str(R/'adapters/eventgenesis/reconstruction/eventgenesis/train'/(case+'.h5')))
    cases.append(spec)
manifest={'required_tasks':['flow','pose','reconstruction'],'augmentation_seeds':[0,1,2],'cases':cases,
          'scope':'independent fixture replays, not the full originating episode history; matched-object pose input is now included',
          'timestamps_origin_restored_without_quantization':True}
(W/'cases_extended.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('EXTENDED_CASES_READY',len(cases),flush=True)
