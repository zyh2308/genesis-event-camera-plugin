import os,json,hashlib,time
from pathlib import Path
import numpy as np,h5py
R=Path(os.environ['EPOSE_REPAIR_RUN']);bank=R/'bank_expanded';rows=[]
with h5py.File(bank/'shared_trajectory.h5','w') as out:
 dst=out.create_group('episodes')
 for j in range(8):
  m=json.loads((bank/f'manifest_worker{j}.json').read_text());rows.extend(m['rollouts'])
  with h5py.File(bank/f'shared_trajectory_worker{j}.h5','r') as src:
   for name in src['episodes']:src.copy(src['episodes'][name],dst,name=name)
 names=sorted(dst);assert len(names)==16 and len(set(names))==32
 for name in names:
  g=dst[name];assert np.allclose(np.asarray(g['frame_timestamps_s']),np.arange(81)/1000,atol=1e-12);assert g['eventgenesis_events_txyp'].shape[0]>0
manifest={'protocol':json.loads((R/'bank_protocol.json').read_text()),'rollouts':sorted(rows,key=lambda r:r['rollout_id']),'max_geometry_error_px':max(r['max_geometry_error_px'] for r in rows),'worker_count':8,'shared_bank_sha256':hashlib.sha256((bank/'shared_trajectory.h5').read_bytes()).hexdigest(),'all_rollouts_completed':True,'merge_unix':time.time()};(bank/'manifest.json').write_text(json.dumps(manifest,indent=2))
assert not (R/'bank').exists();bank.rename(R/'bank');print(json.dumps({'episodes':len(rows),'max_geometry_error_px':manifest['max_geometry_error_px'],'shared_sha256':manifest['shared_bank_sha256']}),flush=True)
