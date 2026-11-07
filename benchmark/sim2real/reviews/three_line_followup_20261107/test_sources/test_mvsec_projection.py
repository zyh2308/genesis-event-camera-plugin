import json,tempfile
from pathlib import Path
import h5py,numpy as np
from project_transfer_mvsec import project,array_digest
with tempfile.TemporaryDirectory() as folder:
    p=Path(folder);source=p/'source.hdf5';target=p/'left.hdf5'
    with h5py.File(source,'x') as f:
        f.attrs['schema']='procedural-fixture';d=f.create_group('davis');d.attrs['clock']='seconds'
        left=d.create_group('left');left.attrs['sensor']='DAVIS'
        left.create_dataset('events',data=np.array([[1,2,148000000.0000001,1],[2,3,148000000.000001,0]],dtype=np.float64))
        left.create_dataset('image_raw',data=np.arange(24,dtype=np.uint8).reshape(2,3,4),compression='gzip')
        left.create_dataset('image_raw_ts',data=[148000000.000001,148000000.02])
        left.create_dataset('image_raw_event_inds',data=np.array([1,2],dtype=np.uint64))
        values=np.array([0.,np.nan,np.inf,-np.inf],dtype=np.float32).reshape(1,2,1,2)
        left.create_dataset('flow_dist',data=values)
        left.create_dataset('flow_dist_ts',data=[148000000.])
        d.create_group('right').create_dataset('unused',data=[4,5,6])
    evidence=project(source,target)
    with h5py.File(source,'r') as a,h5py.File(target,'r') as b:
        assert 'right' not in b['davis'] and b.attrs['schema']==a.attrs['schema'] and b['davis'].attrs['clock']==a['davis'].attrs['clock']
        assert b['davis/left'].attrs['sensor']==a['davis/left'].attrs['sensor']
        for name in a['davis/left']:
            assert array_digest(a['davis/left/'+name])==array_digest(b['davis/left/'+name])
    try:project(source,target)
    except AssertionError:pass
    else:raise AssertionError('Existing target overwritten')
result=dict(status='passed',procedural_only=True,real_data_used=False,checks=[
    'every left array preserves dtype, shape and canonical byte hash','float64 epoch timestamps preserved',
    'NaN/Inf flow bits preserved; invalidity not changed','sensor and root attributes retained',
    'unused right group omitted only','existing output refused'])
Path(__file__).with_name('mvsec_projection_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
