from pathlib import Path
import json,h5py,numpy as np
R=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real');E=R/'benchmark/sim2real/evidence/S3/20261016-stage3';O=Path(__file__).resolve().parent
root=Path('/home/科研/Eventbased_WAM/data/sim2real/stage3/real/MVSEC/indoor_flying');out={}
with h5py.File(root/'indoor_flying1_data.hdf5','r') as full,h5py.File(root/'indoor_flying1_registered_1s_data.hdf5','r') as sub,h5py.File(root/'indoor_flying1_gt.hdf5','r') as gt,h5py.File(root/'indoor_flying1_registered_1s_gt.hdf5','r') as sg:
    fullb=full['davis/left'];b=sub['davis/left'];g=gt['davis/left'];gs=sg['davis/left'];aps=fullb['image_raw_ts'][:].reshape(-1);times=g['flow_dist_ts'][:].reshape(-1)
    out['source_data_fields_equal']={k:bool(np.array_equal(b[k][:],fullb[k][:len(b[k])],equal_nan=True)) for k in ['events','image_raw_ts','image_raw_event_inds','image_raw']}
    out['source_GT_fields_equal']={k:bool(np.array_equal(gs[k][:],g[k][:len(gs[k])],equal_nan=True)) for k in ['flow_dist','flow_dist_ts']}
    events=fullb['events'];first=max(float(events[0,2]),float(times[0]));i=int(np.searchsorted(aps,first,side='left'));end=float(aps[i])+1.0
    expected={str(dt):[j for j in range(i,len(aps)-dt) if aps[j+dt]<=end] for dt in [1,4]}
    m=json.loads((E/'access_and_download/mvsec_fixed_fragment_manifest.json').read_text())
    out.update({'first_common_APS_index':i,'registered_indices_equal_rule':{str(dt):expected[str(dt)]==m['derived_slice']['aps_indices_for_inference'][f'dt{dt}'] for dt in [1,4]},'window_counts':{k:len(v) for k,v in expected.items()},'sensor_hw':list(fullb['image_raw'].shape[1:])})
(O/'source_slice_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
