"""Exercise exact production label/adapter routines before bulk generation ends."""
import hashlib,json,sys,time
from pathlib import Path
import h5py,numpy as np
R=Path(sys.argv[1]);P=Path('/home/zhaoyuhan/Eventbased_WAM');pointer=json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text());sys.path[:0]=[str(R/'sources/benchmark/sim2real/scripts'),pointer['core_path']]
import build_e2vid_stage4_10ms_data as adapter
import prepare_stage4_steflow_bank as flow_adapter
out=R/'preflight/early_data';out.mkdir(parents=True,exist_ok=True);rows=[];started=time.monotonic()
with h5py.File(R/'rich_bank/physical/s0_00.h5','r') as h:
 g=h['episodes/s0_00'];f10,v10,parity=adapter.build_labels(g,True);assert parity==66
 label={'direct_10ms_shape':list(f10.shape),'direct_30ms_parity_pairs':parity,'finite':bool(np.isfinite(f10).all()),'valid_fraction':float(v10.mean())}
 (out/'label_audit.json').write_text(json.dumps(label,indent=2)+'\n')
while not all((R/'rich_bank/events'/method/'s0_00.npy').exists() for method in ['eventgenesis']):
 assert time.monotonic()-started<2*3600;time.sleep(10)
raw=np.load(R/'rich_bank/events/eventgenesis/s0_00.npy');row=json.loads((R/'rich_bank/events/eventgenesis/s0_00.json').read_text())
flow_adapter.load_raw_events=lambda *args:raw
def events(*args):
 q=raw.copy();q[:,3]=(q[:,3]>0).astype(np.uint8);return q,{'source':'new-core early data audit','time_unit':'seconds'}
adapter.load_generator_events=events
class Episode:
 def __init__(self,g):self.g=g;self.attrs=g.attrs;self.name=g.name
 def __getitem__(self,key):return np.asarray(row['interval_event_counts']) if key=='eventgenesis_interval_counts_10ms' else self.g[key]
class Source:
 def __init__(self,g):self.g=g
 def __getitem__(self,key):return Episode(self.g)
with h5py.File(R/'rich_bank/physical/s0_00.h5','r') as h:
 flow=flow_adapter.convert_rollout('eventgenesis','s0_00',h,out,out/'s0_00.npz',{})
 recon=adapter.export_h5(Source(h['episodes/s0_00']),{'s0_00':(f10,v10)},'eventgenesis','s0_00',out,out/'s0_00.h5')
(out/'result.json').write_text(json.dumps({'passed':True,'formal_training_data_modified':False,'flow':flow,'reconstruction':recon,'labels':label,'real_data_used':False},indent=2)+'\n');print('EARLY_DATA_AUDIT_PASS',flush=True)
