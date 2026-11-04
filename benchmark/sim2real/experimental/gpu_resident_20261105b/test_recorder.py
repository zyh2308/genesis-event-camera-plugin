"""Use the actual frozen recorder; do not introduce new rounding rules."""
import hashlib,json,sys
from pathlib import Path
import h5py,numpy as np
root=Path(__file__).resolve().parent;P=Path('/home/zhaoyuhan/Eventbased_WAM')
sys.path.insert(0,json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text())['core_path'])
from genesis_event_plugin.recorder import GenesisEventRecorder
out=root/'recorder_regression';out.mkdir(exist_ok=False);payloads=[]
for name,folder in [('baseline','long_baseline'),('resident','long_resident_exact')]:
    path=out/name;assert not path.exists();events=np.load(root/folder/'trial0.npz')['events']
    recorder=GenesisEventRecorder(str(path),resolution=(260,346));recorder.start_episode();recorder.record_aer_events(events);recorder.end_episode();recorder.close()
    with h5py.File(path/'aer_events.h5','r') as h:
        payloads.append(np.asarray(h['events']));assert h['events'].attrs['timestamp_unit']=='us'
assert np.array_equal(payloads[0],payloads[1])
report=dict(passed=True,dataset_bitwise_equal=True,events=len(payloads[0]),dataset_sha256=hashlib.sha256(payloads[0].tobytes()).hexdigest(),scope='actual frozen uint64 microsecond AER recorder dataset; float64-seconds raw outputs remain a separate contract',real_data_used=False)
(root/'recorder_regression.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
