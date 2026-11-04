"""Independent reference-repeat attribution; never turn failed raw gates into passes."""
import hashlib,itertools,json,sys
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'simulator_gpu_resident_20261105b'
assert json.loads((ROOT/'14_reference_repeat.done.json').read_text())['exit_code']==0
assert json.loads((ROOT/'static_long_observables.json').read_text())['observable_passed']
paths={'reference0':OLD/'long_baseline/trial0.npz','reference1':OLD/'long_baseline_repeat/trial0.npz','reference2':ROOT/'reference_repeat_current/trial0.npz','static':ROOT/'static_long/trial0.npz'}
archives={k:np.load(v) for k,v in paths.items()};events={k:v['events'] for k,v in archives.items()};pairs=[]
for left,right in itertools.combinations(events,2):
    a,b=events[left],events[right]
    assert a.shape==b.shape and np.array_equal(a[:,1:],b[:,1:])
    assert np.array_equal(archives[left]['state_rng_state'],archives[right]['state_rng_state'])
    delta=np.abs(a[:,0]-b[:,0]);pairs.append(dict(left=left,right=right,count=len(a),xy_p_order_exact=True,rng_exact=True,
        changed=int(np.count_nonzero(delta)),max_s=float(delta.max()),rms_s=float(np.sqrt(np.mean(delta**2))),raw_3ns_passed=bool(delta.max()<=3e-9)))
sys.path.insert(0,str(ROOT/'release_candidate_static'))
from genesis_event_plugin.recorder import GenesisEventRecorder
payloads=[]
for name,array in events.items():
    out=ROOT/('precision_review_aer_'+name);recorder=GenesisEventRecorder(str(out),resolution=(260,346))
    recorder.start_episode();recorder.record_aer_events(array);recorder.end_episode();recorder.close()
    with h5py.File(out/'aer_events.h5','r') as h:payloads.append(np.asarray(h['events']))
assert all(np.array_equal(payloads[0],x) for x in payloads)
reference_max=max(x['max_s'] for x in pairs if x['right']!='static')
static0=next(x for x in pairs if x['left']=='reference0' and x['right']=='static')
fresh=next(x for x in pairs if x['left']=='reference2' and x['right']=='static')
assert reference_max>3e-9,'reference-repeat diagnostic does not support the observed failed 3ns assumption'
assert fresh['max_s']<reference_max,'fresh-reference comparison exceeds observed reference repeat scale'
active=json.loads((ROOT/'acceptance_scripts/actual_sensor_active/results.json').read_text())
assert active['passed'] and all(len(x['intervals'])==2 for x in active['cases'])
review=dict(qualified_output_contract_passed=True,strict_raw_bitwise_passed=False,diagnostic_3ns_gate_passed=False,
    pairs=pairs,reference_repeat_max_s=reference_max,static_vs_original_max_s=static0['max_s'],static_vs_fresh_reference_max_s=fresh['max_s'],
    recorder_all_four_datasets_exact=True,recorder_sha256=hashlib.sha256(payloads[0].tobytes()).hexdigest(),
    component_same_input_sensor_exact=True,source_precision_noise_and_occlusion_unchanged=True,
    decision='freeze only the declared Stage9 numerical/non-deterministic reference and tested microsecond/count-output scope; no strict float64 bitwise release',
    interpretation='independent unchanged reference also fails the 3ns diagnostic; static-vs-fresh reference is within observed reference repeat scale; this is supporting evidence, not a universal error bound',
    failed_gates_preserved=['13_final_long_accepted.log','strict_raw_gate from previous frozen baseline comparison'],real_data_used=False)
(ROOT/'PRECISION_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps({k:v for k,v in review.items() if k!='pairs'}))
