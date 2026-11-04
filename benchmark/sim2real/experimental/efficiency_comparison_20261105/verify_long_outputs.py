"""One-second final-source conformance; no training and no real evaluation."""
import hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'simulator_gpu_resident_20261105b'
assert json.loads((ROOT/'final_metrology.json').read_text())['passed']
assert json.loads((ROOT/'native_acceptance.json').read_text())['passed']
tools=ROOT.parent/'sim2real/20261103-stage9-direct-v1/runtime/esim_build_tools'
env=dict(os.environ,CUDA_HOME=str(tools),CXX=str(tools/'bin/x86_64-conda-linux-gnu-c++'),CC=str(tools/'bin/x86_64-conda-linux-gnu-cc'),MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.0',EVENTGENESIS_GPU_BUILD_DIR=str(ROOT/'release_build'))
env['PATH']=str(tools/'bin')+os.pathsep+env['PATH']
a=np.load(OLD/'long_baseline/trial0.npz');b=np.load(ROOT/'static_long/trial0.npz')
ea,eb=a['events'],b['events'];assert ea.shape==eb.shape and np.array_equal(ea[:,1:],eb[:,1:])
error=float(np.abs(ea[:,0]-eb[:,0]).max(initial=0));raw_3ns_passed=error<=3e-9
assert np.array_equal(a['state_rng_state'],b['state_rng_state'])
sys.path.insert(0,str(ROOT/'release_candidate_static'))
from genesis_event_plugin.recorder import GenesisEventRecorder
payloads=[]
for tag,events in [('reference',ea),('static',eb)]:
    folder=ROOT/('static_long_recorder_'+tag)
    recorder=GenesisEventRecorder(str(folder),resolution=(260,346));recorder.start_episode();recorder.record_aer_events(events);recorder.end_episode();recorder.close()
    with h5py.File(folder/'aer_events.h5','r') as h:payloads.append(np.asarray(h['events']))
assert np.array_equal(*payloads)
# Same fixed checkpoint and representation as the earlier regression, CPU only.
P=Path('/home/zhaoyuhan/Eventbased_WAM');repair=P/'runs/robot_r1_5_sparse_time_repair_20261101a'
checkpoint=repair/'fit/spatial_event_state.pt';source=repair/'source/benchmark/robot_closed_loop/event_state_model.py'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='dda51f2d5a9665295eeeeccd98682183946f8b080ceb610f82c2baeec3f8d896'
spec=importlib.util.spec_from_file_location('static_fixed_regression',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
import torch
torch.set_num_threads(2);model,_,_,_=module.load_checkpoint(str(checkpoint),'cpu');model.eval();results=[]
for cutoff in np.arange(1.025,2.+1e-9,.005):
    va=module.five_bin_tensor(ea,float(cutoff),346,260);vb=module.five_bin_tensor(eb,float(cutoff),346,260)
    assert np.array_equal(va,vb)
    with torch.inference_mode():
        pa=model(torch.tensor(np.log1p(va).reshape(1,10,64,64)))[0];pb=model(torch.tensor(np.log1p(vb).reshape(1,10,64,64)))[0]
        ua=module._softargmax_uv(pa,(346,260)).numpy()[0];ub=module._softargmax_uv(pb,(346,260)).numpy()[0]
    delta=float(np.linalg.norm(ua-ub));assert delta==0
    results.append(dict(cutoff_s=float(cutoff),voxel_exact=True,pixel_change=delta))
report=dict(observable_passed=True,numeric_guard_3ns_passed=bool(raw_3ns_passed),strict_raw_bitwise_passed=bool(np.array_equal(ea,eb)),events=len(ea),raw_bitwise_equal=bool(np.array_equal(ea,eb)),max_time_error_s=error,
    xy_p_order_exact=True,rng_exact=True,recorder_payload_exact=True,recorder_sha256=hashlib.sha256(payloads[0].tobytes()).hexdigest(),
    fixed_model_queries=len(results),max_pixel_change=0.,results=results,
    scope='one-second raw, actual AER and fixed count-voxel/localization regression; not all downstream representations',
    reference_summary_sha256=hashlib.sha256((OLD/'long_baseline/summary.json').read_bytes()).hexdigest())
(ROOT/'static_long_observables.json').write_text(json.dumps(report,indent=2)+'\n');print('LONG_FINAL_PASS',json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
