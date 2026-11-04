"""Keep strict raw-bitwise status distinct from scoped numerical acceptance."""
import hashlib,json,os,platform,subprocess
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent;P=Path('/home/zhaoyuhan/Eventbased_WAM');checks={}
for label,path in [('sensor_v1','sensor_tests/results.json'),('sensor_v2','sensor_tests_v2/results.json'),('restore_v1','process_resume_tests/results.json'),('restore_v2','process_resume_tests_v2/results.json'),('actual_v1','actual_sensor_tests/results.json'),('actual_v2','actual_sensor_tests_v2/results.json'),('actual_final','actual_sensor_tests_final/results.json'),('metadata_v2','deferred_tests.json'),('public_api_final','deferred_tests_final.json'),('model_v2','model_regression.json'),('model_final','model_regression_final.json'),('recorder_v2','recorder_regression.json'),('recorder_final','recorder_regression_final.json')]:
    checks[label]=bool(json.loads((root/path).read_text())['passed'])
for folder in ['hole_tests','hole_tests_v2']:
    result=json.loads((root/folder/'results.json').read_text());checks[folder+'_exact']=result['exact_mode_passed'];checks[folder+'_gpu_math_rejected']=not result['gpu_math_bitwise_passed']
analysis=json.loads((root/'paired_suite_final/analysis.json').read_text())
for condition in ['resident_events','resident_full']:
    result=analysis['aggregate'][condition];checks[condition+'_short_events_masks_sampling']=result['all_events_bitwise_equal'] and result['all_masks_and_sampling_equal']
arrays={name:np.load(root/name/'trial0.npz') for name in ['long_baseline','long_baseline_repeat','long_resident_exact','long_resident_final']}
def compare(a,b):
    x=a['events'];y=b['events'];same=x.shape==y.shape
    return dict(events=len(y),same_shape=same,raw_events_bitwise_equal=bool(np.array_equal(x,y)),addresses_order_equal=bool(same and np.array_equal(x[:,1:],y[:,1:])),
                changed_timestamps=int(np.count_nonzero(x[:,0]!=y[:,0])) if same else None,max_time_error_s=float(np.abs(x[:,0]-y[:,0]).max(initial=0)) if same else None,
                state_max_abs={key:float(np.abs(a[key].astype(float)-b[key].astype(float)).max(initial=0)) for key in a.files if key.startswith('state_')})
long={name:compare(arrays['long_baseline'],arrays[name]) for name in arrays if name!='long_baseline'}
checks['long_counts_addresses_rng']=all(row['same_shape'] and row['addresses_order_equal'] and row['state_max_abs']['state_rng_state']==0 for row in long.values())
checks['reference_repeat_variability_observed']=any(long['long_baseline_repeat']['state_max_abs'].values())
base=np.load(root/'worker_throughput/baseline_2/worker_0/trial0.npz')['events'];worker_quality=[]
for folder in ['worker_throughput','worker_throughput_final']:
    for group in json.loads((root/folder/'results.json').read_text())['groups']:
        exact=[]
        if group['passed']:
            for worker in range(group['workers']):
                for trial in range(3):
                    events=np.load(root/folder/(group['variant']+'_'+str(group['workers']))/('worker_'+str(worker))/('trial'+str(trial)+'.npz'))['events'];exact.append(bool(np.array_equal(base,events)))
        passed=group['passed'] and bool(exact) and all(exact);checks[folder+'_'+group['variant']+'_'+str(group['workers'])]=passed
        worker_quality.append(dict(folder=folder,variant=group['variant'],workers=group['workers'],events_bitwise_equal=passed,compared_trials=len(exact)))
pointer=json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text());core=Path(pointer['core_path'])
profile_path=P/'runs/sim2real/20261103-stage9-direct-v1/sources/stage9_sensor_profile.json';profile=json.loads(profile_path.read_text())
hashes={name:hashlib.sha256((core/'genesis_event_plugin'/name).read_text().encode()).hexdigest() for name in profile['core_canonical_LF_sha256']}
checks['frozen_17_files_unchanged']=hashes==profile['core_canonical_LF_sha256']
checks['profiler_observed_actual_gpu_copies']=all(bool(json.loads(path.read_text())['copies']) for path in root.glob('profile_*/copy_summary.json'))
exits={path.name:json.loads(path.read_text())['exit_code'] for path in sorted(root.glob('*.done.json'))};checks['all_completed_jobs_exit_zero']=all(code==0 for code in exits.values())
strict=dict(passed=long['long_resident_final']['raw_events_bitwise_equal'],scope='complete one-second float64 raw event arrays, including timestamps',candidate=long['long_resident_final'],reference_repeat=long['long_baseline_repeat'])
(root/'strict_raw_gate.json').write_text(json.dumps(strict,indent=2)+'\n')
report=dict(scope_passed=all(checks.values()),strict_raw_bitwise_passed=strict['passed'],scope='tested exact sensor/fallback/mask contracts, native AER payload and fixed voxel/model regression; no universal end-to-end float64 bitwise claim',checks=checks,
            long_comparisons=long,worker_quality=worker_quality,job_exit_codes=exits,frozen_release=pointer['release_commit'],frozen_core_hashes=hashes,profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in root.glob('*.py')},real_data_used=False,host=platform.node(),cpu_count=os.cpu_count())
(root/'final_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','frozen_core_hashes','worker_quality']},indent=2),flush=True)
assert report['scope_passed'],checks
