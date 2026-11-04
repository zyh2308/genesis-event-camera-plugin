"""Independent, fail-closed acceptance summary over retained raw outputs."""
import hashlib,json,os,platform,subprocess,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent;P=Path('/home/zhaoyuhan/Eventbased_WAM');v2=root/'final_v2'
checks={}
for name,path in [('sensor_v1',root/'sensor_tests/results.json'),('actual_inputs_v1',root/'actual_input_tests/results.json'),('sensor_v2',v2/'sensor_tests/results.json'),('actual_inputs_v2',v2/'actual_input_tests/results.json'),('process_resume_v2',v2/'process_resume_tests/results.json'),('geometry_v2',v2/'geometry_tests.json'),('model_v1',root/'model_regression.json'),('model_v2',v2/'model_regression.json')]:
    checks[name]=json.loads(path.read_text())['passed']
analysis=json.loads((v2/'analysis.json').read_text())
for variant,summary in analysis['aggregate'].items():
    checks[variant+'_events_masks_sampling']=all(summary[k] for k in ['all_events_bitwise_equal','all_masks_and_sampling_equal'])
reference=np.load(v2/'worker_throughput/baseline_1/worker_0/trial0.npz')['events']
worker_results=[]
workers=json.loads((v2/'worker_throughput/results.json').read_text())
for group in workers['groups']:
    comparisons=[]
    if group['passed']:
        for i in range(group['workers']):
            for trial in range(3):
                path=v2/'worker_throughput'/(group['variant']+'_'+str(group['workers']))/('worker_'+str(i))/('trial'+str(trial)+'.npz')
                events=np.load(path)['events'];comparisons.append(bool(np.array_equal(reference,events)))
    passed=group['passed'] and bool(comparisons) and all(comparisons)
    checks['workers_'+group['variant']+'_'+str(group['workers'])]=passed
    worker_results.append(dict(variant=group['variant'],workers=group['workers'],all_events_bitwise_equal=passed,compared_trials=len(comparisons)))
left=np.load(v2/'long_baseline/trial0.npz');right=np.load(v2/'long_fused/trial0.npz')
event_equal=bool(np.array_equal(left['events'],right['events']))
long_checks=dict(intervals=20,simulated_s=.2,events_baseline=len(left['events']),events_candidate=len(right['events']),events_bitwise_equal=event_equal,
                 state_max_abs={k:float(np.abs(left[k].astype(float)-right[k].astype(float)).max(initial=0)) for k in left.files if k!='events'},
                 rng_exact=bool(np.array_equal(left['state_rng_state'],right['state_rng_state'])))
for variant in ['long_baseline','long_fused']:
    summary=json.loads((v2/variant/'summary.json').read_text());long_checks[variant]=dict(wall_s=summary['median_wall_s'],phases=summary['rows'][0]['phases'],peak_allocated_bytes=summary['rows'][0]['peak_allocated_bytes'],fallbacks=summary['rows'][0]['fallbacks'])
checks['long_trace_events_rng']=event_equal and long_checks['rng_exact']
profile=json.loads((P/'runs/sim2real/20261103-stage9-direct-v1/sources/stage9_sensor_profile.json').read_text())
pointer=json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text());core=Path(pointer['core_path'])
core_hashes={name:hashlib.sha256((core/'genesis_event_plugin'/name).read_text().encode()).hexdigest() for name in profile['core_canonical_LF_sha256']}
checks['frozen_core_all_17_unchanged']=core_hashes==profile['core_canonical_LF_sha256']
exits={p.name:json.loads(p.read_text())['exit_code'] for p in sorted(root.glob('*.done.json'))}
checks['completed_jobs_exit_zero']=all(code==0 for code in exits.values())
report=dict(passed=all(checks.values()),checks=checks,long_trace=long_checks,worker_quality=worker_results,job_exit_codes=exits,
            frozen_core_release=pointer['release_commit'],frozen_core_hashes=core_hashes,real_data_used=False,
            environment=dict(host=platform.node(),python=sys.version,cpu_count=os.cpu_count(),load=os.getloadavg(),gpu=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()))
(root/'final_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['frozen_core_hashes','environment']},indent=2),flush=True)
assert report['passed'],checks
