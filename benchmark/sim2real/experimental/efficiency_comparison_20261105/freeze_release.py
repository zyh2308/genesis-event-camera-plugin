"""Qualified freeze is conditional on measured advantage AND actual acceptance."""
import datetime,hashlib,json,shutil,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
P=Path('/home/zhaoyuhan/Eventbased_WAM')
package=ROOT/'release_candidate_static'
analysis=json.loads((ROOT/'matched_suite_static_analysis.json').read_text())
acceptance=json.loads((ROOT/'release_acceptance_final.json').read_text())
post=json.loads((ROOT/'final_metrology.json').read_text())
native=json.loads((ROOT/'genesis_smoke_accepted.json').read_text())
assert json.loads((ROOT/'native_acceptance.json').read_text())['passed']
assert analysis['qualifies_practical_advantage'] and acceptance['passed'] and post['passed'] and native['passed']
precision_review=json.loads((ROOT/'PRECISION_REVIEW.json').read_text())
assert precision_review['qualified_output_contract_passed'] and not precision_review['diagnostic_3ns_gate_passed']
assert json.loads((ROOT/'static_long_observables.json').read_text())['observable_passed']
assert json.loads((ROOT/'v2e_pin_verification.json').read_text())['exit_code']==0
renderer=json.loads((ROOT/'renderer_dependency_verification.json').read_text())
assert all(x['sha256']==x['expected'] for x in renderer['files'])
resume=next(x for x in json.loads((ROOT/'release_acceptance_retry1.json').read_text())['checks'] if x['name']=='test_process_resume_v2')
assert resume['exit_code']==0
active=json.loads((ROOT/'acceptance_scripts/actual_sensor_active/results.json').read_text())
assert active['passed'] and len(active['cases'])==5 and all(len(x['intervals'])==2 for x in active['cases'])
long=json.loads((ROOT/'controls/batch16_long.json').read_text());assert long['completed']
med={v:statistics.median(x['wall_s'] for x in long['rows'] if x['variant']==v) for v in ['ours_resident','v2e_ssm_memory']}
assert med['v2e_ssm_memory']/med['ours_resident']>=1.20,med
profile=json.loads((package/'stage9_sensor_profile.json').read_text())
reference=Path(json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text())['core_path'])/'genesis_event_plugin'
for name,digest in profile['core_canonical_LF_sha256'].items():
    for path in [reference/name,package/'genesis_event_plugin'/name]:assert hashlib.sha256(path.read_text().encode()).hexdigest()==digest,str(path)
# Re-audit saved native float outputs, not just exit codes or microsecond files.
summary=json.loads((ROOT/'matched_suite_static/summary.json').read_text())
for row in summary['rows']:
    assert row['substeps']==[377]*20 and abs(row['simulated_s']-.2)<1e-12
    if row['variant']=='v2e_ssm_memory':assert row['input_frames']==7540 and abs(row['max_dt_s']-.01/377)<1e-12
    path=ROOT/'matched_suite_static'/(row['case']+'_'+row['variant']+'_'+str(row['trial'])+'_native.npy')
    a=np.load(path);assert len(a)==row['events']['count'] and a.ndim==2 and a.shape[1]==4
    assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==row['events']['sha256']
    assert np.isfinite(a).all() and np.all(np.diff(a[:,0])>=0) and a[:,0].min()>=0 and a[:,0].max()<=row['simulated_s']+1e-7
kernel=ROOT/'release_build/eventgenesis_precision_sensor.so';assert kernel.exists()
files={str(x.relative_to(package)):hashlib.sha256(x.read_bytes()).hexdigest() for x in package.rglob('*') if x.is_file() and '__pycache__' not in str(x) and x.name!='PREPARATION_MANIFEST.json'}
raw_gate=json.loads((ROOT.parent/'simulator_gpu_resident_20261105b/strict_raw_gate.json').read_text())
assert raw_gate['passed'] is False
release=dict(release_id='eventgenesis-gpu-stage9-20261105',baseline_id='eventgenesis-gpu-stage9-20261105',tag='freeze/eventgenesis-gpu-stage9-20261105',repository='https://github.com/zyh2308/genesis-event-camera-plugin.git',status='frozen_qualified_profile',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    entry_point='genesis_event_plugin.optimized.OptimizedGenesisPhysicsEventPlugin',reference_commit='62da4f35ce3c5c96f04268bd1687a27657af9238',
    files=files,package_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
    kernel_binary=dict(path=str(kernel),sha256=hashlib.sha256(kernel.read_bytes()).hexdigest()),
    full_raw_float_bitwise_gate_passed=False,raw_float_failure_retained=True,same_input_sensor_and_rng_exact=True,
    diagnostic_3ns_gate_passed=False,precision_review=precision_review,
    process_restore_check=resume,renderer_dependency=renderer,v2e_source_pin_verified=True,
    recorder_regression_exact=True,native_genesis_smoke_passed=True,
    speedup=analysis['geometric_mean_speedup'],scene_bootstrap_95ci=analysis['scene_bootstrap_95ci'],batch16_long_speedup=med['v2e_ssm_memory']/med['ours_resident'],
    supported_profile='archived complete Stage9 settings; CUDA Torch; hold/coverage; 320x240 or 346x260; no margin/state provider',
    unsupported_configuration_policy='unchanged reference path with explicit reason',
    formal_training_and_frozen_reference_changed=False)
destination=P/'frozen_baselines'/release['release_id'];destination.mkdir(exist_ok=False)
frozen_core=destination/'core'
for rel,digest in files.items():
    path=frozen_core/rel;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write((package/rel).read_bytes())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
release.update(core_path=str(frozen_core),repository_release_path='frozen_versions/'+release['release_id'],repository_baseline_path='frozen_baselines/'+release['release_id']+'/BASELINE.json')
(destination/'BASELINE.json').write_text(json.dumps(release,indent=2)+'\n')
(ROOT/'FROZEN_RELEASE.json').write_text(json.dumps(release,indent=2)+'\n')
print(json.dumps(dict(frozen=True,release_id=release['release_id'],package_sha256=release['package_sha256'],speedup=release['speedup'],raw_bitwise_gate=False)))
