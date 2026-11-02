import pathlib,json,hashlib,subprocess,datetime,fcntl,csv,math
base=pathlib.Path('/home/zhaoyuhan/Eventbased_WAM');root=base/'runs/robot_cuda_r2_server_20261102a'
registry=base/'.gpu_reservations'
audit_code=__AUDIT_CODE__
exec(compile(audit_code,'audit_r2_g1.py','exec'),{})
audit=json.loads((root/'R2_independent_audit.json').read_text())
assert audit['R2_exit']=='0' and audit['completed_episodes']==3
paused=json.loads((root/'USER_PAUSE.json').read_text());resumed=json.loads((root/'USER_RESUME.json').read_text())
def dt(v):return datetime.datetime.fromisoformat(v.replace('Z','+00:00'))
pause_s=(dt(resumed['time_utc'])-dt(paused['time_utc'])).total_seconds()
progress=json.loads((root/'R2_progress.json').read_text())
timing=dict(raw_wall_s=progress['wall_s'],user_requested_pause_s=pause_s,active_wall_excluding_pause_s=progress['wall_s']-pause_s,
 scope='End-to-end 3-episode pipeline; raw per-episode and pipeline times retained; third episode includes pause',
 episodes=[dict(episode=e['episode'],raw_wall_s=float(next(csv.DictReader((root/'R2_events_seed2026110200/episodes'/e['episode']/'episode.csv').open()))['task_rollout_wall_s'])) for e in audit['episodes']])
timing['episodes'][2]['active_wall_excluding_pause_s']=timing['episodes'][2]['raw_wall_s']-pause_s
(root/'R2_timing_pause_corrected.json').write_text(json.dumps(timing,indent=2))
summaries={}
source=base/'runs/robot_sparse_cuda_acceptance_20261102c'
expected=json.loads((source/'local_cuda_20261102a/acceptance_summary.json').read_text())['core_canonical_LF_sha256']
for name,digest in expected.items():assert hashlib.sha256((source/'core/genesis_event_plugin'/name).read_text().encode()).hexdigest()==digest
for runid in ['robot_cuda_r2_server_20261102a','robot_sparse_cuda_v100_20261102b']:
 r=base/'runs'/runid;p=json.loads((r/'progress.json').read_text());assert all(s['exit_code']==0 for s in p['stages'])
 c=json.loads((r/'cuda_controls/cuda_validation.json').read_text());cache=json.loads((r/'cuda_cache/summary.json').read_text());pixel=json.loads((r/'cuda_cache/pixel_equivalence.json').read_text())
 assert all(x['status']=='PASS' for x in [c,cache,pixel])
 for name,digest in expected.items():assert c['core_sha256'][name]==digest
 summaries[runid]=dict(GPU=c['GPU'],control_checks=len(c['tests']),cache_events_per_cuda_backend=sum(x['events'] for x in cache['results'] if x['backend']=='torch' and x['device'].startswith('cuda')),
  mask_mismatch=sum(x['support_mask_mismatch'] for x in cache['results']),event_addresses_polarities_exact=all(x['same_event_addresses_polarities'] for x in cache['results']),
  event_time_max_difference_s=max(x['max_event_time_difference_s'] or 0. for x in cache['results']),queries=pixel['queries'],pixel_generation_change=pixel['max_pixel_change_from_generation'],pixel_inference_change=pixel['max_pixel_change_from_inference'])
 manifest={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*') if p.is_file() and p.name not in ['SHA256SUMS.server','server_closeout_summary.json']}
 (r/'SHA256SUMS.server').write_text(''.join(d+'  '+n+'\n' for n,d in sorted(manifest.items())))
with (registry/'.gpu_reservations.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);p=registry/(root.name+'.json');record=json.loads(p.read_text())
 assert record['run_id']==root.name
 assert not pathlib.Path('/proc/4005623').exists()
 record.update(status='released_R2_completed',stage='R2_completed_0_of_3_strict',released_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 p.write_text(json.dumps(record,indent=2))
result=dict(CUDA=summaries,R2=dict(strict_successes=audit['strict_successes'],episodes=3,causal_time_violations=sum(x['causal_time_violations'] for x in audit['episodes']),
 all_contact=all(x['first_contact_s'] is not None for x in audit['episodes']),all_no_lift=all(x['lift_start_s'] is None for x in audit['episodes'])),timing=timing,core_hashes_match=True,GPU0_released=True)
(root/'server_closeout_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
