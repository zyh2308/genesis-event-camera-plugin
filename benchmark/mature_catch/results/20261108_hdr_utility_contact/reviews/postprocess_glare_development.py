"""Audit all sealed native-light development episodes using existing analyzers.

Requires the already deployed analyze_cadence.py / audit_cadence_final.py.
No renderer, inference, policy, simulator edits or outcome-based case selection.
"""
import argparse,csv,datetime,hashlib,importlib.util,json,math,subprocess,sys,time
from pathlib import Path

def read(path):return json.loads(Path(path).read_text())
def sha(path,canonical=False):
 h=hashlib.sha256()
 if canonical:return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
 return h.hexdigest()

def require(condition,message):
 if not condition:raise AssertionError(message)

def verify_inputs(spec):
 for name,digest in spec['runtime_files'].items():require(sha(Path(spec['source'])/name)==digest,'runtime source changed: '+name)
 for name,digest in spec['core_files'].items():require(sha(Path(spec['core_root'])/'genesis_event_plugin'/name,True)==digest,'frozen core changed: '+name)
 for name,digest in spec.get('v2e_source_files',{}).items():require(sha(Path(spec['v2e_source'])/name)==digest,'native V2E source changed: '+name)
 for field,digest_field in [('checkpoint','checkpoint_sha256'),('profile_file','profile_sha256'),('v2e_checkpoint','v2e_checkpoint_sha256')]:require(sha(spec[field])==spec[digest_field],field+' changed')

def audit_superseded_admissions(root,spec,stems,final=False):
 """Whitelist only proven pre-child admission races, never an actor failure."""
 notes=[]
 for stem in stems:
  require(Path(stem).name==stem and '/' not in stem and '\\' not in stem and not stem.endswith('.json'),'Pass an exact admission job stem, not a path or pattern')
  failed_path=root/(stem+'.result.json');job_path=root/(stem+'.job.json');failed=read(failed_path);job=read(job_path)
  require(failed.get('status')=='failed' and 'New process appeared before launch' in failed.get('error',''),'Whitelisted result is not the specific guarded admission race')
  require(not (root/(stem+'.log')).exists() and not (root/(stem+'.allocation.json')).exists(),'Cannot supersede an attempt that reached child-log/allocation creation')
  command=job['command'];require('shard0' in ' '.join(command) or ('--shard' in command and command[command.index('--shard')+1]=='0'),'This exception is limited to the planned shard0 admission')
  require({float(c['front_glare_peak']) for c in spec['cases'] if c['shard']==0}=={float(job['env']['CATCHIT_FRONT_GLARE_PEAK'])},'Admission peak differs from planned shard0')
  def comparable(value):return {k:v for k,v in value.items() if k not in ['name','gpu_candidates']}
  replacements=[]
  for candidate in root.glob('catchit_glare*_development_*.job.json'):
   replacement_stem=candidate.name[:-len('.job.json')]
   if replacement_stem!=stem and 'retry' in replacement_stem and comparable(read(candidate))==comparable(job):replacements.append((replacement_stem,candidate))
  require(len(replacements)==1,'Require exactly one explicit retry with identical command, environment, cwd and budget')
  replacement_stem,replacement_job=replacements[0];replacement_path=root/(replacement_stem+'.result.json');replacement=read(replacement_path) if replacement_path.exists() else None
  if replacement is not None:require(replacement.get('status')=='completed' and replacement.get('exit_code')==0,'Replacement supervisor did not finish successfully')
  if final:
   require(replacement is not None,'Successful retry receipt is required before superseding admission')
   require((root/(replacement_stem+'.log')).exists() and (root/(replacement_stem+'.allocation.json')).exists(),'Retry has no actual child/allocation evidence')
   complete=read(root/'shard0_complete.json');require(complete.get('all_exit_zero') is True and complete.get('postflight_hashes_pass') is True,'Retry shard0 did not complete with valid hashes')
   require(complete['completed']==sum(len(c['method_order']) for c in spec['cases'] if c['shard']==0),'Retry shard0 denominator differs')
  notes.append(dict(original_job_stem=stem,original_result_sha256=sha(failed_path),original_job_sha256=sha(job_path),original_error=failed['error'],original_child_log_absent=True,original_allocation_file_absent=True,replacement_job_stem=replacement_stem,replacement_job_sha256=sha(replacement_job),replacement_result_sha256=sha(replacement_path) if replacement_path.exists() else None,scope='specific admission race before any child; original files retained and hashed; no actor/runtime failure excluded'))
 return notes

def ready(root,spec,superseded=()):
 missing=[]
 # The actor shard writes its completion proof just before returning. Wait for
 # resource-supervisor receipts too, so their logs are closed before hashing.
 for job_path in root.glob('catchit_glare*_development_*.job.json'):
  if job_path.name[:-len('.job.json')] in superseded:continue
  receipt_path=job_path.with_name(job_path.name.replace('.job.json','.result.json'))
  if not receipt_path.exists():missing.append(receipt_path.name)
  else:
   receipt=read(receipt_path);require(receipt.get('status')=='completed' and receipt.get('exit_code')==0,'Actor supervisor did not finish successfully: '+receipt_path.name)
 for shard in sorted({x['shard'] for x in spec['cases']}):
  p=root/('shard%d_complete.json'%shard)
  if not p.exists():missing.append(p.name);continue
  complete=read(p);require(complete.get('all_exit_zero') is True and complete.get('postflight_hashes_pass') is True,'Shard did not pass exit/hash contract: '+str(shard))
  expected={(c['case_id'],m) for c in spec['cases'] if c['shard']==shard for m in c['method_order']}
  require(complete['completed']==len(expected),'Shard completion count mismatch')
  progress=read(root/('shard%d_progress.json'%shard));require(len(progress)==len(expected) and {(x['case_id'],x['method']) for x in progress}==expected,'Shard progress keys mismatch')
  require(all(x['exit_code']==0 for x in progress),'Infrastructure abort is not a valid physical failure')
  preflight=read(root/('shard%d_preflight.json'%shard));require(preflight['frozen_inputs_pass'] is True and preflight['protocol_sha256']==sha(root/'EVALUATION_PROTOCOL.json'),'Shard preflight protocol mismatch')
 for case in spec['cases']:
  for method in case['method_order']:
   folder=root/'cases'/case['case_id']/method
   for name in ['episode_000/result.json','episode_000/trace.json','protocol.json','episodes.json','migration_audit/adapter_audit.json']:
    if not (folder/name).exists():missing.append(str((folder/name).relative_to(root)))
 return missing

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--analyzer',type=Path);parser.add_argument('--audit-script',type=Path);parser.add_argument('--wait-seconds',type=float,default=0);parser.add_argument('--superseded-admission',action='append',default=[],help='Exact guarded pre-child admission stem with a proven identical successful retry; no wildcards');args=parser.parse_args();root=args.root.resolve();spec=read(root/'EVALUATION_PROTOCOL.json');expected=[(c['case_id'],m) for c in spec['cases'] for m in c['method_order']];require(len(expected)==spec['expected_episodes'] and len(expected)==len(set(expected)),'Planned denominator invalid')
 require(len(args.superseded_admission)==len(set(args.superseded_admission)),'Duplicate admission whitelist entry');superseded=set(args.superseded_admission);audit_superseded_admissions(root,spec,superseded)
 deadline=time.monotonic()+args.wait_seconds;last=None
 while True:
  for p in root.glob('*.result.json'):
   result=read(p)
   if result.get('status')=='failed' and p.name[:-len('.result.json')] not in superseded:raise RuntimeError('Parent resource job failed; preserve and resolve before final scoring: '+str(p))
  missing=ready(root,spec,superseded)
  if not missing:break
  if missing!=last:print(json.dumps({'status':'waiting_for_actual_completion','missing':missing}),flush=True);last=missing
  if time.monotonic()>=deadline:raise RuntimeError('Episodes or completion proofs not ready; no final result written')
  time.sleep(min(15.,max(.1,deadline-time.monotonic())))
 admission_notes=audit_superseded_admissions(root,spec,superseded,final=True)
 (root/'SUPERSEDED_ADMISSION_AUDIT.json').write_text(json.dumps({'status':'PASS','explicit_whitelist':sorted(superseded),'all_original_files_retained':True,'notes':admission_notes},indent=2))
 verify_inputs(spec)
 analyzer=args.analyzer or root/'analyze_cadence.py';audit=args.audit_script or root/'audit_cadence_final.py';require(analyzer.exists() and audit.exists(),'Deploy existing analyzer and generic expected_episodes audit; do not rewrite metric definitions')
 with (root/'analyze_cadence.py.final.log').open('w') as log:subprocess.run([sys.executable,'-u',str(analyzer),'--root',str(root)],stdout=log,stderr=subprocess.STDOUT,check=True)
 data=read(root/'PAIRED_PORTABLE.json');require(data['complete'] and not data['violations'],'Existing common analyzer rejected the dataset');require(len(data['records'])==len(expected),'Portable count mismatch')
 with (root/'audit_cadence_final.py.final.log').open('w') as log:subprocess.run([sys.executable,'-u',str(audit),'--root',str(root)],stdout=log,stderr=subprocess.STDOUT,check=True)
 require(read(root/'FINAL_AUDIT.json')['status']=='PASS','Raw event / core audit failed')
 audited_hashes={};audited_stats={}
 for line in (root/'SHA256SUMS').read_text().splitlines():
  digest,label=line.split('  ',1);p=Path(label) if Path(label).is_absolute() else root/label
  audited_hashes[p]=digest;st=p.stat();audited_stats[p]=(st.st_size,st.st_mtime_ns)
 module_path=Path(spec['source'])/'light_schedule.py';module_spec=importlib.util.spec_from_file_location('sealed_public_light_schedule',module_path);module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
 records={(x['case_id'],x['mode']):x for x in data['records']};lighting=[];rows=[];physics_by_seed={};initialization_by_seed={}
 for case in spec['cases']:
  peak=float(case['front_glare_peak']);declared=module.front_glare_schedule(peak)
  for method in case['method_order']:
   row=records[(case['case_id'],method)];run=Path(row['output']);protocol=row['protocol'];result=row['result'];physical_trace=read(run/'episode_000/trace.json');adapter=read(run/'migration_audit/adapter_audit.json')
   require(protocol['lighting']=='front_glare' and protocol['lighting_schedule']==declared,'Actor lighting differs from planned case: '+case['case_id']+' '+method)
   require(adapter['original_mj_step_calls']==0 and adapter['physics_backend'].startswith('Genesis'),'Physics backend / native stepping mismatch')
   require(adapter['lighting_schedule']==declared,'Adapter schedule differs from actor declaration')
   require(protocol['RGB_frame_budget']==protocol['control_depth_frame_budget']==1,'One-frame initialization budget changed')
   for flag in ['revocable_grasp','reject_partial_contours','event_flight_fit']:require(protocol[flag] is False,'Estimator flag changed: '+flag)
   require(protocol['event_endpoint_rate_hz']==case['endpoint_hz']==100,'Development input cadence changed')
   traces=adapter['lighting_trace_by_episode'];require(set(traces)=={'0'},'Unexpected lighting episode identities');samples=traces['0'];require(samples,'Missing actual native-light readbacks');times=[float(x['time_s']) for x in samples]
   require(abs(times[0])<1e-12,'RGB-D initialization did not see reset-time illumination');require(all(b>a for a,b in zip(times,times[1:])),'Lighting readback times not strictly increasing');require(abs(times[-1]-result['sim_time'])<1e-7,'Actual illumination trace does not cover terminal scene state')
   max_error=0.
   for sample,t in zip(samples,times):
    wanted=module.front_glare_intensity(t,peak);scheduled=float(sample['scheduled_point_intensity']);actual=float(sample['actual_point_intensity']);require(all(math.isfinite(v) for v in [t,scheduled,actual]),'Nonfinite native-light data');error=max(abs(wanted-scheduled),abs(wanted-actual));max_error=max(max_error,error);require(error<1e-7,'Actual lamp does not follow public-time schedule');require(sample['time_source']=='public reset-relative _gs_time','Unexpected private trigger in lighting trace')
   require(samples[0]['actual_point_intensity']==0,'Initial lamp not zero')
   if result['sim_time']>=.20:
    require(any(.10<t<.20 for t in times),'Missing sampled ramp interval');require(any(t>=.20 and abs(x['actual_point_intensity']-peak)<1e-7 for t,x in zip(times,samples)),'Peak was never observed')
   lighting.append({'case_id':case['case_id'],'method':method,'peak':peak,'samples':len(samples),'start_time_s':times[0],'end_time_s':times[-1],'maximum_schedule_or_readback_error':max_error,'initial_native_point_intensity':samples[0]['actual_point_intensity'],'physics_backend':adapter['physics_backend'],'mj_step_calls':adapter['original_mj_step_calls'],'actor_lighting_schedule_matches':True})
   if case['seed'] in physics_by_seed:require(result['initial']==physics_by_seed[case['seed']],'Paired physical reset differs')
   physics_by_seed[case['seed']]=result['initial'];initialization=row['initialization']
   if case['seed'] in initialization_by_seed:require(initialization==initialization_by_seed[case['seed']],'RGB-D initialization differs across methods/peaks')
   initialization_by_seed[case['seed']]=initialization
   confirmed=bool(result['success'] and not result.get('terminated',False) and result['hand_contact_steps']>0 and physical_trace[-1]['hand_contact']);require(confirmed==row['verified_catch'],'Final physical score mismatch')
   metric=lambda key,stat:(row.get(key) or {}).get(stat)
   rows.append(dict(case_id=case['case_id'],seed=case['seed'],lighting='front_glare',peak_point_intensity_renderer_units=peak,input_image_hz=case['endpoint_hz'],method=method,raw_time_limit_success_flag=result['success'],terminated=result.get('terminated',False),verified_catch=confirmed,actual_hand_contact_steps=result['hand_contact_steps'],final_control_step_contact=physical_trace[-1]['hand_contact'],first_contact_s=result.get('first_touch'),terminal_s=result['sim_time'],raw_events=row.get('raw_event_count',0),pixel_error_median=metric('all_precontact_state_pixel_error','median'),pixel_error_p95=metric('all_precontact_state_pixel_error','p95'),position_error_median_mm=metric('all_precontact_state_position_error_mm','median'),position_error_p95_mm=metric('all_precontact_state_position_error_mm','p95'),velocity_error_median_m_s=metric('all_precontact_state_velocity_error_m_s','median'),velocity_error_p95_m_s=metric('all_precontact_state_velocity_error_m_s','p95'),assumed_grasp_first_s=row.get('assumed_grasp_first_time'),initialization_valid=initialization['valid'],native_light_schedule_pass=True,server_output=str(run)))
 require(len(rows)==len(expected),'Every planned success and failure must remain')
 for filename,encoding in [('GLARE_EPISODE_RESULTS.csv','utf-8'),('强光开发逐条结果.csv','utf-8-sig')]:
  with (root/filename).open('w',newline='',encoding=encoding) as stream:writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 summary={'scope':'two already seen development seed blocks; no independent-test or generator-ranking claim','actual_episodes':len(rows),'seeds':sorted(physics_by_seed),'cadence_hz':100,'input_family':'actual native front point illumination, normal-light shared one-frame RGB-D then public-time ramp; no frame editing','V2E_scope':'native SSM39 plus LDR uint8 video pipeline, not a claim that all V2E modes lack HDR','all_planned_cases_retained':True,'by_peak':{}}
 for peak in sorted({float(x['front_glare_peak']) for x in spec['cases']}):
  summary['by_peak'][str(int(peak))]={}
  for method in ['ours','v2e']:
   selected=[x for x in rows if x['peak_point_intensity_renderer_units']==peak and x['method']==method];require(len(selected)>0,'Missing paired source');summary['by_peak'][str(int(peak))][method]={'n':len(selected),'verified_catches':sum(x['verified_catch'] for x in selected),'episodes_with_hand_contact':sum(x['actual_hand_contact_steps']>0 for x in selected),'case_ids':[x['case_id'] for x in selected]}
 (root/'GLARE_COMPARISON.json').write_text(json.dumps(summary,indent=2));(root/'GLARE_LIGHTING_AUDIT.json').write_text(json.dumps({'status':'PASS','native_readback_and_public_time_schedule_pass':True,'shared_physical_resets_and_initial_RGBD_pass':True,'records':lighting},indent=2));verify_inputs(spec)
 complete={'status':'PASS','actual_episodes':len(rows),'development_seed_blocks':len(physics_by_seed),'generic_raw_event_core_audit_sha256':sha(root/'FINAL_AUDIT.json'),'native_lighting_audit_sha256':sha(root/'GLARE_LIGHTING_AUDIT.json'),'specific_admission_exception_audit_sha256':sha(root/'SUPERSEDED_ADMISSION_AUDIT.json'),'common_analyzer_sha256':sha(analyzer),'generic_audit_script_sha256':sha(audit),'sealed_actor_source':spec['source'],'model_and_frozen_core_unchanged':True,'all_planned_successes_and_failures_retained':True,'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(root/'GLARE_FINAL_AUDIT.json').write_text(json.dumps(complete,indent=2))
 exclusions={'SHA256SUMS','GLARE_POSTPROCESS_COMPLETE.json'};paths=[p for p in root.rglob('*') if p.is_file() and p.name not in exclusions and not p.name.endswith('.final.log') and not ('postprocess' in p.name.lower() and p.name.endswith('.log'))]
 lines=[];reused=0
 for p in sorted(paths):
  st=p.stat()
  if p in audited_hashes and audited_stats[p]==(st.st_size,st.st_mtime_ns):digest=audited_hashes[p];reused+=1
  else:digest=sha(p)
  lines.append(digest+'  '+p.relative_to(root).as_posix())
 (root/'SHA256SUMS').write_text('\n'.join(lines)+'\n');(root/'GLARE_POSTPROCESS_COMPLETE.json').write_text(json.dumps(dict(complete,full_manifest_sha256=sha(root/'SHA256SUMS'),hashed_files=len(paths),unchanged_artifact_hashes_reused_from_completed_raw_audit=reused),indent=2));print(json.dumps({'status':'PASS','actual_episodes':len(rows),'hashed_files':len(paths),'summary':summary}),flush=True)

if __name__=='__main__':main()
