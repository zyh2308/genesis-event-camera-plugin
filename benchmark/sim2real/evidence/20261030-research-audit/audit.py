from pathlib import Path
import json,hashlib,datetime,math,zipfile,xml.etree.ElementTree as ET
R=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real');B=R/'benchmark/sim2real'
D=B/'evidence/20261029-execution-refresh'; O=B/'evidence/20261030-research-audit';O.mkdir(exist_ok=True)
index=json.loads((D/'evidence_index.json').read_text())
descriptors=[]
def walk(x):
 if isinstance(x,dict):
  if isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str):descriptors.append(x)
  for y in x.values():walk(y)
 elif isinstance(x,list):
  for y in x:walk(y)
walk(index)
checks=[]
for p in D.rglob('*'):
 if not p.is_file() or p.name=='evidence_index.json':continue
 blob=p.read_bytes();digest=hashlib.sha256(blob).hexdigest()
 matches=[v for v in descriptors if v['sha256']==digest]
 checks.append({'path':str(p.relative_to(R)),'bytes':len(blob),'sha256':digest,'matches_index_content':bool(matches),'matching_source_paths':[v['path'] for v in matches]})
ev=json.loads((D/'flow/independent_rollout_eval.json').read_text());agg={}
for arm in ('eventgenesis','v2e'):
 agg[arm]={}
 for region in ('full_valid','common_event_active','gt_motion_ge_0_5'):
  rows=[x['generator_metrics'][arm][region] for x in ev['windows']]
  n=sum(x['pixels'] for x in rows);e=sum(x['pixels']*x['epe_px_per_30ms'] for x in rows if x['pixels'])/n
  recorded=ev['summary'][arm][region]
  agg[arm][region]={'pixels':n,'recomputed_epe':e,'summary_matches':n==recorded['pixels'] and abs(e-recorded['pixel_weighted_epe_px_per_30ms'])<1e-12,'relative_to_zero':e/recorded['zero_flow_epe_px_per_30ms']}
fit=json.loads((D/'flow/fixed_sample_fit/summary.json').read_text());fit_summary={};logs={}
for arm in ('eventgenesis','v2e'):
 history=fit['arms'][arm]['history'];first=history[0];last=history[-1]
 fit_summary[arm]={'initial':first,'final':last,'checkpoint':fit['arms'][arm]['checkpoint'],'active_epe_ratio_to_zero':last['common_event_active_epe_px_per_30ms']/last['zero_flow_common_event_active_epe_px_per_30ms']}
 rows=[json.loads(x) for x in (D/'flow/fixed_sample_fit'/f'{arm}_fixed_sample_updates.jsonl').read_text().splitlines()]
 updates=[x for x in rows if x.get('record_type')=='optimizer_update']
 assert all(math.isfinite(x['multiscale_objective_before_step']) for x in updates)
 assert all(all(int(s)==x['update'] for s in x['step_values']) for x in updates)
 logs[arm]={'rows':len(rows),'optimizer_updates':len(updates),'updates_sequential_1_200':[x['update'] for x in updates]==list(range(1,201)),'all_gradients_finite_positive':all(math.isfinite(x['gradient_l2_norm']) and x['gradient_l2_norm']>0 for x in updates),'gradient_tensor_counts':sorted({x['gradient_tensor_count'] for x in updates}),'row_keys':list(rows[1]),'record_types':sorted({x.get('record_type','') for x in rows}),'first_two':rows[:2],'last':rows[-1]}
# Recompute pinhole projection directly from the captured MAT keypoint XYZ,
# independent of the execution agent's stored projection-error summary.
lines=(B/'evidence/20261024-execution-refresh/g1_sample_audit_stdout.txt').read_text().splitlines()
fields=json.loads(next(x[len('mat_fields='):] for x in lines if x.startswith('mat_fields=')))
K=fields['intrinsicMatrix']['value'];kp=fields['keypointprojections']['value'];errs=[];abscoords=[]
for row in kp:
 u,v,x,y,z=row; projected=[K[0][0]*x/z+K[0][2],K[1][1]*y/z+K[1][2]]
 du,dv=u-projected[0],v-projected[1];errs.append(math.hypot(du,dv));abscoords.extend([abs(du),abs(dv)])
proj={'sample':'block/r_01_gl/0001_data.mat captured fields','points':len(kp),'pinhole_euclidean_mean_px':sum(errs)/len(errs),'pinhole_euclidean_max_px':max(errs),'mean_absolute_coordinate_error_px':sum(abscoords)/len(abscoords),'note':'Pinhole-vs-stored diagnostic; distortion/undistortion contract must be handled consistently for final corrected labels.'}
shift=Path('/home/科研/Eventbased_WAM/external_sources/E_Pose/time_shift_used.ods');shift_rows=[]
if shift.exists():
 with zipfile.ZipFile(shift) as z:root=ET.fromstring(z.read('content.xml'))
 ns={'table':'urn:oasis:names:tc:opendocument:xmlns:table:1.0','text':'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}
 for table in root.findall('.//table:table',ns):
  name=table.attrib.get('{'+ns['table']+'}name')
  for row in table.findall('table:table-row',ns):
   values=[' '.join(c.itertext()).strip() for c in row.findall('table:table-cell',ns)]
   if any(values):shift_rows.append({'sheet':name,'cells':values[:30]})
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_head':'acbc74eeeaefbcaddcba7fe77fa043cd2488b9f5','evidence_file_checks':checks,'flow_window_count':len(ev['windows']),'flow_recomputed':agg,'fixed_fit':fit_summary,'log_checks':logs,'epose_projection_recomputed':proj,'official_time_shift_table_path':str(shift),'official_time_shift_rows':shift_rows,'remote_access':'Two Windows SSH attempts closed connection (domain fake-IP and previously verified g1 IP); no further retry. No server result rehash or current GPU-state claim.'}
(O/'local_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({'files_checked':len(checks),'unmatched':[x['path'] for x in checks if not x['matches_index_content']],'flow_window_count':len(ev['windows']),'flow_recomputed':agg,'fixed_active_ratio_to_zero':{a:v['active_epe_ratio_to_zero'] for a,v in fit_summary.items()},'epose_projection_recomputed':proj,'time_shift_table':shift_rows[:16],'log_keys':{a:{k:v for k,v in x.items() if k not in ('first_two','last')} for a,x in logs.items()}},ensure_ascii=False,indent=2))
