from pathlib import Path
import importlib.util,json,tempfile,hashlib,numpy as np,traceback
R=Path(r'\\wsl.localhost\Ubuntu-22.04\home\科研\Eventbased_WAM\repositories\gec-sim2real')
path=R/'benchmark/sim2real/scripts/stage4_mvsec_dt1_metrics.py'
spec=importlib.util.spec_from_file_location('audit_mvsec_metrics',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out={'scope':'synthetic-only disposable fixture; no real data/checkpoint read','self_test':m.self_test()}
with tempfile.TemporaryDirectory(prefix='mvsec-schema-audit-') as tmp:
 T=Path(tmp);p=json.loads((R/'benchmark/sim2real/protocol/mvsec_dt1_protocol_candidate_20261019.json').read_text())
 p.update(status='frozen_by_research_review',real_scores_authorized=True,sequences=['indoor_flying1'])
 for arm in ['eventgenesis','v2e']:p['model']['checkpoint_index'][arm]['sha256']='synthetic-checkpoint-'+arm
 event=np.zeros((2,260,346,5),dtype=np.uint8);event[0,10,10,0]=1
 gt=np.zeros((2,260,346),dtype=np.float32);gt[0,10,10]=1
 artifacts={}
 for arm in ['eventgenesis','v2e']:
  file=T/(arm+'.npz');np.savez_compressed(file,event_input_native_uint8=event,prediction_px_per_interval=gt,gt_px_per_interval=gt)
  artifacts[arm]={'path':str(file),'sha256':sha(file),'checkpoint_sha256':p['model']['checkpoint_index'][arm]['sha256']}
 manifest={'status':'frozen_by_research_review','real_scores_authorized':True,'registered_aps_start_indices':{'indoor_flying1':[21]},'windows':[{'dt':1,'sequence':'indoor_flying1','aps_start_index':21,'start_time_s':0.,'end_time_s':.03,'label_time_s':.03,'event_window_convention':'(start_time_s, end_time_s]','event_count':1,'artifacts':artifacts}]}
 protocol=T/'protocol.json';test_manifest=T/'manifest.json';auth=T/'synthetic_fixture_auth.json'
 test_manifest.write_text(json.dumps(manifest))
 for label,alias in [('candidate_schema',False),('temporary_top_level_alias_control',True)]:
  if alias:p['checkpoint_index']=p['model']['checkpoint_index']
  protocol.write_text(json.dumps(p))
  auth.write_text(json.dumps({'authorized_real_test':True,'authorization_id':'synthetic_fixture_only','approved_by':'synthetic_audit_fixture','protocol_sha256':sha(protocol),'manifest_sha256':sha(test_manifest)}))
  try:
   result=m.evaluate_pair_manifest(test_manifest,protocol,auth)
   out[label]={'status':'passed','summary':result['sequence_summary']}
  except Exception as exc:out[label]={'status':'failed','exception':repr(exc),'traceback':traceback.format_exc()}
Path(__file__).with_name('mvsec_schema_issue.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
