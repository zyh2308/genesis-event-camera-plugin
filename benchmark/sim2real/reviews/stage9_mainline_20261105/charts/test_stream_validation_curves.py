"""Small stdlib fixtures for filtering, source hashes and incomplete log tails."""
import hashlib,importlib.util,json
from pathlib import Path
root=Path(__file__).parent/'stream_fixture_v1';assert not root.exists();root.mkdir()
spec=importlib.util.spec_from_file_location('stream_fixture_module',Path(__file__).with_name('stream_validation_curves.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
base=dict(event='synthetic_validation',optimizer_updates=198,epoch=1,generator='v2e',seed=0,
    model_mean_epe_px_per_30ms=.3,zero_flow_mean_epe_px_per_30ms=.48,real_data_used=False)
path=root/'training/flow/seed0/v2e/training_metrics.jsonl';path.parent.mkdir(parents=True)
payload=(json.dumps({'event':'optimizer_update','loss':12345})+'\n'+json.dumps(base)+'\n'+json.dumps(base)+'\n').encode()
path.write_bytes(payload)
rows,rec=m.extract(path,'flow','v2e',0,1024*1024)
assert len(rows)==2 and rec['validation_records']==2 and rec['source_sha256']==hashlib.sha256(payload).hexdigest()
assert {r['metric'] for r in rows}=={'model_mean_epe_px_per_30ms','zero_flow_mean_epe_px_per_30ms'}
tests=['strict_validation_filter_and_dedup','exact_source_sha256']
path.write_bytes(payload+b'{"event":')
rows,rec=m.extract(path,'flow','v2e',0,1024*1024)
assert len(rows)==2 and rec['trailing_partial_bytes']==9 and rec['source_prefix_bytes']==len(payload)+9
assert rec['source_sha256']==hashlib.sha256(payload+b'{"event":').hexdigest();tests.append('partial_tail_hashed_and_not_parsed')
for label,change in [('wrong_identity',{'generator':'eventgenesis'}),('nonfinite_metric',{'model_mean_epe_px_per_30ms':float('nan')}),('real_data_flag',{'real_data_used':True}),('conflicting_update_keys',{'optimizer_update':200})]:
    bad=dict(base,**change);path.write_text(json.dumps(bad)+'\n')
    try:m.extract(path,'flow','v2e',0,1024*1024)
    except RuntimeError:tests.append(label)
    else:raise AssertionError(label)
path.write_bytes(b'{broken complete line}\n')
try:m.extract(path,'flow','v2e',0,1024*1024)
except RuntimeError:tests.append('malformed_complete_line_rejected')
else:raise AssertionError('malformed JSON accepted')
rows,rec=m.extract(root/'missing.jsonl','flow','v2e',0,1024*1024)
assert not rows and rec['status']=='MISSING_LOG';tests.append('missing_log_no_zero_points')
result=dict(status='pass',checks=tests,count=len(tests),real_logs_read=False,pose_training_logs_read=False)
(root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
