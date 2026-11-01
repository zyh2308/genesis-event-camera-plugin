from pathlib import Path
import json,hashlib,tarfile
O=Path('/mnt/c/Users/zhaoyuhan/Documents/ChatGPT/New project/outputs/epose_diverse_short_20261101');a=O/'review_package.tar.gz'
archive_sha=hashlib.sha256(a.read_bytes()).hexdigest();assert archive_sha==(O/'review_package.tar.gz.sha256').read_text().split()[0]
assert not (O/'evidence').exists(),'Preserve any earlier extraction rather than overwriting it.'
with tarfile.open(a) as tar:tar.extractall(O,filter='data')
E=O/'evidence';F=E/'final';manifest=json.loads((E/'manifest.json').read_text())
for name,r in manifest['files'].items():
 p=E/name;assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],name
source=json.loads((F/'source_snapshot_before_training.json').read_text());launch=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real/benchmark/sim2real/reviews/epose_diverse_short_20261101/launched_motion');before=json.loads((launch/'source_snapshot_before_training.json').read_text());assert source['files']==before['files']
for name,sha in source['files'].items():assert hashlib.sha256((F/name).read_bytes()).hexdigest()==sha,name
cfg=json.loads((F/'protocol.json').read_text());protocol_sha=hashlib.sha256((F/'protocol.json').read_bytes()).hexdigest();assert protocol_sha==json.loads((F/'protocol_pretraining_hash.json').read_text())['sha256']
assert json.loads((F/'completion.json').read_text())['exit_code']==0
assert json.loads((F/'training_process_exit_codes.json').read_text())['passed']
assert json.loads((F/'dataset_immutability_after_training.json').read_text())['passed']
assert json.loads((F/'independent_synthetic_verification.json').read_text())['passed']
assert json.loads((F/'real_eval/independent_verification.json').read_text())['passed']
assert json.loads((F/'synthetic_gate.json').read_text())['passed']
core=json.loads((F/'frozen_core_verification.json').read_text());assert core['passed'] and core['frozen_commit']=='e79e656ac00f7847b98db9e12bf43b093f3afff1'
S=json.loads((F/'real_eval/summary.json').read_text());assert not S['checkpoint_selection_used_real_data'] and S['sample_count']==200
for a in ['eventgenesis','v2e']:
 r=json.loads((F/f'training/{a}_result.json').read_text());assert r['executed_updates']==cfg['optimizer_updates_per_arm']==2400 and r['synthetic_gate_passed'];assert S['results'][a]['checkpoint']['sha256']==r['checkpoint_sha256']
receipt={'passed':True,'archive_sha256':archive_sha,'verified_files':len(manifest['files']),'verified_source_files':len(source['files']),'source_matches_pretest_git_snapshot':True,'protocol_sha256':protocol_sha,'paired_training_updates':2400,'real_windows':200,'server_run':manifest['server_run']}
(O/'staging_verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2));print(json.dumps({a:S['results'][a] for a in S['eligible_new_arms']},indent=2))
