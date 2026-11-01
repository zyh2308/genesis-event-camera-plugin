import os,json,hashlib,shutil,tarfile,time
from pathlib import Path
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=Path(os.environ['EPOSE_REPAIR_RUN']);E=R/'review_package';E.mkdir(exist_ok=False);final=E/'final';final.mkdir();files={}
def add(src,dst):
 if src.is_file():
  target=E/dst;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);files[str(dst)]={'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
for pattern in ['*.json','*.txt','*.stdout','*.stderr','*.py','sources/*.py','training/*.json','training/*updates.jsonl.gz','training/*updates_live.jsonl','real_eval/*.json','real_eval/raw_undistorted/*.png','input_previews/*.png','bank/preview/*.png']:
 for p in R.glob(pattern):add(p,Path('final')/p.relative_to(R))
for n in ['manifest.json','config.json','v2e_manifest.json']:add(R/'bank'/n,Path('final/bank')/n)
attempts={'v2':'20261101-epose-localization-repair-v2','v3':'20261101-epose-localization-repair-v3','v4':'20261101-epose-localization-full-v4','v5':'20261101-epose-center-classifier-v5','v6':'20261101-epose-center-full-v6','v7':'20261101-epose-center-selected-v7','v8':'20261101-epose-diverse-bank-v8','v9':'20261101-epose-diverse-paired-v9'}
for tag,name in attempts.items():
 a=P/'runs/sim2real'/name
 for n in ['protocol.json','effective_training_config.json','completion.json','synthetic_gate.json','independent_synthetic_verification.json','synthetic_localization_audit.json','training/result.json','first_augmented_batch_loss_audit.json','eventgenesis_training.stderr','v2e_training.stderr','worker6_attempt1.stderr','sources/train_diverse_pose.py','foreground_observability_audit.json','real_scoring_intentionally_held.json','early_checkpoint_recovery_audit.json','audit_periodic_pose_checkpoint.py','audit_diverse_loss_equivalence.py','physical_coverage_statistics.json','event_window_count_statistics.json']:
  add(a/n,Path('attempts')/tag/n)
old=P/'runs/sim2real/20261101-epose-wood-short-research-v1';add(old/'frozen_core_manifest.json',Path('initial/frozen_core_manifest.json'))
manifest={'files':files,'server_run':str(R),'collection_unix':time.time(),'large_files':'checkpoints and physical bank stay on g1; SHA256 and paths in protocol/results/manifests','real_status':json.loads((R/'completion.json').read_text()) if (R/'completion.json').exists() else None};(E/'manifest.json').write_text(json.dumps(manifest,indent=2))
archive=R/'review_package.tar.gz'
with tarfile.open(archive,'w:gz') as t:t.add(E,arcname='evidence')
(R/'review_package.tar.gz.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  review_package.tar.gz\n')
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(files),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}),flush=True)
