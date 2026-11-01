import os,json,hashlib,tarfile,socket,sys
from pathlib import Path
R=Path(os.environ['EPOSE_SHORT_RUN']);assert (R/'pipeline_exit_code.txt').read_text().strip()=='0'
assert json.loads((R/'real_eval/independent_verification.json').read_text())['status']=='passed'
names=['protocol.json','protocol_pretraining_hash.json','short_config.json','weights_frozen.json','execution_environment.json','trajectory_reset_audit.json','synthetic_pnp_oracle.json','pipeline_status.json','pipeline_exit_code.txt','resource_reservation.json','resource_reservation_finished.json','frozen_core_manifest.json','bank/manifest.json','bank/v2e_manifest.json','bank/config.json','training/result.json','real_eval/summary.json','real_eval/eventgenesis_predictions.json','real_eval/v2e_predictions.json','real_eval/independent_verification.json','cad_keypoints.npy','cad_vertices.npy','attempt1_failure.json','attempt2_cancelled.json']
files=[R/n for n in names if (R/n).is_file()]
files+=[p for p in R.glob('*.log') if p.is_file()]
files+=[p for p in (R/'sources').glob('*.py') if p.is_file()]
files+=[p for p in (R/'bank/preview').glob('*.png') if p.is_file()]
files+=[p for p in (R/'real_eval/raw_undistorted').glob('*.png') if p.is_file()]
files+=[p for p in (R/'bank').glob('*/v2e_command.json') if p.is_file()]
files+=[p for p in (R/'bank').glob('*/v2e.stderr.log') if p.is_file()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'run':str(R),'host':socket.gethostname(),'python':sys.executable,'scope':'matched-CAD synthetic-only server short training and fixed200-window real pilot','small_artifacts':{str(p.relative_to(R)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files},'large_artifacts':{'shared_bank':{'path':str(R/'bank/shared_trajectory.h5'),'sha256':sha(R/'bank/shared_trajectory.h5')},'checkpoints':json.loads((R/'weights_frozen.json').read_text())['checkpoints']},'notes':'full checkpoints and radiance arrays remain on g1; no unrelated GPU process logs are included'}
(R/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2));files.append(R/'evidence_manifest.json')
with tarfile.open(R/'research_evidence.tar.gz','w:gz') as t:
 for p in files:t.add(p,arcname=str(p.relative_to(R)))
print(json.dumps({'archive':str(R/'research_evidence.tar.gz'),'bytes':(R/'research_evidence.tar.gz').stat().st_size,'sha256':sha(R/'research_evidence.tar.gz')},indent=2))
