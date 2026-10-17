"""Publish this research-only audit to the local shared WSL repository."""
from pathlib import Path
import hashlib,json,shutil,subprocess
O=Path(__file__).resolve().parent
R=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real')
D=R/'benchmark/sim2real/reviews/stage4_completed_ce23331'
d=json.loads((O/'remote_verification.json').read_text())
summary={
 'audited_commit':d['head'],'checked_at_JST':d['checked_at_JST'],
 'training_completion':'PASS','research_result_acceptance':'FAIL_PENDING_CORRECTNESS_REPAIR',
 'paired_all_update_sample_orders_match':d['paired_all_8910_sample_orders_match'],
 'arms':{k:{'updates':v['update_count'],'validations':v['validation_count'],'exit_code':v['train_exit'],'best_epoch':v['best_validation']['epoch'],'legacy_best_epe':v['best_validation']['mean_epe_px_per_30ms'],'legacy_final_epe':v['last_validation']['mean_epe_px_per_30ms']} for k,v in d['arms'].items()},
 'legacy_zero_flow_epe':d['zero_flow_EPE_equal_rollouts'],
 'legacy_metric_caveat':'All EPE above uses faulty segmentation-to-link mapping and is not a physically valid generator comparison.',
 'confirmed_findings':[{'priority':'P0','issue':'Renderer segmentation indices passed as link IDs to shared GT and EventGenesis interpolation; static ground assigned object motion.','evidence':['renderer_source_verification.json','segmentation_verification.json','local_source_excerpts.json']},{'priority':'P1','issue':'STEFlow loss/model grid coordinates use W-1/H-1 while grid_sample defaults to align_corners=False.','evidence':['warp_coordinate_verification.json','renderer_source_verification.json']}],
 'real_dataset_scoring_performed_by_audit':False,'production_code_changed_by_audit':False,
 'audit_GPU_use':'24 fixed synthetic validation windows per existing best model, inference only; no training',
 'recommended_next_step':'Repair mappings and warp contracts; version/rebuild affected data; paired synthetic short run before new full training or seeds.'
}
(O/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
D.mkdir(parents=True,exist_ok=True)
for p in O.iterdir():
    if p.is_file() and p.name!='artifact_manifest.json':shutil.copy2(p,D/p.name)
entries=[{'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(D.iterdir()) if p.is_file() and p.name!='artifact_manifest.json']
(D/'artifact_manifest.json').write_text(json.dumps({'files':entries},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert all(hashlib.sha256((D/e['file']).read_bytes()).hexdigest()==e['sha256'] for e in entries)
print(json.dumps({'destination':str(D),'files':len(entries)+1,'status':subprocess.run(['git','-C',str(R),'status','--short'],text=True,capture_output=True,check=True).stdout},ensure_ascii=False))
