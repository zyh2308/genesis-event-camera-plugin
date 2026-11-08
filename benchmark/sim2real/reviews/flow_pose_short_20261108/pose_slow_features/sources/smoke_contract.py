"""No new coefficient/calibration; only actual short GPU admission evidence."""
import hashlib,json
from pathlib import Path

def verify_smoke(path,expected_sha,source_manifest_sha):
    if path is None or expected_sha is None:raise ValueError('Actual slow-features smoke receipt required before study')
    p=Path(path)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected_sha:raise ValueError('Smoke receipt SHA differs')
    s=json.loads(p.read_text())
    if not(s['status']=='complete_synthetic_only' and s['phase']=='smoke' and s['source_manifest_sha256']==source_manifest_sha and
           s['canonical_candidate']=='slow_features' and s['real_scores_computed'] is False and
           s['feature_KD_weight']==0 and s['actual_half_teacher_weight']==0 and s['features_frozen'] is False):
        raise ValueError('Wrong smoke scope/source')
    records=s['records'];expected={(m,0,a,b) for m in ('eventgenesis','v2e') for a in ('clean_control','physical_supervised','paired_consistency') for b in ('depth','center')}
    if len(records)!=12 or {(r['method'],r['seed'],r['arm'],r['branch']) for r in records}!=expected:raise ValueError('All12 fresh actual smoke branches required')
    if s['new_optimizer_updates']!=12:raise ValueError('Smoke must be one update per branch')
    for r in records:
        if not(r['updates']==1 and r['new_optimizer_updates']==1 and r['teacher_unchanged'] and r['independent_model_optimizer_rebuild'] and r['optimizer_state_equal']):
            raise ValueError('Actual smoke optimizer/restore evidence incomplete')
        if r['arm']=='paired_consistency':
            groups=r['optimizer_group_signature']['groups']
            if not(r['canonical_arm']=='slow_features' and r['feature_KD_weight']==0 and r['actual_half_teacher_output_weight']==0 and r['normal_teacher_weight']==2 and
                   r['feature_learning_rate']==.000015 and r['head_learning_rate']==.00015 and r['optimizer_groups_independently_rebuilt'] and
                   [g['name'] for g in groups]==['features','heads'] and [g['lr'] for g in groups]==[.000015,.00015] and
                   r['changed_parameter_names']['features'] and r['changed_parameter_names']['heads']):
                raise ValueError('Candidate smoke did not verify both parameter groups actually updated')
    return s
