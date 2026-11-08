"""Fail-closed phase dependencies; no automatic real evaluation."""
import hashlib
import json
import math
from pathlib import Path
import statistics
from feature_retention import common_coefficient


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_calibration(path,expected_sha,source_manifest_sha):
    assert path is not None and expected_sha is not None and sha(path)==expected_sha
    value=json.loads(Path(path).read_text())
    assert value['schema']=='pose_feature_retention_calibration_v1' and value['status']=='passed'
    assert value['source_manifest_sha256']==source_manifest_sha
    assert value['optimizer_updates']==0 and value['real_data_used'] is False
    assert value['validation_data_used'] is False and value['validation_metrics_used'] is False and value['train_only'] is True
    rows=value['records']
    assert {(r['method'],r['seed'],r['branch']) for r in rows}=={(m,s,b) for m in ('eventgenesis','v2e') for s in range(3) for b in ('depth','center')}
    coefficient,ratios=common_coefficient(rows)
    assert value['coefficient']==coefficient and value['target_median_gradient_ratio']==.1
    assert math.isclose(statistics.median(ratios),.1,rel_tol=1e-12,abs_tol=1e-12)
    assert all(r['source_initial_feature_retention']==0 and r['optimizer_updates']==0 and r['source_and_C1_weights_unchanged'] for r in rows)
    assert set(value['dependency_sources'])=={'scale','pilot','fresh','rotation','causal','encoder'}
    for item in value['dependency_sources'].values():
        p=Path(item['path']);assert p.stat().st_size==item['size_bytes'] and sha(p)==item['sha256']
    return value


def verify_smoke(path,expected_sha,calibration,calibration_sha,source_manifest_sha):
    assert path is not None and expected_sha is not None and sha(path)==expected_sha
    value=json.loads(Path(path).read_text())
    assert value['status']=='complete_synthetic_only' and value['phase']=='smoke'
    assert value['real_scores_computed'] is False and value['source_manifest_sha256']==source_manifest_sha
    assert value['calibration_sha256']==calibration_sha and value['feature_coefficient']==calibration['coefficient']
    assert value['new_optimizer_updates']==12 and len(value['records'])==12
    expected={(m,0,a,b) for m in ('eventgenesis','v2e') for a in ('clean_control','physical_supervised','paired_consistency') for b in ('depth','center')}
    assert {(r['method'],r['seed'],r['arm'],r['branch']) for r in value['records']}==expected
    assert all(r['updates']==1 and r['new_optimizer_updates']==1 and r['teacher_unchanged'] and r['independent_model_optimizer_rebuild'] and r['optimizer_state_equal'] for r in value['records'])
    for r in value['records']:
        if r['arm']=='paired_consistency':
            assert r['canonical_arm']=='feature_retention' and r['feature_coefficient']==calibration['coefficient']
            assert r['features_frozen'] is False and r['dark_feature_retention'] is False
    return value
